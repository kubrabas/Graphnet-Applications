"""
Compute astrophysical neutrino weights for one Spring 2026 MC flavor.

Called by SLURM via submit_astro_weights.sh; can also be run locally:
    python compute_weights.py --flavor Muon --indir /path/to/Generator --format zst --outdir /out

Reads all LeptonInjector Generator files for the given flavor, computes
per-event w_astro [Hz] under the IceCube 2017 power-law flux, and saves:
    {outdir}/spring2026mc_{flavor.lower()}_astro_weights.npz
with arrays: n_hits, n_strings, true_energy [GeV], true_zenith [rad], w_astro [Hz].

Flux model (Aartsen+2016, ApJ 833, 3):
    Phi(E) = 1.44e-18 * (E / 1e5 GeV)^{-2.19}  [GeV^-1 cm^-2 s^-1 sr^-1]

Cross section (Connolly+2011, PRD 83, 113009, ALLM parametrisation):
    sigma_CC(E) = 2.363e-39 * (E/GeV)^0.402  cm^2/nucleon
    sigma_NC    = sigma_CC * 0.285
"""

import argparse
import glob
import os
import sys
import time
from pathlib import Path

import numpy as np

from icecube import icetray, dataio, dataclasses, LeptonInjector  # noqa: F401

icetray.I3Logger.global_logger = icetray.I3NullLogger()

# ── Flux model ────────────────────────────────────────────────────────────────

_PHI_0 = 1.44e-18   # GeV^-1 cm^-2 s^-1 sr^-1
_E_0   = 1e5        # GeV
_GAMMA = 2.19


def phi_astro(E_GeV):
    """
    Single power-law astrophysical flux, per flavor, nu + nubar.

    Aartsen+2016 ApJ 833, 3  [GeV^-1 cm^-2 s^-1 sr^-1]
    """
    return _PHI_0 * (E_GeV / _E_0) ** (-_GAMMA)


# ── Cross-section parametrisation ─────────────────────────────────────────────
# Connolly+2011 (PRD 83, 113009), Table 2, ALLM, per-nucleon, valid E > 100 GeV

_SIGMA_0_CC  = 2.363e-39   # cm^2
_SIGMA_ALPHA = 0.402
_R_NC_CC     = 0.285       # sigma_NC / sigma_CC  (SM high-E ratio)

_NU_TYPES = {
    dataclasses.I3Particle.NuE,
    dataclasses.I3Particle.NuEBar,
    dataclasses.I3Particle.NuMu,
    dataclasses.I3Particle.NuMuBar,
    dataclasses.I3Particle.NuTau,
    dataclasses.I3Particle.NuTauBar,
}


def _sigma(E_GeV, is_nc):
    sig_cc = _SIGMA_0_CC * (E_GeV ** _SIGMA_ALPHA)
    return sig_cc * (_R_NC_CC if is_nc else 1.0)


# ── Physical constants ────────────────────────────────────────────────────────

_N_A     = 6.022e23
_A_WATER = 18.015     # g/mol
_RHO_ICE = 0.917      # g/cm^3
_N_NUC   = _RHO_ICE * _N_A / _A_WATER   # nucleons/cm^3


# ── Generation-weight helpers ─────────────────────────────────────────────────

def _energy_integral(E_min, E_max, gamma):
    """Integral of E^{-gamma} from E_min to E_max."""
    if abs(gamma - 1.0) < 1e-9:
        return float(np.log(E_max / E_min))
    return float((E_max ** (1.0 - gamma) - E_min ** (1.0 - gamma)) / (1.0 - gamma))


def compute_event_weight(ep, li_props):
    """
    Per-event astrophysical weight [Hz].

    For volume injection (Electron, Tau, NC):
        w = Phi(E) * sigma(E) * N_nuc * V_cyl * Omega / (n_events * pdf_E)

    For ranged injection (Muon):
        w = Phi(E) * sigma(E) * (N_A * X_col / A_water) * A_disk * Omega
            / (n_events * pdf_E)
        X_col from ep.columnDepth [g/cm^2] if available;
        fallback: N_nuc * 2 * endcapLength.
    """
    E         = float(ep.totalEnergy)
    gamma_inj = float(li_props.powerlawIndex)
    E_min     = float(li_props.energyMinimum)
    E_max     = float(li_props.energyMaximum)
    n_events  = int(li_props.events)
    theta_min = float(li_props.zenithMinimum)
    theta_max = float(li_props.zenithMaximum)

    Omega      = 2.0 * np.pi * abs(np.cos(theta_min) - np.cos(theta_max))
    E_integral = _energy_integral(E_min, E_max, gamma_inj)
    pdf_E      = E ** (-gamma_inj) / E_integral

    is_nc = (ep.finalType1 in _NU_TYPES)
    sig   = _sigma(E, is_nc)

    is_ranged = hasattr(li_props, "injectionRadius")

    if is_ranged:
        R_inj_cm = float(li_props.injectionRadius) * 100.0
        A_disk   = np.pi * R_inj_cm ** 2
        if hasattr(ep, "columnDepth") and float(ep.columnDepth) > 0.0:
            N_target_per_area = _N_A * float(ep.columnDepth) / _A_WATER
        else:
            endcap_cm         = float(li_props.endcapLength) * 100.0
            N_target_per_area = _N_NUC * 2.0 * endcap_cm
        w = (phi_astro(E) * sig * N_target_per_area
             * A_disk * Omega / (n_events * pdf_E))
    else:
        R_cyl_cm = float(li_props.cylinderRadius) * 100.0
        H_cyl_cm = float(li_props.cylinderHeight) * 100.0
        V_cyl    = np.pi * R_cyl_cm ** 2 * H_cyl_cm
        w = (phi_astro(E) * sig * _N_NUC
             * V_cyl * Omega / (n_events * pdf_E))

    return float(w)


def _hit_stats(pulse_map):
    if pulse_map is None or len(pulse_map) == 0:
        return 0, 0
    n_hits    = sum(len(v) for v in pulse_map.values())
    n_strings = len({k.string for k in pulse_map.keys()})
    return n_hits, n_strings


# ── Main ──────────────────────────────────────────────────────────────────────

LIVETIME_1YR = 365.25 * 24.0 * 3600.0


def main() -> int:
    ap = argparse.ArgumentParser(description="Compute per-event astrophysical weights")
    ap.add_argument("--flavor",  required=True,
                    choices=["Muon", "Electron", "Tau", "NC"])
    ap.add_argument("--indir",   required=True,  help="Folder with Generator i3 files")
    ap.add_argument("--format",  required=True,  choices=["zst", "gz"],
                    help="File extension (zst or gz)")
    ap.add_argument("--outdir",  required=True,  help="Output directory for .npz file")
    ap.add_argument("--task-id", type=int, default=None,
                    help="Process only this file index (0-based); for local testing")
    args = ap.parse_args()

    pattern  = os.path.join(args.indir, "**", f"*.i3.{args.format}")
    i3_files = sorted(glob.glob(pattern, recursive=True))

    if not i3_files:
        print(f"ERROR: no *.i3.{args.format} files found in {args.indir}")
        return 1

    if args.task_id is not None:
        if not (0 <= args.task_id < len(i3_files)):
            print(f"ERROR: task-id {args.task_id} out of range (0..{len(i3_files)-1})")
            return 1
        i3_files = [i3_files[args.task_id]]

    Path(args.outdir).mkdir(parents=True, exist_ok=True)
    out_path = os.path.join(args.outdir,
                            f"spring2026mc_{args.flavor.lower()}_astro_weights.npz")

    print(f"=== ASTRO WEIGHTS: {args.flavor} ===")
    print(f"input dir  : {args.indir}")
    print(f"files      : {len(i3_files)}")
    print(f"output     : {out_path}")
    print(f"flux       : Phi_0={_PHI_0:.2e}, E_0={_E_0:.0e} GeV, gamma={_GAMMA}")
    print(f"sigma_CC   : {_SIGMA_0_CC:.3e} * E^{_SIGMA_ALPHA}  cm^2/nucleon")
    sys.stdout.flush()

    t0 = time.time()

    n_hits_list    = []
    n_strings_list = []
    energies       = []
    zeniths        = []
    weights        = []
    n_skipped      = 0

    for i, i3f in enumerate(i3_files):
        li_props = None
        f = dataio.I3File(i3f)
        while f.more():
            frame = f.pop_frame()

            if li_props is None and frame.Has("LeptonInjectorProperties"):
                li_props = frame["LeptonInjectorProperties"]

            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            if not frame.Has("EventProperties"):
                n_skipped += 1
                continue
            if li_props is None:
                n_skipped += 1
                continue

            ep = frame["EventProperties"]
            w  = compute_event_weight(ep, li_props)

            n_h, n_s = _hit_stats(frame.Get("Accepted_PulseMap"))

            n_hits_list.append(n_h)
            n_strings_list.append(n_s)
            energies.append(float(ep.totalEnergy))
            zeniths.append(float(ep.zenith))
            weights.append(w)
        f.close()

        if (i + 1) % 10 == 0 or (i + 1) == len(i3_files):
            print(f"  [{i + 1}/{len(i3_files)}]  events so far: {len(weights)}"
                  f"  elapsed: {time.time() - t0:.0f}s")
            sys.stdout.flush()

    n_hits      = np.array(n_hits_list,    dtype=np.int32)
    n_strings   = np.array(n_strings_list, dtype=np.int32)
    true_energy = np.array(energies,       dtype=np.float64)
    true_zenith = np.array(zeniths,        dtype=np.float64)
    w_astro     = np.array(weights,        dtype=np.float64)

    np.savez(out_path,
             n_hits=n_hits,
             n_strings=n_strings,
             true_energy=true_energy,
             true_zenith=true_zenith,
             w_astro=w_astro)

    total_rate = float(np.sum(w_astro))
    print(f"\n--- Results ---")
    print(f"events processed : {len(w_astro)}  (skipped {n_skipped})")
    print(f"total rate       : {total_rate:.3e} Hz")
    print(f"events / 1 yr    : {total_rate * LIVETIME_1YR:.1f}")
    print(f"saved to         : {out_path}")
    print(f"elapsed          : {time.time() - t0:.0f}s")
    print(f"=== DONE ===")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
