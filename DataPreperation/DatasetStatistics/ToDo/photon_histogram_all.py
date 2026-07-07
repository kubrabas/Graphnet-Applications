#!/usr/bin/env python3
"""
Usage:
    python photon_histogram_all.py

For each event:
  - Energy from EventProperties (DAQ frame)
  - Total photon count = sum of pulses in Accepted_PulseMap (P frame)
  - Events with empty / missing Accepted_PulseMap are dropped
  - Plots average photon count per energy bin, annotated with N events per bin

Saves photon_avg_all.png next to this script.
"""

import os
import glob
import numpy as np
import matplotlib.pyplot as plt

from icecube import icetray, dataio, dataclasses, LeptonInjector
icetray.I3Logger.global_logger = icetray.I3NullLogger()

# ── Dataset config ─────────────────────────────────────────────────────────────
DATASETS = {
    "Muon":     {"path": "/project/6008051/pone_simulation/MC000008-nu_mu-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",    "format": "zst"},
    "Electron": {"path": "/project/6008051/pone_simulation/MC000009-nu_e-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",    "format": "zst"},
    "Tau":      {"path": "/project/6008051/pone_simulation/MC000010-nu_tau-2_6-LeptonInjector_PROPOSAL_clsim-v17.1/Generator",   "format": "zst"},
    "NC":       {"path": "/project/6008051/pone_simulation/MC000011-nu_NC-2_6-LeptonInjector_PROPOSAL_clsim_NC-v17.1/Generator", "format": "zst"},
}

COLORS = {
    "Muon":     "steelblue",
    "Electron": "tomato",
    "Tau":      "mediumseagreen",
    "NC":       "mediumpurple",
}

DECADE_EDGES = [2, 3, 4, 5, 6]
PULSEMAP_KEY = "Accepted_PulseMap"
# ──────────────────────────────────────────────────────────────────────────────


def count_photons(pulsemap):
    """Total number of pulses across all DOMs in an I3RecoPulseSeriesMap."""
    return sum(len(pulses) for pulses in pulsemap.values())


def collect_data(flavor, path, fmt):
    """
    Returns arrays of (energy, n_photons) for events that have
    at least one photon in Accepted_PulseMap.

    DAQ frame  → energy (EventProperties)
    P frame    → photon count (Accepted_PulseMap)
    """
    pattern = "*.i3" if fmt == "i3" else f"*.i3.{fmt}"
    files = sorted(glob.glob(os.path.join(path, "**", pattern), recursive=True))

    if not files:
        print(f"[{flavor}] No files found under '{path}'")
        return np.array([]), np.array([])

    print(f"\n=== {flavor}: {len(files)} files ===")

    energies  = []
    photons   = []

    total_daq        = 0
    total_no_ep      = 0   # DAQ frame missing EventProperties
    total_no_pulsemap = 0  # P frame missing Accepted_PulseMap
    total_empty      = 0   # Accepted_PulseMap present but 0 photons
    total_ok         = 0

    for fpath in files:
        f = dataio.I3File(fpath)

        while f.more():
            frame = f.pop_frame()
            if frame.Stop != icetray.I3Frame.DAQ:
                continue

            total_daq += 1

            if "EventProperties" not in frame:
                total_no_ep += 1
                continue

            if PULSEMAP_KEY not in frame:
                total_no_pulsemap += 1
                continue

            pm = frame[PULSEMAP_KEY]
            n  = count_photons(pm)

            if n == 0:
                total_empty += 1
                continue

            energies.append(frame["EventProperties"].totalEnergy)
            photons.append(n)
            total_ok += 1

        f.close()

    print(f"  DAQ frames           : {total_daq}")
    print(f"  Missing EventProps   : {total_no_ep}")
    print(f"  Missing Pulsemap     : {total_no_pulsemap}")
    print(f"  Empty Pulsemap (0 ph): {total_empty}")
    print(f"  Events kept          : {total_ok}")
    print(f"  Events dropped total : {total_daq - total_ok}")

    return np.array(energies), np.array(photons)


def plot_flavor(ax, flavor, energies, photons):
    color  = COLORS[flavor]
    log_E  = np.log10(energies)
    bins   = np.linspace(2, 6, 41)   # 0.1-wide bins → ~25% per step
    centers = 0.5 * (bins[:-1] + bins[1:])

    avg_photons = []
    n_events    = []

    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (log_E >= lo) & (log_E < hi)
        n    = mask.sum()
        avg  = photons[mask].mean() if n > 0 else 0.0
        avg_photons.append(avg)
        n_events.append(n)

    avg_photons = np.array(avg_photons)
    n_events    = np.array(n_events)

    # Bar plot
    width = bins[1] - bins[0]
    ax.bar(centers, avg_photons, width=width * 0.85, color=color, alpha=0.7, edgecolor=color)

    total_kept = len(energies)
    y_max = avg_photons.max()
    ax.set_ylim(0, y_max * 1.55)

    # Decade boundary lines + per-decade annotation (staggered heights)
    y_levels = [1.12, 1.35, 1.12, 1.35]
    for i, (lo, hi) in enumerate(zip(DECADE_EDGES[:-1], DECADE_EDGES[1:])):
        if i > 0:
            ax.axvline(lo, color="gray", linestyle="--", linewidth=1, alpha=0.7)

        mask  = (log_E >= lo) & (log_E < hi)
        n_dec = mask.sum()
        avg_dec = photons[mask].mean() if n_dec > 0 else 0.0
        x_mid   = (lo + hi) / 2

        ax.text(
            x_mid, y_max * y_levels[i],
            f"avg detected photons={avg_dec:.0f}\nN={n_dec:,}",
            ha="center", va="bottom", fontsize=8, color=color, fontweight="bold",
            bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=color, alpha=0.85),
        )

    ax.set_xticks(DECADE_EDGES)
    ax.set_xticklabels(["100 GeV", "1 TeV", "10 TeV", "100 TeV", "1 PeV"], fontsize=9)
    ax.set_xlabel("Energy", fontsize=11)
    ax.set_ylabel("Avg photons detected", fontsize=11)
    ax.set_title(f"{flavor}  -  {total_kept:,} events with ≥1 photon", fontsize=11)
    ax.set_xlim(2, 6)


def plot_flavor_zoom(ax, flavor, energies, photons):
    """100 GeV – 1 TeV zoom, 0.1-wide log bins."""
    color = COLORS[flavor]
    log_E = np.log10(energies)

    mask      = (log_E >= 2) & (log_E < 3)
    log_E_z   = log_E[mask]
    photons_z = photons[mask]
    total_z   = mask.sum()

    if total_z == 0:
        ax.set_title(f"{flavor}  -  no events in range")
        return

    bins    = np.linspace(2, 3, 11)   # 0.1-wide bins
    centers = 0.5 * (bins[:-1] + bins[1:])

    avg_photons = []
    for lo, hi in zip(bins[:-1], bins[1:]):
        m   = (log_E_z >= lo) & (log_E_z < hi)
        avg = photons_z[m].mean() if m.sum() > 0 else 0.0
        avg_photons.append(avg)

    avg_photons = np.array(avg_photons)
    width = bins[1] - bins[0]
    ax.bar(centers, avg_photons, width=width * 0.85, color=color, alpha=0.7, edgecolor=color)

    ax.set_xticks([2.0, 2.2, 2.4, 2.6, 2.8, 3.0])
    ax.set_xticklabels(["100 GeV", "158 GeV", "251 GeV", "398 GeV", "631 GeV", "1 TeV"], fontsize=9)
    ax.set_xlabel("Energy", fontsize=11)
    ax.set_ylabel("Avg photons detected", fontsize=11)
    ax.set_title(f"{flavor}  -  {total_z:,} events (100 GeV-1 TeV)", fontsize=11)
    ax.set_xlim(2, 3)
    ax.set_ylim(0, avg_photons.max() * 1.15 if avg_photons.max() > 0 else 1)


if __name__ == "__main__":
    # ── Collect all data once ─────────────────────────────────────────────────
    all_energies = {}
    all_photons  = {}
    for flavor, info in DATASETS.items():
        e, p = collect_data(flavor, info["path"], info["format"])
        all_energies[flavor] = e
        all_photons[flavor]  = p

    # ── Plot 1: full range ────────────────────────────────────────────────────
    fig1, axes1 = plt.subplots(2, 2, figsize=(14, 9))
    for ax, flavor in zip(axes1.flatten(), DATASETS):
        if len(all_energies[flavor]) == 0:
            ax.set_title(f"{flavor}  -  no data")
            continue
        plot_flavor(ax, flavor, all_energies[flavor], all_photons[flavor])

    fig1.suptitle(
        f"Avg detected photons per energy bin - MC000008-11  (Spring 2026, full geometry layout)\n"
        f"Events with 0 photons in {PULSEMAP_KEY} are excluded",
        fontsize=12,
    )
    fig1.tight_layout()
    out1 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photon_avg_all.png")
    fig1.savefig(out1, dpi=150)
    print(f"\nSaved: {out1}")

    # ── Plot 2: zoom 100 GeV – 1 TeV ─────────────────────────────────────────
    fig2, axes2 = plt.subplots(2, 2, figsize=(14, 9))
    for ax, flavor in zip(axes2.flatten(), DATASETS):
        if len(all_energies[flavor]) == 0:
            ax.set_title(f"{flavor}  -  no data")
            continue
        plot_flavor_zoom(ax, flavor, all_energies[flavor], all_photons[flavor])

    fig2.suptitle(
        f"Avg detected photons - 100 GeV to 1 TeV - MC000008-11  (Spring 2026, full geometry layout)\n"
        f"Events with 0 photons in {PULSEMAP_KEY} are excluded",
        fontsize=12,
    )
    fig2.tight_layout()
    out2 = os.path.join(os.path.dirname(os.path.abspath(__file__)), "photon_avg_100GeV_1TeV.png")
    fig2.savefig(out2, dpi=150)
    print(f"Saved: {out2}")

    plt.show()