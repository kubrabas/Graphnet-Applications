"""
Calculate LeptonInjector one-weights for ALL batch IDs in a single Slurm job.

Processes files in parallel using multiprocessing. Outputs:
  <outdir>/<Flavor>_LIW.csv           — final per-event weights
  <logdir>/<Flavor>_file_stats.csv    — per-file processing stats
  <logdir>/<Flavor>_compute.out       — run log

The output keeps the raw LeptonWeighter oneweight and oneweight_x100 columns.

Usage:
    python3 calculate_LIW.py --mc 340StringMC --flavor Muon \\
        --lic-dir /path/to/Generator --photon-dir /path/to/Photon \\
        --photon-pattern '*.i3' --outdir /tmp/out --logdir /tmp/log \\
        [--workers 8] [--overwrite]
"""

import argparse
import importlib.util
import math
import multiprocessing
import os
import re
import struct
import sys
import time
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from icecube import dataio, icetray
from icecube import LeptonInjector  # noqa: F401 – registers EventProperties
from icecube import simclasses      # noqa: F401
import LeptonWeighter as LW
import pandas as pd

icetray.I3Logger.global_logger = icetray.I3NullLogger()

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

XS_PATH = "/project/6008051/pone_simulation/pone_offline/CrossSectionModels/csms_differential_v1.0/"
PATHS_PY = Path(__file__).resolve().parents[3] / "Metadata" / "paths.py"
DEFAULT_NUSQUIDS_WEIGHT_TABLE = (
    "/cvmfs/software.pacific-neutrino.org/pone_offline/v2.0/"
    "data/nsq_allneu_propagation_weight_gamma2.h5"
)

VALID_FLAVORS = {"muon", "electron", "tau", "nc"}
N_GEN_PER_FILE = 100  # generated events per particle type per LIC file
EXPECTED_EVENTS_PER_FILE = 2 * N_GEN_PER_FILE
DEFAULT_INJECTION_RADIUS = 500.0
DEFAULT_INJECTION_MODE = "unknown"
DEFAULT_INJECTION_SOURCE = "default"
LIC_SIZE_T_BYTES = 8

LW_PARTICLE_FROM_PDG = {
    12: LW.NuE,
    -12: LW.NuEBar,
    14: LW.NuMu,
    -14: LW.NuMuBar,
    16: LW.NuTau,
    -16: LW.NuTauBar,
    11: LW.EMinus,
    -11: LW.EPlus,
    13: LW.MuMinus,
    -13: LW.MuPlus,
    15: LW.TauMinus,
    -15: LW.TauPlus,
    -2000001006: LW.Hadrons,
}

# Per-worker cross-section object; initialised once per worker process.
_xs = None
_survival = None


# ---------------------------------------------------------------------------
# Shared helpers
# ---------------------------------------------------------------------------

def to_lw_particle(particle_type):
    particle_type = int(particle_type)
    if particle_type not in LW_PARTICLE_FROM_PDG:
        raise ValueError(f"Unsupported particle type for LeptonWeighter: {particle_type}")
    return LW_PARTICLE_FROM_PDG[particle_type]


def get_lw_particles_from_event_properties(props):
    primary = to_lw_particle(props.initialType)
    fs0 = to_lw_particle(props.finalType1)
    fs1 = to_lw_particle(props.finalType2)
    return primary, fs0, fs1


def extract_batch_id(filepath: Path) -> Optional[str]:
    m = re.search(r'(\d+)(?:\.[^.]+)+$', filepath.name)
    return m.group(1) if m else None


def build_id_map(directory: str, *patterns: str) -> Dict[str, Path]:
    base = Path(directory)
    if not base.exists():
        raise FileNotFoundError(f"Directory not found: {base}")
    result: Dict[str, Path] = {}
    for pattern in patterns:
        for f in sorted(base.rglob(pattern)):
            bid = extract_batch_id(f)
            if bid is not None and bid not in result:
                result[bid] = f
    return result


def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bad_file_rule(i3_file: Path) -> Tuple[str, Optional[int]]:
    target = str(i3_file.resolve())
    bad_i3_files = getattr(load_paths(), "BAD_I3_FILES", {})

    for flavors in bad_i3_files.values():
        for info in flavors.values():
            if target in set(map(str, info.get("no_daq_for_some_reason", set()))):
                return "skip_no_daq", None

            available_daq_counts = info.get("available_daq_counts", {})
            if target in available_daq_counts:
                return "limit_daq", int(available_daq_counts[target])

    return "not_a_bad_file", None


def load_xs():
    # Use physical nu/nubar cross sections for flux-free effective-area weights.
    return LW.CrossSectionFromSpline(
        XS_PATH + "dsdxdy_nu_CC_iso.fits",
        XS_PATH + "dsdxdy_nubar_CC_iso.fits",
        XS_PATH + "dsdxdy_nu_NC_iso.fits",
        XS_PATH + "dsdxdy_nubar_NC_iso.fits",
    )


class SurvivalProbability:
    def __init__(self, table_path: str):
        import nuSQuIDS as nsq

        self.nsq_atm = nsq.nuSQUIDSAtm(table_path)
        self.units = nsq.Const()
        cosrange = [cth for cth in self.nsq_atm.GetCosthRange()]
        energyrange = [energy for energy in self.nsq_atm.GetERange()]
        self.cosmax = max(cosrange)
        self.cosmin = min(cosrange)
        self.emax = max(energyrange)
        self.emin = min(energyrange)

    @staticmethod
    def _nusquids_index(primary_type):
        type_index = 0
        anti_index = 0
        if primary_type in (LW.NuMu, LW.NuMuBar):
            type_index = 1
        elif primary_type in (LW.NuTau, LW.NuTauBar):
            type_index = 2
        if primary_type in (LW.NuEBar, LW.NuMuBar, LW.NuTauBar):
            anti_index = 1
        return type_index, anti_index

    @staticmethod
    def _eflux(energy):
        return 1e18 * energy ** (-2.0)

    def __call__(self, energy_gev: float, zenith: float, primary_type) -> float:
        nusquids_energy = energy_gev * self.units.GeV
        cos_zenith = math.cos(zenith)
        type_index, anti_index = self._nusquids_index(primary_type)

        if self.cosmin < cos_zenith < self.cosmax:
            if self.emin < nusquids_energy < self.emax:
                return (
                    self.nsq_atm.EvalFlavor(
                        type_index, cos_zenith, nusquids_energy, anti_index
                    )
                    / self._eflux(nusquids_energy)
                )
            if nusquids_energy <= self.emin:
                return 1.0
            return 0.0

        if self.emin < nusquids_energy < self.emax:
            return (
                self.nsq_atm.EvalFlavor(type_index, 0.0, nusquids_energy, anti_index)
                / self._eflux(nusquids_energy)
            )

        return 0.0


def append_error(existing: str, new: str) -> str:
    if not new:
        return existing
    if not existing:
        return new
    return f"{existing} | {new}"


def read_lic_injection_configs(lic_path: str) -> List[dict]:
    configs = []
    data = Path(lic_path).read_bytes()
    offset = 0

    while offset + 17 <= len(data):
        block_start = offset
        try:
            block_len = struct.unpack_from("<Q", data, offset)[0]
            offset += 8
            name_len = struct.unpack_from("<Q", data, offset)[0]
            offset += LIC_SIZE_T_BYTES
            name = data[offset:offset + name_len].decode("utf-8", errors="replace")
            offset += name_len
            offset += 1  # block type version
        except (struct.error, UnicodeDecodeError):
            break

        payload_len = block_len - (17 + name_len)
        payload_start = offset
        payload_end = payload_start + payload_len
        if block_len < 17 + name_len or payload_end > len(data):
            break

        payload = data[payload_start:payload_end]
        if name in ("RangedInjectionConfiguration", "VolumeInjectionConfiguration"):
            if len(payload) < 16:
                offset = block_start + block_len
                continue
            mode = "range" if name == "RangedInjectionConfiguration" else "volume"
            radius = struct.unpack_from("<d", payload, len(payload) - 16)[0]
            configs.append({
                "mode": mode,
                "radius": radius,
                "block_type": name,
            })

        offset = block_start + block_len

    return configs


def get_lic_injection_config(lic_path: str) -> Tuple[str, float, str]:
    configs = read_lic_injection_configs(lic_path)
    unique = {
        (config["mode"], config["radius"])
        for config in configs
    }

    if not unique:
        return DEFAULT_INJECTION_MODE, DEFAULT_INJECTION_RADIUS, DEFAULT_INJECTION_SOURCE

    if len(unique) == 1:
        mode, radius = next(iter(unique))
        return mode, radius, "lic"

    return DEFAULT_INJECTION_MODE, DEFAULT_INJECTION_RADIUS, DEFAULT_INJECTION_SOURCE


def get_injection_config(
    frame,
    current_mode: str,
    current_radius: float,
    current_source: str,
) -> Tuple[str, float, str]:
    if not frame.Has("LeptonInjectorProperties"):
        return current_mode, current_radius, current_source

    props = frame["LeptonInjectorProperties"]
    try:
        return "range", props.injectionRadius, "LeptonInjectorProperties"
    except AttributeError:
        return "volume", props.cylinderRadius, "LeptonInjectorProperties"


def read_event_properties(i3_file: Path, max_daq_frames: int = 0) -> Tuple[Dict, str]:
    props = {}
    error = ""
    opened = False
    daq_count = 0
    try:
        f = dataio.I3File(str(i3_file))
        opened = True
        while f.more():
            try:
                frame = f.pop_frame()
            except RuntimeError as e:
                error = (
                    "i3 file problem: read error while reading EventProperties "
                    f"at event {len(props) + 1}. {len(props)} EventProperties "
                    f"entries were read before the problem. Error: {e}"
                )
                break
            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            daq_count += 1
            if not frame.Has("EventProperties") or not frame.Has("I3EventHeader"):
                if max_daq_frames > 0 and daq_count >= max_daq_frames:
                    break
                continue
            hdr = frame["I3EventHeader"]
            eid = (hdr.run_id, hdr.sub_run_id, hdr.event_id, hdr.sub_event_id)
            props[eid] = frame["EventProperties"]
            if max_daq_frames > 0 and daq_count >= max_daq_frames:
                break
        try:
            f.close()
        except RuntimeError as e:
            error = append_error(
                error,
                f"i3 file problem: error while closing EventProperties i3 file. Error: {e}",
            )
    except RuntimeError as e:
        if opened:
            error = (
                "i3 file problem: stream error while reading EventProperties "
                f"at event {len(props) + 1}. {len(props)} EventProperties "
                f"entries were read before the problem. Error: {e}"
            )
        else:
            error = f"i3 file problem: could not open EventProperties i3 file. Error: {e}"
    return props, error


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_init(compute_survival: bool, nusquids_weight_tables: str):
    global _xs, _survival
    _xs = load_xs()
    _survival = (
        SurvivalProbability(nusquids_weight_tables)
        if compute_survival else None
    )


def _process_one(args: tuple) -> tuple:
    """Process one batch. Returns (batch_id, records, stat_dict)."""
    batch_id, lic_path, i3_path = args
    t0 = time.time()

    stat = {
        "batch_id": batch_id,
        "lic_file": lic_path,
        "i3_file": i3_path,
        "status": "not ok",
        "error": "",
        "i3_file_opened_ok": False,
        "bad_file_rule": "not_a_bad_file",
        "max_daq_frames": 0,
        "weights_calculated": 0,
        "duration_s": 0.0,
    }

    rule, available_daq_count = bad_file_rule(Path(i3_path))
    stat["bad_file_rule"] = rule
    if rule == "skip_no_daq":
        stat["status"] = "skipped"
        stat["error"] = "input file is listed in paths.BAD_I3_FILES as no_daq_for_some_reason"
        stat["duration_s"] = time.time() - t0
        return batch_id, [], stat

    max_daq_frames = available_daq_count if rule == "limit_daq" else 0
    stat["max_daq_frames"] = max_daq_frames

    event_props, event_props_error = read_event_properties(Path(i3_path), max_daq_frames)
    stat["error"] = event_props_error
    if not event_props:
        if not stat["error"]:
            stat["error"] = (
                "i3 file problem: no DAQ frames with EventProperties and "
                "I3EventHeader were found in the EventProperties i3 file."
            )
        stat["duration_s"] = time.time() - t0
        return batch_id, [], stat

    try:
        generators = LW.MakeGeneratorsFromLICFile(lic_path)
        weighter = LW.Weighter(_xs, generators)
        lic_injection_mode, lic_injection_radius, lic_injection_source = (
            get_lic_injection_config(lic_path)
        )
    except Exception as e:
        stat["status"] = "not ok"
        stat["error"] = append_error(
            stat["error"],
            f"LIC file problem: could not build LeptonWeighter generators from the LIC file. Error: {e}",
        )
        stat["duration_s"] = time.time() - t0
        return batch_id, [], stat

    records = []
    weight_calculated = 0

    i3_error = ""

    try:
        f = dataio.I3File(i3_path)
        stat["i3_file_opened_ok"] = True
        daq_count = 0
        injection_mode = lic_injection_mode
        injection_radius = lic_injection_radius
        injection_source = lic_injection_source
        while f.more():
            try:
                frame = f.pop_frame()
            except RuntimeError as e:
                problem_event = weight_calculated + 1
                i3_error = (
                    f"i3 file problem at event {problem_event}: read error. "
                    f"{weight_calculated} event weights were calculated before the problem. Error: {e}"
                )
                break

            injection_mode, injection_radius, injection_source = get_injection_config(
                frame, injection_mode, injection_radius, injection_source
            )

            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            daq_count += 1
            if not frame.Has("I3EventHeader"):
                if max_daq_frames > 0 and daq_count >= max_daq_frames:
                    break
                continue

            hdr = frame["I3EventHeader"]
            eid = (hdr.run_id, hdr.sub_run_id, hdr.event_id, hdr.sub_event_id)

            if eid not in event_props:
                if max_daq_frames > 0 and daq_count >= max_daq_frames:
                    break
                continue

            props = event_props[eid]

            try:
                primary, fs0, fs1 = get_lw_particles_from_event_properties(props)
            except ValueError:
                continue

            event = LW.Event()
            event.energy = props.totalEnergy
            event.zenith = props.zenith
            event.azimuth = props.azimuth
            event.interaction_x = props.finalStateX
            event.interaction_y = props.finalStateY
            event.primary_type = primary
            event.final_state_particle_0 = fs0
            event.final_state_particle_1 = fs1
            event.radius = injection_radius
            event.total_column_depth = props.totalColumnDepth
            event.x = props.x
            event.y = props.y
            event.z = props.z

            try:
                oneweight = weighter.get_oneweight(event)
                survival_prob = (
                    _survival(event.energy, event.zenith, event.primary_type)
                    if _survival is not None else None
                )
            except Exception as e:
                problem_event = weight_calculated + 1
                i3_error = (
                    f"weighting problem at event {problem_event}: oneweight could not be computed. "
                    f"{weight_calculated} event weights were calculated before the problem. Error: {e}"
                )
                break

            records.append({
                "RunID": hdr.run_id,
                "SubrunID": hdr.sub_run_id,
                "EventID": hdr.event_id,
                "SubEventID": hdr.sub_event_id,
                "energy": props.totalEnergy,
                "zenith": props.zenith,
                "azimuth": props.azimuth,
                "finalStateX": props.finalStateX,
                "finalStateY": props.finalStateY,
                "columnDepth": props.totalColumnDepth,
                "injectionMode": injection_mode,
                "injectionRadius": injection_radius,
                "injectionConfigSource": injection_source,
                "initialType": int(props.initialType),
                "finalType1": int(props.finalType1),
                "finalType2": int(props.finalType2),
                "oneweight": oneweight,
            })
            if survival_prob is not None:
                records[-1]["survivalProb"] = survival_prob
            weight_calculated += 1
            if max_daq_frames > 0 and daq_count >= max_daq_frames:
                break

        f.close()

    except RuntimeError as e:
        if stat["i3_file_opened_ok"]:
            i3_error = append_error(
                i3_error,
                f"i3 file problem after {weight_calculated} calculated event weights: stream error. Error: {e}",
            )
        else:
            i3_error = f"i3 file problem: could not open i3 file. Error: {e}"

    stat["error"] = append_error(stat["error"], i3_error)

    if weight_calculated == EXPECTED_EVENTS_PER_FILE:
        stat["status"] = "ok"
        stat["error"] = ""
    elif weight_calculated > 0:
        stat["status"] = "partially ok"
        if not stat["error"]:
            if max_daq_frames > 0:
                stat["error"] = (
                    "processed only first "
                    f"{max_daq_frames} DAQ frames per paths.BAD_I3_FILES"
                )
            else:
                stat["error"] = "unknown"
    else:
        stat["status"] = "not ok"
        if not stat["error"]:
            stat["error"] = "unknown"

    stat["weights_calculated"] = weight_calculated
    stat["duration_s"] = time.time() - t0
    return batch_id, records, stat


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--mc", required=True)
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--lic-dir", required=True)
    ap.add_argument("--photon-dir", required=True)
    ap.add_argument("--photon-pattern", required=True, help="e.g. '*.i3' or '*.i3.zst'")
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--workers", type=int, default=None,
                    help="parallel workers (default: all available CPUs)")
    ap.add_argument("--no-survival-prob", dest="survival_prob",
                    action="store_false",
                    help="do not add P-ONE-style nuSQuIDS survivalProb column")
    ap.add_argument("--nusquids-weight-tables", default=DEFAULT_NUSQUIDS_WEIGHT_TABLE,
                    help="nuSQuIDS propagation table for survivalProb")
    ap.set_defaults(survival_prob=True)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.flavor.lower() not in VALID_FLAVORS:
        print(f"ERROR: unknown flavor '{args.flavor}'. Choices: {sorted(VALID_FLAVORS)}")
        return 1

    outdir  = Path(args.outdir)
    logdir  = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    outfile   = outdir / f"{args.flavor}_LIW.csv"
    statsfile = logdir / f"{args.flavor}_file_stats.csv"
    logfile   = logdir / f"{args.flavor}_compute.out"

    if outfile.exists() and not args.overwrite:
        print(f"[skip] output already exists: {outfile}  (use --overwrite to reprocess)")
        return 0

    workers = args.workers or os.cpu_count() or 1

    log_fh = open(logfile, "w")

    def log(msg: str) -> None:
        print(msg, file=log_fh, flush=True)
        print(msg, flush=True)

    t_job_start = time.time()
    log("=== LIW JOB STARTED ===")
    log(f"job_id    : {os.environ.get('SLURM_JOB_ID', 'unknown')}")
    log(f"host      : {os.uname().nodename}")
    log(f"mc        : {args.mc}")
    log(f"flavor    : {args.flavor}")
    log(f"workers   : {workers}")
    log(f"lic_dir   : {args.lic_dir}")
    log(f"photon_dir: {args.photon_dir}")
    log(f"pattern   : {args.photon_pattern}")
    log(f"survival  : {args.survival_prob}")
    if args.survival_prob:
        log(f"nsq table : {args.nusquids_weight_tables}")
    log(f"outfile   : {outfile}")
    log(f"statsfile : {statsfile}")
    log(f"logfile   : {logfile}")

    # ------------------------------------------------------------------
    # Build file maps once in the main process
    # ------------------------------------------------------------------
    try:
        lic_map = build_id_map(args.lic_dir, "*.lic")
        i3_map = build_id_map(args.photon_dir, args.photon_pattern)
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        log_fh.close()
        return 1

    common = sorted(set(lic_map) & set(i3_map))
    log(f"\nbatches   : {len(common)} (lic={len(lic_map)} i3={len(i3_map)})")

    i3_without_lic = sorted(set(i3_map) - set(lic_map))
    if i3_without_lic:
        preview = ", ".join(i3_without_lic[:10])
        suffix = " ..." if len(i3_without_lic) > 10 else ""
        log(
            "WARNING: "
            f"{len(i3_without_lic)} i3 files do not have a matching LIC file. "
            f"They will not be processed. batch_id examples: {preview}{suffix}"
        )

    # ------------------------------------------------------------------
    # Build task list
    # ------------------------------------------------------------------
    tasks: List[tuple] = []
    all_stats: List[dict] = []

    for bid in common:
        tasks.append((bid, str(lic_map[bid]), str(i3_map[bid])))

    log(f"tasks     : {len(tasks)} to process")

    if not tasks:
        log("ERROR: no tasks to process.")
        log_fh.close()
        return 1

    # ------------------------------------------------------------------
    # Process in parallel
    # ------------------------------------------------------------------
    all_records: List[dict] = []
    progress_step = max(1, len(tasks) // 20)

    with multiprocessing.Pool(
        workers,
        initializer=_worker_init,
        initargs=(args.survival_prob, args.nusquids_weight_tables),
    ) as pool:
        for i, (batch_id, records, stat) in enumerate(
            pool.imap_unordered(_process_one, tasks), 1
        ):
            all_records.extend(records)
            all_stats.append(stat)
            if i % progress_step == 0 or i == len(tasks):
                log(f"progress: {i} / {len(tasks)} tasks processed")

    # ------------------------------------------------------------------
    # Write file-level stats CSV
    # ------------------------------------------------------------------
    stats_df = pd.DataFrame(all_stats).sort_values("batch_id").reset_index(drop=True)
    stats_df.to_csv(str(statsfile), index=False)

    n_ok_files = (stats_df["status"] == "ok").sum()
    n_partially_ok_files = (stats_df["status"] == "partially ok").sum()
    n_not_ok_files = (stats_df["status"] == "not ok").sum()
    n_skipped_files = (stats_df["status"] == "skipped").sum()
    n_total_files = len(common)
    n_weights_calculated = int(stats_df["weights_calculated"].sum())
    log(f"\n--- Summary ---")
    log(f"files ok           : {n_ok_files} / {n_total_files}")
    log(f"files partially ok : {n_partially_ok_files}")
    log(f"files not ok       : {n_not_ok_files}")
    log(f"files skipped      : {n_skipped_files}")
    log(f"weights calculated : {n_weights_calculated}")
    log(f"stats CSV written  : {statsfile}")
    log("per-file details   : see stats CSV")

    # ------------------------------------------------------------------
    # Write final LIW CSV
    # ------------------------------------------------------------------
    if not all_records:
        log("=== WARNING: no events processed successfully — output CSV not written ===")
        elapsed = time.time() - t_job_start
        log(f"=== FAILED  elapsed={elapsed:.1f}s ===")
        log_fh.close()
        return 1

    df = pd.DataFrame(all_records)
    df["oneweight_x100"] = df["oneweight"] * N_GEN_PER_FILE

    df.to_csv(str(outfile), index=False)

    elapsed = time.time() - t_job_start
    log(f"LIW CSV written   : {outfile}  ({len(df)} events)")
    log(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")

    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
