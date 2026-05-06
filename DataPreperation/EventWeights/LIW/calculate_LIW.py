"""
Calculate LeptonInjector one-weights for ALL batch IDs in a single Slurm job.

Processes files in parallel using multiprocessing. Outputs:
  <outdir>/<Flavor>_LIW.csv           — final per-event weights
  <logdir>/<Flavor>_file_stats.csv    — per-file processing stats
  <logdir>/<Flavor>_compute.out       — run log

LIW = oneweight / n_accessible_events  (total events successfully processed)

Usage:
    python3 calculate_LIW.py --mc 340StringMC --flavor Muon \\
        --lic-dir /path/to/Generator --photon-dir /path/to/Photon \\
        --photon-pattern '*.i3' --outdir /tmp/out --logdir /tmp/log \\
        [--workers 8] [--overwrite]
"""

import argparse
import multiprocessing
import os
import re
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

VALID_FLAVORS = {"muon", "electron", "tau", "nc"}

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


def load_xs():
    # Use physical nu/nubar cross sections for flux-free effective-area weights.
    return LW.CrossSectionFromSpline(
        XS_PATH + "dsdxdy_nu_CC_iso.fits",
        XS_PATH + "dsdxdy_nubar_CC_iso.fits",
        XS_PATH + "dsdxdy_nu_NC_iso.fits",
        XS_PATH + "dsdxdy_nubar_NC_iso.fits",
    )


def read_gen_props(gen_file: Path) -> Optional[Dict]:
    props = {}
    try:
        f = dataio.I3File(str(gen_file))
        while f.more():
            frame = f.pop_frame()
            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            if not frame.Has("EventProperties") or not frame.Has("I3EventHeader"):
                continue
            hdr = frame["I3EventHeader"]
            eid = (hdr.run_id, hdr.sub_run_id, hdr.event_id, hdr.sub_event_id)
            props[eid] = frame["EventProperties"]
        f.close()
    except RuntimeError:
        return None
    return props


# ---------------------------------------------------------------------------
# Worker
# ---------------------------------------------------------------------------

def _worker_init():
    global _xs
    _xs = load_xs()


def _process_one(args: tuple) -> tuple:
    """Process one batch. Returns (batch_id, records, stat_dict)."""
    batch_id, lic_path, photon_path, gen_path = args
    t0 = time.time()

    stat = {
        "batch_id": batch_id,
        "i3_file": photon_path,
        "lic_file": lic_path,
        "gen_file": gen_path,
        "status": "",
        "file_opened_ok": False,
        "n_events_total": 0,
        "n_events_ok": 0,
        "n_events_skipped": 0,
        "n_frames_corrupt": 0,
        "duration_s": 0.0,
    }

    gen_props = read_gen_props(Path(gen_path))
    if gen_props is None:
        stat["status"] = "corrupt_gen"
        stat["duration_s"] = time.time() - t0
        return batch_id, [], stat

    try:
        generators = LW.MakeGeneratorsFromLICFile(lic_path)
        weighter = LW.Weighter(LW.ConstantFlux(1.0), _xs, generators)
    except Exception as e:
        stat["status"] = f"lic_error: {e}"
        stat["duration_s"] = time.time() - t0
        return batch_id, [], stat

    records = []
    n_total = 0
    n_ok = 0
    n_skipped = 0
    n_corrupt = 0

    try:
        f = dataio.I3File(photon_path)
        stat["file_opened_ok"] = True
        while f.more():
            try:
                frame = f.pop_frame()
            except RuntimeError:
                n_corrupt += 1
                continue

            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            if not frame.Has("I3EventHeader"):
                continue

            n_total += 1
            hdr = frame["I3EventHeader"]
            eid = (hdr.run_id, hdr.sub_run_id, hdr.event_id, hdr.sub_event_id)

            if eid not in gen_props:
                n_skipped += 1
                continue

            props = gen_props[eid]

            try:
                primary, fs0, fs1 = get_lw_particles_from_event_properties(props)
            except ValueError:
                n_skipped += 1
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
            event.radius = props.impactParameter
            event.total_column_depth = props.totalColumnDepth
            event.x = props.x
            event.y = props.y
            event.z = props.z

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
                "initialType": int(props.initialType),
                "finalType1": int(props.finalType1),
                "finalType2": int(props.finalType2),
                "oneweight": weighter.get_oneweight(event),
            })
            n_ok += 1

        f.close()
        stat["status"] = "ok"

    except RuntimeError:
        stat["status"] = "corrupt_i3"

    stat["n_events_total"] = n_total
    stat["n_events_ok"] = n_ok
    stat["n_events_skipped"] = n_skipped
    stat["n_frames_corrupt"] = n_corrupt
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
    log(f"outfile   : {outfile}")
    log(f"statsfile : {statsfile}")
    log(f"logfile   : {logfile}")

    # ------------------------------------------------------------------
    # Build file maps once in the main process
    # ------------------------------------------------------------------
    try:
        lic_map    = build_id_map(args.lic_dir, "*.lic")
        photon_map = build_id_map(args.photon_dir, args.photon_pattern)
        gen_map    = build_id_map(args.lic_dir, "*.i3", "*.i3.gz", "*.i3.zst")
    except FileNotFoundError as e:
        log(f"ERROR: {e}")
        log_fh.close()
        return 1

    common = sorted(set(lic_map) & set(photon_map))
    log(f"\nbatches   : {len(common)} (lic={len(lic_map)} photon={len(photon_map)} gen={len(gen_map)})")

    # ------------------------------------------------------------------
    # Build task list; record gen_missing batches immediately
    # ------------------------------------------------------------------
    tasks: List[tuple] = []
    all_stats: List[dict] = []

    for bid in common:
        if bid not in gen_map:
            all_stats.append({
                "batch_id": bid,
                "i3_file": str(photon_map[bid]),
                "lic_file": str(lic_map[bid]),
                "gen_file": "",
                "status": "gen_missing",
                "file_opened_ok": False,
                "n_events_total": 0,
                "n_events_ok": 0,
                "n_events_skipped": 0,
                "n_frames_corrupt": 0,
                "duration_s": 0.0,
            })
        else:
            tasks.append((bid, str(lic_map[bid]), str(photon_map[bid]), str(gen_map[bid])))

    n_gen_missing = len(all_stats)
    log(f"tasks     : {len(tasks)} to process  ({n_gen_missing} skipped: gen file missing)")

    if not tasks:
        log("ERROR: no tasks to process.")
        log_fh.close()
        return 1

    # ------------------------------------------------------------------
    # Process in parallel
    # ------------------------------------------------------------------
    all_records: List[dict] = []

    with multiprocessing.Pool(workers, initializer=_worker_init) as pool:
        for i, (batch_id, records, stat) in enumerate(
            pool.imap_unordered(_process_one, tasks), 1
        ):
            all_records.extend(records)
            all_stats.append(stat)
            log(
                f"[{i:>{len(str(len(tasks)))}}/{len(tasks)}] {batch_id}"
                f"  {stat['status']:<20}"
                f"  n_ok={stat['n_events_ok']}"
                f"  n_skip={stat['n_events_skipped']}"
                f"  n_corrupt_frames={stat['n_frames_corrupt']}"
                f"  t={stat['duration_s']:.1f}s"
            )

    # ------------------------------------------------------------------
    # Write file-level stats CSV
    # ------------------------------------------------------------------
    stats_df = pd.DataFrame(all_stats).sort_values("batch_id").reset_index(drop=True)
    stats_df.to_csv(str(statsfile), index=False)

    n_ok_files   = (stats_df["status"] == "ok").sum()
    n_total_files = len(common)
    log(f"\n--- Summary ---")
    log(f"files ok          : {n_ok_files} / {n_total_files}")
    log(f"files corrupt_i3 : {(stats_df['status'] == 'corrupt_i3').sum()}")
    log(f"files corrupt_gen    : {(stats_df['status'] == 'corrupt_gen').sum()}")
    log(f"files lic_error      : {stats_df['status'].str.startswith('lic_error').sum()}")
    log(f"files gen_missing    : {(stats_df['status'] == 'gen_missing').sum()}")
    log(f"stats CSV written : {statsfile}")

    # ------------------------------------------------------------------
    # Write final LIW CSV
    # ------------------------------------------------------------------
    if not all_records:
        log("=== WARNING: no events processed successfully — output CSV not written ===")
        elapsed = time.time() - t_job_start
        log(f"=== FAILED  elapsed={elapsed:.1f}s ===")
        log_fh.close()
        return 1

    N_GEN_PER_FILE = 100  # generated events per particle type per LIC file

    df = pd.DataFrame(all_records)

    nu_mask     = df["initialType"] > 0
    antinu_mask = df["initialType"] < 0

    n_accessible_nu     = nu_mask.sum()
    n_accessible_antinu = antinu_mask.sum()

    df.loc[nu_mask,     "LIW"] = df.loc[nu_mask,     "oneweight"] * N_GEN_PER_FILE / n_accessible_nu
    df.loc[antinu_mask, "LIW"] = df.loc[antinu_mask, "oneweight"] * N_GEN_PER_FILE / n_accessible_antinu

    log(f"accessible nu     : {n_accessible_nu}")
    log(f"accessible antinu : {n_accessible_antinu}")

    df.to_csv(str(outfile), index=False)

    elapsed = time.time() - t_job_start
    log(f"LIW CSV written   : {outfile}  ({len(df)} events)")
    log(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")

    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
