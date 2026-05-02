"""
Calculate LeptonInjector one-weights for ONE batch ID per Slurm array task.

Each task:
  1. Scans lic_dir for *.lic files and photon_dir for photon i3 files.
  2. Intersects batch IDs (LIC ∩ photon), sorts, picks entry at SLURM_ARRAY_TASK_ID.
  3. Finds gen i3 file by batch_id in lic_dir.
  4. Reads gen EventProperties, builds LW.Weighter, iterates photon events.
  5. Writes partial CSV (raw oneweight, no scaling):  <outdir>/<Flavor>_<batch_id>.csv
  6. Logs to:                                         <logdir>/<Flavor>_<batch_id>.out

Run merge_LIW.py after all tasks finish to merge CSVs and compute final LIW.

Usage (local test):
    python3 calculate_LIW.py --mc 340StringMC --flavor Muon \
        --lic-dir /path/to/Generator --photon-dir /path/to/Photon \
        --photon-pattern '*.i3' --outdir /tmp/out --logdir /tmp/log \
        --task-id 0
"""

import argparse
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


def to_lw_particle(particle_type):
    """Convert EventProperties particle type / PDG code to LeptonWeighter particle type."""
    particle_type = int(particle_type)

    if particle_type not in LW_PARTICLE_FROM_PDG:
        raise ValueError(f"Unsupported particle type for LeptonWeighter: {particle_type}")

    return LW_PARTICLE_FROM_PDG[particle_type]


def get_lw_particles_from_event_properties(props):
    """
    Get event-by-event LW particle types from EventProperties.

    This is needed because the LIC contains both neutrino and antineutrino generators.
    Therefore primary/final-state particles must not be hard-coded by flavor.
    """
    primary = to_lw_particle(props.initialType)
    fs0 = to_lw_particle(props.finalType1)
    fs1 = to_lw_particle(props.finalType2)

    return primary, fs0, fs1


# ---------------------------------------------------------------------------
# File discovery
# ---------------------------------------------------------------------------

def extract_batch_id(filepath: Path) -> Optional[str]:
    """Return the numeric string immediately before the extension(s), preserving leading zeros."""
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


def get_sorted_common(lic_dir: str, photon_dir: str, photon_pattern: str) -> List[str]:
    """Intersection of LIC batch IDs and photon batch IDs, sorted."""
    lic_ids = set(build_id_map(lic_dir, "*.lic"))
    photon_ids = set(build_id_map(photon_dir, photon_pattern))
    return sorted(lic_ids & photon_ids)


# ---------------------------------------------------------------------------
# Processing
# ---------------------------------------------------------------------------

def load_xs():
    # NOTE:
    # This intentionally uses neutrino cross-section files for both nu and nubar,
    # matching the production convention of this dataset.
    return LW.CrossSectionFromSpline(
        XS_PATH + "dsdxdy_nu_CC_iso.fits",
        XS_PATH + "dsdxdy_nu_CC_iso.fits",
        XS_PATH + "dsdxdy_nu_NC_iso.fits",
        XS_PATH + "dsdxdy_nu_NC_iso.fits",
    )


def read_gen_props(gen_file: Path) -> Optional[Dict]:
    """Read EventProperties from generator i3. Returns None if file is completely corrupt."""
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


def process_batch(
    batch_id: str,
    lic_dir: str,
    photon_dir: str,
    photon_pattern: str,
    flavor: str,
    xs,
) -> Tuple[Optional[List[dict]], Optional[str]]:
    """
    Returns (records, None) on success or (None, corrupt_file_path) on a completely
    corrupt file. Corrupt events within a photon file are silently skipped.
    Records contain raw oneweight (no scaling).
    """
    lic_map = build_id_map(lic_dir, "*.lic")
    photon_map = build_id_map(photon_dir, photon_pattern)
    gen_map = build_id_map(lic_dir, "*.i3", "*.i3.gz", "*.i3.zst")

    lic_f = lic_map[batch_id]
    photon_f = photon_map[batch_id]

    if batch_id not in gen_map:
        return None, f"gen file missing for batch_id={batch_id} in {lic_dir}"
    gen_f = gen_map[batch_id]

    gen_props = read_gen_props(gen_f)
    if gen_props is None:
        return None, str(gen_f)

    try:
        generators = LW.MakeGeneratorsFromLICFile(str(lic_f))
        weighter = LW.Weighter(LW.ConstantFlux(1.0), xs, generators)
    except Exception as e:
        return None, f"{lic_f}  ({e})"

    records = []

    try:
        f = dataio.I3File(str(photon_f))
        while f.more():
            try:
                frame = f.pop_frame()
            except RuntimeError:
                continue  # corrupt event – skip silently

            if frame.Stop != icetray.I3Frame.DAQ:
                continue
            if not frame.Has("I3EventHeader"):
                continue

            hdr = frame["I3EventHeader"]
            eid = (hdr.run_id, hdr.sub_run_id, hdr.event_id, hdr.sub_event_id)

            if eid not in gen_props:
                continue

            props = gen_props[eid]

            try:
                primary, fs0, fs1 = get_lw_particles_from_event_properties(props)
            except ValueError as e:
                print(f"[skip] RunID={hdr.run_id} EventID={hdr.event_id}: {e}")
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

        f.close()

    except RuntimeError:
        return None, str(photon_f)

    return records, None


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
    ap.add_argument("--task-id", type=int, default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    if args.flavor.lower() not in VALID_FLAVORS:
        print(f"ERROR: unknown flavor '{args.flavor}'. Choices: {sorted(VALID_FLAVORS)}")
        return 1

    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    array_job_id = os.environ.get("SLURM_ARRAY_JOB_ID", "unknown")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "unknown")
    job_id = os.environ.get("SLURM_JOB_ID", "unknown")

    if args.task_id is not None:
        task_id = args.task_id
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if task_id < 0:
            print("ERROR: SLURM_ARRAY_TASK_ID not set. Use --task-id for local testing.")
            return 2

    common = get_sorted_common(args.lic_dir, args.photon_dir, args.photon_pattern)
    if not common:
        print("ERROR: No matched batch IDs found.")
        return 3
    if not (0 <= task_id < len(common)):
        print(f"ERROR: task_id={task_id} out of range (0..{len(common) - 1})")
        return 4

    batch_id = common[task_id]
    outfile = outdir / f"{args.flavor}_{batch_id}.csv"
    logfile = logdir / f"{args.flavor}_{batch_id}.out"

    if outfile.exists() and logfile.exists() and not args.overwrite:
        print(f"[skip] both exist: {outfile.name}, {logfile.name}")
        return 0

    log_fh = open(logfile, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    t_start = time.time()
    print("=== LIW JOB STARTED ===")
    print(f"array_job_id  : {array_job_id}")
    print(f"array_task_id : {array_task_id}")
    print(f"job_id        : {job_id}")
    print(f"mc            : {args.mc}")
    print(f"flavor        : {args.flavor}")
    print(f"batch_id      : {batch_id}")
    print(f"task_id       : {task_id} / {len(common) - 1}")
    print(f"outfile       : {outfile}")
    print(f"logfile       : {logfile}")
    log_fh.flush()

    try:
        xs = load_xs()
        records, corrupt = process_batch(
            batch_id,
            args.lic_dir,
            args.photon_dir,
            args.photon_pattern,
            args.flavor,
            xs,
        )

        if corrupt is not None:
            print("\n=== Completely Corrupt Files ===")
            print(f"  {corrupt}")
            elapsed = time.time() - t_start
            print(f"=== FAILED  elapsed={elapsed:.1f}s ===")
            log_fh.flush()
            log_fh.close()
            return 1

        df = pd.DataFrame(records)
        df.to_csv(str(outfile), index=False)

        elapsed = time.time() - t_start
        print(f"events_written : {len(df)}")
        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"=== FAILED  elapsed={elapsed:.1f}s  error={e} ===")
        log_fh.flush()
        log_fh.close()
        return 1

    log_fh.flush()
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())