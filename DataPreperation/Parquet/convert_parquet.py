"""
Convert PMT-response I3 files to Parquet format — parallel, single SLURM node.

Processes all I3 files in indir using GraphNeT DataConverter multiprocessing:
  1. Discovers all I3 files in indir, sorts them.
  2. Runs one DataConverter with PONE_Reader + PONE extractors + ParquetWriter
     over the full file list.
  3. Writes truth/ and features/ parquet files to a shared outdir.
  4. Writes a summary log to logdir/../job_<job_id>_<flavor>_<geometry>.log

Run merge_parquet.py after this script finishes to merge batches and build splits.

Usage (local test):
    python3 convert_parquet.py --mc 340StringMC --flavor Electron \\
        --geometry 102_string \\
        --indir /home/kbas/scratch/String340MC/102_string/Electron_PMT_Response \\
        --gcd /path/to/gcd.i3.gz \\
        --outdir /home/kbas/scratch/String340MC/102_string/Electron_Parquet \\
        --logdir /home/kbas/scratch/String340MC/Logs/Electron_102_string_Parquet \\
        --nworkers 4

Shell script note:
    The SLURM script should call this with --nworkers $SLURM_CPUS_PER_TASK.
"""

import h5py  # must be imported before icecube/graphnet to avoid HDF5 version conflict

import argparse
import os
import sys
import time
from datetime import datetime
from zoneinfo import ZoneInfo
from glob import glob
from pathlib import Path
from typing import List

from graphnet.data.dataconverter import DataConverter
from graphnet.data.extractors.icecube import I3FeatureExtractorPONE, I3TruthExtractorPONE
from graphnet.data.extractors.icecube.utilities.i3_filters import NullSplitI3Filter
from graphnet.data.readers import PONE_Reader
from graphnet.data.writers import ParquetWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_FLAVORS = {"Muon", "Electron", "Tau", "NC"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_i3_files(indir: str) -> List[Path]:
    """Return sorted list of non-GCD I3 files in indir (recursive)."""
    files: List[str] = []
    for pattern in ["*.i3.gz", "*.i3.bz2", "*.i3.zst"]:
        files.extend(glob(os.path.join(indir, "**", pattern), recursive=True))
    return sorted(
        Path(f) for f in files if "gcd" not in os.path.basename(f).lower()
    )


def stem_from_i3(path: str) -> str:
    """Reproduce DataConverter._create_file_name to predict the output file stem."""
    name = os.path.basename(path)
    for ext in [".gz", ".bz2", ".zst"]:
        if name.endswith(ext):
            name = name[: -len(ext)]
            break
    return name.replace(".i3", "")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Convert PMT-response I3 files to Parquet — parallel, single SLURM node."
    )
    ap.add_argument("--mc",        required=True, help="MC type, e.g. 340StringMC")
    ap.add_argument("--flavor",    required=True, help="Particle flavor: Muon, Electron, Tau, NC")
    ap.add_argument("--geometry",  required=True, help="Sub-geometry key, e.g. 102_string")
    ap.add_argument("--indir",     required=True, help="Input PMT-response directory")
    ap.add_argument("--gcd",       required=True, help="Path to GCD rescue file")
    ap.add_argument("--outdir",    required=True, help="Shared output directory for all tasks")
    ap.add_argument("--logdir",    required=True, help="Directory for per-file log files")
    ap.add_argument("--pulsemap",  default="EventPulseSeries_nonoise")
    ap.add_argument("--nworkers",  type=int, default=None,
                    help="Parallel workers (default: $SLURM_CPUS_PER_TASK or 4)")
    ap.add_argument("--overwrite", action="store_true",
                    help="Always re-run: pre-deletes existing outputs so skip check finds nothing")
    args = ap.parse_args()

    if args.flavor not in VALID_FLAVORS:
        print(f"ERROR: unknown flavor '{args.flavor}'. Choices: {sorted(VALID_FLAVORS)}")
        return 1

    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    files = discover_i3_files(args.indir)
    if not files:
        print(f"ERROR: No I3 files found in {args.indir}")
        return 3

    def _is_done(f: Path) -> bool:
        stem = stem_from_i3(str(f))
        t = outdir / "truth" / f"{stem}_truth.parquet"
        x = outdir / "features" / f"{stem}_features.parquet"
        return t.exists() and x.exists()

    # If --overwrite, pre-delete outputs so the skip check finds nothing.
    if args.overwrite:
        for f in files:
            stem = stem_from_i3(str(f))
            for p in [
                outdir / "truth" / f"{stem}_truth.parquet",
                outdir / "features" / f"{stem}_features.parquet",
            ]:
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass

    to_process     = [f for f in files if not _is_done(f)]
    n_already_done = len(files) - len(to_process)

    print(f"Total files        : {len(files)}")
    print(f"Already done (skip): {n_already_done}")
    print(f"To process         : {len(to_process)}")

    if not to_process:
        print("Nothing to do.")
        return 0

    nworkers = args.nworkers or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    print(f"Workers            : {nworkers}")
    sys.stdout.flush()

    job_id           = os.environ.get("SLURM_JOB_ID", "local")
    date_stamp       = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%m_%d_%Y")
    general_log_path = logdir / f"{date_stamp}_job_{job_id}.log"
    t_job_start      = time.time()

    with open(general_log_path, "w") as glog:
        def _glog(msg=""):
            glog.write(msg + "\n")
            glog.flush()

        _glog("=== PARQUET CONVERSION JOB ===")
        _glog(f"job_id   : {job_id}")
        _glog(f"started  : {datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        _glog(f"mc       : {args.mc}")
        _glog(f"flavor   : {args.flavor}")
        _glog(f"geometry : {args.geometry}")
        _glog(f"outdir   : {outdir}")
        _glog(f"logdir   : {logdir}")
        _glog(f"total    : {len(files)}  (already_done={n_already_done}, to_process={len(to_process)})")
        _glog(f"workers  : {nworkers}")
        _glog()
        reader = PONE_Reader(
            gcd_rescue=args.gcd,
            i3_filters=[NullSplitI3Filter()],
            pulsemap=args.pulsemap,
            skip_empty_pulses=True,
        )
        extractors = [
            I3FeatureExtractorPONE(
                pulsemap=args.pulsemap,
                name="features",
                exclude=[],
            ),
            I3TruthExtractorPONE(
                name="truth",
                exclude=[],
            ),
        ]
        writer = ParquetWriter(truth_table="truth", index_column="event_no")
        converter = DataConverter(
            file_reader=reader,
            save_method=writer,
            extractors=extractors,
            outdir=str(outdir),
            num_workers=nworkers,
            index_column="event_no",
        )

        orig_stdout = sys.stdout
        orig_stderr = sys.stderr
        sys.stdout = glog
        sys.stderr = glog
        try:
            converter([str(f) for f in to_process])
        finally:
            sys.stdout = orig_stdout
            sys.stderr = orig_stderr

        total_elapsed = time.time() - t_job_start
        truth_outputs = len(list((outdir / "truth").glob("*_truth.parquet")))
        feature_outputs = len(list((outdir / "features").glob("*_features.parquet")))
        print("\nDone.")
        print("=== FINAL SUMMARY ===")
        print(f"skipped_previously_done : {n_already_done}")
        print(f"input_files_discovered : {len(files)}")
        print(f"input_files_processed : {len(to_process)}")
        print(f"truth_outputs_found : {truth_outputs}")
        print(f"feature_outputs_found : {feature_outputs}")
        print(f"elapsed : {total_elapsed:.1f}s")
        _glog()
        _glog("=== FINAL SUMMARY ===")
        _glog(f"finished : {datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        _glog(f"skipped_previously_done : {n_already_done}")
        _glog(f"input_files_discovered : {len(files)}")
        _glog(f"input_files_processed : {len(to_process)}")
        _glog(f"truth_outputs_found : {truth_outputs}")
        _glog(f"feature_outputs_found : {feature_outputs}")
        _glog(f"elapsed  : {total_elapsed:.1f}s")
        _glog(f"log      : {general_log_path}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
