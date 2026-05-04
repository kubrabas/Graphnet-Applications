"""
Convert a single PMT-response I3 file to Parquet format.

Each SLURM array task:
  1. Discovers all I3 files in indir, sorts them.
  2. Picks the file at SLURM_ARRAY_TASK_ID (or --task-id).
  3. Runs DataConverter with PONE_Reader + PONE extractors + ParquetWriter.
  4. Writes truth/ and features/ parquet files to a shared outdir.
  5. Logs everything to logdir/<flavor>_<geometry>_<stem>.out

All tasks write into the same outdir/truth/ and outdir/features/ directories
(no filename collisions because each task processes a different I3 file).

Run merge_parquet.py after all tasks finish to merge batches and build splits.

Usage (local test):
    python3 convert_parquet.py --mc 340StringMC --flavor Electron \\
        --geometry 102_string \\
        --indir /home/kbas/scratch/String340MC/102_String/Electron_PMT_Response \\
        --gcd /path/to/gcd.i3.gz \\
        --outdir /home/kbas/scratch/String340MC/102_String/Electron_Parquet \\
        --logdir /home/kbas/scratch/String340MC/Logs/Electron_102_String_Parquet \\
        --task-id 0
"""

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

FEATURE_EXCLUDE = []

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def discover_i3_files(indir: str) -> List[str]:
    """Return sorted list of non-GCD I3 files in indir (recursive)."""
    files: List[str] = []
    for pattern in ["*.i3.gz", "*.i3.bz2", "*.i3.zst"]:
        files.extend(glob(os.path.join(indir, "**", pattern), recursive=True))
    return sorted(f for f in files if "gcd" not in os.path.basename(f).lower())


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
        description="Convert one PMT-response I3 file to Parquet (one SLURM task)."
    )
    ap.add_argument("--mc",        required=True, help="MC type, e.g. 340StringMC")
    ap.add_argument("--flavor",    required=True, help="Particle flavor: Muon, Electron, Tau, NC")
    ap.add_argument("--geometry",  required=True, help="Sub-geometry key, e.g. 102_string")
    ap.add_argument("--indir",     required=True, help="Input PMT-response directory")
    ap.add_argument("--gcd",       required=True, help="Path to GCD rescue file")
    ap.add_argument("--outdir",    required=True, help="Shared output directory for all tasks")
    ap.add_argument("--logdir",    required=True, help="Directory for per-task log files")
    ap.add_argument("--pulsemap",  default="EventPulseSeries_nonoise")
    ap.add_argument("--task-id",   type=int, default=None,
                    help="Override SLURM_ARRAY_TASK_ID (for local testing)")
    ap.add_argument("--overwrite", action="store_true",
                    help=(
                        "Overwrite behaviour: "
                        "if --overwrite is NOT set, a task is skipped only when BOTH "
                        "the output parquet (truth table) AND the log file already exist. "
                        "If only one of them exists, the task re-runs and overwrites it. "
                        "If --overwrite IS set, the task always re-runs regardless."
                    ))
    args = ap.parse_args()

    if args.flavor not in VALID_FLAVORS:
        print(f"ERROR: unknown flavor '{args.flavor}'. Choices: {sorted(VALID_FLAVORS)}")
        return 1

    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    outdir.mkdir(parents=True, exist_ok=True)
    logdir.mkdir(parents=True, exist_ok=True)

    array_job_id  = os.environ.get("SLURM_ARRAY_JOB_ID",  "unknown")
    array_task_id = os.environ.get("SLURM_ARRAY_TASK_ID", "unknown")
    job_id        = os.environ.get("SLURM_JOB_ID",        "unknown")

    if args.task_id is not None:
        task_id = args.task_id
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if task_id < 0:
            print("ERROR: SLURM_ARRAY_TASK_ID not set. Use --task-id for local testing.")
            return 2

    files = discover_i3_files(args.indir)
    if not files:
        print(f"ERROR: No I3 files found in {args.indir}")
        return 3
    if not (0 <= task_id < len(files)):
        print(f"ERROR: task_id={task_id} out of range (0..{len(files) - 1})")
        return 4

    i3_file = files[task_id]
    stem    = stem_from_i3(i3_file)
    logfile = logdir / f"{args.flavor}_{args.geometry}_{stem}.out"

    truth_out = outdir / "truth" / f"{stem}_truth.parquet"
    if not args.overwrite and truth_out.exists() and logfile.exists():
        print(f"[skip] parquet and log both exist: {stem}")
        return 0

    log_fh = open(logfile, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    t_start = time.time()
    print("=== PARQUET CONVERSION JOB STARTED ===")
    print(f"array_job_id  : {array_job_id}")
    print(f"array_task_id : {array_task_id}")
    print(f"job_id        : {job_id}")
    print(f"mc            : {args.mc}")
    print(f"flavor        : {args.flavor}")
    print(f"geometry      : {args.geometry}")
    print(f"task_id       : {task_id} / {len(files) - 1}")
    print(f"i3_file       : {i3_file}")
    print(f"gcd           : {args.gcd}")
    print(f"outdir        : {outdir}")
    print(f"pulsemap      : {args.pulsemap}")
    log_fh.flush()

    try:
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
                exclude=FEATURE_EXCLUDE,
            ),
            I3TruthExtractorPONE(
                name="truth",
                mctree="I3MCTree",
                exclude=[],
            ),
        ]

        writer = ParquetWriter(truth_table="truth", index_column="event_no")

        converter = DataConverter(
            file_reader=reader,
            save_method=writer,
            extractors=extractors,
            outdir=str(outdir),
            num_workers=1,
            index_column="event_no",
        )

        converter([i3_file])

        elapsed = time.time() - t_start
        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"=== FAILED  elapsed={elapsed:.1f}s  error={e} ===")
        _berlin = datetime.now(ZoneInfo("Europe/Berlin"))
        print(f"@ {_berlin.strftime('%Y-%m-%d %H:%M:%S')} (Berlin)")
        log_fh.flush()
        log_fh.close()
        return 1

    _berlin = datetime.now(ZoneInfo("Europe/Berlin"))
    print(f"@ {_berlin.strftime('%Y-%m-%d %H:%M:%S')} (Berlin)")
    log_fh.flush()
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
