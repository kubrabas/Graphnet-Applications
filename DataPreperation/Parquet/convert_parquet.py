"""
Convert PMT-response I3 files to Parquet format — parallel, single SLURM node.

Processes all I3 files in indir in parallel using ProcessPoolExecutor:
  1. Discovers all I3 files in indir, sorts them.
  2. Runs DataConverter with PONE_Reader + PONE extractors + ParquetWriter
     for each file in a subprocess.
  3. Writes truth/ and features/ parquet files to a shared outdir.
  4. Logs per-file results to logdir/<flavor>_<geometry>_<stem>.out
  5. Writes a summary log to logdir/../job_<job_id>_<flavor>_<geometry>.log

Run merge_parquet.py after this script finishes to merge batches and build splits.

Usage (local test):
    python3 convert_parquet.py --mc 340StringMC --flavor Electron \\
        --geometry 102_string \\
        --indir /home/kbas/scratch/String340MC/102_String/Electron_PMT_Response \\
        --gcd /path/to/gcd.i3.gz \\
        --outdir /home/kbas/scratch/String340MC/102_String/Electron_Parquet \\
        --logdir /home/kbas/scratch/String340MC/Logs/Electron_102_String_Parquet \\
        --nworkers 4

Shell script note:
    The SLURM script should call this with --nworkers $SLURM_CPUS_PER_TASK
    (no --array, no --task-id).
"""

import h5py  # must be imported before icecube/graphnet to avoid HDF5 version conflict

import argparse
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
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
# Per-file worker (runs in a subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(infile: Path, outdir: Path, logdir: Path, cfg: dict) -> tuple:
    """Process a single I3 file.

    Returns:
        ("skipped", filename)
        ("success", filename, elapsed)
        ("failed",  filename, error_message, elapsed)
    """
    stem      = stem_from_i3(str(infile))
    logfile   = logdir / f"{cfg['flavor']}_{cfg['geometry']}_{stem}.out"
    truth_out = outdir / "truth" / f"{stem}_truth.parquet"

    # Skip check: both outputs must exist AND logfile must confirm SUCCESS
    if truth_out.exists() and logfile.exists():
        try:
            if "=== SUCCESS" in logfile.read_text(errors="replace"):
                return ("skipped", infile.name)
        except OSError:
            pass

    # Clean up any orphaned partial outputs before reprocessing
    for p in (logfile, truth_out):
        try:
            if p.exists():
                p.unlink()
        except OSError:
            pass

    orig_stdout = sys.stdout
    orig_stderr = sys.stderr
    log_fh = None
    t_start = time.time()

    try:
        log_fh = open(logfile, "w")
        sys.stdout = log_fh
        sys.stderr = log_fh

        print("=== PARQUET CONVERSION JOB STARTED ===")
        print(f"job_id    : {os.environ.get('SLURM_JOB_ID', 'local')}")
        print(f"mc        : {cfg['mc']}")
        print(f"flavor    : {cfg['flavor']}")
        print(f"geometry  : {cfg['geometry']}")
        print(f"i3_file   : {infile}")
        print(f"gcd       : {cfg['gcd']}")
        print(f"outdir    : {outdir}")
        print(f"pulsemap  : {cfg['pulsemap']}")
        log_fh.flush()

        reader = PONE_Reader(
            gcd_rescue=cfg["gcd"],
            i3_filters=[NullSplitI3Filter()],
            pulsemap=cfg["pulsemap"],
            skip_empty_pulses=True,
        )

        extractors = [
            I3FeatureExtractorPONE(
                pulsemap=cfg["pulsemap"],
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

        converter([str(infile)])

        elapsed = time.time() - t_start
        _berlin = datetime.now(ZoneInfo("Europe/Berlin"))
        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")
        print(f"@ {_berlin.strftime('%Y-%m-%d %H:%M:%S')} (Berlin)")
        log_fh.flush()
        return ("success", infile.name, elapsed)

    except Exception as e:
        import traceback as _tb
        elapsed = time.time() - t_start
        _berlin = datetime.now(ZoneInfo("Europe/Berlin"))
        if log_fh and not log_fh.closed:
            try:
                print(f"=== FAILED  elapsed={elapsed:.1f}s  error={e} ===")
                print(f"@ {_berlin.strftime('%Y-%m-%d %H:%M:%S')} (Berlin)")
                _tb.print_exc()
                log_fh.flush()
            except Exception:
                pass

        try:
            if log_fh and not log_fh.closed:
                log_fh.close()
        except Exception:
            pass
        log_fh = None  # prevent double-close in finally

        # Remove partial outputs so only complete files remain
        for p in (logfile, truth_out):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

        return ("failed", infile.name, str(e), elapsed)

    finally:
        if log_fh is not None and not log_fh.closed:
            try:
                log_fh.close()
            except Exception:
                pass
        sys.stdout = orig_stdout
        sys.stderr = orig_stderr


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

    cfg = {
        "mc":       args.mc,
        "flavor":   args.flavor,
        "geometry": args.geometry,
        "gcd":      args.gcd,
        "pulsemap": args.pulsemap,
    }

    def _is_done(f: Path) -> bool:
        t = outdir / "truth" / f"{stem_from_i3(str(f))}_truth.parquet"
        l = logdir / f"{args.flavor}_{args.geometry}_{stem_from_i3(str(f))}.out"
        if not (t.exists() and l.exists()):
            return False
        try:
            return "=== SUCCESS" in l.read_text(errors="replace")
        except OSError:
            return False

    # If --overwrite, pre-delete outputs so _process_one's skip check finds nothing
    if args.overwrite:
        for f in files:
            stem = stem_from_i3(str(f))
            for p in [
                outdir / "truth" / f"{stem}_truth.parquet",
                logdir / f"{args.flavor}_{args.geometry}_{stem}.out",
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
    general_log_path = logdir.parent / f"job_{job_id}_{args.flavor}_{args.geometry}.log"
    t_job_start      = time.time()

    n_success = n_failed = n_skipped = 0

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

        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            futures = {
                executor.submit(_process_one, infile, outdir, logdir, cfg): infile.name
                for infile in to_process
            }
            for future in as_completed(futures):
                result = future.result()
                status = result[0]
                fname  = result[1]
                ts     = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%H:%M:%S")
                if status == "success":
                    n_success += 1
                    elapsed = result[2] if len(result) > 2 else 0.0
                    print(f"[ok]     {fname}")
                    _glog(f"[{ts}] [ok]     {fname}  ({elapsed:.1f}s)")
                elif status == "skipped":
                    n_skipped += 1
                    print(f"[skip]   {fname}")
                    _glog(f"[{ts}] [skip]   {fname}")
                else:
                    n_failed += 1
                    err     = result[2] if len(result) > 2 else "unknown"
                    elapsed = result[3] if len(result) > 3 else 0.0
                    print(f"[failed] {fname}  ({err})")
                    _glog(f"[{ts}] [failed] {fname}  ({elapsed:.1f}s) -- {err}")
                sys.stdout.flush()

        total_skipped = n_skipped + n_already_done
        total_elapsed = time.time() - t_job_start
        print(f"\nDone.  success={n_success}  failed={n_failed}  skipped={total_skipped}")
        _glog()
        _glog("=== FINAL SUMMARY ===")
        _glog(f"finished : {datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        _glog(f"success  : {n_success}")
        _glog(f"failed   : {n_failed}")
        _glog(f"skipped  : {total_skipped}")
        _glog(f"elapsed  : {total_elapsed:.1f}s")
        _glog(f"log      : {general_log_path}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
