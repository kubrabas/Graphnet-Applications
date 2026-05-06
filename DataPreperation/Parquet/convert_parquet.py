"""
Convert PMT-response I3 files to Parquet format — parallel, single SLURM node.

Processes all I3 files in indir in parallel using ProcessPoolExecutor:
  1. Discovers all I3 files in indir, sorts them.
  2. Runs DataConverter with PONE_Reader + PONE extractors + ParquetWriter
     for each file in a subprocess.
  3. Writes truth/ and features/ parquet files to a shared outdir.
  4. Logs per-file results to logdir/<stem>.out
  5. Writes a summary log to logdir/../job_<job_id>_<flavor>_<geometry>.log

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
    The SLURM script should call this with --nworkers $SLURM_CPUS_PER_TASK
    (no --array, no --task-id).
"""

import h5py  # must be imported before icecube/graphnet to avoid HDF5 version conflict

import argparse
import os
import re
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from zoneinfo import ZoneInfo
from glob import glob
from pathlib import Path
from typing import Dict, List, Optional

from graphnet.data.dataconverter import DataConverter
from graphnet.data.extractors.icecube import I3FeatureExtractorPONE, I3TruthExtractorPONE
from graphnet.data.extractors.icecube.utilities.i3_filters import NullSplitI3Filter
from graphnet.data.readers import PONE_Reader
from graphnet.data.writers import ParquetWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

VALID_FLAVORS = {"Muon", "Electron", "Tau", "NC"}

SUMMARY_RE = re.compile(
    r"\[.+?\]\s+kept=(\d+)\s+noise_only=(\d+)\s+"
    r"(?:absent_pulsemap|pulsemap_does_not_exist)=(\d+)"
    r"(?:\s+corrupt_frames=\d+)?"
)

SUCCESS_CATEGORIES = {
    "successfully_transferred_kept_events",
    "completely_empty_file",
    "only_noise_events",
    "only_missing_pulsemap_events",
    "only_filtered_events",
}

FAILED_CATEGORIES = {
    "failed_to_open_file",
    "failed_at_first_event",
    "failed_after_partial_progress",
    "failed_after_only_filtered_events",
    "failed_missing_parquet_outputs_after_success",
}

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


def normalize_log_terms(logfile: Path) -> None:
    """Use project-preferred names for counters emitted by upstream readers."""
    try:
        text = logfile.read_text(errors="replace")
        text = text.replace("absent_pulsemap", "pulsemap_does_not_exist")
        logfile.write_text(text)
    except OSError:
        pass


def read_conversion_summary(logfile: Path) -> Optional[Dict[str, int]]:
    """Read the final per-file reader summary from a conversion log."""
    try:
        text = logfile.read_text(errors="replace")
    except OSError:
        return None

    matches = list(SUMMARY_RE.finditer(text))
    if not matches:
        return None

    m = matches[-1]
    return {
        "kept": int(m.group(1)),
        "noise_only": int(m.group(2)),
        "pulsemap_does_not_exist": int(m.group(3)),
    }


def log_has_file_error(logfile: Path) -> bool:
    try:
        return "[FILE ERROR]" in logfile.read_text(errors="replace")
    except OSError:
        return False


def classify_conversion(
    *, logfile: Path, truth_out: Path, features_out: Path, failed: bool
) -> str:
    """Assign exactly one long category name to a per-file conversion."""
    if log_has_file_error(logfile):
        return "failed_to_open_file"

    summary = read_conversion_summary(logfile)
    if summary is None:
        return "failed_at_first_event" if failed else "completely_empty_file"

    kept = summary["kept"]
    noise_only = summary["noise_only"]
    missing_pulsemap = summary["pulsemap_does_not_exist"]
    filtered = noise_only + missing_pulsemap

    if failed:
        if kept > 0:
            return "failed_after_partial_progress"
        if filtered > 0:
            return "failed_after_only_filtered_events"
        return "failed_at_first_event"

    if kept > 0:
        if truth_out.exists() and features_out.exists():
            return "successfully_transferred_kept_events"
        return "failed_missing_parquet_outputs_after_success"

    if noise_only == 0 and missing_pulsemap == 0:
        return "completely_empty_file"
    if noise_only > 0 and missing_pulsemap == 0:
        return "only_noise_events"
    if noise_only == 0 and missing_pulsemap > 0:
        return "only_missing_pulsemap_events"
    return "only_filtered_events"


def format_summary_for_log(logfile: Path) -> str:
    summary = read_conversion_summary(logfile)
    if summary is None:
        return "kept=NA noise_only=NA pulsemap_does_not_exist=NA"
    return (
        f"kept={summary['kept']} "
        f"noise_only={summary['noise_only']} "
        f"pulsemap_does_not_exist={summary['pulsemap_does_not_exist']}"
    )


def per_file_log_path(logdir: Path, stem: str) -> Path:
    """Per-file logs live in a flavor/geometry-specific directory, so keep names short."""
    return logdir / f"{stem}.out"


# ---------------------------------------------------------------------------
# Per-file worker (runs in a subprocess via ProcessPoolExecutor)
# ---------------------------------------------------------------------------

def _process_one(infile: Path, outdir: Path, logdir: Path, cfg: dict) -> tuple:
    """Process a single I3 file.

    Returns:
        ("skipped", filename)
        (category, filename, elapsed, error_message)
    """
    stem      = stem_from_i3(str(infile))
    logfile   = per_file_log_path(logdir, stem)
    truth_out = outdir / "truth" / f"{stem}_truth.parquet"
    features_out = outdir / "features" / f"{stem}_features.parquet"

    # Skip check: expected outputs must exist AND logfile must confirm SUCCESS
    if truth_out.exists() and features_out.exists() and logfile.exists():
        try:
            if "=== SUCCESS" in logfile.read_text(errors="replace"):
                return ("skipped", infile.name)
        except OSError:
            pass

    # Clean up old outputs/logs before reprocessing this file.
    for p in (logfile, truth_out, features_out):
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
            num_workers=1,
            index_column="event_no",
        )

        converter([str(infile)])

        elapsed = time.time() - t_start
        _berlin = datetime.now(ZoneInfo("Europe/Berlin"))
        log_fh.flush()
        category = classify_conversion(
            logfile=logfile,
            truth_out=truth_out,
            features_out=features_out,
            failed=False,
        )
        if category in FAILED_CATEGORIES:
            print(f"=== FAILED  elapsed={elapsed:.1f}s ===")
            print(f"category : {category}")
            print(f"@ {_berlin.strftime('%Y-%m-%d %H:%M:%S')} (Berlin)")
            log_fh.flush()
            log_fh.close()
            log_fh = None
            normalize_log_terms(logfile)
            for p in (truth_out, features_out):
                try:
                    if p.exists():
                        p.unlink()
                except OSError:
                    pass
            error_message = (
                "file could not be opened"
                if category == "failed_to_open_file"
                else "missing expected parquet output"
            )
            return (category, infile.name, elapsed, error_message)

        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")
        print(f"category : {category}")
        print(f"@ {_berlin.strftime('%Y-%m-%d %H:%M:%S')} (Berlin)")
        log_fh.flush()
        log_fh.close()
        log_fh = None
        normalize_log_terms(logfile)
        return (category, infile.name, elapsed, "")

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
        normalize_log_terms(logfile)

        # Keep the failed log for diagnosis, but remove partial outputs.
        for p in (truth_out, features_out):
            try:
                if p.exists():
                    p.unlink()
            except OSError:
                pass

        category = classify_conversion(
            logfile=logfile,
            truth_out=truth_out,
            features_out=features_out,
            failed=True,
        )
        return (category, infile.name, elapsed, str(e))

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
        stem = stem_from_i3(str(f))
        t = outdir / "truth" / f"{stem}_truth.parquet"
        x = outdir / "features" / f"{stem}_features.parquet"
        l = per_file_log_path(logdir, stem)
        if not l.exists():
            return False
        try:
            if "=== SUCCESS" not in l.read_text(errors="replace"):
                return False
        except OSError:
            return False

        category = classify_conversion(
            logfile=l,
            truth_out=t,
            features_out=x,
            failed=False,
        )
        return category in SUCCESS_CATEGORIES

    # If --overwrite, pre-delete outputs so _process_one's skip check finds nothing
    if args.overwrite:
        for f in files:
            stem = stem_from_i3(str(f))
            for p in [
                outdir / "truth" / f"{stem}_truth.parquet",
                outdir / "features" / f"{stem}_features.parquet",
                per_file_log_path(logdir, stem),
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

    n_failed = n_skipped = 0
    category_counts: Dict[str, int] = {
        category: 0
        for category in [
            "successfully_transferred_kept_events",
            "completely_empty_file",
            "only_noise_events",
            "only_missing_pulsemap_events",
            "only_filtered_events",
            "failed_to_open_file",
            "failed_at_first_event",
            "failed_after_partial_progress",
            "failed_after_only_filtered_events",
            "failed_missing_parquet_outputs_after_success",
        ]
    }

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
                if status == "skipped":
                    n_skipped += 1
                    print(f"[skip]   {fname}")
                    _glog(f"[{ts}] [skip]   {fname}")
                else:
                    category = status
                    category_counts.setdefault(category, 0)
                    category_counts[category] += 1
                    elapsed = result[2] if len(result) > 2 else 0.0
                    err = result[3] if len(result) > 3 else ""
                    stem = stem_from_i3(fname)
                    per_file_log = per_file_log_path(logdir, stem)
                    summary_text = format_summary_for_log(per_file_log)
                    if category in FAILED_CATEGORIES:
                        n_failed += 1
                        print(f"[{category}] {fname}  {summary_text}  ({err})")
                        _glog(
                            f"[{ts}] [{category}] {fname}  {summary_text}  "
                            f"({elapsed:.1f}s) -- {err}"
                        )
                    else:
                        print(f"[{category}] {fname}  {summary_text}")
                        _glog(
                            f"[{ts}] [{category}] {fname}  {summary_text}  "
                            f"({elapsed:.1f}s)"
                        )
                sys.stdout.flush()

        total_skipped = n_skipped + n_already_done
        total_elapsed = time.time() - t_job_start
        category_total = sum(category_counts.values())
        per_file_logs_found = len(
            [
                p for p in logdir.glob("*.out")
                if not p.name.startswith("merge_")
            ]
        )
        accounting_check = "PASS" if category_total + total_skipped == len(files) else "FAIL"
        per_file_log_check = "PASS" if per_file_logs_found == len(files) else "FAIL"
        print("\nDone.")
        print("=== FINAL SUMMARY ===")
        for category, count in category_counts.items():
            print(f"{category} : {count}")
        print(f"category_total : {category_total}")
        print(f"failed_total : {n_failed}")
        print(f"skipped_previously_done : {total_skipped}")
        print(f"input_files_discovered : {len(files)}")
        print(f"per_file_logs_found : {per_file_logs_found}")
        print(f"accounting_check : {accounting_check}")
        print(f"per_file_log_check : {per_file_log_check}")
        print(f"elapsed : {total_elapsed:.1f}s")
        _glog()
        _glog("=== FINAL SUMMARY ===")
        _glog(f"finished : {datetime.now(ZoneInfo('Europe/Berlin')).strftime('%Y-%m-%d %H:%M:%S %Z')}")
        for category, count in category_counts.items():
            _glog(f"{category} : {count}")
        _glog(f"category_total : {category_total}")
        _glog(f"failed_total : {n_failed}")
        _glog(f"skipped_previously_done : {total_skipped}")
        _glog(f"input_files_discovered : {len(files)}")
        _glog(f"per_file_logs_found : {per_file_logs_found}")
        _glog(f"accounting_check : {accounting_check}")
        _glog(f"per_file_log_check : {per_file_log_check}")
        _glog(f"elapsed  : {total_elapsed:.1f}s")
        _glog(f"log      : {general_log_path}")

    return 0 if n_failed == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
