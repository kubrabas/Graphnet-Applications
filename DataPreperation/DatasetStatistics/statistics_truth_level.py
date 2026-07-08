"""Write truth-level event statistics for P-ONE PMT-response files."""
# example usage: sbatch --cpus-per-task=100 /home/kbas/SlurmScripts/DataPreperation/submit_truth_level_statistics.sh NC 100

import argparse
import csv
import importlib.util
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py  # noqa: F401  # import before icecube/graphnet to avoid HDF5 conflicts
from icecube import LeptonInjector, dataclasses, dataio, icetray  # noqa: F401


GRAPHNET_SRC = Path("/project/def-nahee/kbas/graphnet/src")
if str(GRAPHNET_SRC) not in sys.path:
    sys.path.insert(0, str(GRAPHNET_SRC))

from graphnet.data.extractors.icecube.i3truthextractor import (  # noqa: E402
    I3TruthExtractorPONE,
)


PATHS_PY = Path("/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py")
DEFAULT_OUTDIR = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/"
    "DatasetStatistics/TruthLevelStatistics"
)
BERLIN_TZ = ZoneInfo("Europe/Berlin")

COLUMNS = [
    "RunID",
    "SubrunID",
    "EventID",
    "SubEventID",
    "position_x",
    "position_y",
    "position_z",
    "pid",
    "is_CC",
    "totalEnergy",
    "zenith",
    "azimuth",
    "finalStateX",
    "finalStateY",
    "finalType1",
    "finalType2",
    "initialType",
    "totalColumnDepth",
    "impactParameter",
    "triggered_noisy_340_string",
    "trigger_time_noisy_340_string",
    "triggered_noisy_160_string",
    "trigger_time_noisy_160_string",
    "triggered_noisy_102_string",
    "trigger_time_noisy_102_string",
    "triggered_nonoise_340_string",
    "trigger_time_nonoise_340_string",
    "triggered_nonoise_160_string",
    "trigger_time_nonoise_160_string",
    "triggered_nonoise_102_string",
    "trigger_time_nonoise_102_string",
]


def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def bad_file_rule(infile: Path, flavor: str, mc_key: str):
    """Return the BAD_I3_FILES rule for infile, if one exists."""
    paths = load_paths()
    bad_i3_files = getattr(paths, "BAD_I3_FILES", {})
    info = bad_i3_files.get(mc_key, {}).get(flavor, {})
    target = str(infile.resolve())

    if target in set(map(str, info.get("no_daq_for_some_reason", set()))):
        return "skip_no_daq", None

    available_daq_counts = info.get("available_daq_counts", {})
    if target in available_daq_counts:
        return "limit_daq", int(available_daq_counts[target])

    return "normal", None


def scalar(value):
    """Convert IceTray scalar-like objects to CSV-friendly Python values."""
    if hasattr(value, "value"):
        value = value.value

    if isinstance(value, (bool, int, float, str)) or value is None:
        return value

    try:
        return int(value)
    except (TypeError, ValueError):
        pass

    try:
        return float(value)
    except (TypeError, ValueError):
        return str(value)


def process_file(task):
    """Extract selected truth rows from one I3 file into a chunk CSV."""
    index, infile_s, chunk_s, flavor, mc_key, gcd_file = task
    infile = Path(infile_s)
    chunk = Path(chunk_s)
    t_start = time.time()

    rule, available_daq_count = bad_file_rule(infile, flavor, mc_key)
    if rule == "skip_no_daq":
        return {
            "index": index,
            "file": str(infile),
            "chunk": str(chunk),
            "status": "skipped",
            "rows": 0,
            "daq_seen": 0,
            "elapsed": time.time() - t_start,
            "reason": "no_daq_for_some_reason",
        }

    max_daq_frames = available_daq_count if rule == "limit_daq" else 0

    extractor = I3TruthExtractorPONE(name="truth", exclude=[])
    extractor.set_gcd(gcd_file)

    rows = 0
    daq_seen = 0
    chunk.parent.mkdir(parents=True, exist_ok=True)

    with chunk.open("w", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=COLUMNS)
        writer.writeheader()

        i3_file = dataio.I3File(str(infile), "r")
        while i3_file.more():
            try:
                frame = i3_file.pop_daq()
            except RuntimeError as exc:
                if "no frame to pop" in str(exc).lower():
                    break
                raise

            daq_seen += 1
            if max_daq_frames > 0 and daq_seen > max_daq_frames:
                break

            extracted = extractor(frame)
            writer.writerow({column: scalar(extracted.get(column, -1)) for column in COLUMNS})
            rows += 1

    return {
        "index": index,
        "file": str(infile),
        "chunk": str(chunk),
        "status": "success",
        "rows": rows,
        "daq_seen": daq_seen,
        "elapsed": time.time() - t_start,
        "reason": rule,
    }


def discover_inputs(indir: Path, pattern: str):
    return sorted(path for path in indir.rglob(pattern) if path.is_file())


def merge_chunks(results, out: Path):
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", newline="") as out_fh:
        writer = csv.DictWriter(out_fh, fieldnames=COLUMNS)
        writer.writeheader()

        for result in sorted(results, key=lambda item: item["index"]):
            if result["status"] != "success":
                continue

            with Path(result["chunk"]).open("r", newline="") as in_fh:
                reader = csv.DictReader(in_fh)
                for row in reader:
                    writer.writerow({column: row.get(column, -1) for column in COLUMNS})


def parse_args():
    ap = argparse.ArgumentParser(
        description="Extract selected truth-level statistics from PMT-response I3 files."
    )
    ap.add_argument("--flavor", required=True, help="Flavor key, e.g. Muon, Electron, Tau, NC")
    ap.add_argument("--indir", required=True, help="Input directory containing PMT-response I3 files")
    ap.add_argument("--pattern", default="*.i3.gz", help="Input glob pattern under --indir")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--logdir", default=None, help="Log/chunk directory")
    ap.add_argument("--nworkers", type=int, default=None, help="Parallel workers")
    ap.add_argument("--mc-key", default="String340MC", help="BAD_I3_FILES top-level key")
    ap.add_argument("--gcd", default=None, help="GCD file for I3TruthExtractorPONE.set_gcd")
    return ap.parse_args()


def main() -> int:
    args = parse_args()
    paths = load_paths()

    indir = Path(args.indir).resolve()
    out = Path(args.out).resolve() if args.out else (DEFAULT_OUTDIR / f"{args.flavor}.csv").resolve()
    logdir = Path(args.logdir).resolve() if args.logdir else out.parent
    chunks = logdir / "chunks"
    logdir.mkdir(parents=True, exist_ok=True)
    chunks.mkdir(parents=True, exist_ok=True)

    gcd_file = args.gcd or paths.GCD["340StringMC"]
    inputs = discover_inputs(indir, args.pattern)
    if not inputs:
        print(f"ERROR: no files found in {indir} matching {args.pattern}")
        return 3

    nworkers = args.nworkers or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    started = datetime.now(BERLIN_TZ)
    summary_log = logdir / f"truth_level_statistics_{args.flavor}_{job_id}.log"

    tasks = [
        (
            index,
            str(infile),
            str(chunks / f"{index:06d}_{infile.stem}.csv"),
            args.flavor,
            args.mc_key,
            gcd_file,
        )
        for index, infile in enumerate(inputs)
    ]

    results = []
    t_job_start = time.time()
    with summary_log.open("w") as log:
        def log_line(message=""):
            print(message)
            log.write(message + "\n")
            log.flush()

        log_line("=== TRUTH LEVEL STATISTICS JOB ===")
        log_line(f"job_id   : {job_id}")
        log_line(f"started  : {started.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        log_line(f"flavor   : {args.flavor}")
        log_line(f"indir    : {indir}")
        log_line(f"pattern  : {args.pattern}")
        log_line(f"gcd      : {gcd_file}")
        log_line(f"out      : {out}")
        log_line(f"logdir   : {logdir}")
        log_line(f"files    : {len(inputs)}")
        log_line(f"workers  : {nworkers}")
        log_line()

        with ProcessPoolExecutor(max_workers=nworkers) as executor:
            futures = [executor.submit(process_file, task) for task in tasks]
            for future in as_completed(futures):
                result = future.result()
                results.append(result)
                log_line(
                    f"{result['status']:7s} rows={result['rows']:7d} "
                    f"daq_seen={result['daq_seen']:7d} elapsed={result['elapsed']:8.1f}s "
                    f"reason={result['reason']} file={result['file']}"
                )

        merge_chunks(results, out)

        n_success = sum(result["status"] == "success" for result in results)
        n_skipped = sum(result["status"] == "skipped" for result in results)
        n_rows = sum(result["rows"] for result in results)
        elapsed = time.time() - t_job_start
        log_line()
        log_line("=== DONE ===")
        log_line(f"success_files : {n_success}")
        log_line(f"skipped_files : {n_skipped}")
        log_line(f"rows          : {n_rows}")
        log_line(f"elapsed_s     : {elapsed:.1f}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
