"""Write accepted-pulse feature statistics for P-ONE PMT-response files."""
# example usage:
# sbatch --cpus-per-task=100 /home/kbas/SlurmScripts/DataPreperation/submit_feature_level_statistics.sh NC 100

import argparse
import csv
import importlib.util
import os
import shutil
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime
from pathlib import Path
from zoneinfo import ZoneInfo

import h5py  # noqa: F401  # import before icecube to avoid HDF5 conflicts
from icecube import LeptonInjector, dataio, dataclasses, icetray, simclasses  # noqa: F401


PATHS_PY = Path("/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py")
DEFAULT_OUTDIR = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/"
    "DatasetStatistics/FeatureLevelStatistics"
)
BERLIN_TZ = ZoneInfo("Europe/Berlin")

ACCEPTED_PULSE_MAPS = {
    "accepted_pulse_count_340_string": "Accepted_PulseMap_340String",
    "accepted_pulse_count_160_string": "Accepted_PulseMap_160_String",
    "accepted_pulse_count_102_string": "Accepted_PulseMap_102_String",
}

COLUMNS = [
    "RunID",
    "SubrunID",
    "EventID",
    "SubEventID",
    *ACCEPTED_PULSE_MAPS.keys(),
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


def event_header_row(frame):
    """Return event identifiers from I3EventHeader, or -1 when unavailable."""
    if "I3EventHeader" not in frame:
        return {
            "RunID": -1,
            "SubrunID": -1,
            "EventID": -1,
            "SubEventID": -1,
        }

    header = frame["I3EventHeader"]
    if header is None:
        return {
            "RunID": -1,
            "SubrunID": -1,
            "EventID": -1,
            "SubEventID": -1,
        }

    return {
        "RunID": int(getattr(header, "run_id", -1)),
        "SubrunID": int(getattr(header, "sub_run_id", -1)),
        "EventID": int(getattr(header, "event_id", -1)),
        "SubEventID": int(getattr(header, "sub_event_id", -1)),
    }


def pulse_count(frame, pulsemap_key: str) -> int:
    """Count all pulse entries in a pulse map; return -1 if the map is missing."""
    if pulsemap_key not in frame:
        return -1

    pulse_map = frame[pulsemap_key]
    try:
        return sum(len(pulse_map[omkey]) for omkey in pulse_map.keys())
    except AttributeError:
        return sum(len(pulses) for pulses in pulse_map.values())


def process_file(task):
    """Count accepted pulses from one I3 file into a chunk CSV."""
    index, infile_s, chunk_s, flavor, mc_key = task
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

            row = event_header_row(frame)
            for column, pulsemap_key in ACCEPTED_PULSE_MAPS.items():
                row[column] = pulse_count(frame, pulsemap_key)

            writer.writerow(row)
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
        description="Extract accepted-pulse feature statistics from PMT-response I3 files."
    )
    ap.add_argument("--flavor", required=True, help="Flavor key, e.g. Muon, Electron, Tau, NC")
    ap.add_argument("--indir", required=True, help="Input directory containing PMT-response I3 files")
    ap.add_argument("--pattern", default="*.i3.gz", help="Input glob pattern under --indir")
    ap.add_argument("--out", default=None, help="Output CSV path")
    ap.add_argument("--logdir", default=None, help="Log/chunk directory")
    ap.add_argument("--nworkers", type=int, default=None, help="Parallel workers")
    ap.add_argument("--mc-key", default="String340MC", help="BAD_I3_FILES top-level key")
    return ap.parse_args()


def main() -> int:
    args = parse_args()

    indir = Path(args.indir).resolve()
    out = Path(args.out).resolve() if args.out else (DEFAULT_OUTDIR / f"{args.flavor}.csv").resolve()
    logdir = Path(args.logdir).resolve() if args.logdir else out.parent
    logdir.mkdir(parents=True, exist_ok=True)

    inputs = discover_inputs(indir, args.pattern)
    if not inputs:
        print(f"ERROR: no files found in {indir} matching {args.pattern}")
        return 3

    nworkers = args.nworkers or int(os.environ.get("SLURM_CPUS_PER_TASK", 4))
    job_id = os.environ.get("SLURM_JOB_ID", "local")
    chunks = logdir / "chunks" / f"{args.flavor}_{job_id}"
    chunks.mkdir(parents=True, exist_ok=True)
    started = datetime.now(BERLIN_TZ)
    summary_log = logdir / f"feature_level_statistics_{args.flavor}_{job_id}.log"

    tasks = [
        (
            index,
            str(infile),
            str(chunks / f"{index:06d}_{infile.stem}.csv"),
            args.flavor,
            args.mc_key,
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

        log_line("=== FEATURE LEVEL STATISTICS JOB ===")
        log_line(f"job_id   : {job_id}")
        log_line(f"started  : {started.strftime('%Y-%m-%d %H:%M:%S %Z')}")
        log_line(f"flavor   : {args.flavor}")
        log_line(f"indir    : {indir}")
        log_line(f"pattern  : {args.pattern}")
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
        shutil.rmtree(chunks)

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
