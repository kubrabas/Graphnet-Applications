"""
Skim ONE i3 file per Slurm array task, using FilterFrame.

- FilterFrame is imported from:
    /project/def-nahee/kbas/GeometrySkimmer/FilterFrame.py

- Each Slurm array task selects ONE input file by index:
    index = SLURM_ARRAY_TASK_ID
  from the sorted list of files in --indir matching --pattern.

- For that one file:
    reads [GCD, input] -> applies FilterFrame(AllowedStrings=...) -> writes DAQ stream to:
    <outdir>/<input_stem>.i3.gz

- Log is written to:
    <logdir>/<particle>_<input_stem>.out

Notes:
- Deterministic: input files are sorted.
- Skips both output and log if they already exist (unless --overwrite).
- Selection file: extracts ALL integers (comma-separated, multi-line, spaces are OK).
- --indir and --pattern are always passed explicitly (from paths.py via submit_skim_I3.py).
"""

import argparse
import os
import re
import sys
from pathlib import Path
from typing import List

from icecube import icetray, dataio  # noqa: F401
from icecube.icetray import I3LogLevel

icetray.I3Logger.global_logger.set_level(I3LogLevel.LOG_INFO)

DEFAULT_FILTERFRAME_PY = "/project/def-nahee/kbas/GeometrySkimmer/FilterFrame.py"


def import_filterframe(filterframe_py: str):
    p = Path(filterframe_py).resolve()
    if not p.exists():
        raise FileNotFoundError(f"FilterFrame.py not found at: {p}")
    sys.path.insert(0, str(p.parent))
    from FilterFrame import FilterFrame  # type: ignore
    return FilterFrame


def read_allowed_strings(selection_path: str) -> List[int]:
    txt = Path(selection_path).read_text()
    nums = list(map(int, re.findall(r"\d+", txt)))
    seen = set()
    ordered: List[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def list_inputs(indir: str, pattern: str) -> List[Path]:
    base = Path(indir).resolve()
    if not base.exists():
        raise FileNotFoundError(f"Input folder not found: {base}")
    if not base.is_dir():
        raise NotADirectoryError(f"Not a directory: {base}")
    return sorted([p for p in base.rglob(pattern) if p.is_file()])


def stem_with_relpath(infile: Path, indir: Path) -> str:
    rel = infile.relative_to(indir)
    parts = list(rel.parts)
    suffixes = "".join(infile.suffixes)
    parts[-1] = parts[-1][: -len(suffixes)] if suffixes else infile.stem
    return "_".join(parts)


def output_name_for(infile: Path, indir: Path, outdir: Path, particle: str) -> Path:
    stem = stem_with_relpath(infile, indir)
    return outdir / f"{particle}_{stem}.i3.gz"


def log_name_for(infile: Path, indir: Path, logdir: Path, particle: str) -> Path:
    stem = stem_with_relpath(infile, indir)
    return logdir / f"{particle}_{stem}.out"


def process_one_file(FilterFrame, infile: Path, gcd: Path, allowed: List[int], outfile: Path) -> None:
    tray = icetray.I3Tray()
    tray.Add("I3Reader", FilenameList=[str(gcd), str(infile)])
    tray.Add(FilterFrame, AllowedStrings=allowed, OnlyDAQ=True)
    tray.Add("I3Writer", Filename=str(outfile), Streams=[icetray.I3Frame.DAQ])
    tray.Execute()
    tray.Finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Folder containing input i3 files")
    ap.add_argument(
        "--particle",
        required=True,
        type=str.lower,
        choices=["electron", "muon", "tau", "nc"],
        help="Particle flavor (case-insensitive)",
    )
    ap.add_argument("--pattern", required=True, help="Glob pattern for input files (e.g. '*.i3.zst')")
    ap.add_argument("--sub-geometry", default=None, help="Sub-geometry name used for logging")
    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--logdir", required=True, help="Folder for per-file log outputs")
    ap.add_argument("--gcd", required=True, help="GCD file path")
    ap.add_argument("--selection", required=True, help="Selection file containing allowed string IDs")
    ap.add_argument("--filterframe", default=DEFAULT_FILTERFRAME_PY, help="Absolute path to FilterFrame.py")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing output and log files")
    ap.add_argument("--task-id", type=int, default=None, help="Override task index (for local tests)")

    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    logdir = Path(args.logdir).resolve()
    gcd = Path(args.gcd).resolve()
    selection = Path(args.selection).resolve()

    if not gcd.exists():
        print(f"ERROR: GCD not found: {gcd}")
        return 2
    if not selection.exists():
        print(f"ERROR: selection file not found: {selection}")
        return 2

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

    inputs = list_inputs(str(indir), args.pattern)
    if not inputs:
        print(f"ERROR: No files found in {indir} matching pattern {args.pattern}")
        return 3

    if not (0 <= task_id < len(inputs)):
        print(f"ERROR: task_id={task_id} out of range (0..{len(inputs)-1}), inputs={len(inputs)}")
        return 4

    infile = inputs[task_id]
    outfile = output_name_for(infile, indir, outdir, args.particle)
    logfile = log_name_for(infile, indir, logdir, args.particle)

    # skip if both exist and not overwriting
    if outfile.exists() and logfile.exists() and not args.overwrite:
        print(f"[skip] both exist: {outfile.name}, {logfile.name}")
        return 0

    # redirect stdout+stderr to log file from here on
    log_fh = open(logfile, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    import time
    t_start = time.time()

    print(f"=== SKIM JOB STARTED ===")
    print(f"array_job_id  : {array_job_id}")
    print(f"array_task_id : {array_task_id}")
    print(f"job_id        : {job_id}")
    print(f"particle      : {args.particle}")
    print(f"sub_geometry  : {args.sub_geometry}")
    print(f"pattern       : {args.pattern}")
    print(f"task_id       : {task_id} / {len(inputs) - 1}")
    print(f"indir         : {indir}")
    print(f"infile        : {infile}")
    print(f"outfile       : {outfile}")
    print(f"logfile       : {logfile}")
    print(f"gcd           : {gcd}")
    print(f"selection     : {selection}")
    print(f"filterframe   : {args.filterframe}")
    log_fh.flush()

    try:
        FilterFrame = import_filterframe(args.filterframe)
        allowed = read_allowed_strings(str(selection))
        print(f"allowed_strings_count : {len(allowed)}")
        log_fh.flush()

        print(f"--- Processing started")
        log_fh.flush()

        process_one_file(FilterFrame, infile, gcd, allowed, outfile)

        elapsed = time.time() - t_start
        print(f"--- Processing done")
        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")
    except Exception as e:
        elapsed = time.time() - t_start
        print(f"=== FAILED  elapsed={elapsed:.1f}s ===")
        print(f"ERROR: {e}")
        log_fh.flush()
        log_fh.close()
        return 1

    log_fh.flush()
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
