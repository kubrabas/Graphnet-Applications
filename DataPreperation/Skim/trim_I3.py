"""
Skim ONE i3 file per Slurm array task, using FilterFrame.

- FilterFrame is imported from this Skim directory by default.

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
- Optional OM exclusion: --exclude-oms accepts comma/space-separated OM IDs to drop.
- Known bad files listed in Metadata/paths.py are handled specially:
  no-DAQ files are skipped, and files with available_daq_counts are stopped
  after that many DAQ frames.
- --indir and --pattern are always passed explicitly (from paths.py via submit_skim_I3.py).
"""

import argparse
import importlib.util
import os
import re
import sys
from datetime import datetime
from pathlib import Path
from typing import List
from zoneinfo import ZoneInfo

from icecube import icetray, dataio  # noqa: F401
from icecube.icetray import I3LogLevel

icetray.I3Logger.global_logger.set_level(I3LogLevel.LOG_INFO)

DEFAULT_FILTERFRAME_PY = str(Path(__file__).resolve().parent / "FilterFrame.py")
PATHS_PY = Path(__file__).resolve().parents[2] / "Metadata" / "paths.py"
BERLIN_TZ = ZoneInfo("Europe/Berlin")


def import_filterframe(filterframe_py: str):
    p = Path(filterframe_py).resolve()
    if not p.exists():
        raise FileNotFoundError(f"FilterFrame.py not found at: {p}")
    sys.path.insert(0, str(p.parent))
    from FilterFrame import FilterFrame  # type: ignore
    return FilterFrame


def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def read_allowed_strings(selection_path: str) -> List[int]:
    txt = Path(selection_path).read_text()
    return parse_int_list(txt)


def parse_int_list(text: str) -> List[int]:
    nums = list(map(int, re.findall(r"\d+", text)))
    seen = set()
    ordered: List[int] = []
    for n in nums:
        if n not in seen:
            seen.add(n)
            ordered.append(n)
    return ordered


def berlin_timestamp() -> str:
    return datetime.now(BERLIN_TZ).strftime("%H:%M %d %B %Y")


def status_line(status: str, elapsed: float) -> str:
    return f"=== {status}  @ {berlin_timestamp()}: elapsed={elapsed:.1f}s ==="


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


def bad_file_rule(infile: Path):
    target = str(infile.resolve())
    bad_i3_files = getattr(load_paths(), "BAD_I3_FILES", {})

    for flavors in bad_i3_files.values():
        for info in flavors.values():
            if target in set(map(str, info.get("no_daq_for_some_reason", set()))):
                return "skip_no_daq", None

            available_daq_counts = info.get("available_daq_counts", {})
            if target in available_daq_counts:
                return "limit_daq", int(available_daq_counts[target])

    return "normal", None


class DAQFrameLimiter(icetray.I3ConditionalModule):
    def __init__(self, ctx):
        super(DAQFrameLimiter, self).__init__(ctx)
        self.AddOutBox("OutBox")
        self.AddParameter("MaxDAQFrames", "Maximum DAQ frames to process; <=0 means unlimited", 0)

    def Configure(self):
        self.max_daq_frames = int(self.GetParameter("MaxDAQFrames"))
        self.daq_count = 0

    def Process(self):
        frame = self.PopFrame()

        if frame.Stop == icetray.I3Frame.DAQ:
            self.daq_count += 1
            if self.max_daq_frames > 0 and self.daq_count > self.max_daq_frames:
                self.RequestSuspension()
                return

        self.PushFrame(frame)

        if frame.Stop == icetray.I3Frame.DAQ and self.max_daq_frames > 0 and self.daq_count >= self.max_daq_frames:
            self.RequestSuspension()


def process_one_file(
    FilterFrame,
    infile: Path,
    gcd: Path,
    allowed: List[int],
    excluded_oms: List[int],
    max_daq_frames: int,
    outfile: Path,
) -> None:
    tray = icetray.I3Tray()
    tray.Add("I3Reader", FilenameList=[str(gcd), str(infile)])
    tray.Add(DAQFrameLimiter, MaxDAQFrames=max_daq_frames)
    tray.Add(FilterFrame, AllowedStrings=allowed, ExcludedOMs=excluded_oms, OnlyDAQ=True)
    tray.Add("I3Writer", Filename=str(outfile), Streams=[icetray.I3Frame.TrayInfo, icetray.I3Frame.Simulation, icetray.I3Frame.DAQ])
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
    ap.add_argument(
        "--exclude-oms",
        default=os.environ.get("EXCLUDE_OMS", ""),
        help="Optional comma/space-separated OM IDs to drop within allowed strings",
    )
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
    print(f"exclude_oms   : {args.exclude_oms or '(none)'}")
    print(f"filterframe   : {args.filterframe}")
    log_fh.flush()

    try:
        rule, available_daq_count = bad_file_rule(infile)
        if rule == "skip_no_daq":
            print("--- Skipping: input file is listed in paths.BAD_I3_FILES as no_daq_for_some_reason")
            elapsed = time.time() - t_start
            print(status_line("SKIPPED", elapsed))
            log_fh.flush()
            log_fh.close()
            return 0

        max_daq_frames = available_daq_count if rule == "limit_daq" else 0
        if max_daq_frames > 0:
            print(f"available_daq_count : {max_daq_frames}")
        else:
            print("available_daq_count : unlimited")

        FilterFrame = import_filterframe(args.filterframe)
        allowed = read_allowed_strings(str(selection))
        excluded_oms = parse_int_list(args.exclude_oms)
        print(f"allowed_strings_count : {len(allowed)}")
        print(f"excluded_oms          : {excluded_oms}")
        log_fh.flush()

        print(f"--- Processing started")
        log_fh.flush()

        process_one_file(FilterFrame, infile, gcd, allowed, excluded_oms, max_daq_frames, outfile)

        elapsed = time.time() - t_start
        print(f"--- Processing done")
        print(status_line("SUCCESS", elapsed))
    except Exception as e:
        elapsed = time.time() - t_start
        print(status_line("FAILED", elapsed))
        print(f"ERROR: {e}")
        log_fh.flush()
        log_fh.close()
        return 1

    log_fh.flush()
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
