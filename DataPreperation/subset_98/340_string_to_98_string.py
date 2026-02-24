#!/usr/bin/env python3
# -*- coding: utf-8 -*-

"""
Skim ONE i3 file per Slurm array task, using FilterFrame.

- FilterFrame is imported from:
    /project/def-nahee/kbas/GeometrySkimmer/FilterFrame.py

- Each Slurm array task selects ONE input file by index:
    index = SLURM_ARRAY_TASK_ID
  from the sorted list of files in --indir matching --pattern.

- For that one file:
    reads [GCD, input] -> applies FilterFrame(AllowedStrings=...) -> writes DAQ stream to:
    <outdir>/<input_stem>_skim.i3   (or keeps suffixes like .i3.gz if present)

Key fix in this version:
- Do NOT use Streams=... when adding FilterFrame (your IceTray build treats it as a module parameter).
- Instead, run FilterFrame ONLY on DAQ frames via If=... so it won't touch Geometry/Cal/DetStatus frames.

Run (inside IceTray container env):
  python3 340_string_to_98_string.py \
    --indir /path/to/Photon \
    --pattern "*.i3" \
    --outdir /project/def-nahee/kbas/98_string/Raw_Muon \
    --gcd /project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz \
    --selection /project/.../string_ids.csv

Notes:
- Deterministic: input files are sorted.
- Skips output if it already exists (unless --overwrite).
- Selection file: extracts ALL integers (comma-separated, multi-line, spaces are OK).
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
    """Import FilterFrame from an absolute path to FilterFrame.py."""
    p = Path(filterframe_py).resolve()
    if not p.exists():
        raise FileNotFoundError(f"FilterFrame.py not found at: {p}")
    sys.path.insert(0, str(p.parent))
    from FilterFrame import FilterFrame  # type: ignore
    return FilterFrame


def read_allowed_strings(selection_path: str) -> List[int]:
    """Extract all integers; keep unique in first-seen order."""
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
    return sorted([p for p in base.glob(pattern) if p.is_file()])


def output_name_for(infile: Path, outdir: Path, suffix: str = "_skim") -> Path:
    suffixes = "".join(infile.suffixes) 
    name_no_suffixes = infile.name[: -len(suffixes)] if suffixes else infile.stem
    return outdir / f"{name_no_suffixes}{suffix}.i3.gz"


def process_one_file(FilterFrame, infile: Path, gcd: Path, allowed: List[int], outfile: Path) -> None:
    tray = icetray.I3Tray()

    # Read GCD first, then data
    tray.Add("I3Reader", FilenameList=[str(gcd), str(infile)])

    # IMPORTANT: only run FilterFrame on DAQ frames (avoid Geometry/Cal/DetStatus)
    tray.Add(
        FilterFrame,
        AllowedStrings=allowed,
        OnlyDAQ=True,
    )



    # Match your GeoSkimmer behavior: write DAQ only
    tray.Add("I3Writer", Filename=str(outfile), Streams=[icetray.I3Frame.DAQ])

    tray.Execute()
    tray.Finish()


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--indir", required=True, help="Folder containing input i3 files")
    ap.add_argument("--pattern", default="*.i3.zst", help="Glob pattern, default: *.i3.zst")

    ap.add_argument("--outdir", required=True, help="Output folder")
    ap.add_argument("--gcd", required=True, help="GCD file path")
    ap.add_argument("--selection", required=True, help="Selection file containing allowed string IDs")

    ap.add_argument("--filterframe", default=DEFAULT_FILTERFRAME_PY, help="Absolute path to FilterFrame.py")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing outputs")

    # Optional: local testing without Slurm
    ap.add_argument("--task-id", type=int, default=None, help="Override task index (for local tests)")

    args = ap.parse_args()

    indir = Path(args.indir).resolve()
    outdir = Path(args.outdir).resolve()
    gcd = Path(args.gcd).resolve()
    selection = Path(args.selection).resolve()

    if not gcd.exists():
        print(f"ERROR: GCD not found: {gcd}")
        return 2
    if not selection.exists():
        print(f"ERROR: selection file not found: {selection}")
        return 2

    outdir.mkdir(parents=True, exist_ok=True)

    # task index
    if args.task_id is not None:
        task_id = args.task_id
    else:
        task_id = int(os.environ.get("SLURM_ARRAY_TASK_ID", "-1"))
        if task_id < 0:
            print("ERROR: SLURM_ARRAY_TASK_ID not set. Use --task-id for local testing.")
            return 2

    # list inputs (deterministic)
    inputs = list_inputs(str(indir), args.pattern)
    if not inputs:
        print(f"ERROR: No files found in {indir} matching pattern {args.pattern}")
        return 3

    if not (0 <= task_id < len(inputs)):
        print(f"ERROR: task_id={task_id} out of range (0..{len(inputs)-1}), inputs={len(inputs)}")
        return 4

    infile = inputs[task_id]
    outfile = output_name_for(infile, outdir, suffix="_skim")

    if outfile.exists() and not args.overwrite:
        icetray.logging.log_info(f"[skip] exists: {outfile}")
        return 0

    # import + selection
    FilterFrame = import_filterframe(args.filterframe)
    allowed = read_allowed_strings(str(selection))

    icetray.logging.log_info(f"task_id={task_id}/{len(inputs)-1}")
    icetray.logging.log_info(f"infile={infile}")
    icetray.logging.log_info(f"outfile={outfile}")
    icetray.logging.log_info(f"AllowedStrings count={len(allowed)}")

    process_one_file(FilterFrame, infile, gcd, allowed, outfile)
    icetray.logging.log_info("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
