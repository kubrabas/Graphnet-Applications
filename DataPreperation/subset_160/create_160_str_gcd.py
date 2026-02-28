"""
Create a reduced GCD file for a selected subset of strings.

Reads an input GCD (.i3 / .i3.gz / .i3.zst), applies FilterFrame with
OnlyDAQ=False (so it filters Geometry/Calibration/DetectorStatus),
and writes ONLY the G, C, D frames to a new output GCD file.

Usage example:
  python3 create_160_str_gcd.py \
    --gcd-in  /project/6008051/pone_simulation/GCD_Library/PONE_800mGrid.i3.gz \
    --gcd-out /scratch/kbas/160_string/GCD_160strings.i3.gz \
    --selection /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/subset_160/string_ids_160.csv

Optional:
  --filterframe /project/def-nahee/kbas/GeometrySkimmer/FilterFrame.py
  --overwrite
"""

import argparse
import re
import sys
from pathlib import Path
from typing import List

from icecube import icetray  # noqa: F401
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


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--gcd-in", required=True, help="Input (full) GCD file path")
    ap.add_argument("--gcd-out", required=True, help="Output (reduced) GCD file path")
    ap.add_argument("--selection", required=True, help="Text file containing allowed string IDs")
    ap.add_argument("--filterframe", default=DEFAULT_FILTERFRAME_PY, help="Absolute path to FilterFrame.py")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite output if exists")
    args = ap.parse_args()

    gcd_in = Path(args.gcd_in).resolve()
    gcd_out = Path(args.gcd_out).resolve()
    selection = Path(args.selection).resolve()

    if not gcd_in.exists():
        print(f"ERROR: input GCD not found: {gcd_in}")
        return 2
    if not selection.exists():
        print(f"ERROR: selection file not found: {selection}")
        return 2

    if gcd_out.exists() and not args.overwrite:
        icetray.logging.log_info(f"[skip] exists: {gcd_out}")
        return 0

    gcd_out.parent.mkdir(parents=True, exist_ok=True)

    FilterFrame = import_filterframe(args.filterframe)
    allowed = read_allowed_strings(str(selection))

    icetray.logging.log_info(f"gcd_in={gcd_in}")
    icetray.logging.log_info(f"gcd_out={gcd_out}")
    icetray.logging.log_info(f"AllowedStrings count={len(allowed)}")

    tray = icetray.I3Tray()

    # Read ONLY the GCD file
    tray.Add("I3Reader", Filename=str(gcd_in))

    # Filter G/C/D frames (OnlyDAQ=False) using your module logic
    tray.Add(
        FilterFrame,
        AllowedStrings=allowed,
        OnlyDAQ=False,
    )

    # Write ONLY G, C, D frames to output GCD
    tray.Add(
        "I3Writer",
        Filename=str(gcd_out),
        Streams=[
            icetray.I3Frame.Geometry,
            icetray.I3Frame.Calibration,
            icetray.I3Frame.DetectorStatus,
        ],
    )

    tray.Execute()
    tray.Finish()

    icetray.logging.log_info("DONE")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())