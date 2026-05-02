"""
Merge partial LIW CSVs into a single file and compute final LIW weights.

After all SLURM array tasks finish, each flavor has many partial CSVs:
    <outdir>/<Flavor>_<batch_id>.csv   (raw oneweight, no scaling)

This script:
  1. Collects all partial CSVs for the given flavor.
  2. Concatenates them.
  3. Computes LIW = oneweight / n_files  (n_files = number of partial CSVs found).
  4. Drops the raw oneweight column.
  5. Writes: <outdir>/<Flavor>_LIW.csv
  6. Deletes all partial CSVs.

Usage:
    python3 merge_LIW.py --mc 340StringMC --flavor Muon
    python3 merge_LIW.py --mc 340StringMC --flavor Muon Electron Tau NC
    python3 merge_LIW.py --mc 340StringMC --flavor all
    python3 merge_LIW.py --dry-run --mc 340StringMC --flavor Muon
"""

import argparse
import re
from pathlib import Path

import pandas as pd

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

BASE_OUTDIR = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/EventWeights"

OUTDIR_SUBDIR = {
    "340StringMC":  "String340MC",
    "Spring2026MC": "Spring2026MC",
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def is_partial(path: Path, flavor: str) -> bool:
    """True if file looks like a partial batch CSV (has a numeric batch_id in name)."""
    stem = path.stem  # e.g. "Muon_000001"
    return bool(re.match(rf'^{re.escape(flavor)}_\d+$', stem))


def merge_flavor(flavor: str, outdir: Path, dry_run: bool) -> int:
    partials = sorted(f for f in outdir.glob(f"{flavor}_*.csv") if is_partial(f, flavor))

    if not partials:
        print(f"  [skip] {flavor}: no partial CSVs found in {outdir}")
        return 0

    merged_path = outdir / f"{flavor}_LIW.csv"
    if merged_path.exists():
        print(f"  [warning] {flavor}: {merged_path.name} already exists — overwriting")

    n_files = len(partials)
    print(f"  {flavor}: merging {n_files} files  →  {merged_path.name}")

    if dry_run:
        for p in partials:
            print(f"    {p.name}")
        return 0

    dfs = []
    for p in partials:
        try:
            dfs.append(pd.read_csv(p))
        except Exception as e:
            print(f"  [warning] {flavor}: could not read {p.name} — {e}")

    if not dfs:
        print(f"  [error] {flavor}: all partial CSVs failed to read")
        return 1

    df = pd.concat(dfs, ignore_index=True)
    df["LIW"] = df["oneweight"] / n_files
    df = df.drop(columns=["oneweight"])

    df.to_csv(merged_path, index=False)
    print(f"  {flavor}: wrote {len(df)} events  (n_files={n_files})")

    for p in partials:
        p.unlink()
    print(f"  {flavor}: deleted {len(partials)} partial CSVs")

    return 0

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Merge partial LIW CSVs and compute final weights")
    ap.add_argument("--mc",      required=True, choices=list(OUTDIR_SUBDIR))
    ap.add_argument("--flavor",  required=True, nargs="+",
                    help=f"Flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--dry-run", action="store_true", help="Print what would be done without writing")
    args = ap.parse_args()

    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    outdir = Path(BASE_OUTDIR) / OUTDIR_SUBDIR[args.mc]
    if not outdir.exists():
        print(f"ERROR: output directory not found: {outdir}")
        return 1

    print(f"\n[{args.mc}]  outdir={outdir}")

    for flavor in flavors:
        rc = merge_flavor(flavor, outdir, args.dry_run)
        if rc != 0:
            return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
