"""
Compute p25/p50/p75 feature percentiles from the combined (mixed) training set
of multiple flavors, for use as RobustScaler normalization in classification.

Reads per-flavor train_reindexed/features/ directories from paths.py,
scans them lazily with Polars, and saves a single CSV to:
  METADATA_ROBUSTSCALER/<mc>/<geometry>_mixed_train_feature_percentiles_p25_p50_p75.csv

Usage:
    python3 compute_mixed_percentiles.py --mc 340StringMC --geometry 102_string
    python3 compute_mixed_percentiles.py --mc 340StringMC --geometry full_geometry --flavors Muon Electron Tau NC
"""

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import polars as pl

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

PATHS_PY             = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
METADATA_ROBUSTSCALER = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/RobustScaler"
ALL_FLAVORS          = ["Muon", "Electron", "Tau", "NC"]

PARQUET_TABLE = {
    "340StringMC":  "STRING340MC_PARQUET",
    "Spring2026MC": "SPRING2026MC_PARQUET",
}

EXCLUDE_COLS = {"event_no", "global_event_no"}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute mixed-flavor RobustScaler percentiles.")
    ap.add_argument("--mc",       required=True, choices=list(PARQUET_TABLE))
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--flavors",  nargs="+", default=ALL_FLAVORS,
                    help=f"Flavors to include (default: {ALL_FLAVORS})")
    args = ap.parse_args()

    paths = load_paths()
    table = getattr(paths, PARQUET_TABLE[args.mc])

    feature_dirs = []
    for flavor in args.flavors:
        entry = table.get(args.geometry, {}).get(flavor, {})
        train_path = entry.get("train")
        if train_path is None:
            print(f"[skip] {flavor}: train path not set in paths.py")
            continue
        feat_dir = Path(train_path) / "features"
        if not feat_dir.exists():
            print(f"[skip] {flavor}: {feat_dir} does not exist")
            continue
        feature_dirs.append((flavor, feat_dir))
        print(f"[found] {flavor}: {feat_dir}")

    if not feature_dirs:
        print("ERROR: No feature directories found.")
        return 1

    patterns = [str(d / "*.parquet") for _, d in feature_dirs]
    print(f"\nScanning {len(patterns)} directories with Polars...")
    lf = pl.scan_parquet(patterns)

    schema   = lf.collect_schema()
    num_cols = [c for c, t in schema.items() if t.is_numeric() and c not in EXCLUDE_COLS]

    if not num_cols:
        print("ERROR: No numeric feature columns found.")
        return 1

    print(f"Computing percentiles for {len(num_cols)} columns...")
    q25 = lf.select([pl.col(c).quantile(0.25).alias(c) for c in num_cols]).collect()
    q50 = lf.select([pl.col(c).quantile(0.50).alias(c) for c in num_cols]).collect()
    q75 = lf.select([pl.col(c).quantile(0.75).alias(c) for c in num_cols]).collect()

    result = pd.DataFrame({
        "feature": num_cols,
        "p25": [q25[c][0] for c in num_cols],
        "p50": [q50[c][0] for c in num_cols],
        "p75": [q75[c][0] for c in num_cols],
    })

    out_dir = Path(METADATA_ROBUSTSCALER) / args.mc
    out_dir.mkdir(parents=True, exist_ok=True)

    flavors_tag = "_".join(args.flavors).lower()
    csv_name    = f"{args.geometry}_mixed_{flavors_tag}_train_feature_percentiles_p25_p50_p75.csv"
    csv_path    = out_dir / csv_name
    result.to_csv(str(csv_path), index=False, float_format="%.16f")

    print(f"\nmixed percentiles -> {csv_path}")
    print(result.to_string(index=False))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
