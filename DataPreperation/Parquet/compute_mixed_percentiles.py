"""
Compute p25/p50/p75 feature percentiles from the combined (mixed) training set
of multiple flavors, for use as RobustScaler normalization.

Reads per-flavor train_reindexed/features/ directories from paths.py,
loads all matching feature Parquet files eagerly with Polars, and writes the
category-free classification scaler to:
  METADATA_ROBUSTSCALER/<mc>/mixed/<geometry>/classification/
    train_feature_percentiles_p25_p50_p75.csv

The same run then computes one mixed percentile CSV per class for all three
configured categories. For each class, the script combines train/features
directories across all requested flavors, skips paths marked "does_not_exist",
and writes to:
  METADATA_ROBUSTSCALER/<mc>/mixed/<geometry>/<category>/<class_label>/
    train_feature_percentiles_p25_p50_p75.csv

Usage:
    python3 compute_mixed_percentiles.py --mc 340StringMC --geometry 102_string_emax1e6
    python3 compute_mixed_percentiles.py --mc 340StringMC --geometry full_geometry_emax1e6

Full-geometry processing can require substantially more memory than the
reduced geometries. Request a high-memory compute node before running it:

salloc \
  --time=1:00:00 \
  --account=def-nahee \
  --nodes=1 \
  --ntasks=1 \
  --cpus-per-task=8 \
  --mem=128G

Then activate the Parquet environment and run the script:

source /project/def-nahee/kbas/.venv_try/bin/activate
cd /project/def-nahee/kbas/Graphnet-Applications/DataPreperation/Parquet

python3 compute_mixed_percentiles.py \
  --mc 340StringMC \
  --geometry full_geometry_emax1e6
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
PERCENTILE_FILENAME  = "train_feature_percentiles_p25_p50_p75.csv"

CATEGORY_CLASS_DIRS = {
    "category1_isMuonCC": {
        "0": "class_0_not_muon_cc",
        "1": "class_1_muon_cc",
    },
    "category2_tauCC_others_muonCC": {
        "0": "class_0_tau_cc",
        "1": "class_1_electron_cc_or_nc",
        "2": "class_2_muon_cc",
    },
    "category_3_contains_muon": {
        "0": "class_0_no_muon",
        "1": "class_1_contains_muon",
    },
}

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


def category_class_dir(category: str, category_id: str) -> str:
    """Return a stable, descriptive output directory for a category class."""
    category_id = str(category_id)
    try:
        return CATEGORY_CLASS_DIRS[category][category_id]
    except KeyError as e:
        raise ValueError(
            f"No output label configured for {category}[{category_id}]."
        ) from e


def compute_percentiles(feature_dirs, csv_path: Path) -> int:
    all_files = []
    for _, d in feature_dirs:
        all_files.extend(sorted(d.glob("*.parquet")))

    if not all_files:
        print("ERROR: No .parquet files found in feature directories.")
        return 1

    print(f"\nReading {len(all_files)} parquet files (memory_map=False)...")
    df = pl.read_parquet([str(f) for f in all_files], memory_map=False)

    num_cols = [c for c, t in df.schema.items() if t.is_numeric() and c not in EXCLUDE_COLS]

    if not num_cols:
        print("ERROR: No numeric feature columns found.")
        return 1

    print(f"Computing percentiles for {len(num_cols)} columns...")
    q25 = df.select([pl.col(c).quantile(0.25) for c in num_cols])
    q50 = df.select([pl.col(c).quantile(0.50) for c in num_cols])
    q75 = df.select([pl.col(c).quantile(0.75) for c in num_cols])

    result = pd.DataFrame({
        "feature": num_cols,
        "p25": [q25[c][0] for c in num_cols],
        "p50": [q50[c][0] for c in num_cols],
        "p75": [q75[c][0] for c in num_cols],
    })

    csv_path.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(str(csv_path), index=False, float_format="%.16f")

    print(f"\nmixed percentiles -> {csv_path}")
    print(result.to_string(index=False))
    return 0


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Compute mixed-flavor RobustScaler percentiles.")
    ap.add_argument("--mc",       required=True, choices=list(PARQUET_TABLE))
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--flavors",  nargs="+", default=ALL_FLAVORS,
                    help=f"Flavors to include (default: {ALL_FLAVORS})")
    ap.add_argument(
        "--category",
        nargs="+",
        choices=list(CATEGORY_CLASS_DIRS),
        default=list(CATEGORY_CLASS_DIRS),
        help="Categories to process after classification (default: all configured categories).",
    )
    args = ap.parse_args()

    paths = load_paths()
    table = getattr(paths, PARQUET_TABLE[args.mc])
    geometry_entry = table.get(args.geometry, {})
    out_dir = Path(METADATA_ROBUSTSCALER) / args.mc / "mixed" / args.geometry

    print("\n=== classification: all training events ===")
    feature_dirs = []
    for flavor in args.flavors:
        entry = geometry_entry.get(flavor, {})
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
        print("ERROR: No classification feature directories found.")
        return 1

    csv_path = out_dir / "classification" / PERCENTILE_FILENAME
    rc = compute_percentiles(feature_dirs, csv_path)
    if rc != 0:
        return rc

    for category in args.category:
        print(f"\n=== routed reconstruction: {category} ===")
        category_ids = set()
        for flavor in args.flavors:
            entry = geometry_entry.get(flavor, {})
            category_entry = entry.get(category)
            if category_entry is None:
                print(f"ERROR: {flavor}: {category} is not set in paths.py")
                return 1
            category_ids.update(category_entry.keys())

        if not category_ids:
            print(f"ERROR: No category ids found for {category}.")
            return 1

        for category_id in sorted(category_ids, key=str):
            feature_dirs = []
            for flavor in args.flavors:
                entry = geometry_entry.get(flavor, {})
                category_entry = entry[category]
                split_entry = category_entry.get(category_id)
                if split_entry is None:
                    print(f"ERROR: {flavor}: {category}[{category_id}] is not set in paths.py")
                    return 1

                train_path = split_entry.get("train")
                if train_path is None:
                    print(f"ERROR: {flavor}: {category}[{category_id}].train is not set in paths.py")
                    return 1
                if train_path == "does_not_exist":
                    print(f"[skip] {flavor}: {category}[{category_id}].train = does_not_exist")
                    continue

                feat_dir = Path(train_path) / "features"
                if not feat_dir.exists():
                    print(f"ERROR: {flavor}: {feat_dir} does not exist")
                    return 1
                feature_dirs.append((flavor, feat_dir))
                print(f"[found] {flavor}: {category}[{category_id}]: {feat_dir}")

            if not feature_dirs:
                print(f"[skip] {category}[{category_id}]: no feature directories found")
                continue

            try:
                class_dir = category_class_dir(category, category_id)
            except ValueError as e:
                print(f"ERROR: {e}")
                return 1
            csv_path = out_dir / category / class_dir / PERCENTILE_FILENAME
            rc = compute_percentiles(feature_dirs, csv_path)
            if rc != 0:
                return rc

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
