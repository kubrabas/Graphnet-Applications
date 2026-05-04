"""
Merge per-batch Parquet files into a single shuffled dataset with train/val/test splits.

After all convert_parquet.py SLURM array tasks finish, this script:
  1. Merges per-batch truth/ and features/ parquet files into shuffled batches.
  2. Renames merged/ -> merged_raw/ (immutable source; never modified).
  3. Builds 80/10/10 train/val/test splits as symlinks into merged_raw/.
  4. Builds *_reindexed/ dirs (sequential 0,1,2,... symlinks for ParquetDataset).
  5. Writes split_manifest.json.
  6. Extracts unique event IDs from truth table -> saves triggered_events.csv.
  7. Computes p25/p50/p75 feature percentiles from training set -> saves CSV for RobustScaler.

Usage:
    python3 merge_parquet.py --mc 340StringMC --flavor Electron --geometry 102_string \\
        --outdir /home/kbas/scratch/String340MC/102_String/Electron_Parquet \\
        --logdir /home/kbas/scratch/String340MC/Logs/Electron_102_String_Parquet
"""

import argparse
import json
import os
import random
import re
import shutil
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import pandas as pd
import polars as pl

from graphnet.data.writers import ParquetWriter

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

METADATA_BASE          = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/TriggeredEventList"
METADATA_ROBUSTSCALER  = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/RobustScaler"
ALL_FLAVORS   = ["Muon", "Electron", "Tau", "NC"]
EVENT_ID_COLS = ["RunID", "SubrunID", "EventID", "SubEventID"]
TABLES        = ["truth", "features"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def link_or_copy(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        return
    try:
        os.link(src, dst)
    except Exception:
        try:
            os.symlink(src.resolve(), dst)
        except Exception:
            shutil.copy2(src, dst)


def reindex_split(split_dir: Path, reindexed_dir: Path) -> None:
    """Create sequential 0,1,2,... symlinks in reindexed_dir pointing into split_dir."""
    rx = re.compile(r"_(\d+)\.parquet$")
    ids = sorted(
        int(rx.search(p.name).group(1))
        for p in (split_dir / "truth").glob("truth_*.parquet")
        if rx.search(p.name)
    )
    for table in TABLES:
        (reindexed_dir / table).mkdir(parents=True, exist_ok=True)
    for new_id, old_id in enumerate(ids):
        for table in TABLES:
            src = (split_dir / table / f"{table}_{old_id}.parquet").resolve()
            dst =  reindexed_dir / table / f"{table}_{new_id}.parquet"
            if dst.exists():
                dst.unlink()
            dst.symlink_to(src)


def build_splits(merged_raw: Path, merged_dir: Path, seed: int = 42) -> Dict[str, List[int]]:
    pat = re.compile(r"^truth_(\d+)\.parquet$")
    batch_ids = sorted(
        int(pat.match(p.name).group(1))
        for p in (merged_raw / "truth").glob("truth_*.parquet")
        if pat.match(p.name)
    )
    if not batch_ids:
        raise RuntimeError(f"No merged batches found in {merged_raw / 'truth'}")

    last_batch  = max(batch_ids)
    main_batches = [b for b in batch_ids if b != last_batch]
    rng = random.Random(seed)
    rng.shuffle(main_batches)

    n       = len(main_batches)
    n_train = int(0.8 * n)
    n_val   = int(0.1 * n)

    splits: Dict[str, List[int]] = {
        "train": main_batches[:n_train],
        "val":   main_batches[n_train: n_train + n_val],
        "test":  main_batches[n_train + n_val:] + [last_batch],
    }

    for split_name, ids in splits.items():
        for table in TABLES:
            for bid in ids:
                src = merged_raw / table / f"{table}_{bid}.parquet"
                dst = merged_dir / split_name / table / f"{table}_{bid}.parquet"
                if src.exists():
                    link_or_copy(src, dst)

    manifest = {
        "seed": seed,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "last_batch_forced_to_test": int(last_batch),
        "batch_counts": {k: len(v) for k, v in splits.items()},
        "splits": {k: sorted(v) for k, v in splits.items()},
    }
    with open(merged_dir / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return splits


def save_triggered_event_list(merged_raw: Path, mc: str, geometry: str, flavor: str) -> None:
    truth_files = sorted((merged_raw / "truth").glob("truth_*.parquet"))
    if not truth_files:
        print(f"[WARN] No truth files in {merged_raw / 'truth'} — skipping event list.")
        return

    probe = pl.read_parquet(truth_files[0], n_rows=1)
    available_id_cols = [c for c in EVENT_ID_COLS if c in probe.columns]
    if not available_id_cols:
        print(f"[WARN] No event ID columns found in truth table. Columns: {probe.columns}")
        return

    dfs = [pl.read_parquet(f, columns=available_id_cols) for f in truth_files]
    combined = pl.concat(dfs).unique()

    out_dir = Path(METADATA_BASE) / mc
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / f"{geometry}_{flavor.lower()}_triggered_events.csv"
    combined.write_csv(str(out_path))
    print(f"triggered event list -> {out_path}  ({len(combined)} unique events)")


def compute_feature_percentiles(
    train_feat_dir: Path, mc: str, geometry: str, flavor: str
) -> None:
    pattern = str(train_feat_dir / "features_*.parquet")
    lf = pl.scan_parquet(pattern)
    schema = lf.collect_schema()
    exclude = {"event_no", "global_event_no"}
    num_cols = [c for c, t in schema.items() if t.is_numeric() and c not in exclude]

    if not num_cols:
        print("[WARN] No numeric feature columns found — skipping percentile CSV.")
        return

    q25 = lf.select([pl.col(c).quantile(0.25).alias(c) for c in num_cols]).collect()
    q50 = lf.select([pl.col(c).quantile(0.50).alias(c) for c in num_cols]).collect()
    q75 = lf.select([pl.col(c).quantile(0.75).alias(c) for c in num_cols]).collect()

    result = pd.DataFrame({
        "feature": num_cols,
        "p25": [q25[c][0] for c in num_cols],
        "p50": [q50[c][0] for c in num_cols],
        "p75": [q75[c][0] for c in num_cols],
    })

    out_dir = Path(METADATA_ROBUSTSCALER) / mc
    out_dir.mkdir(parents=True, exist_ok=True)
    csv_name = f"{geometry}_{flavor.lower()}_train_feature_percentiles_p25_p50_p75.csv"
    csv_path = out_dir / csv_name
    result.to_csv(str(csv_path), index=False, float_format="%.16f")
    print(f"feature percentiles -> {csv_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Merge parquet batches and build train/val/test splits.")
    ap.add_argument("--mc",               required=True)
    ap.add_argument("--flavor",           required=True)
    ap.add_argument("--geometry",         required=True)
    ap.add_argument("--outdir",           required=True, help="Shared parquet output directory")
    ap.add_argument("--logdir",           required=True)
    ap.add_argument("--events-per-batch", type=int, default=1024)
    ap.add_argument("--num-workers",      type=int, default=1)
    ap.add_argument("--dry-run",          action="store_true")
    args = ap.parse_args()

    if args.flavor not in ALL_FLAVORS:
        print(f"ERROR: unknown flavor '{args.flavor}'. Choices: {ALL_FLAVORS}")
        return 1

    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    logfile = logdir / f"merge_{args.flavor}_{args.geometry}.out"
    log_fh  = open(logfile, "w")
    sys.stdout = log_fh
    sys.stderr = log_fh

    t_start = time.time()
    print("=== PARQUET MERGE JOB STARTED ===")
    print(f"mc            : {args.mc}")
    print(f"flavor        : {args.flavor}")
    print(f"geometry      : {args.geometry}")
    print(f"outdir        : {outdir}")
    print(f"events/batch  : {args.events_per_batch}")
    log_fh.flush()

    try:
        merged_dir     = outdir / "merged"
        merged_raw_dir = outdir / "merged_raw"

        # ----------------------------------------------------------------
        # 1. Merge per-batch files into shuffled batches
        # ----------------------------------------------------------------
        if merged_raw_dir.exists():
            print(f"merged_raw already exists, skipping merge: {merged_raw_dir}")
        elif args.dry_run:
            print(f"[DRY-RUN] would merge into {merged_dir} then rename to {merged_raw_dir}")
        else:
            writer = ParquetWriter(truth_table="truth", index_column="event_no")
            writer.merge_files(
                files=[],
                output_dir=str(merged_dir),
                events_per_batch=args.events_per_batch,
                num_workers=args.num_workers,
            )
            os.rename(merged_dir, merged_raw_dir)
            print(f"merged_raw -> {merged_raw_dir}")
        log_fh.flush()

        # ----------------------------------------------------------------
        # 2. Build train/val/test splits + split_manifest.json
        # ----------------------------------------------------------------
        if args.dry_run:
            print("[DRY-RUN] would build train/val/test split dirs")
        else:
            splits = build_splits(merged_raw=merged_raw_dir, merged_dir=merged_dir)
            print(f"splits: { {k: len(v) for k, v in splits.items()} }")
        log_fh.flush()

        # ----------------------------------------------------------------
        # 3. Build *_reindexed dirs
        # ----------------------------------------------------------------
        if not args.dry_run:
            for split_name in ["train", "val", "test"]:
                reindex_split(
                    split_dir     = merged_dir / split_name,
                    reindexed_dir = merged_dir / f"{split_name}_reindexed",
                )
            print("reindexed splits: done")
        log_fh.flush()

        # ----------------------------------------------------------------
        # 4. Save triggered event list
        # ----------------------------------------------------------------
        if not args.dry_run:
            save_triggered_event_list(
                merged_raw = merged_raw_dir,
                mc         = args.mc,
                geometry   = args.geometry,
                flavor     = args.flavor,
            )
        log_fh.flush()

        # ----------------------------------------------------------------
        # 5. Feature percentiles for RobustScaler
        # ----------------------------------------------------------------
        if not args.dry_run:
            compute_feature_percentiles(
                train_feat_dir = merged_dir / "train" / "features",
                mc             = args.mc,
                geometry       = args.geometry,
                flavor         = args.flavor,
            )
        log_fh.flush()

        elapsed = time.time() - t_start
        print(f"=== SUCCESS  elapsed={elapsed:.1f}s ===")

    except Exception as e:
        elapsed = time.time() - t_start
        print(f"=== FAILED  elapsed={elapsed:.1f}s  error={e} ===")
        traceback.print_exc()
        log_fh.flush()
        log_fh.close()
        return 1

    log_fh.flush()
    log_fh.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
