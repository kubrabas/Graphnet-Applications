"""
Split existing merged parquet datasets into category-specific parquet outputs.

This script does not read I3 files. It filters truth rows by a category column,
then filters the matching features rows by event_no.
"""

import argparse
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import List, Tuple
from zoneinfo import ZoneInfo

TABLES = ("truth", "features")


def category_value_label(value: str) -> str:
    text = str(value).replace("-", "minus").replace(".", "p")
    return f"category{text}"


def batch_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Could not parse batch id from {path}")
    return int(match.group(1))


def prepare_output(outdir: Path, overwrite: bool) -> bool:
    existing = []
    for table in TABLES:
        table_dir = outdir / table
        table_dir.mkdir(parents=True, exist_ok=True)
        existing.extend(table_dir.glob("*.parquet"))

    if existing and not overwrite:
        print(f"Output already exists, skipping: {outdir}")
        return False

    if overwrite:
        for path in existing:
            path.unlink()
    return True


def pop_event_batch(truth_buffer, features_buffer, events_per_batch: int):
    event_batch = truth_buffer.head(events_per_batch).select("event_no")
    truth_batch = truth_buffer.join(event_batch, on="event_no", how="inner")
    features_batch = features_buffer.join(event_batch, on="event_no", how="inner")
    truth_buffer = truth_buffer.join(event_batch, on="event_no", how="anti")
    features_buffer = features_buffer.join(event_batch, on="event_no", how="anti")
    return truth_batch, features_batch, truth_buffer, features_buffer


def write_batch(outdir: Path, out_batch: int, truth_batch, features_batch) -> Tuple[int, int]:
    truth_out = outdir / "truth" / f"truth_{out_batch}.parquet"
    features_out = outdir / "features" / f"features_{out_batch}.parquet"
    truth_batch.write_parquet(truth_out)
    features_batch.write_parquet(features_out)
    return truth_batch.height, features_batch.height


def split_category(
    source_split_dir: Path,
    outdir: Path,
    category_column: str,
    category_value: str,
    events_per_batch: int,
    overwrite: bool,
) -> dict:
    import polars as pl

    truth_dir = source_split_dir / "truth"
    feature_dir = source_split_dir / "features"
    truth_files = sorted(truth_dir.glob("truth_*.parquet"), key=batch_id)
    if not truth_files:
        raise FileNotFoundError(f"No truth parquet files found in {truth_dir}")

    if not prepare_output(outdir, overwrite):
        return {
            "source_truth_files": len(truth_files),
            "output_batches": 0,
            "truth_rows": 0,
            "feature_rows": 0,
            "skipped_existing": True,
        }

    out_batch = 0
    truth_rows = 0
    feature_rows = 0
    buffer_truth: List[pl.DataFrame] = []
    buffer_features: List[pl.DataFrame] = []

    for truth_file in truth_files:
        bid = batch_id(truth_file)
        feature_file = feature_dir / f"features_{bid}.parquet"
        if not feature_file.exists():
            raise FileNotFoundError(f"Missing matching features file: {feature_file}")

        truth = pl.read_parquet(truth_file)
        if category_column not in truth.columns:
            raise ValueError(f"Missing column {category_column} in {truth_file}")
        if "event_no" not in truth.columns:
            raise ValueError(f"Missing event_no in {truth_file}")

        truth_filtered = truth.filter(
            pl.col(category_column).cast(pl.Utf8) == str(category_value)
        )
        if truth_filtered.is_empty():
            continue

        event_ids = truth_filtered.select("event_no").unique()
        features = pl.read_parquet(feature_file)
        if "event_no" not in features.columns:
            raise ValueError(f"Missing event_no in {feature_file}")
        features_filtered = features.join(event_ids, on="event_no", how="inner")

        buffer_truth.append(truth_filtered)
        buffer_features.append(features_filtered)

        truth_buffer = pl.concat(buffer_truth)
        features_buffer = pl.concat(buffer_features)
        while truth_buffer.height >= events_per_batch:
            truth_batch, features_batch, truth_buffer, features_buffer = pop_event_batch(
                truth_buffer=truth_buffer,
                features_buffer=features_buffer,
                events_per_batch=events_per_batch,
            )
            n_truth, n_features = write_batch(
                outdir=outdir,
                out_batch=out_batch,
                truth_batch=truth_batch,
                features_batch=features_batch,
            )
            truth_rows += n_truth
            feature_rows += n_features
            out_batch += 1

        buffer_truth = [truth_buffer] if not truth_buffer.is_empty() else []
        buffer_features = [features_buffer] if not features_buffer.is_empty() else []

    if buffer_truth:
        truth_buffer = pl.concat(buffer_truth)
        features_buffer = pl.concat(buffer_features)
        n_truth, n_features = write_batch(
            outdir=outdir,
            out_batch=out_batch,
            truth_batch=truth_buffer,
            features_batch=features_buffer,
        )
        truth_rows += n_truth
        feature_rows += n_features
        out_batch += 1

    return {
        "source_truth_files": len(truth_files),
        "output_batches": out_batch,
        "events_per_batch": events_per_batch,
        "truth_rows": truth_rows,
        "feature_rows": feature_rows,
        "skipped_existing": False,
    }


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Split existing parquet files into category-specific parquet outputs."
    )
    ap.add_argument("--mc", required=True)
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--source-split-dir", required=True)
    ap.add_argument("--outdir", required=True)
    ap.add_argument("--logdir", required=True)
    ap.add_argument("--split", required=True, choices=["train", "val", "test"])
    ap.add_argument("--category-column", required=True)
    ap.add_argument("--category-value", required=True)
    ap.add_argument("--events-per-batch", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    source_split_dir = Path(args.source_split_dir)
    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)

    job_id = "local"
    import os

    job_id = os.environ.get("SLURM_JOB_ID", job_id)
    date_stamp = datetime.now(ZoneInfo("Europe/Berlin")).strftime("%m_%d_%Y")
    value_label = category_value_label(args.category_value)
    log_path = logdir / f"{value_label}_{args.split}_{date_stamp}_job_{job_id}.log"

    t_start = time.time()
    with open(log_path, "w") as log_fh:
        sys.stdout = log_fh
        sys.stderr = log_fh
        try:
            print("=== CATEGORIZED PARQUET SPLIT JOB ===")
            print(f"job_id          : {job_id}")
            print(f"mc              : {args.mc}")
            print(f"flavor          : {args.flavor}")
            print(f"geometry        : {args.geometry}")
            print(f"split           : {args.split}")
            print(f"category_column : {args.category_column}")
            print(f"category_value  : {args.category_value}")
            print(f"source_split_dir: {source_split_dir}")
            print(f"outdir          : {outdir}")
            print(f"events_per_batch: {args.events_per_batch}")
            print(f"overwrite       : {args.overwrite}")

            stats = split_category(
                source_split_dir=source_split_dir,
                outdir=outdir,
                category_column=args.category_column,
                category_value=args.category_value,
                events_per_batch=args.events_per_batch,
                overwrite=args.overwrite,
            )

            elapsed = time.time() - t_start
            print("=== FINAL SUMMARY ===")
            for key, value in stats.items():
                print(f"{key} : {value}")
            print(f"elapsed : {elapsed:.1f}s")
            print(f"log     : {log_path}")
            return 0
        except Exception as e:
            elapsed = time.time() - t_start
            print(f"=== FAILED elapsed={elapsed:.1f}s error={e} ===")
            traceback.print_exc()
            return 1


if __name__ == "__main__":
    raise SystemExit(main())
