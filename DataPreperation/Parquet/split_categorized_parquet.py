"""
Split existing merged parquet datasets into category-specific parquet outputs.

This script does not read I3 files. It filters truth rows by a category column,
then filters the matching feature rows by event_no.
"""

import argparse
import json
import random
import re
import sys
import time
import traceback
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple
from zoneinfo import ZoneInfo

SPLITS = ["train", "val", "test"]
EVENT_ID_COLS = ["RunID", "SubrunID", "EventID", "SubEventID"]


def category_value_label(value: str) -> str:
    text = str(value).replace("-", "minus").replace(".", "p")
    return f"category{text}"


def batch_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Could not parse batch id from {path}")
    return int(match.group(1))


def truth_stem(path: Path) -> str:
    if not path.name.endswith("_truth.parquet"):
        raise ValueError(f"Unexpected truth parquet name: {path}")
    return path.name[: -len("_truth.parquet")]


def truth_files_in(truth_dir: Path) -> List[Path]:
    files = sorted(truth_dir.glob("*_truth.parquet"))
    if files:
        return files
    return sorted(truth_dir.glob("truth_*.parquet"), key=batch_id)


def feature_file_for_truth(source_dataset_dir: Path, table: str, truth_file: Path) -> Path:
    if truth_file.name.startswith("truth_"):
        bid = batch_id(truth_file)
        return source_dataset_dir / table / f"{table}_{bid}.parquet"
    stem = truth_stem(truth_file)
    return source_dataset_dir / table / f"{stem}_{table}.parquet"


def parquet_tables(source_split_dir: Path) -> List[str]:
    tables = [
        p.name
        for p in source_split_dir.iterdir()
        if p.is_dir()
    ]
    if "truth" not in tables:
        raise FileNotFoundError(f"Missing truth table in {source_split_dir}")
    feature_tables = sorted(t for t in tables if t.startswith("features"))
    if not feature_tables:
        raise FileNotFoundError(f"Missing feature table(s) in {source_split_dir}")
    return ["truth"] + feature_tables


def prepare_output(outdir: Path, tables: List[str], overwrite: bool) -> bool:
    existing = []
    for table in tables:
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


def prepare_dataset_output(outbase: Path, tables: List[str], overwrite: bool) -> bool:
    existing = []
    for split in SPLITS:
        for table in tables:
            table_dir = outbase / split / table
            table_dir.mkdir(parents=True, exist_ok=True)
            existing.extend(table_dir.glob("*.parquet"))

    if existing and not overwrite:
        print(f"Output already exists, skipping: {outbase}")
        return False

    if overwrite:
        for path in existing:
            path.unlink()
    return True


def pop_event_batch(truth_buffer, feature_buffers: Dict[str, object], events_per_batch: int):
    event_batch = truth_buffer.head(events_per_batch).select("event_no")
    truth_batch = truth_buffer.join(event_batch, on="event_no", how="inner")
    feature_batches = {
        table: features.join(event_batch, on="event_no", how="inner")
        for table, features in feature_buffers.items()
    }
    truth_buffer = truth_buffer.join(event_batch, on="event_no", how="anti")
    feature_buffers = {
        table: features.join(event_batch, on="event_no", how="anti")
        for table, features in feature_buffers.items()
    }
    return truth_batch, feature_batches, truth_buffer, feature_buffers


def write_batch(
    outdir: Path,
    out_batch: int,
    truth_batch,
    feature_batches: Dict[str, object],
) -> Tuple[int, Dict[str, int]]:
    truth_out = outdir / "truth" / f"truth_{out_batch}.parquet"
    truth_batch.write_parquet(truth_out)
    feature_rows = {}
    for table, features_batch in feature_batches.items():
        features_out = outdir / table / f"{table}_{out_batch}.parquet"
        features_batch.write_parquet(features_out)
        feature_rows[table] = features_batch.height
    return truth_batch.height, feature_rows


def event_key_columns(truth) -> List[str]:
    if all(c in truth.columns for c in EVENT_ID_COLS):
        return EVENT_ID_COLS
    if "event_no" in truth.columns:
        return ["event_no"]
    raise ValueError(f"Missing event identifiers. Need {EVENT_ID_COLS} or event_no.")


def split_event_keys(keys, seed: int = 42) -> Dict[str, object]:
    import polars as pl

    key_cols = keys.columns
    schema = keys.schema

    def empty_keys():
        return pl.DataFrame({
            col: pl.Series(col, [], dtype=schema[col])
            for col in key_cols
        })

    rows = keys.unique(maintain_order=True).sort(key_cols).to_dicts()
    rng = random.Random(seed)
    rng.shuffle(rows)

    n = len(rows)
    n_train = int(0.8 * n)
    n_val = int(0.1 * n)
    split_rows = {
        "train": rows[:n_train],
        "val": rows[n_train:n_train + n_val],
        "test": rows[n_train + n_val:],
    }
    return {
        split: pl.DataFrame(split_rows[split], schema=schema)
        if split_rows[split]
        else empty_keys()
        for split in SPLITS
    }


def write_split_batches(
    outdir: Path,
    truth_split,
    feature_splits: Dict[str, object],
    events_per_batch: int,
) -> Tuple[int, Dict[str, int], int]:
    import polars as pl

    event_ids = truth_split.select("event_no").unique(maintain_order=True)
    truth_rows = 0
    feature_rows = {table: 0 for table in feature_splits}
    out_batch = 0

    for start in range(0, event_ids.height, events_per_batch):
        event_batch = event_ids.slice(start, events_per_batch)
        truth_batch = truth_split.join(event_batch, on="event_no", how="inner")
        feature_batches = {
            table: features.join(event_batch, on="event_no", how="inner")
            for table, features in feature_splits.items()
        }
        n_truth, n_features = write_batch(
            outdir=outdir,
            out_batch=out_batch,
            truth_batch=truth_batch,
            feature_batches=feature_batches,
        )
        truth_rows += n_truth
        for table, n_rows in n_features.items():
            feature_rows[table] += n_rows
        out_batch += 1

    return truth_rows, feature_rows, out_batch


def split_category_dataset(
    source_dataset_dir: Path,
    outbase: Path,
    category_column: str,
    category_value: str,
    events_per_batch: int,
    overwrite: bool,
    seed: int = 42,
) -> dict:
    import polars as pl

    tables = parquet_tables(source_dataset_dir)
    feature_table_names = [t for t in tables if t != "truth"]
    truth_dir = source_dataset_dir / "truth"
    truth_files = truth_files_in(truth_dir)
    if not truth_files:
        raise FileNotFoundError(f"No truth parquet files found in {truth_dir}")

    if not prepare_dataset_output(outbase, tables, overwrite):
        return {
            "source_truth_files": len(truth_files),
            "tables": tables,
            "events_per_batch": events_per_batch,
            "skipped_existing": True,
        }

    truth_parts = []
    feature_parts: Dict[str, List[pl.DataFrame]] = {
        table: [] for table in feature_table_names
    }

    for truth_file in truth_files:
        feature_files = {
            table: feature_file_for_truth(source_dataset_dir, table, truth_file)
            for table in feature_table_names
        }
        missing = [str(path) for path in feature_files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing matching feature file(s): {missing}")

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
        truth_parts.append(truth_filtered)
        for table, feature_file in feature_files.items():
            features = pl.read_parquet(feature_file)
            if "event_no" not in features.columns:
                raise ValueError(f"Missing event_no in {feature_file}")
            feature_parts[table].append(features.join(event_ids, on="event_no", how="inner"))

    if not truth_parts:
        manifest = {
            "seed": seed,
            "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
            "category_column": category_column,
            "category_value": str(category_value),
            "events_per_batch": events_per_batch,
            "split_event_counts": {split: 0 for split in SPLITS},
            "output_batches": {split: 0 for split in SPLITS},
        }
        with open(outbase / "split_manifest.json", "w") as f:
            json.dump(manifest, f, indent=2)
        return {
            "source_truth_files": len(truth_files),
            "tables": tables,
            "events_per_batch": events_per_batch,
            "truth_rows": {split: 0 for split in SPLITS},
            "feature_rows": {
                split: {table: 0 for table in feature_table_names}
                for split in SPLITS
            },
            "output_batches": {split: 0 for split in SPLITS},
            "skipped_existing": False,
        }

    truth_all = pl.concat(truth_parts)
    feature_all = {
        table: pl.concat(parts) if parts else pl.DataFrame()
        for table, parts in feature_parts.items()
    }
    key_cols = event_key_columns(truth_all)
    split_keys = split_event_keys(truth_all.select(key_cols), seed=seed)

    truth_rows = {}
    feature_rows = {}
    output_batches = {}
    split_event_counts = {}

    for split in SPLITS:
        keys = split_keys[split]
        split_event_counts[split] = keys.height
        split_outdir = outbase / split
        if keys.is_empty():
            truth_rows[split] = 0
            feature_rows[split] = {table: 0 for table in feature_table_names}
            output_batches[split] = 0
            continue

        truth_split = truth_all.join(keys, on=key_cols, how="inner")
        event_ids = truth_split.select("event_no").unique()
        feature_splits = {
            table: features.join(event_ids, on="event_no", how="inner")
            for table, features in feature_all.items()
        }
        n_truth, n_features, n_batches = write_split_batches(
            outdir=split_outdir,
            truth_split=truth_split,
            feature_splits=feature_splits,
            events_per_batch=events_per_batch,
        )
        truth_rows[split] = n_truth
        feature_rows[split] = n_features
        output_batches[split] = n_batches

    manifest = {
        "seed": seed,
        "fractions": {"train": 0.8, "val": 0.1, "test": 0.1},
        "category_column": category_column,
        "category_value": str(category_value),
        "event_key_columns": key_cols,
        "events_per_batch": events_per_batch,
        "split_event_counts": split_event_counts,
        "output_batches": output_batches,
    }
    with open(outbase / "split_manifest.json", "w") as f:
        json.dump(manifest, f, indent=2)

    return {
        "source_truth_files": len(truth_files),
        "tables": tables,
        "event_key_columns": key_cols,
        "events_per_batch": events_per_batch,
        "truth_rows": truth_rows,
        "feature_rows": feature_rows,
        "output_batches": output_batches,
        "skipped_existing": False,
    }


def split_category(
    source_split_dir: Path,
    outdir: Path,
    category_column: str,
    category_value: str,
    events_per_batch: int,
    overwrite: bool,
) -> dict:
    import polars as pl

    tables = parquet_tables(source_split_dir)
    feature_table_names = [t for t in tables if t != "truth"]
    truth_dir = source_split_dir / "truth"
    truth_files = sorted(truth_dir.glob("truth_*.parquet"), key=batch_id)
    if not truth_files:
        raise FileNotFoundError(f"No truth parquet files found in {truth_dir}")

    if not prepare_output(outdir, tables, overwrite):
        return {
            "source_truth_files": len(truth_files),
            "output_batches": 0,
            "truth_rows": 0,
            "feature_rows": {table: 0 for table in feature_table_names},
            "skipped_existing": True,
        }

    out_batch = 0
    truth_rows = 0
    feature_rows = {table: 0 for table in feature_table_names}
    buffer_truth: List[pl.DataFrame] = []
    buffer_features: Dict[str, List[pl.DataFrame]] = {
        table: [] for table in feature_table_names
    }

    for truth_file in truth_files:
        bid = batch_id(truth_file)
        feature_files = {
            table: source_split_dir / table / f"{table}_{bid}.parquet"
            for table in feature_table_names
        }
        missing = [str(path) for path in feature_files.values() if not path.exists()]
        if missing:
            raise FileNotFoundError(f"Missing matching feature file(s): {missing}")

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
        features_filtered = {}
        for table, feature_file in feature_files.items():
            features = pl.read_parquet(feature_file)
            if "event_no" not in features.columns:
                raise ValueError(f"Missing event_no in {feature_file}")
            features_filtered[table] = features.join(event_ids, on="event_no", how="inner")

        buffer_truth.append(truth_filtered)
        for table, features in features_filtered.items():
            buffer_features[table].append(features)

        truth_buffer = pl.concat(buffer_truth)
        feature_buffers = {
            table: pl.concat(parts) for table, parts in buffer_features.items()
        }
        while truth_buffer.height >= events_per_batch:
            truth_batch, feature_batches, truth_buffer, feature_buffers = pop_event_batch(
                truth_buffer=truth_buffer,
                feature_buffers=feature_buffers,
                events_per_batch=events_per_batch,
            )
            n_truth, n_features = write_batch(
                outdir=outdir,
                out_batch=out_batch,
                truth_batch=truth_batch,
                feature_batches=feature_batches,
            )
            truth_rows += n_truth
            for table, n_rows in n_features.items():
                feature_rows[table] += n_rows
            out_batch += 1

        buffer_truth = [truth_buffer] if not truth_buffer.is_empty() else []
        buffer_features = {
            table: [features] if not features.is_empty() else []
            for table, features in feature_buffers.items()
        }

    if buffer_truth:
        truth_buffer = pl.concat(buffer_truth)
        feature_buffers = {
            table: pl.concat(parts) for table, parts in buffer_features.items()
        }
        n_truth, n_features = write_batch(
            outdir=outdir,
            out_batch=out_batch,
            truth_batch=truth_buffer,
            feature_batches=feature_buffers,
        )
        truth_rows += n_truth
        for table, n_rows in n_features.items():
            feature_rows[table] += n_rows
        out_batch += 1

    return {
        "source_truth_files": len(truth_files),
        "tables": tables,
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
    ap.add_argument("--events-per-batch", type=int, default=256)
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
