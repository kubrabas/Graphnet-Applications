"""Create truth-category views of canonical train and validation splits.

The canonical split has already been assigned upstream. This module only
filters each split by a truth category; it never randomizes or reassigns an
event and intentionally does not create categorized test datasets.
"""

import json
import re
from pathlib import Path
from typing import Dict, List, Tuple

SPLITS = ["train", "val"]
EVENT_ID_COLS = ["RunID", "SubrunID", "EventID", "SubEventID"]


def batch_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Could not parse batch id from {path}")
    return int(match.group(1))


def truth_files_in(truth_dir: Path) -> List[Path]:
    return sorted(truth_dir.glob("truth_*.parquet"), key=batch_id)


def parquet_tables(source_split_dir: Path) -> List[str]:
    tables = [item.name for item in source_split_dir.iterdir() if item.is_dir()]
    if "truth" not in tables:
        raise FileNotFoundError(f"Missing truth table in {source_split_dir}")
    feature_tables = sorted(table for table in tables if table.startswith("features"))
    if not feature_tables:
        raise FileNotFoundError(f"Missing feature table(s) in {source_split_dir}")
    return ["truth"] + feature_tables


def prepare_dataset_output(outbase: Path, tables: List[str], overwrite: bool) -> bool:
    existing = []
    test_dir = outbase / "test"
    if test_dir.exists():
        raise FileExistsError(
            f"Legacy categorized test directory exists: {test_dir}. Archive it before rerunning."
        )
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


def write_batch(
    outdir: Path,
    output_id: int,
    truth_batch,
    feature_batches: Dict[str, object],
) -> Tuple[int, Dict[str, int]]:
    truth_batch.write_parquet(outdir / "truth" / f"truth_{output_id}.parquet")
    feature_rows = {}
    for table, features in feature_batches.items():
        features.write_parquet(outdir / table / f"{table}_{output_id}.parquet")
        feature_rows[table] = features.height
    return truth_batch.height, feature_rows


def filter_one_canonical_split(
    source_split_dir: Path,
    output_split_dir: Path,
    category_column: str,
    category_value: str,
    events_per_batch: int,
) -> dict:
    import polars as pl

    tables = parquet_tables(source_split_dir)
    feature_tables = [table for table in tables if table != "truth"]
    truth_files = truth_files_in(source_split_dir / "truth")
    if not truth_files:
        raise FileNotFoundError(f"No truth parquet files in {source_split_dir / 'truth'}")

    pending_truth = None
    pending_features = {table: None for table in feature_tables}
    output_id = 0
    truth_rows = 0
    feature_rows = {table: 0 for table in feature_tables}

    def flush(n_events=None):
        nonlocal pending_truth, pending_features, output_id, truth_rows
        if pending_truth is None or pending_truth.is_empty():
            return
        event_ids = pending_truth.select("event_no").unique(maintain_order=True)
        if n_events is not None:
            event_ids = event_ids.head(n_events)
        truth_batch = pending_truth.join(event_ids, on="event_no", how="inner")
        feature_batches = {
            table: frame.join(event_ids, on="event_no", how="inner")
            for table, frame in pending_features.items()
        }
        n_truth, n_features = write_batch(
            output_split_dir, output_id, truth_batch, feature_batches
        )
        truth_rows += n_truth
        for table, count in n_features.items():
            feature_rows[table] += count
        pending_truth = pending_truth.join(event_ids, on="event_no", how="anti")
        pending_features = {
            table: frame.join(event_ids, on="event_no", how="anti")
            for table, frame in pending_features.items()
        }
        output_id += 1

    for truth_file in truth_files:
        source_id = batch_id(truth_file)
        truth = pl.read_parquet(truth_file)
        required = ["event_no", category_column, *EVENT_ID_COLS]
        missing = [column for column in required if column not in truth.columns]
        if missing:
            raise ValueError(f"Missing columns {missing} in {truth_file}")
        selected_truth = truth.filter(
            pl.col(category_column).cast(pl.Utf8) == str(category_value)
        )
        if selected_truth.is_empty():
            continue
        event_ids = selected_truth.select("event_no").unique()
        selected_features = {}
        for table in feature_tables:
            feature_file = source_split_dir / table / f"{table}_{source_id}.parquet"
            if not feature_file.exists():
                raise FileNotFoundError(f"Missing matching feature batch: {feature_file}")
            features = pl.read_parquet(feature_file)
            selected_features[table] = features.join(event_ids, on="event_no", how="inner")

        if pending_truth is None:
            pending_truth = selected_truth
            pending_features = selected_features
        else:
            pending_truth = pl.concat([pending_truth, selected_truth])
            pending_features = {
                table: pl.concat([pending_features[table], selected_features[table]])
                for table in feature_tables
            }
        while pending_truth.select("event_no").n_unique() >= events_per_batch:
            flush(events_per_batch)

    flush()
    return {
        "source_truth_files": len(truth_files),
        "truth_rows": truth_rows,
        "feature_rows": feature_rows,
        "output_batches": output_id,
    }


def split_category_dataset(
    source_split_dirs: Dict[str, Path],
    outbase: Path,
    category_column: str,
    category_value: str,
    events_per_batch: int,
    overwrite: bool,
) -> dict:
    if set(source_split_dirs) != set(SPLITS):
        raise ValueError(f"Expected canonical source splits {SPLITS}, got {sorted(source_split_dirs)}")
    if events_per_batch <= 0:
        raise ValueError("events_per_batch must be positive")

    tables = parquet_tables(source_split_dirs["train"])
    for split in SPLITS:
        split_tables = parquet_tables(source_split_dirs[split])
        if split_tables != tables:
            raise ValueError(f"Table mismatch in canonical {split}: {split_tables} != {tables}")
    if not prepare_dataset_output(outbase, tables, overwrite):
        return {"skipped_existing": True}

    split_stats = {}
    for split in SPLITS:
        split_stats[split] = filter_one_canonical_split(
            source_split_dir=source_split_dirs[split],
            output_split_dir=outbase / split,
            category_column=category_column,
            category_value=category_value,
            events_per_batch=events_per_batch,
        )

    manifest = {
        "split_policy": "inherited_from_canonical_master_split",
        "source_split_dirs": {key: str(value) for key, value in source_split_dirs.items()},
        "category_column": category_column,
        "category_value": str(category_value),
        "event_key_columns": EVENT_ID_COLS,
        "events_per_batch": events_per_batch,
        "categorized_splits": SPLITS,
        "test_output": "intentionally_not_created",
        "stats": split_stats,
    }
    with open(outbase / "split_manifest.json", "w") as handle:
        json.dump(manifest, handle, indent=2)
    return {"skipped_existing": False, "tables": tables, "splits": split_stats}
