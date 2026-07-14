"""Merge Parquet batches and apply the canonical event-level split.

After conversion finishes this script merges the per-file outputs, joins every
physical event to the immutable master split manifest, writes train/val/test
batches, and creates sequential ``*_reindexed`` views. It never chooses or
randomizes a split locally.
"""

import h5py  # must be imported before graphnet to avoid HDF5 version conflict

import argparse
import json
import os
import re
import sys
import time
import traceback
from pathlib import Path
from typing import Dict, List

import polars as pl
from graphnet.data.writers import ParquetWriter

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]
EVENT_ID_COLS = ["RunID", "SubrunID", "EventID", "SubEventID"]
SPLIT_NAME_MAP = {"train": "train", "validation": "val", "test": "test"}
GEOMETRY_TRIGGER = {
    "102_string": "trigger_102",
    "160_string": "trigger_160",
    "full_geometry": "trigger_340",
    "340_string": "trigger_340",
}


def batch_id(path: Path) -> int:
    match = re.search(r"_(\d+)\.parquet$", path.name)
    if match is None:
        raise ValueError(f"Could not parse batch id from {path}")
    return int(match.group(1))


def parquet_tables(dataset_dir: Path) -> List[str]:
    ignored = {"merged", "merged_raw", "categorized"}
    tables = [
        item.name
        for item in dataset_dir.iterdir()
        if item.is_dir() and item.name not in ignored
    ]
    if "truth" not in tables:
        raise RuntimeError(f"Missing truth table directory under {dataset_dir}")
    return ["truth"] + sorted(table for table in tables if table != "truth")


def reindex_split(split_dir: Path, reindexed_dir: Path, tables: List[str]) -> None:
    """Create sequential symlinks for GraphNeT's ParquetDataset."""
    ids = sorted(batch_id(path) for path in (split_dir / "truth").glob("truth_*.parquet"))
    for table in tables:
        (reindexed_dir / table).mkdir(parents=True, exist_ok=True)
    for new_id, old_id in enumerate(ids):
        for table in tables:
            src = (split_dir / table / f"{table}_{old_id}.parquet").resolve()
            if not src.exists():
                raise FileNotFoundError(f"Missing table batch while reindexing: {src}")
            dst = reindexed_dir / table / f"{table}_{new_id}.parquet"
            dst.symlink_to(src)


def build_splits(
    merged_raw: Path,
    merged_dir: Path,
    tables: List[str],
    master_manifest_path: Path,
    flavor: str,
    geometry: str,
) -> Dict[str, int]:
    """Partition merged batches according to the canonical master manifest."""
    if merged_dir.exists():
        raise FileExistsError(
            f"Split output already exists: {merged_dir}. Archive or remove it before rerunning."
        )
    if geometry not in GEOMETRY_TRIGGER:
        raise ValueError(f"No master-manifest trigger configured for geometry={geometry}")
    if not master_manifest_path.is_file():
        raise FileNotFoundError(f"Master split manifest not found: {master_manifest_path}")

    trigger_column = GEOMETRY_TRIGGER[geometry]
    master = pl.read_parquet(master_manifest_path)
    required = {"flavor", "split", trigger_column, *EVENT_ID_COLS}
    missing = sorted(required - set(master.columns))
    if missing:
        raise ValueError(f"Master split manifest is missing columns: {missing}")

    master = (
        master.filter(
            (pl.col("flavor") == flavor)
            & (pl.col(trigger_column).cast(pl.Boolean))
        )
        .select(EVENT_ID_COLS + ["split"])
    )
    if master.select(EVENT_ID_COLS).is_duplicated().any():
        raise ValueError(f"Duplicate physical event keys in {master_manifest_path}")
    unknown_splits = set(master["split"].unique().to_list()) - set(SPLIT_NAME_MAP)
    if unknown_splits:
        raise ValueError(f"Unknown split values in master manifest: {sorted(unknown_splits)}")

    truth_files = sorted((merged_raw / "truth").glob("truth_*.parquet"), key=batch_id)
    if not truth_files:
        raise RuntimeError(f"No merged truth batches found in {merged_raw / 'truth'}")

    for split_name in SPLIT_NAME_MAP.values():
        for table in tables:
            (merged_dir / split_name / table).mkdir(parents=True, exist_ok=True)

    counts = {name: 0 for name in SPLIT_NAME_MAP.values()}
    output_batches = {name: 0 for name in SPLIT_NAME_MAP.values()}
    seen_parts = []
    feature_tables = [table for table in tables if table != "truth"]

    for truth_file in truth_files:
        source_batch_id = batch_id(truth_file)
        truth = pl.read_parquet(truth_file)
        missing_ids = [column for column in EVENT_ID_COLS + ["event_no"] if column not in truth.columns]
        if missing_ids:
            raise ValueError(f"Missing columns {missing_ids} in {truth_file}")
        if truth.select(EVENT_ID_COLS).is_duplicated().any():
            raise ValueError(f"Duplicate physical event keys inside {truth_file}")

        joined = truth.join(master, on=EVENT_ID_COLS, how="left")
        if joined.height != truth.height:
            raise ValueError(f"Non-unique manifest join while processing {truth_file}")
        unmatched = joined.filter(pl.col("split").is_null()).height
        if unmatched:
            raise ValueError(
                f"{unmatched} events in {truth_file} are absent from the canonical manifest "
                f"selection for flavor={flavor}, geometry={geometry}"
            )
        seen_parts.append(joined.select(EVENT_ID_COLS))

        feature_frames = {}
        for table in feature_tables:
            feature_path = merged_raw / table / f"{table}_{source_batch_id}.parquet"
            if not feature_path.exists():
                raise FileNotFoundError(f"Missing matching feature batch: {feature_path}")
            feature_frames[table] = pl.read_parquet(feature_path)

        for canonical_name, split_name in SPLIT_NAME_MAP.items():
            truth_split = joined.filter(pl.col("split") == canonical_name).drop("split")
            if truth_split.is_empty():
                continue
            event_nos = truth_split.select("event_no")
            output_id = output_batches[split_name]
            truth_split.write_parquet(
                merged_dir / split_name / "truth" / f"truth_{output_id}.parquet"
            )
            for table, features in feature_frames.items():
                if "event_no" not in features.columns:
                    raise ValueError(f"Missing event_no in {table}_{source_batch_id}.parquet")
                features.join(event_nos, on="event_no", how="inner").write_parquet(
                    merged_dir / split_name / table / f"{table}_{output_id}.parquet"
                )
            counts[split_name] += truth_split.height
            output_batches[split_name] += 1

    seen = pl.concat(seen_parts)
    if seen.select(EVENT_ID_COLS).is_duplicated().any():
        raise ValueError("A physical event occurs in more than one merged source batch")
    if seen.height != master.height:
        missing_count = master.join(seen, on=EVENT_ID_COLS, how="anti").height
        raise ValueError(
            f"Canonical manifest expects {master.height} events but merged data contains "
            f"{seen.height}; missing={missing_count}"
        )

    local_manifest = {
        "source_master_manifest": str(master_manifest_path),
        "flavor": flavor,
        "geometry": geometry,
        "trigger_column": trigger_column,
        "event_key_columns": EVENT_ID_COLS,
        "split_event_counts": counts,
        "output_batches": output_batches,
        "split_name_mapping": SPLIT_NAME_MAP,
    }
    with open(merged_dir / "split_manifest.json", "w") as handle:
        json.dump(local_manifest, handle, indent=2)
    return counts


def summarize_conversion_logs(logdir: Path, flavor: str, geometry: str) -> None:
    summary_pattern = re.compile(
        r"\[.+?\]\s+kept=(\d+)\s+noise_only=(\d+)\s+"
        r"(?:absent_pulsemap|pulsemap_does_not_exist)=(\d+)"
        r"(?:\s+corrupt_frames=\d+)?"
    )
    file_error_pattern = re.compile(r"\[FILE ERROR\] Could not open: (.+?)\s+error=")
    merge_log_name = f"merge_{flavor}_{geometry}.out"
    total_kept = total_noise = total_absent = files_parsed = 0
    corrupt_files: List[str] = []

    for log_file in sorted(logdir.glob("*.out")):
        if log_file.name == merge_log_name:
            continue
        for line in log_file.read_text(errors="replace").splitlines():
            match = summary_pattern.search(line)
            if match:
                total_kept += int(match.group(1))
                total_noise += int(match.group(2))
                total_absent += int(match.group(3))
                files_parsed += 1
            file_error = file_error_pattern.search(line)
            if file_error:
                corrupt_files.append(file_error.group(1))

    print(f"=== CONVERSION SUMMARY: FILE LEVEL ({files_parsed} tasks) ===")
    print(f"  corrupt_files   : {len(corrupt_files)}")
    corrupt_log = logdir / "corrupt_files.log"
    corrupt_log.write_text("\n".join(corrupt_files) + ("\n" if corrupt_files else ""))
    print("=== CONVERSION SUMMARY: EVENT LEVEL ===")
    print(f"  kept            : {total_kept}")
    print(f"  noise_only      : {total_noise}")
    print(f"  pulsemap_does_not_exist : {total_absent}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Merge parquet batches and apply a master split.")
    parser.add_argument("--mc", required=True)
    parser.add_argument("--flavor", required=True)
    parser.add_argument("--geometry", required=True)
    parser.add_argument("--outdir", required=True)
    parser.add_argument("--logdir", required=True)
    parser.add_argument("--master-split", required=True)
    parser.add_argument("--events-per-batch", type=int, default=1024)
    parser.add_argument("--num-workers", type=int, default=1)
    parser.add_argument("--metadata-suffix", default="")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if args.flavor not in ALL_FLAVORS:
        print(f"ERROR: unknown flavor '{args.flavor}'. Choices: {ALL_FLAVORS}")
        return 1

    outdir = Path(args.outdir)
    logdir = Path(args.logdir)
    logdir.mkdir(parents=True, exist_ok=True)
    logfile = logdir / f"merge_{args.flavor}_{args.geometry}.out"
    log_handle = open(logfile, "w")
    sys.stdout = log_handle
    sys.stderr = log_handle

    started = time.time()
    print("=== PARQUET MERGE JOB STARTED ===")
    print(f"mc             : {args.mc}")
    print(f"flavor         : {args.flavor}")
    print(f"geometry       : {args.geometry}")
    print(f"outdir         : {outdir}")
    print(f"master_split   : {args.master_split}")
    print(f"events/batch   : {args.events_per_batch}")
    log_handle.flush()

    try:
        merged_dir = outdir / "merged"
        merged_raw_dir = outdir / "merged_raw"
        summarize_conversion_logs(logdir, args.flavor, args.geometry)

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

        if args.dry_run:
            print("[DRY-RUN] would apply canonical master split and build reindexed views")
        else:
            tables = parquet_tables(merged_raw_dir)
            counts = build_splits(
                merged_raw=merged_raw_dir,
                merged_dir=merged_dir,
                tables=tables,
                master_manifest_path=Path(args.master_split),
                flavor=args.flavor,
                geometry=args.geometry,
            )
            print(f"split event counts: {counts}")
            print(f"tables: {tables}")
            for split_name in ["train", "val", "test"]:
                reindex_split(
                    split_dir=merged_dir / split_name,
                    reindexed_dir=merged_dir / f"{split_name}_reindexed",
                    tables=tables,
                )
            print("reindexed splits: done")

        print(f"=== SUCCESS  elapsed={time.time() - started:.1f}s ===")
    except Exception as error:
        print(f"=== FAILED  elapsed={time.time() - started:.1f}s  error={error} ===")
        traceback.print_exc()
        log_handle.flush()
        log_handle.close()
        return 1

    log_handle.flush()
    log_handle.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
