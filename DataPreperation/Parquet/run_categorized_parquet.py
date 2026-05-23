"""
Run the full categorized parquet workflow for one flavor inside SLURM.

This worker runs in the IceTray/GraphNeT container. It reads existing merged
parquet splits, writes the event-list CSV, then writes categorized parquet
outputs for all category values and train/val/test splits.
"""

import argparse
from pathlib import Path

from submit_categorized_parquet import (
    MC_TABLE,
    SPLITS,
    category_parent,
    category_value_label,
    load_paths,
    source_parquet_outdir,
    split_dir_from_paths,
    write_event_list_csv,
)
from split_categorized_parquet import split_category


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run categorized parquet split workflow for one flavor."
    )
    ap.add_argument("--mc", required=True, choices=list(MC_TABLE))
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--category-column", required=True)
    ap.add_argument("--events-per-batch", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths = load_paths()
    category_root = category_parent(args.category_column)

    print(
        f"[{args.mc}] geometry={args.geometry} flavor={args.flavor} "
        f"category={args.category_column} events_per_batch={args.events_per_batch}"
    )

    event_csv, values = write_event_list_csv(
        paths=paths,
        mc=args.mc,
        geometry=args.geometry,
        flavor=args.flavor,
        category_column=args.category_column,
        dry_run=False,
    )
    print(f"category values: {values}")

    parquet_base = source_parquet_outdir(
        paths=paths,
        mc=args.mc,
        geometry=args.geometry,
        flavor=args.flavor,
    )
    for value in values:
        value_dir = category_value_label(value)
        print(f"category_value={value} -> {value_dir}")

        for split in SPLITS:
            source_split_dir = split_dir_from_paths(
                paths=paths,
                mc=args.mc,
                geometry=args.geometry,
                flavor=args.flavor,
                split=split,
            )
            outdir = parquet_base / "categorized" / category_root / value_dir / split
            outdir.mkdir(parents=True, exist_ok=True)

            print(f"  split={split}")
            print(f"    source={source_split_dir}")
            print(f"    outdir={outdir}")
            stats = split_category(
                source_split_dir=source_split_dir,
                outdir=outdir,
                category_column=args.category_column,
                category_value=value,
                events_per_batch=args.events_per_batch,
                overwrite=args.overwrite,
            )
            print(f"    stats={stats}")

    print(f"event CSV -> {event_csv}")
    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
