"""
Run the full categorized parquet workflow for one flavor inside SLURM.

This worker runs in the IceTray/GraphNeT container. It reads existing raw
per-file parquet files, discovers category values, then writes categorized
parquet outputs. Category filtering happens before train/val/test splitting.
"""

import argparse
from pathlib import Path

from submit_categorized_parquet import (
    MC_TABLE,
    category_parent,
    category_value_label,
    load_paths,
    source_parquet_outdir,
    get_category_values,
)
from split_categorized_parquet import split_category_dataset


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Run categorized parquet split workflow for one flavor."
    )
    ap.add_argument("--mc", required=True, choices=list(MC_TABLE))
    ap.add_argument("--geometry", required=True)
    ap.add_argument("--flavor", required=True)
    ap.add_argument("--category-column", required=True)
    ap.add_argument("--events-per-batch", type=int, default=256)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    paths = load_paths()
    category_root = category_parent(args.category_column)

    print(
        f"[{args.mc}] geometry={args.geometry} flavor={args.flavor} "
        f"category={args.category_column} events_per_batch={args.events_per_batch}"
    )

    values = get_category_values(
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
    source_dataset_dir = parquet_base
    for value in values:
        value_dir = category_value_label(value)
        print(f"category_value={value} -> {value_dir}")

        outbase = parquet_base / "categorized" / category_root / value_dir
        outbase.mkdir(parents=True, exist_ok=True)

        print(f"  source={source_dataset_dir}")
        print(f"  outbase={outbase}")
        stats = split_category_dataset(
            source_dataset_dir=source_dataset_dir,
            outbase=outbase,
            category_column=args.category_column,
            category_value=value,
            events_per_batch=args.events_per_batch,
            overwrite=args.overwrite,
        )
        print(f"  stats={stats}")

    print("done")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
