"""
Run the full categorized parquet workflow for one flavor inside SLURM.

This worker filters the existing canonical train and validation parquet
splits. Split membership is inherited unchanged and categorized test output is
intentionally not produced.
"""

import argparse
from pathlib import Path

from submit_categorized_parquet import (
    MC_TABLE,
    category_parent,
    category_value_label,
    load_paths,
    source_parquet_outdir,
    split_dir_from_paths,
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
    source_split_dirs = {
        split: split_dir_from_paths(
            paths, args.mc, args.geometry, args.flavor, split
        )
        for split in ("train", "val")
    }
    for value in values:
        value_dir = category_value_label(value)
        print(f"category_value={value} -> {value_dir}")

        outbase = parquet_base / "categorized" / category_root / value_dir
        outbase.mkdir(parents=True, exist_ok=True)

        print(f"  canonical_sources={source_split_dirs}")
        print(f"  outbase={outbase}")
        stats = split_category_dataset(
            source_split_dirs=source_split_dirs,
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
