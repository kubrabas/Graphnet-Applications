"""
Build category event-list CSVs and submit categorized Parquet split jobs.

Example:
    python3 submit_categorized_parquet.py \\
        --mc 340StringMC \\
        --geometry 102_string \\
        --flavor Electron \\
        --category-column category1

The script first reads the existing merged/*_reindexed/truth parquet files,
writes one CSV per flavor/category column, then submits one SLURM job for each
observed category value and split. The worker filters existing parquet files;
it does not re-read I3 files.
"""

import argparse
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional, Tuple

# ---------------------------------------------------------------------------
# Paths / tables
# ---------------------------------------------------------------------------

PATHS_PY = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
CATEGORY_INFO_BASE = Path(
    "/project/def-nahee/kbas/Graphnet-Applications/Metadata/CategoryInformation"
)
WORKER_SH = Path("/home/kbas/SlurmScripts/DataPreperation/submit_categorized_parquet.sh")
SCRATCH_BASE = Path("/home/kbas/scratch")
NWORKERS = 16

MC_TABLE = {
    "340StringMC": {
        "pmt_attr": "STRING340MC_PMT",
        "parquet_attr": "STRING340MC_PARQUET",
        "gcd_key": "340StringMC",
        "results": "String340MC",
        "scratch": "String340MC",
    },
    "Spring2026MC": {
        "pmt_attr": "SPRING2026MC_PMT",
        "parquet_attr": "SPRING2026MC_PARQUET",
        "gcd_key": "Spring2026MC",
        "results": "Spring2026MC",
        "scratch": "Spring2026MC",
    },
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]
SPLITS = ["train", "val", "test"]
EVENT_COLUMNS = ["event_no", "RunID", "SubrunID", "EventID", "SubEventID"]
CATEGORY_DIRS = {
    "category1": "first_category",
    "category2": "second_category",
}


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def parse_job_id(sbatch_output: str) -> Optional[str]:
    m = re.search(r"Submitted batch job (\d+)", sbatch_output)
    return m.group(1) if m else None


def base_parquet_outdir(pmt_path: str) -> Path:
    return Path(pmt_path.replace("_PMT_Response", "_Parquet"))


def split_dir_from_paths(
    paths,
    mc: str,
    geometry: str,
    flavor: str,
    split: str,
) -> Path:
    parquet_table = getattr(paths, MC_TABLE[mc]["parquet_attr"], {})
    entry = parquet_table.get(geometry, {}).get(flavor, {}).get(split)
    if entry:
        return Path(entry)

    pmt_path = (
        getattr(paths, MC_TABLE[mc]["pmt_attr"], {})
        .get(geometry, {})
        .get(flavor, {})
        .get("path")
    )
    if not pmt_path:
        raise ValueError(
            f"No parquet or PMT path found for mc={mc}, geometry={geometry}, flavor={flavor}"
        )
    return base_parquet_outdir(pmt_path) / "merged" / f"{split}_reindexed"


def source_parquet_outdir(
    paths,
    mc: str,
    geometry: str,
    flavor: str,
) -> Path:
    train_dir = split_dir_from_paths(paths, mc, geometry, flavor, "train")
    marker = Path("merged/train_reindexed")
    parts = train_dir.parts
    marker_parts = marker.parts
    for i in range(len(parts) - len(marker_parts) + 1):
        if parts[i : i + len(marker_parts)] == marker_parts:
            return Path(*parts[:i])
    return train_dir.parent.parent


def category_parent(category_column: str) -> str:
    return CATEGORY_DIRS.get(category_column, category_column)


def category_value_label(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).replace("-", "minus").replace(".", "p")
    return f"category{text}"


def read_split_events(
    split_dir: Path,
    split: str,
    category_column: str,
):
    try:
        import polars as pl
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "submit_categorized_parquet.py needs polars to read parquet files. "
            "Run it in the same software environment used for parquet merging "
            "(for example after loading scipy-stack/2023b on the cluster)."
        ) from e

    truth_dir = split_dir / "truth"
    truth_files = sorted(truth_dir.glob("*.parquet"))
    if not truth_files:
        raise FileNotFoundError(f"No truth parquet files found in {truth_dir}")

    schema = pl.scan_parquet(str(truth_dir / "*.parquet")).collect_schema()
    missing = [c for c in EVENT_COLUMNS + [category_column] if c not in schema]
    if missing:
        raise ValueError(f"Missing columns {missing} in {truth_dir}")

    return (
        pl.scan_parquet(str(truth_dir / "*.parquet"))
        .select(
            [pl.col(c) for c in EVENT_COLUMNS]
            + [
                pl.lit(split).alias("split"),
                pl.lit(category_column).alias("category_column"),
                pl.col(category_column).alias("category_value"),
            ]
        )
        .collect()
    )


def write_event_list_csv(
    paths,
    mc: str,
    geometry: str,
    flavor: str,
    category_column: str,
    dry_run: bool,
) -> Tuple[Path, List[str]]:
    out_dir = CATEGORY_INFO_BASE / MC_TABLE[mc]["results"] / geometry
    out_path = out_dir / f"{flavor}_{category_column}_events.csv"

    if dry_run:
        split_dirs = [
            split_dir_from_paths(paths, mc, geometry, flavor, split)
            for split in SPLITS
        ]
        print(f"  [DRY-RUN] would read split dirs:")
        for split, split_dir in zip(SPLITS, split_dirs):
            print(f"    {split}: {split_dir}")
        print(f"  [DRY-RUN] would write event CSV: {out_path}")
        print("  [DRY-RUN] category values will be read from the generated CSV")
        return out_path, []

    try:
        import polars as pl
    except ModuleNotFoundError as e:
        raise ModuleNotFoundError(
            "submit_categorized_parquet.py needs polars to read parquet files. "
            "Run it in the same software environment used for parquet merging "
            "(for example after loading scipy-stack/2023b on the cluster)."
        ) from e

    split_frames = [
        read_split_events(
            split_dir=split_dir_from_paths(paths, mc, geometry, flavor, split),
            split=split,
            category_column=category_column,
        )
        for split in SPLITS
    ]
    combined = pl.concat(split_frames).unique(
        subset=["RunID", "SubrunID", "EventID", "SubEventID", "split"],
        maintain_order=True,
    )

    values = [str(v) for v in combined["category_value"].unique().sort().to_list()]
    out_dir.mkdir(parents=True, exist_ok=True)
    combined.write_csv(str(out_path))
    print(f"  event CSV -> {out_path} ({len(combined)} rows)")
    return out_path, values


def submit_category_workflow(
    *,
    mc: str,
    geometry: str,
    flavor: str,
    category_column: str,
    nworkers: int,
    events_per_batch: int,
    overwrite: bool,
    dry_run: bool,
) -> Optional[str]:
    export_vars = (
        f"MC={mc},"
        f"FLAVOR={flavor},"
        f"GEOMETRY={geometry},"
        f"NWORKERS={nworkers},"
        f"EVENTS_PER_BATCH={events_per_batch},"
        f"CATEGORY_COLUMN={category_column},"
        f"OVERWRITE={int(overwrite)}"
    )
    job_name = f"CatParquet_{mc}_{geometry}_{flavor}_{category_column}"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={nworkers}",
        f"--export={export_vars}",
        str(WORKER_SH),
    ]
    if dry_run:
        print(f"    [DRY-RUN] {' '.join(cmd)}")
        return None

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = parse_job_id(result.stdout)
    print(f"  submitted {flavor}: job_id={job_id}")
    return job_id


def iter_flavors(values: Iterable[str]) -> List[str]:
    requested = list(values)
    if "all" in requested:
        return ALL_FLAVORS
    return requested


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Build one category event CSV and submit categorized Parquet jobs."
    )
    ap.add_argument("--mc", required=True, choices=list(MC_TABLE))
    ap.add_argument("--geometry", required=True, help="Geometry key, e.g. 102_string")
    ap.add_argument("--flavor", required=True, nargs="+",
                    help=f"Flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--category-column", default="category1",
                    help="Truth-table category column to split on, e.g. category1")
    ap.add_argument("--nworkers", type=int, default=NWORKERS)
    ap.add_argument("--events-per-batch", type=int, default=1024)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing categorized parquet outputs.")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    flavors = iter_flavors(args.flavor)
    for flavor in flavors:
        if flavor not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{flavor}'. Choices: {ALL_FLAVORS}")
            return 1

    paths = load_paths()
    category_root = category_parent(args.category_column)
    print(
        f"\n[{args.mc}] geometry={args.geometry} category={args.category_column} "
        f"({category_root}) nworkers={args.nworkers} events_per_batch={args.events_per_batch}"
    )

    for flavor in flavors:
        print(f"\n  flavor={flavor}")
        submit_category_workflow(
            mc=args.mc,
            geometry=args.geometry,
            flavor=flavor,
            category_column=args.category_column,
            nworkers=args.nworkers,
            events_per_batch=args.events_per_batch,
            overwrite=args.overwrite,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
