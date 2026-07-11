"""
Submit categorized Parquet split jobs.

Example:
    python3 submit_categorized_parquet.py \
        --mc 340StringMC \
        --geometry 102_string_emax1e6 \
        --flavor all \
        --category-column category1_isMuonCC

    python3 submit_categorized_parquet.py \
        --mc 340StringMC \
        --geometry full_geometry_emax1e6 \
        --flavor all \
        --category-column category1_isMuonCC

The worker reads existing raw truth parquet files to discover category values,
then filters by category before computing train/val/test splits. It does not
re-read I3 files.
"""

import argparse
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Iterable, List, Optional

# ---------------------------------------------------------------------------
# Paths / tables
# ---------------------------------------------------------------------------

PATHS_PY = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH = Path("/home/kbas/SlurmScripts/DataPreperation/submit_categorized_parquet.sh")
SCRATCH_BASE = Path("/home/kbas/scratch")
NWORKERS = 16
DEFAULT_EVENTS_PER_BATCH = 256
DEFAULT_EXCLUDE_NODES = "fc30564,fc30568"

MC_TABLE = {
    "340StringMC": {
        "parquet_attr": "STRING340MC_PARQUET",
        "scratch": "String340MC_pone_offline_version3_plus",
    },
    "Spring2026MC": {
        "parquet_attr": "SPRING2026MC_PARQUET",
        "scratch": "Spring2026MC",
    },
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]
EVENT_COLUMNS = ["event_no", "RunID", "SubrunID", "EventID", "SubEventID"]
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


def split_dir_from_paths(
    paths,
    mc: str,
    geometry: str,
    flavor: str,
    split: str,
) -> Path:
    parquet_table = getattr(paths, MC_TABLE[mc]["parquet_attr"], {})
    entry = parquet_table.get(geometry, {}).get(flavor, {}).get(split)
    if not entry:
        raise ValueError(
            f"No parquet path found in paths.py for mc={mc}, "
            f"geometry={geometry}, flavor={flavor}, split={split}"
        )
    return Path(entry)


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
    return category_column


def category_value_label(value) -> str:
    if isinstance(value, float) and value.is_integer():
        value = int(value)
    text = str(value).replace("-", "minus").replace(".", "p")
    return f"category{text}"


def read_dataset_events(
    parquet_base: Path,
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

    truth_dir = parquet_base / "truth"
    truth_files = sorted(truth_dir.glob("*.parquet"))
    if not truth_files:
        raise FileNotFoundError(f"No truth parquet files found in {truth_dir}")

    first_truth = pl.read_parquet(truth_files[0], n_rows=0)
    missing = [c for c in EVENT_COLUMNS + [category_column] if c not in first_truth.columns]
    if missing:
        raise ValueError(f"Missing columns {missing} in {truth_dir}")

    columns = EVENT_COLUMNS + [category_column]
    parts = []
    for truth_file in truth_files:
        part = (
            pl.read_parquet(truth_file, columns=columns)
            .with_columns(
                pl.lit(category_column).alias("category_column"),
                pl.col(category_column).alias("category_value"),
            )
            .drop(category_column)
        )
        parts.append(part)

    return pl.concat(parts, how="vertical")


def get_category_values(
    paths,
    mc: str,
    geometry: str,
    flavor: str,
    category_column: str,
    dry_run: bool,
) -> List[str]:
    parquet_base = source_parquet_outdir(
        paths=paths,
        mc=mc,
        geometry=geometry,
        flavor=flavor,
    )

    if dry_run:
        print(f"  [DRY-RUN] would read truth dir: {parquet_base / 'truth'}")
        print("  [DRY-RUN] category values will be read from truth parquet files")
        return []

    combined = read_dataset_events(
        parquet_base=parquet_base,
        category_column=category_column,
    ).unique(
        subset=["RunID", "SubrunID", "EventID", "SubEventID"],
        maintain_order=True,
    )

    values = [str(v) for v in combined["category_value"].unique().sort().to_list()]
    print(f"  category events read from {parquet_base / 'truth'} ({len(combined)} rows)")
    return values


def submit_category_workflow(
    *,
    paths,
    mc: str,
    geometry: str,
    flavor: str,
    category_column: str,
    nworkers: int,
    events_per_batch: int,
    overwrite: bool,
    dependency: Optional[str],
    exclude_nodes: Optional[str],
    dry_run: bool,
) -> Optional[str]:
    parquet_base = source_parquet_outdir(
        paths=paths,
        mc=mc,
        geometry=geometry,
        flavor=flavor,
    )
    parquet_label = parquet_base.name
    if parquet_label.startswith(f"{flavor}_"):
        parquet_label = parquet_label[len(flavor) + 1:]
    geometry_label = parquet_base.parent.name.lower()
    log_dir = (
        SCRATCH_BASE
        / MC_TABLE[mc]["scratch"]
        / "Logs"
        / f"{flavor}_{geometry_label}_{parquet_label}"
    )
    job_name = f"CatParquet_{mc}_{geometry}_{flavor}_{category_column}"
    log_path = log_dir / f"categorization_{category_column}_%j.log"

    export_vars = (
        f"MC={mc},"
        f"FLAVOR={flavor},"
        f"GEOMETRY={geometry},"
        f"NWORKERS={nworkers},"
        f"EVENTS_PER_BATCH={events_per_batch},"
        f"CATEGORY_COLUMN={category_column},"
        f"OVERWRITE={int(overwrite)}"
    )
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={nworkers}",
        f"--output={log_path}",
        f"--error={log_path}",
        f"--export={export_vars}",
        str(WORKER_SH),
    ]
    if dependency:
        cmd.insert(2, f"--dependency={dependency}")
    if exclude_nodes:
        cmd.insert(2, f"--exclude={exclude_nodes}")
    if dry_run:
        print(f"    [DRY-RUN] {' '.join(cmd)}")
        return None

    log_dir.mkdir(parents=True, exist_ok=True)
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = parse_job_id(result.stdout)
    print(f"  submitted {flavor}: job_id={job_id}")
    print(f"    log: {log_path}")
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
        description="Submit categorized Parquet jobs."
    )
    ap.add_argument("--mc", required=True, choices=list(MC_TABLE))
    ap.add_argument("--geometry", required=True, help="Geometry key, e.g. 102_string")
    ap.add_argument("--flavor", required=True, nargs="+",
                    help=f"Flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--category-column", default="category1_isMuonCC",
                    help="Truth-table category column to split on, e.g. category1_isMuonCC")
    ap.add_argument("--nworkers", type=int, default=NWORKERS)
    ap.add_argument("--events-per-batch", type=int, default=DEFAULT_EVENTS_PER_BATCH)
    ap.add_argument("--overwrite", action="store_true",
                    help="Overwrite existing categorized parquet outputs.")
    ap.add_argument("--dependency", default=None,
                    help="Optional SLURM dependency, e.g. afterok:123456.")
    ap.add_argument("--exclude-nodes", default=DEFAULT_EXCLUDE_NODES,
                    help=f"Comma-separated SLURM node exclude list. Default: {DEFAULT_EXCLUDE_NODES}.")
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
            paths=paths,
            mc=args.mc,
            geometry=args.geometry,
            flavor=flavor,
            category_column=args.category_column,
            nworkers=args.nworkers,
            events_per_batch=args.events_per_batch,
            overwrite=args.overwrite,
            dependency=args.dependency,
            exclude_nodes=args.exclude_nodes,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
