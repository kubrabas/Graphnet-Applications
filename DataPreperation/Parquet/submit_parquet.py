"""
Submit Parquet conversion jobs to SLURM — single node, parallel execution.

Usage:
    python3 submit_parquet.py --mc 340StringMC --flavor all
    python3 submit_parquet.py --mc 340StringMC --geometry 102_string --flavor Electron
    python3 submit_parquet.py --mc 340StringMC --geometry full_geometry 160_string 102_string --flavor Tau

For each selected geometry/flavor:
  1. Reads PMT-response path and GCD from paths.py.
  2. Submits a single SLURM conversion job with --cpus-per-task=NWORKERS.
     convert_parquet.py filters on that geometry's triggered_nonoise_* flag.
  3. Chains merge after conversion; merge builds train/val/test, reindexed
     splits, split_manifest.json, and RobustScaler percentile CSVs.
  4. Chains categorized parquet jobs after conversion for the default category
     columns.

Shell script note:
    submit_parquet.sh must call convert_parquet.py with --nworkers $NWORKERS
    (no --array, no --task-id).

Logs go to: /home/kbas/scratch/<mc_scratch>/Logs/<flavor>_<geometry>_Parquet/
"""

import argparse
import importlib.util
import os
import re
import subprocess
from glob import glob
from pathlib import Path
from typing import List, Optional

from submit_categorized_parquet import (
    DEFAULT_EVENTS_PER_BATCH,
    DEFAULT_EXCLUDE_NODES,
    submit_category_workflow,
)

# ---------------------------------------------------------------------------
# Paths / tables
# ---------------------------------------------------------------------------

PATHS_PY     = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH    = Path("/home/kbas/SlurmScripts/DataPreperation/submit_parquet.sh")
MERGE_SH     = Path("/home/kbas/SlurmScripts/DataPreperation/submit_parquet_merge.sh")
SCRATCH_BASE = "/home/kbas/scratch"
NWORKERS     = 16   # CPUs per job (parallel files processed simultaneously)

MC_TABLE = {
    "340StringMC": {
        "pmt_attr": "STRING340MC_PMT",
        "scratch": "String340MC_pone_offline_version3_plus",
    },
    "Spring2026MC": {
        "pmt_attr": "SPRING2026MC_PMT",
        "scratch": "Spring2026MC",
    },
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]
DEFAULT_GEOMETRIES = ["full_geometry", "160_string", "102_string"]
DEFAULT_CATEGORY_COLUMNS = [
    "category1_isMuonCC",
    "category2_tauCC_others_muonCC",
    "category_3_contains_muon",
]

PULSEMAP_BY_GEOMETRY = {
    "full_geometry": "PMT_Response_nonoise_340_String",
    "340_string": "PMT_Response_nonoise_340_String",
    "160_string": "PMT_Response_nonoise_160_String",
    "102_string": "PMT_Response_nonoise_102_String",
}

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def discover_i3_files(indir: str) -> int:
    files = []
    for pattern in ["*.i3.gz", "*.i3.bz2", "*.i3.zst"]:
        files.extend(glob(os.path.join(indir, "**", pattern), recursive=True))
    return len([f for f in files if "gcd" not in os.path.basename(f).lower()])


def format_energy_suffix(max_energy: Optional[float]) -> Optional[str]:
    if max_energy is None:
        return None
    mantissa, exponent = f"{max_energy:.6e}".split("e")
    mantissa = mantissa.rstrip("0").rstrip(".").replace(".", "p")
    return f"Emax{mantissa}e{int(exponent)}"


def add_suffix(path: str, suffix: Optional[str]) -> str:
    if suffix is None:
        return path
    p = Path(path)
    return str(p.with_name(f"{p.name}_{suffix}"))


def paths_suffix(suffix: Optional[str]) -> str:
    if suffix is None:
        return ""
    return suffix[:1].lower() + suffix[1:]


def categorized_geometry_key(geometry: str, metadata_suffix: Optional[str]) -> str:
    suffix = paths_suffix(metadata_suffix)
    if not suffix:
        return geometry
    return f"{geometry}_{suffix}"


def geometry_output_folder(mc: str, geometry: str, pmt_path: str) -> str:
    if mc == "340StringMC":
        if geometry == "full_geometry":
            return "Full_Geometry"
        return geometry
    return Path(pmt_path).parent.name or geometry


def outdir_from_geometry(
    *,
    mc: str,
    scratch_base: str,
    scratch: str,
    geometry: str,
    flavor: str,
    pmt_path: str,
    suffix: Optional[str] = None,
) -> str:
    folder = geometry_output_folder(mc, geometry, pmt_path)
    if mc == "340StringMC":
        outdir = f"{scratch_base}/{scratch}/Parquet/{folder}/{flavor}_Parquet"
    else:
        outdir = f"{scratch_base}/{scratch}/{folder}/{flavor}_Parquet"
    return add_suffix(outdir, suffix)


def parse_job_id(sbatch_output: str) -> Optional[str]:
    m = re.search(r"Submitted batch job (\d+)", sbatch_output)
    return m.group(1) if m else None


def pulsemap_for_geometry(geometry: str, pulsemap: Optional[str]) -> str:
    if pulsemap is not None:
        return pulsemap
    try:
        return PULSEMAP_BY_GEOMETRY[geometry]
    except KeyError as e:
        raise ValueError(
            f"No default pulsemap known for geometry={geometry}. "
            "Pass --pulsemap explicitly."
        ) from e


def gcd_for_geometry(paths, mc: str, geometry: str) -> Optional[str]:
    if geometry == "full_geometry":
        return getattr(paths, "GCD")[mc]
    return getattr(paths, "GCD_TRIMMED").get(mc, {}).get(geometry)


def submit_convert(
    *,
    mc: str,
    flavor: str,
    geometry: str,
    indir: str,
    gcd: str,
    outdir: str,
    logdir: str,
    n_files: int,
    pulsemap: str,
    nworkers: int,
    max_energy: Optional[float],
    exclude_nodes: Optional[str],
) -> Optional[str]:
    export_vars = (
        f"MC={mc},"
        f"FLAVOR={flavor},"
        f"GEOMETRY={geometry},"
        f"INDIR={indir},"
        f"GCD={gcd},"
        f"OUTDIR={outdir},"
        f"LOGDIR={logdir},"
        f"PULSEMAP={pulsemap},"
        f"NWORKERS={nworkers}"
    )
    if max_energy is not None:
        export_vars += f",MAX_ENERGY={max_energy}"

    job_name = f"Parquet_{mc}_{geometry}_{flavor}"
    slurm_log = Path(logdir) / f"conversion_%j.log"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={nworkers}",
        f"--output={slurm_log}",
        f"--error={slurm_log}",
        f"--export={export_vars}",
        str(WORKER_SH),
    ]
    if exclude_nodes:
        cmd.insert(2, f"--exclude={exclude_nodes}")
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = parse_job_id(result.stdout)
    print(f"  submitted: {job_name}  ({n_files} files, {nworkers} workers)  job_id={job_id}")
    return job_id


def submit_merge(
    *,
    mc: str,
    flavor: str,
    geometry: str,
    outdir: str,
    logdir: str,
    convert_job_id: str,
    metadata_suffix: Optional[str],
) -> Optional[str]:
    export_vars = (
        f"MC={mc},"
        f"FLAVOR={flavor},"
        f"GEOMETRY={geometry},"
        f"OUTDIR={outdir},"
        f"LOGDIR={logdir}"
    )
    if metadata_suffix is not None:
        export_vars += f",METADATA_SUFFIX={metadata_suffix}"

    job_name = f"merge_Parquet_{mc}_{geometry}_{flavor}"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--dependency=afterok:{convert_job_id}",
        f"--export={export_vars}",
        str(MERGE_SH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    merge_job_id = parse_job_id(result.stdout)
    print(f"  chained:   {job_name}  job_id={merge_job_id}  (runs after completion of {convert_job_id})")
    return merge_job_id


def submit_categories_after_dependency(
    *,
    paths,
    mc: str,
    flavor: str,
    geometry: str,
    metadata_suffix: Optional[str],
    dependency_job_id: str,
    category_columns: List[str],
    nworkers: int,
    events_per_batch: int,
    overwrite: bool,
    dry_run: bool,
) -> None:
    category_geometry = categorized_geometry_key(geometry, metadata_suffix)
    dependency = f"afterok:{dependency_job_id}"
    for category_column in category_columns:
        submit_category_workflow(
            paths=paths,
            mc=mc,
            geometry=category_geometry,
            flavor=flavor,
            category_column=category_column,
            nworkers=nworkers,
            events_per_batch=events_per_batch,
            overwrite=overwrite,
            dependency=dependency,
            exclude_nodes=DEFAULT_EXCLUDE_NODES,
            dry_run=dry_run,
        )


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Submit Parquet conversion, merge, and categorized split jobs")
    ap.add_argument("--mc",       required=True, choices=list(MC_TABLE))
    ap.add_argument(
        "--geometry",
        nargs="+",
        default=None,
        help=f"Dataset/geometry key(s). If omitted, runs {DEFAULT_GEOMETRIES}.",
    )
    ap.add_argument("--flavor",   required=True, nargs="+",
                    help=f"Flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument(
        "--pulsemap",
        default=None,
        help="Override pulsemap. If omitted, selected automatically from --geometry.",
    )
    ap.add_argument("--nworkers", type=int, default=NWORKERS,
                    help=f"CPUs per job / parallel workers (default: {NWORKERS})")
    ap.add_argument("--max-energy", type=float, default=1e6,
                    help="Keep only frames with EventProperties.totalEnergy <= this value in GeV. Default: 1e6.")
    ap.add_argument("--category-columns", nargs="+", default=DEFAULT_CATEGORY_COLUMNS,
                    help="Category columns to build after conversion.")
    ap.add_argument("--category-events-per-batch", type=int, default=DEFAULT_EVENTS_PER_BATCH,
                    help=f"Categorized output events per batch (default: {DEFAULT_EVENTS_PER_BATCH}).")
    ap.add_argument("--category-overwrite", action="store_true",
                    help="Overwrite existing categorized parquet outputs.")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    mc      = args.mc
    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    geometries = DEFAULT_GEOMETRIES if args.geometry is None else args.geometry
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    paths     = load_paths()
    pmt_table = getattr(paths, MC_TABLE[mc]["pmt_attr"])
    scratch   = MC_TABLE[mc]["scratch"]

    for geometry in geometries:
        gcd = gcd_for_geometry(paths, mc, geometry)
        if gcd is None:
            print(f"ERROR: No GCD found for mc={mc}, geometry={geometry} in paths.py")
            return 1
        try:
            pulsemap = pulsemap_for_geometry(geometry, args.pulsemap)
        except ValueError as e:
            print(f"ERROR: {e}")
            return 1

        print(f"\n[{mc}]  geometry={geometry}  nworkers={args.nworkers}")

        for flavor in flavors:
            entry    = pmt_table.get(geometry, {}).get(flavor, {})
            pmt_path = entry.get("path")

            if pmt_path is None:
                print(f"  [skip] {flavor}: PMT path not set in paths.py")
                continue

            metadata_suffix = format_energy_suffix(args.max_energy)
            outdir = outdir_from_geometry(
                mc=mc,
                scratch_base=SCRATCH_BASE,
                scratch=scratch,
                geometry=geometry,
                flavor=flavor,
                pmt_path=pmt_path,
                suffix=metadata_suffix,
            )
            logdir = add_suffix(
                f"{SCRATCH_BASE}/{scratch}/Logs/{flavor}_{geometry}_Parquet",
                metadata_suffix,
            )

            try:
                n_files = discover_i3_files(pmt_path)
            except Exception as e:
                print(f"  [error] {flavor}: {e}")
                return 1

            if n_files == 0:
                print(f"  [skip] {flavor}: no I3 files found in {pmt_path}")
                continue

            if args.dry_run:
                category_geometry = categorized_geometry_key(geometry, metadata_suffix)
                print(f"  [DRY-RUN] {flavor}: {n_files} files, {args.nworkers} workers")
                print(f"    indir    = {pmt_path}")
                print(f"    outdir   = {outdir}")
                print(f"    gcd      = {gcd}")
                print(f"    logdir   = {logdir}")
                print(f"    slurm_log = {Path(logdir) / 'conversion_%j.log'}")
                print(f"    pulsemap = {pulsemap}")
                print(f"    exclude_nodes = {DEFAULT_EXCLUDE_NODES}")
                if args.max_energy is None:
                    print("    max_energy = none")
                else:
                    print(f"    max_energy = {args.max_energy:.6g} GeV")
                    print(f"    suffix   = {metadata_suffix}")
                print("    merge = yes (after conversion)")
                print(f"    category_columns = {args.category_columns}")
                print(f"    category_geometry = {category_geometry}")
                print(f"    category_events_per_batch = {args.category_events_per_batch}")
                continue

            convert_job_id = submit_convert(
                mc=mc, flavor=flavor, geometry=geometry,
                indir=pmt_path, gcd=gcd, outdir=outdir, logdir=logdir,
                n_files=n_files, pulsemap=pulsemap, nworkers=args.nworkers,
                max_energy=args.max_energy,
                exclude_nodes=DEFAULT_EXCLUDE_NODES,
            )
            if convert_job_id is None:
                print(f"  [error] {flavor}: could not parse job ID from sbatch output")
                return 1

            merge_job_id = submit_merge(
                mc=mc, flavor=flavor, geometry=geometry,
                outdir=outdir, logdir=logdir,
                convert_job_id=convert_job_id,
                metadata_suffix=metadata_suffix,
            )
            if merge_job_id is None:
                print(f"  [error] {flavor}: could not parse merge job ID from sbatch output")
                return 1

            submit_categories_after_dependency(
                paths=paths,
                mc=mc,
                flavor=flavor,
                geometry=geometry,
                metadata_suffix=metadata_suffix,
                dependency_job_id=convert_job_id,
                category_columns=args.category_columns,
                nworkers=args.nworkers,
                events_per_batch=args.category_events_per_batch,
                overwrite=args.category_overwrite,
                dry_run=args.dry_run,
            )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
