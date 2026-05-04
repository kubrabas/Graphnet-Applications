"""
Submit Parquet conversion array jobs to SLURM, followed by an automatic merge job.

Usage:
    python3 submit_parquet.py --mc 340StringMC --geometry 102_string --flavor Electron
    python3 submit_parquet.py --mc 340StringMC --geometry 102_string --flavor Muon Electron Tau NC
    python3 submit_parquet.py --mc 340StringMC --geometry 102_string --flavor all
    python3 submit_parquet.py --dry-run --mc 340StringMC --geometry 102_string --flavor Muon

For each flavor:
  1. Reads PMT-response path and GCD from paths.py.
  2. Counts I3 files in the PMT-response directory.
  3. Submits a SLURM array job (one task per file).
  4. Chains a merge job (--dependency=afterany) that runs merge_parquet.py.

Output parquet files go to the same parent as the PMT-response directory,
with _PMT_Response replaced by _Parquet.
Logs go to: /home/kbas/scratch/<mc_scratch>/Logs/<flavor>_<geometry>_Parquet/

Skip / overwrite behaviour (per convert_parquet.py task):
  --overwrite not set (default):
      A task is skipped only if BOTH the output parquet (truth table) AND the
      per-task log file already exist. If either one is missing the task re-runs
      and overwrites whatever is there.
  --overwrite set:
      Every task re-runs unconditionally, overwriting existing parquet and log.
"""

import argparse
import importlib.util
import os
import re
import subprocess
from glob import glob
from pathlib import Path
from typing import Optional

# ---------------------------------------------------------------------------
# Paths / tables
# ---------------------------------------------------------------------------

PATHS_PY     = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH    = Path("/home/kbas/SlurmScripts/DataPreperation/submit_parquet.sh")
MERGE_SH     = Path("/home/kbas/SlurmScripts/DataPreperation/submit_parquet_merge.sh")
SCRATCH_BASE = "/home/kbas/scratch"
CONCURRENT   = 50

MC_TABLE = {
    "340StringMC":  {"pmt_attr": "STRING340MC_PMT",  "gcd_key": "340StringMC",  "scratch": "String340MC"},
    "Spring2026MC": {"pmt_attr": "SPRING2026MC_PMT", "gcd_key": "Spring2026MC", "scratch": "Spring2026MC"},
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

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


def outdir_from_pmt(pmt_path: str) -> str:
    return pmt_path.replace("_PMT_Response", "_Parquet")


def parse_job_id(sbatch_output: str) -> Optional[str]:
    m = re.search(r"Submitted batch job (\d+)", sbatch_output)
    return m.group(1) if m else None


def submit_array(
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
) -> Optional[str]:
    job_name = f"Parquet_{mc}_{geometry}_{flavor}"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array=0-{n_files - 1}%{CONCURRENT}",
        (
            "--export="
            f"MC={mc},"
            f"FLAVOR={flavor},"
            f"GEOMETRY={geometry},"
            f"INDIR={indir},"
            f"GCD={gcd},"
            f"OUTDIR={outdir},"
            f"LOGDIR={logdir},"
            f"PULSEMAP={pulsemap}"
        ),
        str(WORKER_SH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = parse_job_id(result.stdout)
    print(f"  submitted: {job_name}  ({n_files} tasks)  job_id={job_id}")
    return job_id


def submit_merge(
    *,
    mc: str,
    flavor: str,
    geometry: str,
    outdir: str,
    logdir: str,
    array_job_id: str,
) -> None:
    job_name = f"merge_Parquet_{mc}_{geometry}_{flavor}"
    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--dependency=afterany:{array_job_id}",
        (
            "--export="
            f"MC={mc},"
            f"FLAVOR={flavor},"
            f"GEOMETRY={geometry},"
            f"OUTDIR={outdir},"
            f"LOGDIR={logdir}"
        ),
        str(MERGE_SH),
    ]
    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    merge_job_id = parse_job_id(result.stdout)
    print(f"  chained:   {job_name}  job_id={merge_job_id}  (runs after {array_job_id})")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Submit Parquet array jobs + auto merge")
    ap.add_argument("--mc",       required=True, choices=list(MC_TABLE))
    ap.add_argument("--geometry", required=True, help="Sub-geometry key, e.g. 102_string")
    ap.add_argument("--flavor",   required=True, nargs="+",
                    help=f"Flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--pulsemap",  default="EventPulseSeries_nonoise")
    ap.add_argument("--dry-run",  action="store_true")
    args = ap.parse_args()

    mc      = args.mc
    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    paths     = load_paths()
    pmt_table = getattr(paths, MC_TABLE[mc]["pmt_attr"])
    scratch   = MC_TABLE[mc]["scratch"]

    if args.geometry == "full_geometry":
        gcd = getattr(paths, "GCD")[mc]
    else:
        gcd = getattr(paths, "GCD_TRIMMED").get(mc, {}).get(args.geometry)
        if gcd is None:
            print(f"ERROR: No trimmed GCD found for mc={mc}, geometry={args.geometry} in paths.py")
            return 1

    print(f"\n[{mc}]  geometry={args.geometry}")

    for flavor in flavors:
        entry = pmt_table.get(args.geometry, {}).get(flavor, {})
        pmt_path = entry.get("path")

        if pmt_path is None:
            print(f"  [skip] {flavor}: PMT path not set in paths.py")
            continue

        outdir = outdir_from_pmt(pmt_path)
        logdir = f"{SCRATCH_BASE}/{scratch}/Logs/{flavor}_{args.geometry}_Parquet"

        try:
            n_files = discover_i3_files(pmt_path)
        except Exception as e:
            print(f"  [error] {flavor}: {e}")
            return 1

        if n_files == 0:
            print(f"  [skip] {flavor}: no I3 files found in {pmt_path}")
            continue

        if args.dry_run:
            print(f"  [DRY-RUN] {flavor}: {n_files} files")
            print(f"    indir    = {pmt_path}")
            print(f"    outdir   = {outdir}")
            print(f"    gcd      = {gcd}")
            print(f"    logdir   = {logdir}")
            print(f"    pulsemap = {args.pulsemap}")
            continue

        array_job_id = submit_array(
            mc=mc, flavor=flavor, geometry=args.geometry,
            indir=pmt_path, gcd=gcd, outdir=outdir, logdir=logdir,
            n_files=n_files, pulsemap=args.pulsemap,
        )
        if array_job_id is None:
            print(f"  [error] {flavor}: could not parse job ID from sbatch output")
            return 1

        submit_merge(
            mc=mc, flavor=flavor, geometry=args.geometry,
            outdir=outdir, logdir=logdir,
            array_job_id=array_job_id,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
