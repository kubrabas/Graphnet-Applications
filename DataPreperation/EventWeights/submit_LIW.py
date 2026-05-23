"""
Submit a single LIW computation job to SLURM (one job per flavor).

All batch IDs are processed in parallel within one job using multiprocessing.
No merge step is needed — the worker writes the final CSV directly.

Usage:
    python3 submit_LIW.py --mc 340StringMC --flavor Muon
    python3 submit_LIW.py --mc 340StringMC --flavor Muon Electron Tau NC
    python3 submit_LIW.py --mc 340StringMC --flavor all
    python3 submit_LIW.py --dry-run --mc 340StringMC --flavor Muon
    python3 submit_LIW.py --mc 340StringMC --flavor all --workers 16

Paths come from paths.py (LIC + full_geometry i3 entries).
Output CSVs go to: /project/def-nahee/kbas/Graphnet-Applications/Metadata/EventWeights/<MC>/
"""

import argparse
import importlib.util
import re
import subprocess
from pathlib import Path
from typing import Dict, Optional

# ---------------------------------------------------------------------------
# Paths / tables
# ---------------------------------------------------------------------------

PATHS_PY     = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH    = Path("/home/kbas/SlurmScripts/DataPreperation/submit_LIW.sh")
BASE_OUTDIR  = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/EventWeights"
SCRATCH_BASE = "/home/kbas/scratch"
DEFAULT_WORKERS = 16

MC_TABLE = {
    "340StringMC":  {"lic_key": "STRING340MC",  "i3_attr": "STRING340MC_I3",  "scratch": "String340MC"},
    "Spring2026MC": {"lic_key": "SPRING2026MC", "i3_attr": "SPRING2026MC_I3", "scratch": "Spring2026MC"},
}

OUTDIR_SUBDIR = {
    "340StringMC":  "String340MC",
    "Spring2026MC": "Spring2026MC",
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


def fmt_to_pattern(fmt: str) -> str:
    if fmt in ("gz", "zst"):
        return f"*.i3.{fmt}"
    return "*.i3"


def parse_job_id(sbatch_output: str) -> Optional[str]:
    m = re.search(r'Submitted batch job (\d+)', sbatch_output)
    return m.group(1) if m else None


def submit_job(*, mc: str, flavor: str, lic_dir: str, photon_dir: str,
               photon_pattern: str, workers: int) -> Optional[str]:
    subdir   = OUTDIR_SUBDIR[mc]
    scratch  = MC_TABLE[mc]["scratch"]
    outdir   = f"{BASE_OUTDIR}/{subdir}"
    logdir   = f"{SCRATCH_BASE}/{scratch}/Logs/LIW"
    job_name = f"LIW_{mc}_{flavor}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={workers}",
        (
            "--export="
            f"MC={mc},"
            f"FLAVOR={flavor},"
            f"LIC_DIR={lic_dir},"
            f"PHOTON_DIR={photon_dir},"
            f"PHOTON_PATTERN={photon_pattern},"
            f"OUTDIR={outdir},"
            f"LOGDIR={logdir},"
            f"WORKERS={workers}"
        ),
        str(WORKER_SH),
    ]

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    job_id = parse_job_id(result.stdout)
    print(f"  submitted: {job_name}  workers={workers}  job_id={job_id}")
    return job_id

# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Submit LIW computation jobs")
    ap.add_argument("--mc",      required=True, choices=list(MC_TABLE))
    ap.add_argument("--flavor",  required=True, nargs="+",
                    help=f"Flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--workers", type=int, default=DEFAULT_WORKERS,
                    help=f"parallel workers per job (default: {DEFAULT_WORKERS})")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    mc      = args.mc
    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    paths     = load_paths()
    lic_table = getattr(paths, "LIC")[MC_TABLE[mc]["lic_key"]]
    i3_table  = getattr(paths, MC_TABLE[mc]["i3_attr"])

    print(f"\n[{mc}]  workers={args.workers}")

    for flavor in flavors:
        lic_entry    = lic_table[flavor]
        photon_entry = i3_table["full_geometry"][flavor]

        if lic_entry["path"] is None:
            print(f"  [skip] {flavor}: LIC path not set in paths.py")
            continue
        if photon_entry["path"] is None:
            print(f"  [skip] {flavor}: photon path not set in paths.py")
            continue

        photon_pattern = fmt_to_pattern(photon_entry["format"])

        if args.dry_run:
            print(f"  [DRY-RUN] {flavor}: would submit single job (workers={args.workers})")
            continue

        job_id = submit_job(
            mc=mc,
            flavor=flavor,
            lic_dir=lic_entry["path"],
            photon_dir=photon_entry["path"],
            photon_pattern=photon_pattern,
            workers=args.workers,
        )
        if job_id is None:
            print(f"  [error] {flavor}: could not parse job ID from sbatch output")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
