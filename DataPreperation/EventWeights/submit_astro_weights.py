"""
Submit astrophysical-weight computation jobs to SLURM.

One job per flavor; input paths are read from paths.py (SPRING2026MC_I3,
full_geometry). Geometry-independent: always uses the Generator files.

Usage:
    python submit_astro_weights.py --flavor Muon
    python submit_astro_weights.py --flavor Muon Electron Tau NC
    python submit_astro_weights.py --flavor all
    python submit_astro_weights.py --dry-run --flavor all
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PATHS_PY  = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH = Path("/home/kbas/SlurmScripts/DataPreperation/submit_astro_weights.sh")
OUTDIR    = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/AstroWeights/Spring2026MC"
LOG_BASE  = "/home/kbas/scratch/Spring2026MC/Logs"

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

MC_KEY      = "SPRING2026MC_I3"
GEOMETRY    = "full_geometry"
ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod  = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def submit_one(*, flavor: str, indir: str, fmt: str, outdir: str,
               logdir: str, dry_run: bool) -> None:
    job_name = f"astro_weights_Spring2026MC_{flavor}"
    logfile  = f"{logdir}/astro_weights_{flavor}_%j.out"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--output={logfile}",
        f"--error={logfile}",
        (
            "--export="
            f"FLAVOR={flavor},"
            f"INDIR={indir},"
            f"FORMAT={fmt},"
            f"OUTDIR={outdir},"
            f"LOGDIR={logdir}"
        ),
        str(WORKER_SH),
    ]

    print(f"  {'[DRY-RUN] ' if dry_run else ''}submitting: {job_name}")
    if dry_run:
        print("  cmd:", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(" ", result.stdout.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(
        description="Submit astrophysical-weight SLURM jobs for Spring 2026 MC"
    )
    ap.add_argument(
        "--flavor", required=True, nargs="+",
        help=f"Flavor(s) to process, or 'all'. Choices: {ALL_FLAVORS}",
    )
    ap.add_argument(
        "--dry-run", action="store_true",
        help="Print sbatch commands without submitting",
    )
    args = ap.parse_args()

    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    if not WORKER_SH.exists():
        print(f"ERROR: worker script not found: {WORKER_SH}")
        return 1

    paths = load_paths()
    table = getattr(paths, MC_KEY)

    if GEOMETRY not in table:
        print(f"ERROR: '{GEOMETRY}' not found in {MC_KEY}")
        return 1

    Path(OUTDIR).mkdir(parents=True, exist_ok=True)
    Path(LOG_BASE).mkdir(parents=True, exist_ok=True)

    print(f"[Spring2026MC / {GEOMETRY}]")

    for flavor in flavors:
        entry = table[GEOMETRY][flavor]
        if entry["path"] is None:
            print(f"  [skip] {flavor}: path not set in paths.py")
            continue
        submit_one(
            flavor=flavor,
            indir=entry["path"],
            fmt=entry["format"],
            outdir=OUTDIR,
            logdir=LOG_BASE,
            dry_run=args.dry_run,
        )

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
