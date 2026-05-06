"""
Submit PMT response jobs to SLURM — single node, parallel execution.

Usage:
    python3 submit_pmt_response.py --mc-name SPRING2026MC --geometry strings_102_40m --flavor Muon --with-first-3-layers
    python3 submit_pmt_response.py --mc-name STRING340MC --geometry full_geometry --flavor Tau --with-first-3-layers
    python3 submit_pmt_response.py --mc-name SPRING2026MC --geometry strings_102_40m --flavor Muon --no-with-first-3-layers --dry-run
    python3 submit_pmt_response.py --mc-name SPRING2026MC --geometry full_geometry --flavor all --no-with-first-3-layers

Workers:
    apply_pmt_response_with_first_3_layers.py    (--with-first-3-layers)
    apply_pmt_response_without_first_3_layers.py (--no-with-first-3-layers)

Each job requests a single node with --nworkers CPUs and processes all files
for that flavor in parallel inside one SLURM job.

NOTE: WORKER_SH must call the Python worker WITHOUT --task-id and WITH
      --nworkers $NWORKERS  (and no --array in sbatch).
      The relevant line in the shell script should look like:
          python3 /path/to/apply_pmt_response_*.py \\
              --flavor $FLAVOR --geometry $GEOMETRY --mc $MC \\
              --indir $INDIR --pattern $PATTERN --gcd $GCD \\
              --outdir $OUTDIR --logdir $LOGDIR \\
              --nworkers $NWORKERS

Input data comes from the _I3 datasets in paths.py.
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PATHS_PY     = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH    = Path("/home/kbas/SlurmScripts/DataPreperation/submit_pmt_response.sh")
SCRATCH_BASE = "/home/kbas/scratch"
NWORKERS     = 16   # CPUs per job (parallel files processed simultaneously)

# ---------------------------------------------------------------------------
# MC lookup table
# ---------------------------------------------------------------------------

MC_TABLE = {
    "SPRING2026MC": "SPRING2026MC_I3",
    "STRING340MC":  "STRING340MC_I3",
}

MC_FOLDER = {
    "SPRING2026MC": "Spring2026MC",
    "STRING340MC":  "String340MC",
}

GCD_KEY = {
    "SPRING2026MC": "Spring2026MC",
    "STRING340MC":  "340StringMC",
}

FORMAT_PATTERN = {
    "zst": "*.i3.zst",
    "gz":  "*.i3.gz",
    "i3":  "*.i3",
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_folder_name(mc: str, geometry: str) -> str:
    if mc == "STRING340MC" and geometry != "full_geometry":
        return geometry
    return "_".join(part.capitalize() for part in geometry.split("_"))


def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def count_files(indir: str, fmt: str) -> int:
    pattern = FORMAT_PATTERN.get(fmt, f"*.i3.{fmt}")
    p = Path(indir)
    if not p.exists():
        raise FileNotFoundError(f"Input dir not found: {indir}")
    return len(sorted(p.rglob(pattern)))


def submit_one(*, mc: str, mc_folder: str, geometry: str, geo_folder: str,
               flavor: str, indir: str, fmt: str, gcd: str,
               with_first_3_layers: bool, nworkers: int, dry_run: bool) -> None:
    pattern = FORMAT_PATTERN.get(fmt, f"*.i3.{fmt}")
    n = count_files(indir, fmt)
    if n == 0:
        print(f"  [skip] no files found in: {indir}")
        return

    outdir   = f"{SCRATCH_BASE}/{mc_folder}/{geo_folder}/{flavor}_PMT_Response"
    logdir   = f"{SCRATCH_BASE}/{mc_folder}/Logs/{flavor}_pmt_response_{geo_folder}"
    job_name = f"pmt_{mc}_{geometry}_{flavor}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={nworkers}",
        (
            "--export="
            f"FLAVOR={flavor},"
            f"GEOMETRY={geometry},"
            f"MC={mc},"
            f"INDIR={indir},"
            f"PATTERN={pattern},"
            f"GCD={gcd},"
            f"OUTDIR={outdir},"
            f"LOGDIR={logdir},"
            f"WITH_FIRST_3_LAYERS={'1' if with_first_3_layers else '0'},"
            f"NWORKERS={nworkers}"
        ),
        str(WORKER_SH),
    ]

    print(f"  {'[DRY-RUN] ' if dry_run else ''}submitting: {job_name}  ({n} files, {nworkers} workers)")
    if dry_run:
        print("  cmd:", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(" ", result.stdout.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Submit PONE PMT response jobs (single node, parallel)")
    ap.add_argument("--mc-name", default="SPRING2026MC", choices=list(MC_TABLE),
                    help=f"MC dataset (default: SPRING2026MC). Choices: {list(MC_TABLE)}")
    ap.add_argument("--geometry", required=True,
                    help="Geometry key in paths.py (e.g. strings_102_40m)")
    ap.add_argument("--flavor", required=True, nargs="+",
                    help=f"Particle flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--with-first-3-layers", required=True, action=argparse.BooleanOptionalAction,
                    help="Use with_first_3_layers script (--with-first-3-layers) or without (--no-with-first-3-layers)")
    ap.add_argument("--nworkers", type=int, default=NWORKERS,
                    help=f"CPUs per job / parallel workers (default: {NWORKERS})")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sbatch commands without submitting")
    args = ap.parse_args()

    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    paths      = load_paths()
    table      = getattr(paths, MC_TABLE[args.mc_name])
    mc_folder  = MC_FOLDER[args.mc_name]
    geo_folder = to_folder_name(args.mc_name, args.geometry)
    gcd_key    = GCD_KEY[args.mc_name]
    trimmed    = getattr(paths, "GCD_TRIMMED", {}).get(gcd_key, {})
    gcd        = trimmed.get(args.geometry) or paths.GCD[gcd_key]

    if args.geometry not in table:
        print(f"ERROR: geometry '{args.geometry}' not found in {MC_TABLE[args.mc_name]}")
        print(f"  Available: {list(table.keys())}")
        return 1

    print(f"\n[{args.mc_name}] geometry={args.geometry}  nworkers={args.nworkers}")

    for flavor in flavors:
        entry = table[args.geometry][flavor]
        if entry["path"] is None:
            print(f"  [skip] {flavor}: path not set in paths.py")
            continue
        try:
            submit_one(
                mc=args.mc_name,
                mc_folder=mc_folder,
                geometry=args.geometry,
                geo_folder=geo_folder,
                flavor=flavor,
                indir=entry["path"],
                fmt=entry["format"],
                gcd=gcd,
                with_first_3_layers=args.with_first_3_layers,
                nworkers=args.nworkers,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as e:
            print(f"  [error] {flavor}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
