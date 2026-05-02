"""
Submit geometry-skimmer array jobs to SLURM.

Usage:
    python3 submit_skim_I3.py --csv /path/to/strings_102_40m.csv --flavor Muon
    python3 submit_skim_I3.py --csv /path/to/strings_102_40m.csv --flavor Muon Electron Tau NC
    python3 submit_skim_I3.py --csv /project/def-nahee/kbas/Graphnet-Applications/Metadata/GeometryFiles/Spring2026MC/strings_102_40m.csv --flavor Muon
    python3 submit_skim_I3.py --dry-run --csv /path/to/strings_102_40m.csv --flavor Muon

The MC set (Spring2026MC or 340StringMC) is inferred from the CSV path.
Input data always comes from full_geometry in paths.py.
"""

import argparse
import importlib.util
import subprocess
from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------

PATHS_PY    = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/paths.py"
WORKER_SH   = Path("/home/kbas/SlurmScripts/DataPreperation/submit_skim_I3.sh")
FILTERFRAME = "/project/def-nahee/kbas/GeometrySkimmer/FilterFrame.py"
SCRATCH_BASE = "/home/kbas/scratch"
CONCURRENT  = 50

# ---------------------------------------------------------------------------
# MC lookup table
# ---------------------------------------------------------------------------

MC_TABLE = {
    "Spring2026MC": "SPRING2026MC_I3",
    "340StringMC":  "STRING340MC_I3",
}

SCRATCH_DIR = {
    "Spring2026MC": "Spring2026MC",
    "340StringMC":  "String340MC",
}

ALL_FLAVORS = ["Muon", "Electron", "Tau", "NC"]

# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def load_paths():
    spec = importlib.util.spec_from_file_location("paths", PATHS_PY)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def infer_mc(csv_path: Path) -> str:
    for mc in MC_TABLE:
        if mc in str(csv_path):
            return mc
    raise ValueError(f"Could not infer MC set from CSV path: {csv_path}\nExpected 'Spring2026MC' or '340StringMC' in path.")


def count_files(indir: str, fmt: str) -> int:
    pattern = "*.i3.zst" if fmt == "zst" else "*.i3"
    p = Path(indir)
    if not p.exists():
        raise FileNotFoundError(f"Input dir not found: {indir}")
    return len(sorted(p.rglob(pattern)))


def submit_one(*, mc: str, geometry: str, flavor: str, indir: str, fmt: str,
               gcd: str, csv: Path, dry_run: bool) -> None:
    pattern = "*.i3.zst" if fmt == "zst" else "*.i3"
    n = count_files(indir, fmt)
    if n == 0:
        print(f"  [skip] no files found in: {indir}")
        return

    scratch = SCRATCH_DIR[mc]
    outdir  = f"{SCRATCH_BASE}/{scratch}/{geometry}/{flavor}_I3Photons"
    logdir  = f"{SCRATCH_BASE}/{scratch}/Logs/{flavor}_skim_{geometry}"
    job_name = f"skim_{mc}_{geometry}_{flavor}"

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--array=0-{n - 1}%{CONCURRENT}",
        (
            "--export="
            f"FLAVOR={flavor},"
            f"GEOMETRY={geometry},"
            f"MC={mc},"
            f"INDIR={indir},"
            f"PATTERN={pattern},"
            f"GCD={gcd},"
            f"SELECTION={csv},"
            f"FILTERFRAME={FILTERFRAME},"
            f"OUTDIR={outdir},"
            f"LOGDIR={logdir}"
        ),
        str(WORKER_SH),
    ]

    print(f"  {'[DRY-RUN] ' if dry_run else ''}submitting: {job_name}  ({n} tasks)")
    if dry_run:
        print("  cmd:", " ".join(cmd))
        return

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(" ", result.stdout.strip())


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> int:
    ap = argparse.ArgumentParser(description="Submit PONE geometry-skimmer array jobs")
    ap.add_argument("--csv", required=True, help="Path to selection CSV (e.g. strings_102_40m.csv)")
    ap.add_argument("--flavor", required=True, nargs="+",
                    help=f"Particle flavor(s) or 'all'. Choices: {ALL_FLAVORS}")
    ap.add_argument("--dry-run", action="store_true", help="Print sbatch commands without submitting")
    args = ap.parse_args()

    csv = Path(args.csv).resolve()
    if not csv.exists():
        print(f"ERROR: CSV not found: {csv}")
        return 1

    # infer MC from path
    try:
        mc = infer_mc(csv)
    except ValueError as e:
        print(f"ERROR: {e}")
        return 1

    geometry = csv.stem.capitalize()  # e.g. "Strings_102_40m"

    flavors = ALL_FLAVORS if "all" in args.flavor else args.flavor
    for f in flavors:
        if f not in ALL_FLAVORS:
            print(f"ERROR: unknown flavor '{f}'. Choices: {ALL_FLAVORS}")
            return 1

    paths = load_paths()
    table = getattr(paths, MC_TABLE[mc])
    gcd   = paths.GCD[mc]

    if "full_geometry" not in table:
        print(f"ERROR: 'full_geometry' not found in {MC_TABLE[mc]}")
        return 1

    print(f"\n[{mc}] geometry={geometry}  (input: full_geometry)")

    for flavor in flavors:
        entry = table["full_geometry"][flavor]
        if entry["path"] is None:
            print(f"  [skip] {flavor}: path not set in paths.py")
            continue
        try:
            submit_one(
                mc=mc,
                geometry=geometry,
                flavor=flavor,
                indir=entry["path"],
                fmt=entry["format"],
                gcd=gcd,
                csv=csv,
                dry_run=args.dry_run,
            )
        except FileNotFoundError as e:
            print(f"  [error] {flavor}: {e}")
            return 1

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
