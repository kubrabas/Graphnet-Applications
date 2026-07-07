"""
Submit I3Photons summary scan job to SLURM.

Usage:
    python submit_summary_for_I3.py --mc-name SPRING2026MC
    python submit_summary_for_I3.py --mc-name SPRING2026MC --workers 8 --dry-run

The worker script (prepare_summary_for_I3.py) is invoked inside the container
via submit_summary_for_I3.sh; all parameters are passed through --export.
"""

import argparse
import os
import subprocess
from pathlib import Path

WORKER_SH    = Path("/home/kbas/SlurmScripts/DataPreperation/submit_summary_for_I3.sh")
OUTPUT_BASE  = "/project/def-nahee/kbas/Graphnet-Applications/Metadata/DatasetStatistics"

MC_DATASETS = ["SPRING2026MC", "STRING340MC"]

MC_FOLDER = {
    "SPRING2026MC": "Spring2026MC",
    "STRING340MC":  "340StringMC",
}


def main() -> int:
    ap = argparse.ArgumentParser(description="Submit I3Photons summary scan to SLURM")
    ap.add_argument("--mc-name", default="SPRING2026MC", choices=MC_DATASETS,
                    help="MC dataset to scan (default: SPRING2026MC)")
    ap.add_argument("--photon-key", default="I3Photons",
                    help="Frame key to analyze (default: I3Photons)")
    ap.add_argument("--workers", type=int, default=8,
                    help="Parallel file workers (processes) inside the job (default: 8)")
    ap.add_argument("--dry-run", action="store_true",
                    help="Print sbatch command without submitting")
    args = ap.parse_args()

    mc_folder = MC_FOLDER[args.mc_name]
    out_csv  = f"{OUTPUT_BASE}/{mc_folder}/summary_i3.csv"
    out_txt  = f"{OUTPUT_BASE}/{mc_folder}/summary_i3.txt"
    log_dir  = "/home/kbas/SlurmScripts/Logs"
    log_file = f"{log_dir}/%j_{args.mc_name}_i3summary.out"
    job_name = f"i3summary_{args.mc_name}"

    os.makedirs(log_dir, exist_ok=True)

    cmd = [
        "sbatch",
        f"--job-name={job_name}",
        f"--cpus-per-task={args.workers}",
        f"--output={log_file}",
        f"--error={log_file}",
        (
            "--export="
            f"MC_NAME={args.mc_name},"
            f"PHOTON_KEY={args.photon_key},"
            f"OUT_CSV={out_csv},"
            f"OUT_TXT={out_txt},"
            f"WORKERS={args.workers}"
        ),
        str(WORKER_SH),
    ]

    print(f"{'[DRY-RUN] ' if args.dry_run else ''}submitting: {job_name}")
    print(f"  mc_name    : {args.mc_name}")
    print(f"  photon_key : {args.photon_key}")
    print(f"  workers    : {args.workers}")
    print(f"  out_csv    : {out_csv}")
    print(f"  out_txt    : {out_txt}")
    print(f"  log_file   : {log_file}")

    if args.dry_run:
        print("  cmd:", " ".join(cmd))
        return 0

    result = subprocess.run(cmd, capture_output=True, text=True, check=True)
    print(" ", result.stdout.strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
