#!/bin/bash
#SBATCH --time=06:00:00
#SBATCH --account=def-nahee
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --output=/home/kbas/scratch/mixed_percentiles_%x_%j.out
#SBATCH --error=/home/kbas/scratch/mixed_percentiles_%x_%j.out

# Runs once per geometry after every flavor/category parquet job succeeds.
# MC and GEOMETRY are supplied by submit_parquet.py.
set -euo pipefail


echo "--- HOST: JOB=${SLURM_JOB_ID:-} HOST=$(hostname)"
echo "--- CONFIG: MC=${MC} GEOMETRY=${GEOMETRY}"

module --force purge
module load StdEnv/2020 scipy-stack/2023b

PYTHON=/project/def-nahee/kbas/.venv_try/bin/python
SCRIPT=/project/def-nahee/kbas/Graphnet-Applications/DataPreperation/Parquet/compute_mixed_percentiles.py
"${PYTHON}" -u "${SCRIPT}" --mc "${MC}" --geometry "${GEOMETRY}"
