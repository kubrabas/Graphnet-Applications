#!/bin/bash
#SBATCH --job-name=DynEdge_Energy
#SBATCH --account=def-nahee
#SBATCH --time=08:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=h100:1
#SBATCH --output=/project/def-nahee/kbas/slurm_logs/dynedge_energy_%j.out
#SBATCH --error=/project/def-nahee/kbas/slurm_logs/dynedge_energy_%j.err

set -euo pipefail
mkdir -p /project/def-nahee/kbas/slurm_logs

echo "===== SLURM INFO ====="
echo "JOBID: ${SLURM_JOB_ID:-}"
echo "NODELIST: ${SLURM_NODELIST:-}"
echo "CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-}"
echo "======================"

# ---- Modules ----
module purge
module load StdEnv/2023 gcc/12.3 scipy-stack/2023b

# ---- Env ----
VENV="/project/def-nahee/kbas/pone_offline/graphnet_env/bin/activate"
SCRIPT="/project/def-nahee/kbas/GraphnetApplications/Training/EnergyReconstruction_EventPulseSeries_nonoise/script.py"

echo "===== ENV SETUP ====="
echo "Activating venv: ${VENV}"
source "${VENV}"

echo "Python: $(which python3)"
python3 -c "import sys; print('sys.executable =', sys.executable)"

python3 - <<'PY'
import torch
print("torch:", torch.__version__)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("cuda device:", torch.cuda.get_device_name(0))
PY

echo "===== GPU INFO ====="
nvidia-smi || true
echo "===================="

echo "===== RUN ====="
srun --cpu-bind=cores python3 -u "${SCRIPT}"
echo "===== DONE ====="
