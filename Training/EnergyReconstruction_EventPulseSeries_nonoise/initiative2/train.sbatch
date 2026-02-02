#!/bin/bash
#SBATCH --job-name=DynEdge_Energy
#SBATCH --account=def-nahee
#SBATCH --time=48:00:00
#SBATCH --mem=48G
#SBATCH --cpus-per-task=12
#SBATCH --gpus-per-node=nvidia_h100_80gb_hbm3_3g.40gb:1
#SBATCH --output=/project/6061446/kbas/Graphnet-Applications/Training/EnergyReconstruction_EventPulseSeries_nonoise/initiative2/dynedge_energy_%j.out
#SBATCH --error=/project/6061446/kbas/Graphnet-Applications/Training/EnergyReconstruction_EventPulseSeries_nonoise/initiative2/dynedge_energy_%j.out
#SBATCH --open-mode=append


set -euo pipefail

echo "===== SLURM INFO ====="
echo "JOBID: ${SLURM_JOB_ID:-}"
echo "NODELIST: ${SLURM_NODELIST:-}"
echo "CPUS_PER_TASK: ${SLURM_CPUS_PER_TASK:-}"
echo "GPUS_ON_NODE: ${SLURM_GPUS_ON_NODE:-}"
echo "SUBMIT_DIR: ${SLURM_SUBMIT_DIR:-}"
echo "======================"

# ---- Modules ----
module purge
module load StdEnv/2023 gcc/12.3 scipy-stack/2023b

# ---- Env ----
PONE_ENV_DIR="/project/def-nahee/kbas/pone_offline"
VENV="${PONE_ENV_DIR}/graphnet_env/bin/activate"
SCRIPT="/project/def-nahee/kbas/Graphnet-Applications/Training/EnergyReconstruction_EventPulseSeries_nonoise/initiative2/script.py"

cd "${PONE_ENV_DIR}"

echo "===== ENV SETUP ====="
echo "PWD: $(pwd)"
echo "Activating venv: ${VENV}"
source "${VENV}"

# Keep CPU threading sane (avoid oversubscription with DataLoader workers)
export OMP_NUM_THREADS=1
export MKL_NUM_THREADS=1

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
# If your cluster doesn't support --gpu-bind, remove that option.
srun --cpu-bind=cores --gpu-bind=closest python3 -u "${SCRIPT}"
echo "===== DONE ====="