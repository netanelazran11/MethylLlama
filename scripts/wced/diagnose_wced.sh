#!/bin/bash -l
#SBATCH --job-name=diagnose-wced
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=01:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths - UPDATE THESE
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs"

# Data path
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Checkpoint path - UPDATE THIS to your actual checkpoint
# Find checkpoints: find ${REPO}/outputs -name "*.ckpt" -type f
CHECKPOINT="${CHECKPOINT:-}"

# Output directory for diagnosis results
OUTPUT_DIR="${REPO}/wced_diagnosis"

# Settings
VOCAB_SIZE="${VOCAB_SIZE:-2048}"
MAX_SAMPLES="${MAX_SAMPLES:-500}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTPUT_DIR}"

echo "============================================================"
echo "WCED DIAGNOSTIC ANALYSIS"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "============================================================"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data: ${DATA}"
echo "Vocab size: ${VOCAB_SIZE}"
echo "Max samples: ${MAX_SAMPLES}"
echo "Output: ${OUTPUT_DIR}"
echo "============================================================"

if [ -z "${CHECKPOINT}" ]; then
    echo "ERROR: CHECKPOINT not set!"
    echo ""
    echo "Usage: CHECKPOINT=/path/to/checkpoint.ckpt sbatch scripts/diagnose_wced.sh"
    echo ""
    echo "To find checkpoints:"
    echo "  find ${REPO}/outputs -name '*.ckpt' -type f"
    exit 1
fi

if [ ! -f "${CHECKPOINT}" ]; then
    echo "ERROR: Checkpoint file not found: ${CHECKPOINT}"
    exit 1
fi

# -------------------------
# Modules
# -------------------------
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true

module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

# -------------------------
# Env
# -------------------------
cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false

# -------------------------
# Run Diagnostics
# -------------------------
python scripts/diagnose_wced.py \
    --checkpoint "${CHECKPOINT}" \
    --data "${DATA}" \
    --output "${OUTPUT_DIR}" \
    --vocab_size "${VOCAB_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --device cuda

echo "============================================================"
echo "DIAGNOSIS COMPLETE"
echo "============================================================"
echo "Results saved to: ${OUTPUT_DIR}"
echo "============================================================"
