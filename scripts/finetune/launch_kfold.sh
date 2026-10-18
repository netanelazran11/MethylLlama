#!/bin/bash
# Launch all 5 k-fold fine-tune jobs in parallel.
#
# Usage:
#   CHECKPOINT=/path/to/pretrain.ckpt bash scripts/finetune/launch_kfold.sh
#
# All 5 jobs run simultaneously. Each takes up to 48h on goldfish/H200.
# Monitor: squeue -u $USER

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?ERROR: export CHECKPOINT=/path/to/ckpt before running}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
KFOLD_DIR="${REPO}/outputs/kfold_splits"
SCRIPT="${REPO}/scripts/finetune/finetune_kfold.sh"

# Verify fold files exist before submitting anything
echo "Checking fold files in ${KFOLD_DIR} ..."
for FOLD in 0 1 2 3 4; do
    for SUFFIX in train val; do
        F="${KFOLD_DIR}/fold_${FOLD}_${SUFFIX}.npy"
        if [ ! -f "${F}" ]; then
            echo "ERROR: missing ${F}"
            echo "Run this first:"
            echo "  cd ${REPO} && methyllama_env"
            echo "  python scripts/data/create_kfold_splits.py"
            exit 1
        fi
    done
done
echo "All fold files present."
echo ""

JOB_IDS=()
for FOLD in 0 1 2 3 4; do
    JOB_ID=$(sbatch \
        --export=ALL,FOLD=${FOLD},CHECKPOINT="${CHECKPOINT}" \
        "${SCRIPT}" \
        | awk '{print $NF}')
    JOB_IDS+=("${JOB_ID}")
    echo "Submitted fold ${FOLD} → job ${JOB_ID}"
done

echo ""
echo "All 5 folds submitted:"
for i in "${!JOB_IDS[@]}"; do
    echo "  fold ${i} → job ${JOB_IDS[$i]}"
done
echo ""
echo "Monitor: squeue -u \$USER"
echo "Logs:    ${REPO}/logs_llama-wced/methyllama-kfold_fold*"
