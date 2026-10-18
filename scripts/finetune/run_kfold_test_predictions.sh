#!/bin/bash -l
#SBATCH --job-name=kfold-test-predictions
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=1:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Per-sample test predictions for one MethylLlama k-fold checkpoint, for a paired
# subject-level bootstrap comparison against MethylGPT (extracted separately
# in the MethylGPT-thesis repo, same fixed 2,149-sample test set, joined by
# real GSM sample ID). Inference-only, isolated script -- does not modify or
# resubmit any training job.
#
# Usage: FOLD=0 sbatch scripts/finetune/run_kfold_test_predictions.sh
#   (submit once per fold: FOLD=0,1,2,3,4)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

FOLD="${FOLD:?ERROR: set FOLD=0..4}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
DATA="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank_finetune.npy"
TEST_IDS="${REPO}/outputs/kfold_splits/test_ids.npy"
DUP_CSV="${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv"
OUTDIR="${REPO}/outputs/bootstrap_predictions/methyllama"

# Best-val_medae checkpoint per fold -- verified this session (matches WandB
# fold*-testeval runs exactly). See scripts/finetune/validate_kfold_test_predictions.py
# for the recorded official metrics each of these must reproduce.
declare -A CKPT=(
    # released fold checkpoints (scripts/download_checkpoint.sh); set CKPT_DIR for your own runs
    [0]="${CKPT_DIR:-${REPO}/checkpoints/finetuned_age}/fold0.ckpt"
    [1]="${CKPT_DIR:-${REPO}/checkpoints/finetuned_age}/fold1.ckpt"
    [2]="${CKPT_DIR:-${REPO}/checkpoints/finetuned_age}/fold2.ckpt"
    [3]="${CKPT_DIR:-${REPO}/checkpoints/finetuned_age}/fold3.ckpt"
    [4]="${CKPT_DIR:-${REPO}/checkpoints/finetuned_age}/fold4.ckpt"
)
CKPT_PATH="${CKPT[$FOLD]:?ERROR: no checkpoint registered for FOLD=$FOLD}"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Fold: ${FOLD}"
echo "Checkpoint: ${CKPT_PATH}"
[ -f "${CKPT_PATH}" ] || { echo "ERROR: checkpoint not found"; exit 1; }
[ -f "${TEST_IDS}" ] || { echo "ERROR: test_ids.npy not found"; exit 1; }

# Determinism only needs proving once -- nothing fold-specific affects
# reproducibility (no dropout, no shuffling, fixed data order, same code
# path for every fold). Restricting the (expensive, runs extraction twice)
# check to fold 0 cuts the other 4 folds' runtime roughly in half.
DET_FLAG=""
if [ "${FOLD}" = "0" ]; then
    DET_FLAG="--determinism_check"
fi

python scripts/finetune/extract_kfold_test_predictions.py \
    --fold "${FOLD}" \
    --checkpoint "${CKPT_PATH}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --test_ids "${TEST_IDS}" \
    --duplicate_pairs_csv "${DUP_CSV}" \
    --outdir "${OUTDIR}" \
    --model_name "MethylLlama" \
    ${DET_FLAG}

echo "DONE fold ${FOLD}: $(date)"
