#!/bin/bash -l
#SBATCH --job-name=methyllama-kfold-testeval
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Standalone test-eval for k-fold folds whose TRAINING converged but whose job
# hit the 48h limit before the auto test-eval step ran.
#
# For each fold: find best val_medae checkpoint, evaluate on the fixed test set
# (split=='test', 2149 samples) with the correct fold age-normalization.
#
# Usage:
#   CHECKPOINT=<pretrain ep85 ckpt> bash scripts/finetune/testeval_kfold.sh
#   (or set FOLDS="0 1 2" to run a subset)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT to the ep85 pretrain ckpt}"
FOLDS="${FOLDS:-0 1 2 3 4}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
DATA="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
KFOLD_DIR="${REPO}/outputs/kfold_splits"
GENOMIC_RANK_FT_NPY="${REPO}/resources/cpg_genomic_rank_finetune.npy"
TOKENIZER_PATH="${REPO}/tokenizer"
OUTROOT="${REPO}/outputs/finetune-llama-small"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="finetune-llama-small"
WANDB_GROUP="methyllama-kfold-5fold"

methyllama_env   # environment setup, defined in configs/paths.sh
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

for FOLD in ${FOLDS}; do
    # There can be MANY dirs per fold (failed/smoke submissions). Pick the one that
    # actually has val_medae checkpoints; if several, the highest job id (newest).
    OUTDIR=""
    for CAND in $(find "${OUTROOT}" -maxdepth 1 -type d -name "methyllama-kfold-fold${FOLD}-ep*" 2>/dev/null | sort); do
        if find "${CAND}/checkpoints" -name "epoch=*-val_medae=*.ckpt" 2>/dev/null | grep -q .; then
            OUTDIR="${CAND}"
        fi
    done
    echo ">> fold ${FOLD}: OUTDIR='${OUTDIR}'"
    if [ -z "${OUTDIR}" ] || [ ! -d "${OUTDIR}/checkpoints" ]; then
        echo "SKIP fold ${FOLD}: no output dir with val_medae checkpoints under ${OUTROOT}"
        continue
    fi
    # best val_medae checkpoint = lowest value in filename
    BEST_CKPT=$(find "${OUTDIR}/checkpoints" -name "epoch=*-val_medae=*.ckpt" 2>/dev/null \
        | sed -E 's/.*val_medae=([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
    if [ -z "${BEST_CKPT}" ]; then
        echo "SKIP fold ${FOLD}: no val_medae checkpoint"
        continue
    fi
    FOLD_TRAIN_NPY="${KFOLD_DIR}/fold_${FOLD}_train.npy"
    FOLD_VAL_NPY="${KFOLD_DIR}/fold_${FOLD}_val.npy"

    echo "============================================================"
    echo "Fold ${FOLD} TEST EVAL"
    echo "  dir  : ${OUTDIR}"
    echo "  ckpt : ${BEST_CKPT}"
    echo "============================================================"

    python -m bmfm_methylation.llama.finetune_llama \
        "data_path='${DATA}'" \
        "checkpoint_path='${CHECKPOINT}'" \
        "tokenizer_path='${TOKENIZER_PATH}'" \
        "output_directory='${OUTDIR}'" \
        "+eval_checkpoint='${BEST_CKPT}'" \
        data_module.subset_k=49156 \
        data_module.fixed_subset_seed=42 \
        data_module.max_length=21369 \
        data_module.batch_size=32 \
        data_module.num_workers=8 \
        data_module.filter_age_outliers=true \
        "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
        "+data_module.fold_train_ids_npy='${FOLD_TRAIN_NPY}'" \
        "+data_module.fold_val_ids_npy='${FOLD_VAL_NPY}'" \
        wced_input_ratio=1.0 \
        wced_genomic_rank_path="${GENOMIC_RANK_FT_NPY}" \
        finetune.head_hidden_size=128 \
        finetune.head_dropout=0.1 \
        finetune.pooling=cls \
        finetune.loss_type=huber \
        precision="16-mixed" \
        track_wandb.enabled=true \
        track_wandb.project="${WANDB_PROJECT}" \
        track_wandb.entity="${WANDB_ENTITY}" \
        track_wandb.name="methyllama-kfold-fold${FOLD}-testeval" \
        "+track_wandb.group='${WANDB_GROUP}'"

    echo "Fold ${FOLD} test eval done."
done

echo "============================================================"
echo "ALL TEST EVALS DONE: $(date)"
echo "============================================================"
