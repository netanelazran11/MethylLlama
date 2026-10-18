#!/bin/bash -l
#SBATCH --job-name=methyllama-kfold
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=goldfish --gres=gpu:h200:1

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# MethylLlama — 5-fold cross-validation
#
# Usage:
#   CHECKPOINT=/path/to/ckpt sbatch --export=ALL,FOLD=0 finetune_kfold.sh
#   (submit once per fold: FOLD=0,1,2,3,4)
#
# Architecture: identical to MethylLlama (6L × 256D × 4H, Genomic RoPE, head=256→128→1, dropout=0.1)
# Purpose: prove result stability and get confidence intervals on val and test metrics.
# ─────────────────────────────────────────────────────────────────────────────

FOLD="${FOLD:?ERROR: set FOLD=0..4}"
CHECKPOINT="${CHECKPOINT:?ERROR: set CHECKPOINT to the 6L pretrain checkpoint path}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
LOGDIR="${REPO}/logs_llama-wced"

DATA="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
KFOLD_DIR="${REPO}/outputs/kfold_splits"
GENOMIC_RANK_FT_NPY="${REPO}/resources/cpg_genomic_rank_finetune.npy"
TOKENIZER_PATH="${REPO}/tokenizer"

FOLD_TRAIN_NPY="${KFOLD_DIR}/fold_${FOLD}_train.npy"
FOLD_VAL_NPY="${KFOLD_DIR}/fold_${FOLD}_val.npy"
TEST_IDS_NPY="${KFOLD_DIR}/test_ids.npy"

# ─── Hyperparameters (identical to MethylLlama for fair comparison) ──────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
HEAD_HIDDEN="${HEAD_HIDDEN:-128}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"
POOLING="${POOLING:-cls}"
LOSS_TYPE="${LOSS_TYPE:-huber}"

# ─── WandB ───────────────────────────────────────────────────────────────────
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="methyllama-kfold-fold${FOLD}-ep${FINETUNE_EPOCHS}-${SLURM_JOB_ID}"
WANDB_GROUP="methyllama-kfold-5fold"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA — K-FOLD CV (fold ${FOLD} / ${N_FOLDS:-5})"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Fold       : ${FOLD}"
echo "Train IDs  : ${FOLD_TRAIN_NPY}"
echo "Val IDs    : ${FOLD_VAL_NPY}"
echo "Test IDs   : ${TEST_IDS_NPY}"
echo "Checkpoint : ${CHECKPOINT}"
echo "Output     : ${OUTDIR}"
echo "============================================================"

methyllama_env   # environment setup, defined in configs/paths.sh

cd "${REPO}"

# Verify fold files exist
if [ ! -f "${FOLD_TRAIN_NPY}" ] || [ ! -f "${FOLD_VAL_NPY}" ]; then
    echo "ERROR: fold files missing — run create_kfold_splits.py first"
    echo "  Expected: ${FOLD_TRAIN_NPY}"
    echo "  Expected: ${FOLD_VAL_NPY}"
    exit 1
fi

if [ ! -f "${GENOMIC_RANK_FT_NPY}" ]; then
    echo "Genomic rank file not found — generating: ${GENOMIC_RANK_FT_NPY}"
    python scripts/data/create_finetune_genomic_rank.py
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m bmfm_methylation.llama.finetune_llama \
    "data_path='${DATA}'" \
    "checkpoint_path='${CHECKPOINT}'" \
    "tokenizer_path='${TOKENIZER_PATH}'" \
    "output_directory='${OUTDIR}'" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=21369 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.filter_age_outliers=true \
    "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
    "+data_module.fold_train_ids_npy='${FOLD_TRAIN_NPY}'" \
    "+data_module.fold_val_ids_npy='${FOLD_VAL_NPY}'" \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_genomic_rank_path="${GENOMIC_RANK_FT_NPY}" \
    finetune.head_hidden_size="${HEAD_HIDDEN}" \
    finetune.head_dropout="${HEAD_DROPOUT}" \
    finetune.learning_rate="${LR}" \
    finetune.encoder_lr="${ENCODER_LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.warmup_steps="${WARMUP_STEPS}" \
    finetune.freeze_encoder=true \
    finetune.unfreeze_encoder_epoch=10 \
    finetune.recon_weight=0.0 \
    finetune.pooling="${POOLING}" \
    finetune.loss_type="${LOSS_TYPE}" \
    finetune.beta_noise=0.0 \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    "+track_wandb.group='${WANDB_GROUP}'"

echo "============================================================"
echo "Fold ${FOLD} training finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"

# ── Test evaluation on best val/MedAE checkpoint ─────────────────────────────
# Find the checkpoint with lowest val_medae in the filename
BEST_CKPT=$(ls "${OUTDIR}/checkpoints/"epoch=*-val_medae=*.ckpt 2>/dev/null \
    | sort -t= -k3 -n \
    | head -1)

if [ -z "${BEST_CKPT}" ]; then
    echo "WARNING: no val_medae checkpoint found — trying val_loss"
    BEST_CKPT=$(ls "${OUTDIR}/checkpoints/"epoch=*-val_loss=*.ckpt 2>/dev/null \
        | sort -t= -k3 -n | head -1)
fi

if [ -n "${BEST_CKPT}" ]; then
    echo ""
    echo ">>> Evaluating best checkpoint on fixed TEST SET <<<"
    echo "    Checkpoint : ${BEST_CKPT}"
    echo "    Test IDs   : ${TEST_IDS_NPY}"

    python -m bmfm_methylation.llama.finetune_llama \
        "data_path='${DATA}'" \
        "checkpoint_path='${CHECKPOINT}'" \
        "tokenizer_path='${TOKENIZER_PATH}'" \
        "output_directory='${OUTDIR}'" \
        "+eval_checkpoint='${BEST_CKPT}'" \
        data_module.subset_k="${SUBSET_K}" \
        data_module.fixed_subset_seed=42 \
        data_module.max_length=21369 \
        data_module.batch_size="${BATCH_SIZE}" \
        data_module.num_workers=8 \
        data_module.filter_age_outliers=true \
        "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
        "+data_module.fold_train_ids_npy='${FOLD_TRAIN_NPY}'" \
        "+data_module.fold_val_ids_npy='${FOLD_VAL_NPY}'" \
        wced_input_ratio="${INPUT_RATIO}" \
        wced_genomic_rank_path="${GENOMIC_RANK_FT_NPY}" \
        finetune.head_hidden_size="${HEAD_HIDDEN}" \
        finetune.head_dropout="${HEAD_DROPOUT}" \
        finetune.pooling="${POOLING}" \
        finetune.loss_type="${LOSS_TYPE}" \
        precision="16-mixed" \
        track_wandb.enabled=true \
        track_wandb.project="${WANDB_PROJECT}" \
        track_wandb.entity="${WANDB_ENTITY}" \
        track_wandb.name="${WANDB_RUN_NAME}-testeval" \
        "+track_wandb.group='${WANDB_GROUP}'"

    echo "Test eval complete. Metrics logged to WandB run: ${WANDB_RUN_NAME}-testeval"
else
    echo "WARNING: could not find best checkpoint — skipping test eval"
    echo "  Run manually: EVAL_CHECKPOINT=<ckpt> sbatch finetune_kfold.sh"
fi

echo "============================================================"
echo "Fold ${FOLD} DONE: $(date)"
echo "============================================================"
