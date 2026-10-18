#!/bin/bash -l
#SBATCH --job-name=methyllama-random-init
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

# ─────────────────────────────────────────────────────────────────────────────
# Random-init baseline for the CURRENT MethylLlama architecture (Experiment A1, MethylLlama re-run).
#
# PURPOSE: quantify how much WCED pretraining contributes on top of the MethylLlama
# architecture itself (6L, 256D, 4H, Genomic RoPE), vs the same architecture
# trained from scratch. The earlier A1 result (V4b, no RoPE) showed only a
# small pretrain-vs-scratch gap (MedAE 3.63 vs 3.75yr) — this re-runs the same
# idea on the architecture that actually won the k-fold benchmark, on the
# SAME 21k fold-0 split and fixed test set used there, for direct comparison
# against fold 0's pretrained result (test MedAE=3.125yr, MAE=4.440, R²=0.9321).
#
# DESIGN: identical to finetune_kfold.sh (fold 0) in every way
# except encoder initialization:
#   fold 0 (pretrained) → init_mode=pretrained (WCED ep85 checkpoint)
#   this script         → init_mode=random     (fresh MethylLlamaModel)
# Same data, same fold-0 split, same fixed test set, same head (128/0.1),
# same Genomic RoPE, same hyperparameters, same epoch/early-stop budget.
# Differences (intentional, mirrors the original A1 script):
#   1. No checkpoint_path — init_mode=random builds a fresh encoder
#   2. CpG embeddings NOT frozen (randomly initialised, must learn)
#   3. FREEZE_ENCODER=false, UNFREEZE_EPOCH=0 — nothing pretrained to protect
#
# Usage: sbatch scripts/finetune/finetune_random_init.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
LOGDIR="${REPO}/logs_llama-wced"

DATA="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
KFOLD_DIR="${REPO}/outputs/kfold_splits"
GENOMIC_RANK_FT_NPY="${REPO}/resources/cpg_genomic_rank_finetune.npy"
TOKENIZER_PATH="${REPO}/tokenizer"

FOLD="${FOLD:-0}"
FOLD_TRAIN_NPY="${KFOLD_DIR}/fold_${FOLD}_train.npy"
FOLD_VAL_NPY="${KFOLD_DIR}/fold_${FOLD}_val.npy"
TEST_IDS_NPY="${KFOLD_DIR}/test_ids.npy"

# ─── Data settings (identical to MethylLlama kfold) ───────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─── Hyperparameters (identical to MethylLlama kfold) ─────────────────────────────────
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

# Random-init specific: no encoder freeze (nothing pretrained to protect)
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-0}"

# ─── Model architecture — must exactly match MethylLlama (6L × 256D × 4H, FFN=512) ───
VOCAB_SIZE=49161
HIDDEN_SIZE=256
NUM_LAYERS=6
INTERMEDIATE_SIZE=512
NUM_HEADS=4
N_SIN_BASIS=48

# ─── WandB ───────────────────────────────────────────────────────────────────
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="methyllama-random-init-fold${FOLD}-ep${FINETUNE_EPOCHS}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "MethylLlama RANDOM-INIT BASELINE (Experiment A1, re-run on MethylLlama arch)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "init_mode: random (no WCED pretraining)"
echo "Architecture: ${HIDDEN_SIZE}D × ${NUM_LAYERS}L × ${NUM_HEADS}H, FFN=${INTERMEDIATE_SIZE}"
echo "Fold: ${FOLD} | Train IDs: ${FOLD_TRAIN_NPY} | Val IDs: ${FOLD_VAL_NPY}"
echo "Test IDs: ${TEST_IDS_NPY}"
echo "Output: ${OUTDIR}"
echo "============================================================"

methyllama_env   # environment setup, defined in configs/paths.sh

cd "${REPO}"

if [ ! -f "${FOLD_TRAIN_NPY}" ] || [ ! -f "${FOLD_VAL_NPY}" ]; then
    echo "ERROR: fold files missing — run create_kfold_splits.py first"
    exit 1
fi
if [ ! -f "${GENOMIC_RANK_FT_NPY}" ]; then
    echo "Genomic rank file not found — generating: ${GENOMIC_RANK_FT_NPY}"
    python scripts/data/create_finetune_genomic_rank.py
fi

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# ─── Fine-tuning from random init (same data/arch as MethylLlama, no pretrained weights) ─
python -m bmfm_methylation.llama.finetune_llama \
    "data_path='${DATA}'" \
    "tokenizer_path='${TOKENIZER_PATH}'" \
    "output_directory='${OUTDIR}'" \
    init_mode=random \
    model_arch.vocab_size="${VOCAB_SIZE}" \
    model_arch.hidden_size="${HIDDEN_SIZE}" \
    model_arch.num_hidden_layers="${NUM_LAYERS}" \
    model_arch.intermediate_size="${INTERMEDIATE_SIZE}" \
    model_arch.num_attention_heads="${NUM_HEADS}" \
    model_arch.n_sin_basis="${N_SIN_BASIS}" \
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
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight=0.0 \
    finetune.pooling="${POOLING}" \
    finetune.loss_type="${LOSS_TYPE}" \
    finetune.beta_noise=0.0 \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    seed.seed_value=42 \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Random-init fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"

# ── Test evaluation on best val/MedAE checkpoint (same fixed test set) ───────
BEST_CKPT=$(ls "${OUTDIR}/checkpoints/"epoch=*-val_medae=*.ckpt 2>/dev/null \
    | sort -t= -k3 -n \
    | head -1)

if [ -n "${BEST_CKPT}" ]; then
    echo ""
    echo ">>> Evaluating best checkpoint on fixed TEST SET <<<"
    echo "    Checkpoint : ${BEST_CKPT}"
    echo "    Test IDs   : ${TEST_IDS_NPY}"

    python -m bmfm_methylation.llama.finetune_llama \
        "data_path='${DATA}'" \
        "tokenizer_path='${TOKENIZER_PATH}'" \
        "output_directory='${OUTDIR}'" \
        init_mode=random \
        model_arch.vocab_size="${VOCAB_SIZE}" \
        model_arch.hidden_size="${HIDDEN_SIZE}" \
        model_arch.num_hidden_layers="${NUM_LAYERS}" \
        model_arch.intermediate_size="${INTERMEDIATE_SIZE}" \
        model_arch.num_attention_heads="${NUM_HEADS}" \
        model_arch.n_sin_basis="${N_SIN_BASIS}" \
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
        track_wandb.name="${WANDB_RUN_NAME}-testeval"

    echo "Test eval complete. Metrics logged to WandB run: ${WANDB_RUN_NAME}-testeval"
else
    echo "WARNING: could not find best checkpoint — skipping test eval"
fi

echo "============================================================"
echo "DONE: $(date)"
echo "============================================================"
