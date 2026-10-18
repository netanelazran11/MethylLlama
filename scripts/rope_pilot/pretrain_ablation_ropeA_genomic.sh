#!/bin/bash -l
#SBATCH --job-name=ablation-rope-A-genomic
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=goldfish --gres=gpu:h200:1

# ─────────────────────────────────────────────────────────────────────────────
# Genomic RoPE ablation — RUN A: genomic-order positions (the "on" condition).
#
# Isolated pilot experiment, NOT a modification of the production pretrain
# pipeline: new script, new WandB project, new output dir, small fixed
# 5,000-sample subset (scripts/rope_pilot/create_pretrain_subset_h5ad.py — run
# run_ablation_rope_step0_subset.sh first). Architecture, contrastive
# settings, and all hyperparameters are otherwise identical to the production
# recipe (pretrain_methyllama.sh) except warmup/epochs, which
# are scaled down because the subset has far fewer steps/epoch — this does
# not affect ablation validity since Run B uses the identical scaled-down
# settings; only wced_genomic_rank_path differs between A and B.
#
# Companion: pretrain_ablation_ropeB_nogenomic.sh (same everything, no
# genomic_rank_path -> RoPE falls back to sequential arbitrary-order positions)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
LOGDIR="${REPO}/logs_llama-wced"

# ─── Data — fixed subset from Step 0, SAME file used by Run B ────────────────
PRETRAIN_DATA="${REPO}/outputs/ablation_rope/pretrain_subset_5000.h5ad"
DATA_DIR="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_pretrain.csv"

# Genomic rank array — SAME file the production run uses (column identity is
# unchanged by row-subsetting, so this rank table is still valid)
GENOMIC_RANK_NPY="${REPO}/resources/cpg_genomic_rank.npy"

# ─── Architecture — identical to production 6L recipe ────────────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-512}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"

# ─── WCED settings — identical to production recipe ───────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"
AGE_WEIGHT="${AGE_WEIGHT:-0.0}"
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.05}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
DIAG_CHECK_EVERY="${DIAG_CHECK_EVERY:-5}"

# ─── Training hyperparameters — scaled down for the 5k-sample pilot ──────────
# ~5000 samples / batch=32 / 1 GPU / accum=2 -> eff batch 64 -> ~78 steps/epoch
# (production: ~169k samples, eff batch 256 -> ~660 steps/epoch)
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"       # scaled down from 3000 (fewer total steps)
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-2}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-30}"

# ─── Resume (optional) — set to continue a crashed/timed-out run ────────────
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─── WandB — separate project, does not mix with production runs ─────────────
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="pretrain-ablation-rope"
WANDB_RUN_NAME="ropeA-genomic-5k-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO}/tokenizer}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

if [ ! -f "${PRETRAIN_DATA}" ]; then
    echo "ERROR: subset h5ad not found: ${PRETRAIN_DATA}"
    echo "Run scripts/rope_pilot/run_ablation_rope_step0_subset.sh first."
    exit 1
fi
if [ ! -f "${GENOMIC_RANK_NPY}" ]; then
    echo "ERROR: genomic rank file not found: ${GENOMIC_RANK_NPY}"
    exit 1
fi

echo "============================================================"
echo "ROPE ABLATION — RUN A (genomic-order positions)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:         ${PRETRAIN_DATA}  (fixed 5k subset)"
echo "Genomic RoPE: ${GENOMIC_RANK_NPY}  <-- ON for this run"
echo "Model:        ${NUM_LAYERS}L x ${HIDDEN_SIZE}D x ${NUM_HEADS}H, FFN=${INTERMEDIATE_SIZE}"
echo "Output:       ${OUTDIR}"
echo "W&B:          ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "============================================================"

methyllama_env   # environment setup, defined in configs/paths.sh

cd "${REPO}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${PRETRAIN_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.bmfm_style=false \
    model.hidden_size="${HIDDEN_SIZE}" \
    model.num_hidden_layers="${NUM_LAYERS}" \
    model.num_attention_heads="${NUM_HEADS}" \
    model.intermediate_size="${INTERMEDIATE_SIZE}" \
    model.rope_theta="${ROPE_THETA}" \
    model.n_sin_basis="${N_SIN_BASIS}" \
    model.basis_scale="${BASIS_SCALE}" \
    trainer.learning_rate="${LR}" \
    trainer.weight_decay="${WEIGHT_DECAY}" \
    trainer.warmup_steps="${WARMUP_STEPS}" \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_age_weight="${AGE_WEIGHT}" \
    wced_contrastive="${CONTRASTIVE}" \
    wced_contrastive_weight="${CONTRASTIVE_WEIGHT}" \
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
    wced_normalize_loss="${NORMALIZE_LOSS}" \
    wced_decoder_dropout="${DECODER_DROPOUT}" \
    wced_genomic_rank_path="${GENOMIC_RANK_NPY}" \
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience="${EARLY_STOP}" \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    diag_check_every="${DIAG_CHECK_EVERY}" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "RUN A (genomic) finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
