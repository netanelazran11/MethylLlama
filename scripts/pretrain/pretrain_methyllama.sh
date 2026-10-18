#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-6L-ctr
#SBATCH --gres=gpu:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=168:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=goldfish --gres=gpu:h200:4

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# MethylLlama-Small 6L: 256D × 6L × 4H, FFN=512 (~3.9M encoder params)
#
# Three improvements over baseline (pretrain_llama_small.sh, 4L FFN=320):
#   1. Depth:        4L → 6L  (hierarchical representation, Geneformer-style)
#   2. FFN width:    320 → 512  (expansion ratio 1.25× → 2.0×, standard range)
#   3. Contrastive:  InfoNCE weight=0.05  (view-invariant CLS, reconstruction primary)
#
# RoPE with genomic ordering:
#   CpGs sorted by chromosomal position in each sequence → RoPE encodes
#   true genomic proximity rather than arbitrary probe-ID order.
#   Requires cpg_genomic_rank.npy (produced by sort_probes_genomic.sh).
#
# Baseline: pretrain_llama_small.sh → job 44450919, epoch=98, val_loss=0.0059
# ─────────────────────────────────────────────────────────────────────────────

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
LOGDIR="${REPO}/logs_llama-wced"

# ─── Data ────────────────────────────────────────────────────────────────────
DATA_DIR="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad"
PRETRAIN_DATA="${DATA_DIR}/methylgpt_pretrain_type3.h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_pretrain.csv"

# Genomic rank array (produced by sort_probes_genomic.sh — must exist before this job)
GENOMIC_RANK_NPY="${REPO}/resources/cpg_genomic_rank.npy"

# ─── Architecture — 6L × 256D × 4H, FFN=512 ─────────────────────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-512}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"

# ─── WCED settings ────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"
AGE_WEIGHT="${AGE_WEIGHT:-0.0}"           # no age labels in 169k pretrain data
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.05}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
DIAG_CHECK_EVERY="${DIAG_CHECK_EVERY:-5}"    # print diag metrics every N epochs

# ─── Training hyperparameters ─────────────────────────────────────────────────
LR="${LR:-3e-4}"                          # lower than baseline (5e-4): deeper model
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-3000}"      # longer warmup: more params to stabilize
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-2}"                       # eff batch = 32 × 4 GPUs × 2 = 256
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-30}"

# ─── Resume (optional) ───────────────────────────────────────────────────────
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─── WandB ───────────────────────────────────────────────────────────────────
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="pretrain-llama-wced"
WANDB_RUN_NAME="llama-6L-all49k-r${INPUT_RATIO}-w${CONTRASTIVE_WEIGHT}-genomic-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO}/tokenizer}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

# ─── Validate genomic rank file exists ───────────────────────────────────────
if [ ! -f "${GENOMIC_RANK_NPY}" ]; then
    echo "ERROR: genomic rank file not found: ${GENOMIC_RANK_NPY}"
    echo "Run sort_probes_genomic.sh first, then resubmit."
    exit 1
fi

echo "============================================================"
echo "METHYLLAMA 6L PRETRAINING (256D × 6L × 4H, FFN=512)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:        ${PRETRAIN_DATA}"
echo "CpGs:        ${SUBSET_K}, input_ratio=${INPUT_RATIO}"
echo "Model:       ${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H (head_dim=$((HIDDEN_SIZE / NUM_HEADS)), FFN=${INTERMEDIATE_SIZE})"
echo "Train:       lr=${LR}, warmup=${WARMUP_STEPS}, batch=${BATCH_SIZE}×4GPUs×${ACCUM}=$(( BATCH_SIZE * 4 * ACCUM )) eff"
echo "Contrastive: ${CONTRASTIVE}, weight=${CONTRASTIVE_WEIGHT}, temp=${CONTRASTIVE_TEMP}"
echo "Genomic RoPE: ${GENOMIC_RANK_NPY}"
echo "Output:      ${OUTDIR}"
echo "W&B:         ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "============================================================"

# ─── Environment ─────────────────────────────────────────────────────────────
methyllama_env   # environment setup, defined in configs/paths.sh

cd "${REPO}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ─── Pretraining ─────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${PRETRAIN_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=14 \
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
echo "6L pretraining finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "Next step:  bash scripts/finetune/launch_kfold.sh"
echo "  with CHECKPOINT=<best epoch=XX-val_loss=X.XXXX.ckpt>"
echo "============================================================"
