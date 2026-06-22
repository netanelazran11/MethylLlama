#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-small
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=120:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# MethylLlama-Small: 256D × 4L × 4H (~5M params)
# Same data, WCED settings, and GPU config as pretrain_llama.sh (full model).
# Only difference: smaller architecture — for fair small vs full comparison.
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

# ─── PRETRAIN DATA (same as full model run) ───────────────────────────────────
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad"
PRETRAIN_DATA="${PRETRAIN_DATA:-${DATA_DIR}/methylgpt_pretrain_type3.h5ad}"
PROBE_IDS_CSV="${PROBE_IDS_CSV:-${DATA_DIR}/probe_ids_type3_pretrain.csv}"
DATA="${PRETRAIN_DATA}"

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — small variant (256D × 4L × 4H)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-4}"
NUM_HEADS="${NUM_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-320}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"

# ─────────────────────────────────────────────────────────────────────────────
# WCED settings — identical to pretrain_llama.sh for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"
AGE_WEIGHT="${AGE_WEIGHT:-0.0}"
CONTRASTIVE="${CONTRASTIVE:-false}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.0}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-false}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# Small model fits larger batch per GPU: B=32 × 4 GPUs × 2 accum = 256 eff
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-2}"                       # Effective batch = 32 × 4 GPUs × 2 = 256
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-60}"

# ─────────────────────────────────────────────────────────────────────────────
# Resume (optional)
# ─────────────────────────────────────────────────────────────────────────────
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-wced"
WANDB_RUN_NAME="llama-small-all49k-r${INPUT_RATIO}-w${CONTRASTIVE_WEIGHT}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

# Same tokenizer as full model — CpG ID→index mapping must be consistent
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO}/tokenizer_llama_pretrain49k}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA-SMALL PRETRAINING (256D × 4L × 4H, ~5M params)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:    ${DATA}"
echo "CpGs:    ${SUBSET_K}, input_ratio=${INPUT_RATIO}"
echo "Model:   ${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H (head_dim=$((HIDDEN_SIZE / NUM_HEADS)))"
echo "Train:   lr=${LR}, batch=${BATCH_SIZE}×4GPUs×${ACCUM}accum=$(( BATCH_SIZE * 4 * ACCUM )) eff, epochs=${PRETRAIN_EPOCHS}"
echo "Output:  ${OUTDIR}"
echo "W&B:     ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "============================================================"

# ─────────────────────────────────────────────────────────────────────────────
# Environment
# ─────────────────────────────────────────────────────────────────────────────
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

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

# ─────────────────────────────────────────────────────────────────────────────
# Pretraining
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset=true \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=$(python3 -c "import math; print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=14 \
    data_module.bmfm_style=true \
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
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience="${EARLY_STOP}" \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "Small model pretraining finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
