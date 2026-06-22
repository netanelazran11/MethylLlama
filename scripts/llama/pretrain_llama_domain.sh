#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-domain
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-domain"

# ─── DOMAIN DATA: AltumAge 21k (~11.5k samples × 21368 CpGs) ────────────────
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad"
DATA="${DATA_DIR}/altumage_21k_3way.h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_21k.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Architecture settings — same as 49k run for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NUM_HEADS="${NUM_HEADS:-12}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"

# ─────────────────────────────────────────────────────────────────────────────
# WCED settings
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-21368}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"
AGE_WEIGHT="${AGE_WEIGHT:-0.5}"
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# seq_len = 21368×0.5 + 1 = 10685 tokens
# B=16 × 2 GPUs × 8 accum = 256 effective batch size
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUM="${ACCUM:-8}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-60}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-domain"
WANDB_RUN_NAME="llama-domain-k${SUBSET_K}-r${INPUT_RATIO}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

# New tokenizer for AltumAge CpGs — built automatically on first run
TOKENIZER_PATH="${REPO}/tokenizer_llama_domain21k"

mkdir -p "${LOGDIR}" "${OUTDIR}" "${TOKENIZER_PATH}"

echo "============================================================"
echo "METHYLLAMA DOMAIN PRETRAINING (AltumAge 21k)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:    ${DATA}"
echo "CpGs:    ${SUBSET_K}, input_ratio=${INPUT_RATIO}"
echo "Model:   ${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H"
echo "Train:   lr=${LR}, batch=${BATCH_SIZE}×2GPUs×${ACCUM}accum=$(( BATCH_SIZE * 2 * ACCUM )) eff"
echo "Tokenizer: ${TOKENIZER_PATH}"
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
# Data has no split column → auto 80/10/10 (seed=42): ~9.2k train, ~1.15k val/test
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset=false \
    data_module.max_length=$(python3 -c "import math; print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
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
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Domain pretraining finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "Next: sbatch finetune_llama.sh with CHECKPOINT=<best ckpt> TOKENIZER_PATH=${TOKENIZER_PATH}"
echo "============================================================"
