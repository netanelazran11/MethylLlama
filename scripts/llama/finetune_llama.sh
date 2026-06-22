#!/bin/bash -l
#SBATCH --job-name=finetune-llama
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=10:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs"

# ─── FINETUNE DATA (AltumAge 8k: labeled with age) ──────────────────────────
# Provide cluster path here:
FINETUNE_DATA="${FINETUNE_DATA:-/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad}"

DATA="${FINETUNE_DATA}"

# ─── TOKENIZER: MUST be the same one created during pretrain ─────────────────
# Points to tokenizer built from 49k pretrain data so CpG IDs match the model
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO}/tokenizer_llama_pretrain49k}"

# REQUIRED: set to your pretrained LLaMA checkpoint path
CHECKPOINT="${CHECKPOINT:-???}"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-8000}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"   # Match pretraining distribution

# ─────────────────────────────────────────────────────────────────────────────
# Age normalization
# Compute from AltumAge training set: approximately mean≈47, std≈22
# ─────────────────────────────────────────────────────────────────────────────
AGE_MEAN="${AGE_MEAN:-47.0}"
AGE_STD="${AGE_STD:-22.0}"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning settings
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
EARLY_STOP="${EARLY_STOP:-30}"

# Encoder: M1 = frozen (use WCED CLS as-is)
# Set FREEZE_ENCODER=false for M2 (fine-tune encoder — risks corrupting WCED repr)
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-9999}"
# recon_weight=0.0: decoder was trained on random 8k subsets of 49k pretrain CpGs.
# During finetune the 8k AltumAge CpGs are a fixed specific subset — IDs may not align.
# Disable decoder regularizer to avoid mismatched reconstruction loss.
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-wced"
WANDB_RUN_NAME="llama-ft-k${SUBSET_K}-freeze${FREEZE_ENCODER}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA WCED FINE-TUNING"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Strategy: freeze_encoder=${FREEZE_ENCODER}, recon_weight=${RECON_WEIGHT}"
echo "LR=${LR}, batch=${BATCH_SIZE}×${ACCUM}=${BATCH_SIZE}*${ACCUM}"
echo "Age norm: mean=${AGE_MEAN}, std=${AGE_STD}"
echo "Output: ${OUTDIR}"
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

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.finetune_llama \
    data_path="${DATA}" \
    checkpoint_path="${CHECKPOINT}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=$((SUBSET_K + 2)) \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    wced_input_ratio="${INPUT_RATIO}" \
    age_mean="${AGE_MEAN}" \
    age_std="${AGE_STD}" \
    finetune.learning_rate="${LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight="${RECON_WEIGHT}" \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Fine-tuning finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
