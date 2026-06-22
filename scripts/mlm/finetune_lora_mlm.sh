#!/bin/bash -l
#SBATCH --job-name=finetune-lora
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# -------------------------
# WCED pretrained checkpoint
# -------------------------
CHECKPOINT="${CHECKPOINT:-/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt}"

# CpG subset — 8k fixed (maximum information per step)
SUBSET_K="${SUBSET_K:-8000}"

# LoRA hyperparameters
LORA_RANK="${LORA_RANK:-8}"           # Rank of adapters (4-16 typical)
LORA_ALPHA="${LORA_ALPHA:-16}"        # Scaling = alpha/rank = 2.0
LORA_LR="${LORA_LR:-1e-4}"           # LoRA adapter LR (between head and frozen)

# Training hyperparameters
LEARNING_RATE="${LEARNING_RATE:-1e-3}"   # Head LR (fast)
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUMULATE_GRAD="${ACCUMULATE_GRAD:-4}"  # Effective batch = 16 * 4 = 64
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP_PATIENCE="${EARLY_STOP_PATIENCE:-60}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.1}"

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-lora-bmfm"
WANDB_RUN_NAME="lora-r${LORA_RANK}-k${SUBSET_K}-${SLURM_JOB_ID}"

# Output directory
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "WCED + LoRA FINE-TUNING"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "============================================================"
echo "Checkpoint:  ${CHECKPOINT}"
echo "CpG Subset:  FIXED ${SUBSET_K} CpGs (maximum information)"
echo "LoRA:        rank=${LORA_RANK}, alpha=${LORA_ALPHA}, lr=${LORA_LR}"
echo "             targets: query + value in all 6 attention layers"
echo "             trainable params: ~$((LORA_RANK * 512 * 2 * 2 * 6)) (0.4% of encoder)"
echo "Head LR:     ${LEARNING_RATE}  |  LoRA LR: ${LORA_LR}"
echo "Batch:       ${BATCH_SIZE} x ${ACCUMULATE_GRAD} = $((BATCH_SIZE * ACCUMULATE_GRAD)) effective"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run:     ${WANDB_RUN_NAME}"
echo "Output dir:  ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules
# -------------------------
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true

module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

# -------------------------
# Env
# -------------------------
cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

# -------------------------
# LoRA Fine-tuning
# -------------------------
python -m bmfm_methylation.mlm.finetune_lora \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    finetune_epochs=${FINETUNE_EPOCHS} \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="true" \
    data_module.fixed_subset_seed="42" \
    data_module.max_length=$((SUBSET_K + 2)) \
    data_module.batch_size=${BATCH_SIZE} \
    data_module.num_workers=0 \
    accumulate_grad_batches=${ACCUMULATE_GRAD} \
    trainer.learning_rate=${LEARNING_RATE} \
    lora_rank=${LORA_RANK} \
    lora_alpha=${LORA_ALPHA} \
    lora_lr=${LORA_LR} \
    regression_head.dropout=${HEAD_DROPOUT} \
    early_stopping.patience=${EARLY_STOP_PATIENCE} \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "LoRA Fine-tuning finished: $(date)"
echo "============================================================"
