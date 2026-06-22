#!/bin/bash -l
#SBATCH --job-name=finetune-seed
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Seed from environment (set via sbatch --export=SEED=XX)
# -------------------------
SEED=${SEED:-42}

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Pretrained checkpoint - FULL 8K CpGs (PCC=0.9963, epoch 288)
CHECKPOINT='/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-full8k-bmfm-rna-methylation/add-full8k-44199523/pretrain/checkpoints/epoch=epoch=288-val_loss=validation/loss=0.0013.ckpt'

# CpG subset settings - FULL 8K CpGs (MUST MATCH PRETRAINING!)
SUBSET_K="${SUBSET_K:-8000}"
FIXED_SUBSET="true"
FIXED_SUBSET_SEED="42"

# W&B naming - include seed
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-full8k-multiseed"
WANDB_RUN_NAME="finetune-full8k-seed${SEED}-${SLURM_JOB_ID}"
WANDB_GROUP="full8k-multiseed-experiment"

# Output directory (unique per seed)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/seed${SEED}-${SLURM_JOB_ID}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "METHYLATION FINE-TUNING (MULTI-SEED)"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "============================================================"
echo "SEED:        ${SEED}"
echo "CpG Subset:  FIXED ${SUBSET_K} CpGs"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B group:   ${WANDB_GROUP}"
echo "W&B run:     ${WANDB_RUN_NAME}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output dir:  ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules (CUDA/NVCC)
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

# Perf / stability knobs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Tensor cores utilization
python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
PY

# -------------------------
# Fine-tuning with specific seed
# -------------------------
# Encoder collapse fix: keep encoder frozen, train head only.
#   Previous joint training (encoder_lr=1e-4) caused encoder to forget
#   pretrained representations → collapse to predicting mean age.
#   Strategy: freeze encoder permanently, train head only (lr=1e-3).
#   This avoids corrupting pretrained weights while learning the age mapping.
python3 -m bmfm_methylation.finetune \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    seed.seed_value=${SEED} \
    finetune_epochs=300 \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="${FIXED_SUBSET}" \
    data_module.fixed_subset_seed="${FIXED_SUBSET_SEED}" \
    data_module.max_length=$((SUBSET_K + 2)) \
    data_module.batch_size=16 \
    data_module.num_workers=0 \
    accumulate_grad_batches=2 \
    trainer.learning_rate=1e-3 \
    trainer.warmup_steps=200 \
    regression_head.dropout=0.1 \
    freeze_encoder=true \
    unfreeze_encoder_epoch=3 \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    +track_wandb.group="${WANDB_GROUP}" \
    early_stopping.patience=60

echo "============================================================"
echo "Seed ${SEED} - Job finished: $(date)"
echo "============================================================"
