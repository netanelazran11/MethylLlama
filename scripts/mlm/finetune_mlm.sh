#!/bin/bash -l
#SBATCH --job-name=finetune-full8k-bmfm
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
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Pretrained checkpoint — set via env var or override below
# full8k run (subset_k=8000, loss=0.0013):
#   CHECKPOINT=".../pretrain-full8k-bmfm-rna-methylation/add-full8k-44199523/pretrain/checkpoints/BEST.ckpt"
# fixed2048 run (subset_k=2048, loss=0.0013):
#   CHECKPOINT=".../pretrain-fixed2048-bmfm-rna-methylation/add-fixed2048-44043043/pretrain/checkpoints/epoch=epoch=240-val_loss=validation/loss=0.0013.ckpt"
CHECKPOINT="${CHECKPOINT:-/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-full8k-bmfm-rna-methylation/add-full8k-44199523/pretrain/checkpoints/epoch=epoch=279-val_loss=validation/loss=0.0013.ckpt}"

# CpG subset settings - MUST MATCH PRETRAINING!
# full8k pretrain → SUBSET_K=8000 | fixed2048 pretrain → SUBSET_K=2048
SUBSET_K="${SUBSET_K:-8000}"
FIXED_SUBSET="true"
FIXED_SUBSET_SEED="42"

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-full8k-bmfm-rna-methylation"
WANDB_RUN_NAME="finetune-full8k-${SLURM_JOB_ID}"

# Output directory (unique per run to avoid overwriting checkpoints)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "METHYLATION FINE-TUNING (FIXED SUBSET)"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "============================================================"
echo "CpG Subset:  FIXED ${SUBSET_K} CpGs (MUST match pretraining!)"
echo "Seed:        ${FIXED_SUBSET_SEED}"
echo "============================================================"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run:     ${WANDB_RUN_NAME}"
echo "Data:        ${DATA}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "Output dir:  ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules (CUDA/NVCC)
# -------------------------
# Initialize module system (required for non-login shells)
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
print("matmul_precision:", torch.get_float32_matmul_precision())
PY

# -------------------------
# Fine-tuning
# -------------------------
python3 -m bmfm_methylation.mlm.finetune_mlm \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    finetune_epochs=300 \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="${FIXED_SUBSET}" \
    data_module.fixed_subset_seed="${FIXED_SUBSET_SEED}" \
    data_module.max_length=$((SUBSET_K + 2)) \
    data_module.batch_size=32 \
    data_module.num_workers=0 \
    freeze_encoder=false \
    trainer.learning_rate=5e-4 \
    encoder_lr_multiplier=0.3 \
    pearson_weight=0.5 \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    early_stopping.patience=100

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
