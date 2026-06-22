#!/bin/bash -l
#SBATCH --job-name=downstream-multitask
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=10:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CHECKPOINT="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt"

# NOTE: data_path must be a MERGED h5ad with columns: age, smoking_status, sex, split
# See data_prep/merge_multitask_data.py (TODO: merge your age h5ad + smoking h5ad)
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/merged/multitask_data.h5ad"

WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="methyl-downstream-multitask"
WANDB_RUN_NAME="multitask-wced-${SLURM_JOB_ID}"
OUTDIR="${REPO}/outputs/downstream/multitask/${WANDB_RUN_NAME}"
mkdir -p "${REPO}/logs" "${OUTDIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "============================================================"
echo "TASK B — MULTI-TASK PROBING (age + smoking + sex)"
echo "============================================================"

python -m bmfm_methylation.downstream.multitask.finetune_multitask \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    subset_k=4000 \
    batch_size=32 \
    finetune_epochs=100 \
    learning_rate=1e-4 \
    age_weight=1.0 \
    smoking_weight=1.0 \
    sex_weight=0.5 \
    freeze_encoder=true \
    unfreeze_encoder_epoch=5 \
    early_stop_patience=20 \
    use_wandb=true \
    wandb_project="${WANDB_PROJECT}" \
    wandb_entity="${WANDB_ENTITY}" \
    wandb_run_name="${WANDB_RUN_NAME}"

echo "Done: $(date)"
