#!/bin/bash -l
#SBATCH --job-name=downstream-smoking
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=6:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs"

# ── STEP 0: Build smoking h5ad first (once, locally or on cluster) ────────────
# python data_prep/prepare_geo_smoking.py \
#   --from_mrcieu /path/to/mrcieu_output \
#   --output_dir /sci/labs/benjamin.yakir/netanel.azran/data/smoking \
#   --vocab_49k_path /sci/labs/benjamin.yakir/netanel.azran/data/pretrain_cpg_list.csv

# Use combined dataset if available, else fall back to GSE50660 only
if [ -f "/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_combined_aligned.h5ad" ]; then
    DATA="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_combined_aligned.h5ad"
else
    DATA="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_data_aligned.h5ad"
fi
CHECKPOINT="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt"

WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="methyl-downstream-smoking"
WANDB_RUN_NAME="smoking-cls-wced-${SLURM_JOB_ID}"
OUTDIR="${REPO}/outputs/downstream/smoking/${WANDB_RUN_NAME}"
mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "TASK A — SMOKING STATUS CLASSIFICATION"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data:       ${DATA}"
echo "Output:     ${OUTDIR}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m bmfm_methylation.downstream.smoking.finetune_smoking \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    output_directory="${OUTDIR}" \
    label_col=smoking_status \
    binary=true \
    subset_k=4000 \
    batch_size=32 \
    finetune_epochs=100 \
    learning_rate=3e-4 \
    freeze_encoder=true \
    unfreeze_encoder_epoch=999 \
    early_stop_patience=30 \
    use_wandb=true \
    wandb_project="${WANDB_PROJECT}" \
    wandb_entity="${WANDB_ENTITY}" \
    wandb_run_name="${WANDB_RUN_NAME}"

echo "Done: $(date)"
