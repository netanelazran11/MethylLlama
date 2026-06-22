#!/bin/bash -l
#SBATCH --job-name=finetune-llama-domain-dna
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=24:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-domain"

DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad"
DATA="${DATA_DIR}/altumage_21k_3way.h5ad"

# Tokenizer: same one used during domain pretraining (must match CpG IDs)
TOKENIZER_PATH="${REPO}/tokenizer_llama_domain21k"

# Pretrained checkpoint — override via: CHECKPOINT=<path> sbatch finetune_llama_domain.sh
CHECKPOINT="${CHECKPOINT:-/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-llama-domain-dna/llama-domain-dna-k21368-r0.5-44366438/checkpoints/epoch=156-val_loss=0.1276.ckpt}"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings — match pretraining distribution exactly
# subset_k=21368: all domain CpGs; input_ratio=0.5: same as pretrain
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-21368}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-100}"
EARLY_STOP="${EARLY_STOP:-30}"

# Encoder: frozen (M1 strategy — CLS already encodes age from WCED pretraining)
# Set FREEZE_ENCODER=false + UNFREEZE_EPOCH=5 for gradual unfreeze (M2 strategy)
FREEZE_ENCODER="${FREEZE_ENCODER:-true}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-9999}"

# recon_weight=0.1: pretrain and finetune share the same 21k CpGs, so decoder IDs align
RECON_WEIGHT="${RECON_WEIGHT:-0.1}"

# Resume from a fine-tuning checkpoint (optional)
# Usage: RESUME_CHECKPOINT=<path> sbatch finetune_llama_domain_dna.sh
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-domain"
WANDB_RUN_NAME="llama-domain-ft-k${SUBSET_K}-freeze${FREEZE_ENCODER}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA DOMAIN FINE-TUNING (Age Prediction)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:        ${DATA}"
echo "Checkpoint:  ${CHECKPOINT}"
echo "CpGs:        ${SUBSET_K}, input_ratio=${INPUT_RATIO}"
echo "Train:       lr=${LR}, batch=${BATCH_SIZE}×${ACCUM}accum=$(( BATCH_SIZE * ACCUM )) eff"
echo "Freeze enc:  ${FREEZE_ENCODER} (unfreeze epoch=${UNFREEZE_EPOCH})"
echo "Recon wt:    ${RECON_WEIGHT}"
echo "Resume ckpt: ${RESUME_CHECKPOINT:-<none>}"
echo "Output:      ${OUTDIR}"
echo "W&B:         ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
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
# Compute age mean/std from training split
# ─────────────────────────────────────────────────────────────────────────────
read AGE_MEAN AGE_STD < <(python3 - 2>/dev/null <<'PY'
import warnings, anndata, numpy as np
warnings.filterwarnings("ignore")
adata = anndata.read_h5ad("/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad")
train = adata[adata.obs["split"] == "train"]
ages = train.obs["age"].dropna().astype(float).values
print(f"{ages.mean():.4f} {ages.std():.4f}")
PY
)
echo "Age normalization: mean=${AGE_MEAN}, std=${AGE_STD}"

# ─────────────────────────────────────────────────────────────────────────────
# Fine-tuning
# max_length = int(subset_k * input_ratio) + 1  (same as pretraining)
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.finetune_llama \
    data_path="${DATA}" \
    "checkpoint_path='${CHECKPOINT}'" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    age_mean="${AGE_MEAN}" \
    age_std="${AGE_STD}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.max_length=$(python3 -c "print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    wced_input_ratio="${INPUT_RATIO}" \
    finetune.learning_rate="${LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight="${RECON_WEIGHT}" \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience="${EARLY_STOP}" \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"+resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "Fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
