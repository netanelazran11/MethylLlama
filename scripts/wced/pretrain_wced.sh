#!/bin/bash -l
#SBATCH --job-name=pretrain-wced
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=50:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Combine style
COMBINE_STYLE="${COMBINE_STYLE:-add}"

# CpG subset settings
SUBSET_K="${SUBSET_K:-8000}"
FIXED_SUBSET="true"
FIXED_SUBSET_SEED="42"

# WCED input ratio (fraction of CpGs per view)
# For contrastive: 0.5 = non-overlapping views
INPUT_RATIO="${INPUT_RATIO:-0.5}"

# Age supervision (multi-task learning)
AGE_WEIGHT="${AGE_WEIGHT:-1.0}"

# Contrastive learning settings
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"

# Normalize loss (removes "predict averages" shortcut)
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"

# ============================================================
# MULTI-TASK WCED SETTINGS
# ============================================================
# Architecture:
#   Input:   Random subset (50%) of CpGs
#   Encoder: Transformer → CLS
#   Decoder: Linear(CLS) → ALL vocab_size beta predictions
#   Age Head: Linear(CLS) → age prediction
#   Loss:    Reconstruction + λ_age * MSE(age_pred, age_true)
#
# Key insight: Age supervision forces CLS to encode sample-
# specific information (age), preventing CLS collapse.
# ============================================================

# Standard encoder settings
HIDDEN_SIZE="${HIDDEN_SIZE:-512}"
NUM_ATTENTION_HEADS="${NUM_ATTENTION_HEADS:-8}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"

# Decoder dropout
WCED_DECODER_DROPOUT="0.1"

# Training settings
EARLY_STOP_PATIENCE="20"
PRETRAIN_EPOCHS="300"

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-wced-bmfm"
WANDB_RUN_NAME="wced-contrastive-k${SUBSET_K}-w${CONTRASTIVE_WEIGHT}-${SLURM_JOB_ID}"

# Output directory
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "CONTRASTIVE WCED PRETRAINING"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "============================================================"
echo "ARCHITECTURE:"
echo "  Two views per sample: ${INPUT_RATIO} of ${SUBSET_K} CpGs each"
echo "  Encoder: Transformer → CLS1, CLS2"
echo "  Contrastive: Same-sample views → similar CLS"
echo "  Decoder: Linear([CLS]) → ${SUBSET_K} betas"
echo "  Loss: Reconstruction + ${CONTRASTIVE_WEIGHT} * Contrastive"
echo "============================================================"
echo "CpG Vocab:   ${SUBSET_K} CpGs"
echo "Input ratio: ${INPUT_RATIO} per view"
echo "Age weight:  ${AGE_WEIGHT} (multi-task supervision)"
echo "Contrastive: ${CONTRASTIVE} (weight=${CONTRASTIVE_WEIGHT})"
echo "Normalize:   ${NORMALIZE_LOSS}"
echo "Combine:     ${COMBINE_STYLE}"
echo "Model:       hidden=${HIDDEN_SIZE}, heads=${NUM_ATTENTION_HEADS}"
echo "============================================================"
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
# Run WCED Pretraining
# -------------------------
python -m bmfm_methylation.wced.pretrain_wced \
    data_path="${DATA}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    combine_style="${COMBINE_STYLE}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="${FIXED_SUBSET}" \
    data_module.fixed_subset_seed="${FIXED_SUBSET_SEED}" \
    data_module.max_length=$((SUBSET_K + 2)) \
    wced_input_ratio="${INPUT_RATIO}" \
    wced_age_weight="${AGE_WEIGHT}" \
    wced_contrastive="${CONTRASTIVE}" \
    wced_contrastive_weight="${CONTRASTIVE_WEIGHT}" \
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
    wced_normalize_loss="${NORMALIZE_LOSS}" \
    wced_decoder_dropout=${WCED_DECODER_DROPOUT} \
    early_stop_patience=${EARLY_STOP_PATIENCE} \
    pretrain_epochs=${PRETRAIN_EPOCHS} \
    model.hidden_size=${HIDDEN_SIZE} \
    model.num_attention_heads=${NUM_ATTENTION_HEADS} \
    model.intermediate_size=${INTERMEDIATE_SIZE} \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "WCED Pretraining finished: $(date)"
echo "============================================================"
echo "Checkpoint: ${OUTDIR}"
echo "============================================================"
echo "Next steps:"
echo "  1. Check WandB for training curves"
echo "  2. Verify PCC is improving (should reach ~0.95+)"
echo "  3. If good, the [CLS] embedding is now a strong global representation"
echo "  4. Use for finetuning - [CLS] should work well for age prediction"
echo "============================================================"
