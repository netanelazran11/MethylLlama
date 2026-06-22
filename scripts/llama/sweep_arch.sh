#!/bin/bash -l
#SBATCH --job-name=arch-sweep
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00
#SBATCH --array=0-7

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/%x_%A_%a.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/%x_%A_%a.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# 9-variant architecture sweep via SLURM job array (--array=0-8)
# Each task picks its (hidden_size, num_layers) from the grid below.
#
# Submit:  sbatch scripts/llama/sweep_arch.sh
# Monitor: squeue -u $USER
# Cancel:  scancel <array_job_id>
#
# Grid:
#   hidden_size : 128, 256, 512
#   num_layers  :   2,   4,   8
# ─────────────────────────────────────────────────────────────────────────────

# Parameter grid (index 0–8)
HIDDEN_SIZES=(128 128 128 256 256 256 512 512 512)
NUM_LAYERS_LIST=(2 4 8 2 4 8 2 4 8)

HIDDEN_SIZE="${HIDDEN_SIZES[$SLURM_ARRAY_TASK_ID]}"
NUM_LAYERS="${NUM_LAYERS_LIST[$SLURM_ARRAY_TASK_ID]}"

# Derived: head_dim=64, FFN=4×hidden
NUM_HEADS=$(( HIDDEN_SIZE / 64 ))
INTERMEDIATE_SIZE=$(( HIDDEN_SIZE * 4 ))

VARIANT_TAG="h${HIDDEN_SIZE}_l${NUM_LAYERS}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-smoke"

SMOKE_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_smoke"
DATA="${SMOKE_DIR}/methylgpt_pretrain_type3_subset.h5ad"
PROBE_IDS_CSV="${SMOKE_DIR}/probe_ids_type3_pretrain_subset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Fixed hyperparameters
# ─────────────────────────────────────────────────────────────────────────────
ROPE_THETA=10000.0
N_SIN_BASIS=48
BASIS_SCALE=2.0

SUBSET_K=49156
INPUT_RATIO=0.5
AGE_WEIGHT=0.0
CONTRASTIVE=false
CONTRASTIVE_WEIGHT=0.0
NORMALIZE_LOSS=false
DECODER_DROPOUT=0.1

LR=5e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=50
BATCH_SIZE=8
ACCUM=4
PRETRAIN_EPOCHS=30
EARLY_STOP=30

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="arch-sweep"
WANDB_RUN_NAME="${VARIANT_TAG}-${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"
TOKENIZER_PATH="${REPO}/tokenizer_llama_sweep_${VARIANT_TAG}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo " ARCH SWEEP [task ${SLURM_ARRAY_TASK_ID}/7] — ${VARIANT_TAG}"
echo " Model: ${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H  FFN=${INTERMEDIATE_SIZE}"
echo "============================================================"
echo "Job: ${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID} | Host: $(hostname) | Time: $(date)"
echo "W&B: ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
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
# Pretraining
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset=false \
    data_module.max_length=$(python3 -c "print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=4 \
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

echo "Done: ${VARIANT_TAG} — $(date)"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
