#!/bin/bash -l
#SBATCH --job-name=finetune-llama-random-init
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Random-init baseline for MethylLlama (Experiment A1)
#
# PURPOSE:
#   Measures how much WCED pretraining actually contributes versus having a
#   well-architected transformer fine-tuned from scratch.
#
#   This is the KEY ablation for the thesis. If random-init matches pretrained
#   performance → pretraining adds no value (architecture does the work).
#   If pretrained is substantially better → WCED representations matter.
#
# DESIGN:
#   Identical to V5 in EVERY way except encoder initialization:
#     V5          → init_mode=pretrained  (WCED checkpoint epoch=98)
#     This script → init_mode=random      (PyTorch default Xavier/Kaiming init)
#
#   Same seed (42), same architecture, same LR/epochs/batch/loss, same data.
#   Differences from V5 that are intentional:
#     1. No checkpoint_path (no WCED weights)
#     2. init_mode=random
#     3. CpG embeddings are NOT frozen (randomly initialised, must learn)
#     4. FREEZE_ENCODER=false — no pretrained layers to protect during warmup
#     5. Explicit model_arch params to guarantee identical architecture
#
# EXPECTED RESULT:
#   Random-init will likely reach MedAE ≈ 5–8yr (vs 3.56yr pretrained).
#   The gap quantifies the value of WCED pretraining.
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Data settings — identical to V5
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-1.0}"

# ─────────────────────────────────────────────────────────────────────────────
# Hyperparameters — identical to V5
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-1e-4}"
ENCODER_LR="${ENCODER_LR:-2e-5}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-4}"
FINETUNE_EPOCHS="${FINETUNE_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-100}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
RECON_WEIGHT="${RECON_WEIGHT:-0.0}"
HEAD_HIDDEN="${HEAD_HIDDEN:-256}"
HEAD_DROPOUT="${HEAD_DROPOUT:-0.0}"
POOLING="${POOLING:-cls}"
LOSS_TYPE="${LOSS_TYPE:-huber}"
BETA_NOISE="${BETA_NOISE:-0.0}"

# Random-init specific: no encoder freeze (nothing pretrained to protect)
FREEZE_ENCODER="${FREEZE_ENCODER:-false}"
UNFREEZE_EPOCH="${UNFREEZE_EPOCH:-0}"

# ─────────────────────────────────────────────────────────────────────────────
# Model architecture — must exactly match V5 / WCED pretrained small model
# ─────────────────────────────────────────────────────────────────────────────
VOCAB_SIZE=49161
HIDDEN_SIZE=256
NUM_LAYERS=4
INTERMEDIATE_SIZE=320
NUM_HEADS=4
N_SIN_BASIS=48

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="finetune-llama-small"
WANDB_RUN_NAME="llama-small-ft-random-init-cls-huber-ep${FINETUNE_EPOCHS}-wu${WARMUP_STEPS}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA RANDOM-INIT BASELINE (Experiment A1)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "init_mode: random (no WCED pretraining)"
echo "Architecture: ${HIDDEN_SIZE}D × ${NUM_LAYERS}L × ${NUM_HEADS}H"
echo "Pooling: ${POOLING} | Loss: ${LOSS_TYPE}"
echo "epochs=${FINETUNE_EPOCHS} | early_stop=${EARLY_STOP} | warmup=${WARMUP_STEPS} steps"
echo "batch=${BATCH_SIZE}×${ACCUM}=$(( BATCH_SIZE * ACCUM )) eff"
echo "lr=${LR} (head+encoder, no freeze)"
echo "Data: ${DATA}"
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
# Fine-tuning from random init
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.finetune_llama \
    "data_path='${DATA}'" \
    "tokenizer_path='${TOKENIZER_PATH}'" \
    "output_directory='${OUTDIR}'" \
    init_mode=random \
    model_arch.vocab_size="${VOCAB_SIZE}" \
    model_arch.hidden_size="${HIDDEN_SIZE}" \
    model_arch.num_hidden_layers="${NUM_LAYERS}" \
    model_arch.intermediate_size="${INTERMEDIATE_SIZE}" \
    model_arch.num_attention_heads="${NUM_HEADS}" \
    model_arch.n_sin_basis="${N_SIN_BASIS}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=19609 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    wced_input_ratio="${INPUT_RATIO}" \
    finetune.head_hidden_size="${HEAD_HIDDEN}" \
    finetune.head_dropout="${HEAD_DROPOUT}" \
    finetune.learning_rate="${LR}" \
    finetune.encoder_lr="${ENCODER_LR}" \
    finetune.weight_decay="${WEIGHT_DECAY}" \
    finetune.warmup_steps="${WARMUP_STEPS}" \
    finetune.freeze_encoder="${FREEZE_ENCODER}" \
    finetune.unfreeze_encoder_epoch="${UNFREEZE_EPOCH}" \
    finetune.recon_weight="${RECON_WEIGHT}" \
    finetune.pooling="${POOLING}" \
    finetune.loss_type="${LOSS_TYPE}" \
    finetune.beta_noise="${BETA_NOISE}" \
    finetune_epochs="${FINETUNE_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    gradient_clip_val=1.0 \
    early_stop_patience="${EARLY_STOP}" \
    precision="16-mixed" \
    seed.seed_value=42 \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Random-init fine-tuning finished: $(date)"
echo "Checkpoints: ${OUTDIR}/checkpoints/"
echo "============================================================"
