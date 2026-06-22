#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-wced
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=64
#SBATCH --mem=400G
#SBATCH --time=120:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

# ─── PRETRAIN DATA (large corpus: 169k samples × 49k CpGs) ──────────────────
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad"
PRETRAIN_DATA="${PRETRAIN_DATA:-${DATA_DIR}/methylgpt_pretrain_type3.h5ad}"
PROBE_IDS_CSV="${PROBE_IDS_CSV:-${DATA_DIR}/probe_ids_type3_pretrain.csv}"

DATA="${PRETRAIN_DATA}"

# ─────────────────────────────────────────────────────────────────────────────
# Architecture settings (MethylLlamaConfig)
# H200 80GB optimal: 768D × 8L × 12H — 134M params, ~40GB at B=16
# (6L×512D was only 10.7GB — 13% of H200, heavily underutilised)
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NUM_HEADS="${NUM_HEADS:-12}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"   # SwiGLU: round(2/3 × 4 × 768 / 64) × 64
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"                # 2.0 for methylation [0,1]

# ─────────────────────────────────────────────────────────────────────────────
# WCED settings
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"             # Use ALL 49k CpGs
INPUT_RATIO="${INPUT_RATIO:-0.5}"         # 50%→50% original WCED split: see 24.5k, predict 24.5k
AGE_WEIGHT="${AGE_WEIGHT:-0.0}"           # No age labels in pretrain corpus
CONTRASTIVE="${CONTRASTIVE:-false}"       # Disabled: pure reconstruction first run
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.0}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-false}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters
# seq_len = 49156×0.5 + 1 = 24579 tokens
# H200 memory at B=8, seq=24.5k, 768D×8L: ~36GB — comfortable
# Effective batch = 8 per GPU × 2 GPUs × 16 accum = 256 samples
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-2000}"
BATCH_SIZE="${BATCH_SIZE:-8}"             # Reduced for 24.5k seq len at 768D
ACCUM="${ACCUM:-8}"                       # Effective batch = 8 × 4 GPUs × 8 = 256
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-60}"

# ─────────────────────────────────────────────────────────────────────────────
# Resume (optional) — set to continue from a previous checkpoint
# Usage: RESUME_CHECKPOINT=/path/to/epoch=X-val_loss=Y.ckpt sbatch ...
# ─────────────────────────────────────────────────────────────────────────────
RESUME_CHECKPOINT="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/outputs/pretrain-llama-wced/llama-wced-all49k-r0.5-w0.0-44248557/checkpoints/epoch=2-val_loss=0.0112.ckpt"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-wced"
WANDB_RUN_NAME="llama-wced-all49k-r${INPUT_RATIO}-w${CONTRASTIVE_WEIGHT}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

# Tokenizer saved here — finetune_llama_wced.sh must use the SAME path
# because the CpG ID→index mapping must be consistent across pretrain and finetune
TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO}/tokenizer_llama_pretrain49k}"

mkdir -p "${LOGDIR}" "${OUTDIR}" "${TOKENIZER_PATH}"

echo "============================================================"
echo "METHYLLAMA WCED PRETRAINING"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "------------------------------------------------------------"
echo "Architecture (LLaMA vs SCBert improvements):"
echo "  RMSNorm + Pre-LN   (stable from step 1, no careful warmup)"
echo "  SwiGLU FFN         (gated, more expressive)"
echo "  RoPE               (relative position, no 4M pos-embed table)"
echo "  ScaleAdaptEncoder  (sinusoidal basis, trainable freqs for beta [0,1])"
echo "------------------------------------------------------------"
echo "Model:   ${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H, intermediate=${INTERMEDIATE_SIZE}"
echo "ScaleAdapt: n_sin_basis=${N_SIN_BASIS}, basis_scale=${BASIS_SCALE}"
echo "------------------------------------------------------------"
echo "Data:    ${SUBSET_K} CpGs, input_ratio=${INPUT_RATIO}"
echo "WCED:    age_weight=${AGE_WEIGHT}, contrastive=${CONTRASTIVE} (w=${CONTRASTIVE_WEIGHT})"
echo "Train:   lr=${LR}, batch=${BATCH_SIZE}×8GPUs×${ACCUM}accum=$(( BATCH_SIZE * 8 * ACCUM )) eff, epochs=${PRETRAIN_EPOCHS}"
echo "Output:  ${OUTDIR}"
echo "W&B:     ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
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

python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("CUDA available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
PY

# ─────────────────────────────────────────────────────────────────────────────
# Pretraining (WCEDLlamaModule using MethylLlamaModel)
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset=true \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=$(python3 -c "import math; print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=14 \
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
    wced_contrastive_temp="${CONTRASTIVE_TEMP}" \
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
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "Pretraining finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
