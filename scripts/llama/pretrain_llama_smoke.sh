#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-smoke
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=8:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test: ~2000 samples × all 49k CpGs
# Purpose: verify pipeline end-to-end with exact same vocab size, sequence
# length, and GPU memory profile as the full 170k × 49k run.
#
# Run make_pretrain_subset.py first to generate the subset h5ad:
#   python scripts/utils/make_pretrain_subset.py \
#       --tar     /path/to/processed_type3_parquet_shuffled.tar.gz \
#       --probes  /path/to/probe_ids_type3_pretrain.csv \
#       --out_dir /sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_smoke \
#       --n_samples 2000
#   (top_k_cpg defaults to 0 = keep all 49,156 CpGs)
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-smoke"

SMOKE_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_smoke"
DATA="${SMOKE_DIR}/methylgpt_pretrain_type3_subset.h5ad"
PROBE_IDS_CSV="${SMOKE_DIR}/probe_ids_type3_pretrain_subset.csv"

# ─────────────────────────────────────────────────────────────────────────────
# Model — default: small (~5M). Override: MODEL_SIZE=full sbatch ...
# ─────────────────────────────────────────────────────────────────────────────
MODEL_SIZE="${MODEL_SIZE:-small}"

if [ "${MODEL_SIZE}" = "small" ]; then
    HIDDEN_SIZE=256
    NUM_LAYERS=4
    NUM_HEADS=4
    INTERMEDIATE_SIZE=320
    WANDB_RUN_SUFFIX="small"
else
    HIDDEN_SIZE=768
    NUM_LAYERS=8
    NUM_HEADS=12
    INTERMEDIATE_SIZE=2048
    WANDB_RUN_SUFFIX="full"
fi

ROPE_THETA=10000.0
N_SIN_BASIS=48
BASIS_SCALE=2.0

# ─────────────────────────────────────────────────────────────────────────────
# WCED settings — same as full pretrain for realistic test
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"       # All CpGs — matches full run vocab/sequence length
INPUT_RATIO=0.5
AGE_WEIGHT=0.0                      # No age labels in pretrain corpus
CONTRASTIVE=false
CONTRASTIVE_WEIGHT=0.0
NORMALIZE_LOSS=false
DECODER_DROPOUT=0.1

# ─────────────────────────────────────────────────────────────────────────────
# Training — short run: enough to confirm convergence trend
# ─────────────────────────────────────────────────────────────────────────────
LR=5e-4
WEIGHT_DECAY=0.01
WARMUP_STEPS=50
BATCH_SIZE=8
ACCUM=4                             # Effective batch = 8 × 1GPU × 4 = 32
PRETRAIN_EPOCHS=60
EARLY_STOP=60                       # No early stop during smoke test

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-smoke"
WANDB_RUN_NAME="smoke-${WANDB_RUN_SUFFIX}-k${SUBSET_K}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

# New tokenizer for smoke-test CpGs (built automatically on first run)
TOKENIZER_PATH="${REPO}/tokenizer_llama_smoke_${WANDB_RUN_SUFFIX}"

# Resume from previous run (auto-picks best checkpoint by val_loss)
PREV_RUN="${OUTROOT}/smoke-small-k49156-44387304"
RESUME_CHECKPOINT="$(ls ${PREV_RUN}/checkpoints/*.ckpt 2>/dev/null | sort | tail -1)"
echo "Resuming from: ${RESUME_CHECKPOINT}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo " SMOKE TEST — MODEL: ${MODEL_SIZE} (${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:    ${DATA}  (2000 samples)"
echo "CpGs:    ${SUBSET_K} (all 49k)"
echo "Train:   lr=${LR}, batch=${BATCH_SIZE}×1GPU×${ACCUM}accum=$(( BATCH_SIZE * ACCUM )) eff"
echo "Epochs:  ${PRETRAIN_EPOCHS}"
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
    print("GPU memory:", round(torch.cuda.get_device_properties(0).total_memory / 1e9, 1), "GB")
PY

# ─────────────────────────────────────────────────────────────────────────────
# Smoke test pretraining
# Single GPU: strategy=auto (no DDP overhead)
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
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "Smoke test finished: $(date)"
echo ""
echo "What to check in WandB (${WANDB_PROJECT}/${WANDB_RUN_NAME}):"
echo "  1. train/loss    — should decrease steadily from epoch 1"
echo "  2. train/pcc     — should rise above 0.5 within 5 epochs"
echo "  3. validation/loss — must appear (confirms valid split works)"
echo "  4. GPU memory usage — log from nvidia-smi below"
echo "============================================================"
nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader
