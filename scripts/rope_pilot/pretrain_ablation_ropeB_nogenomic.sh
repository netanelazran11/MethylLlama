#!/bin/bash -l
#SBATCH --job-name=ablation-rope-B-nogenomic
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=100G
#SBATCH --time=48:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=goldfish --gres=gpu:h200:1

# ─────────────────────────────────────────────────────────────────────────────
# Genomic RoPE ablation — RUN B: arbitrary-order positions (the "off" condition).
#
# Byte-for-byte identical to pretrain_ablation_ropeA_genomic.sh except:
#   1. wced_genomic_rank_path is never passed (omitted entirely, not blanked —
#      see note below on why that distinction matters).
#   2. job-name / WANDB_RUN_NAME / echoed labels say "B / nogenomic".
# Same architecture, same WCED hyperparameters, same fixed 5,000-sample
# subset (pretrain_subset_5000.h5ad — the exact same file Run A used), same
# epoch/early-stop targets, same everything else. This is deliberate: the
# only intentional difference between A and B is whether RoPE receives
# genomic-rank positions or falls back to sequential arbitrary-order ones.
#
# Verified at the code level before running (not just assumed):
#   - bmfm_methylation/llama/pretrain_llama.py:232
#       wced_genomic_rank_path = cfg.get("wced_genomic_rank_path", None)
#     -> safely None when the CLI arg is omitted (confirmed by reading the code)
#   - bmfm_methylation/shared/data_module.py:1113-1114
#       if self.genomic_rank is not None: result["position_ids"] = position_ids_v1
#     -> "position_ids" key is entirely ABSENT from the batch when disabled,
#        not zero-filled (a zero-filled tensor would be a different, wrong
#        condition — this confirms the actual fallback path is used)
#   - bmfm_methylation/llama/model.py RotaryEmbedding.forward
#       if position_ids is not None: ... else: sequential 0..L-1 fallback
#     -> confirms position_ids=None correctly triggers arbitrary/sequential RoPE
#   - pretrain_llama.py has a startup log line that prints the actual state:
#       "genomic_RoPE=DISABLED" when off, "ENABLED path=..." when on
#     -> CHECK THIS LINE IN THE JOB LOG after submitting, to confirm the run
#        actually went in disabled, not just that we intended it to.
#
# IMPORTANT — do NOT set wced_genomic_rank_path="" (empty string) as a way to
# "disable" it: cfg.get(...) only returns the None default when the key is
# ABSENT. An empty string is not None, so WCEDCollator would try
# np.load("") and crash. The line must be omitted entirely, which is what
# this script does (compare against Run A's invocation: the
# `wced_genomic_rank_path=...` line is simply not present below).
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
LOGDIR="${REPO}/logs_llama-wced"

# ─── Data — SAME fixed subset file Run A used (do not regenerate) ────────────
PRETRAIN_DATA="${REPO}/outputs/ablation_rope/pretrain_subset_5000.h5ad"
DATA_DIR="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_pretrain.csv"

# ─── Architecture — identical to Run A / production 6L recipe ────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-256}"
NUM_LAYERS="${NUM_LAYERS:-6}"
NUM_HEADS="${NUM_HEADS:-4}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-512}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"

# ─── WCED settings — identical to Run A ───────────────────────────────────────
SUBSET_K="${SUBSET_K:-49156}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"
AGE_WEIGHT="${AGE_WEIGHT:-0.0}"
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.05}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"
DIAG_CHECK_EVERY="${DIAG_CHECK_EVERY:-5}"

# ─── Training hyperparameters — identical to Run A ────────────────────────────
LR="${LR:-3e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-32}"
ACCUM="${ACCUM:-2}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-150}"
EARLY_STOP="${EARLY_STOP:-30}"

# ─── Resume (optional) — set to continue a crashed/timed-out run ────────────
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"

# ─── WandB — same project as Run A, distinct run name ─────────────────────────
WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="pretrain-ablation-rope"
WANDB_RUN_NAME="ropeB-nogenomic-5k-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

TOKENIZER_PATH="${TOKENIZER_PATH:-${REPO}/tokenizer}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

if [ ! -f "${PRETRAIN_DATA}" ]; then
    echo "ERROR: subset h5ad not found: ${PRETRAIN_DATA}"
    echo "Run scripts/rope_pilot/run_ablation_rope_step0_subset.sh first."
    exit 1
fi

echo "============================================================"
echo "ROPE ABLATION — RUN B (arbitrary-order positions, no genomic rank)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:         ${PRETRAIN_DATA}  (fixed 5k subset -- SAME file as Run A)"
echo "Genomic RoPE: DISABLED (wced_genomic_rank_path not passed)  <-- OFF for this run"
echo "Model:        ${NUM_LAYERS}L x ${HIDDEN_SIZE}D x ${NUM_HEADS}H, FFN=${INTERMEDIATE_SIZE}"
echo "Output:       ${OUTDIR}"
echo "W&B:          ${WANDB_PROJECT}/${WANDB_RUN_NAME}"
echo "============================================================"
echo "VERIFY after this job starts: grep 'genomic_RoPE=' on this log must show DISABLED"
echo "============================================================"

methyllama_env   # environment setup, defined in configs/paths.sh

cd "${REPO}"

export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${PRETRAIN_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k="${SUBSET_K}" \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
    data_module.bmfm_style=false \
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
    diag_check_every="${DIAG_CHECK_EVERY}" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}" \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "============================================================"
echo "RUN B (no genomic) finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
