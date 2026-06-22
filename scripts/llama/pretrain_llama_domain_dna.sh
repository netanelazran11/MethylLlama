#!/bin/bash -l
#SBATCH --job-name=pretrain-llama-domain-dna
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=100G
#SBATCH --time=48:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Identical to pretrain_llama_domain.sh except:
#   - Passes cpg_embeddings_npy + cpg_embeddings_ids to pretrain_llama.py
#   - MethylLlamaModel's CpG embedding table is initialized with BMFM-DNA vectors
#   - WandB project name distinguishes this run from the random-init baseline
# ─────────────────────────────────────────────────────────────────────────────
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-domain"

DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad"
DATA="${DATA_DIR}/altumage_21k_3way.h5ad"
PROBE_IDS_CSV="${DATA_DIR}/probe_ids_type3_21k.csv"

# BMFM-DNA embeddings
CPG_EMB_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/cpg_embeddings"
CPG_EMB_NPY="${CPG_EMB_DIR}/cpg_embeddings_bmfdna_21k.npy"
CPG_EMB_IDS="${CPG_EMB_DIR}/cpg_ids_order.txt"

# ─────────────────────────────────────────────────────────────────────────────
# Architecture — identical to baseline for fair comparison
# ─────────────────────────────────────────────────────────────────────────────
HIDDEN_SIZE="${HIDDEN_SIZE:-768}"
NUM_LAYERS="${NUM_LAYERS:-8}"
NUM_HEADS="${NUM_HEADS:-12}"
INTERMEDIATE_SIZE="${INTERMEDIATE_SIZE:-2048}"
ROPE_THETA="${ROPE_THETA:-10000.0}"
N_SIN_BASIS="${N_SIN_BASIS:-48}"
BASIS_SCALE="${BASIS_SCALE:-2.0}"

# ─────────────────────────────────────────────────────────────────────────────
# WCED settings — identical to baseline
# ─────────────────────────────────────────────────────────────────────────────
SUBSET_K="${SUBSET_K:-21368}"
INPUT_RATIO="${INPUT_RATIO:-0.5}"
AGE_WEIGHT="${AGE_WEIGHT:-0.5}"
CONTRASTIVE="${CONTRASTIVE:-true}"
CONTRASTIVE_WEIGHT="${CONTRASTIVE_WEIGHT:-0.1}"
CONTRASTIVE_TEMP="${CONTRASTIVE_TEMP:-0.1}"
NORMALIZE_LOSS="${NORMALIZE_LOSS:-true}"
DECODER_DROPOUT="${DECODER_DROPOUT:-0.1}"

# ─────────────────────────────────────────────────────────────────────────────
# Training hyperparameters — identical to baseline
# ─────────────────────────────────────────────────────────────────────────────
LR="${LR:-5e-4}"
WEIGHT_DECAY="${WEIGHT_DECAY:-0.01}"
WARMUP_STEPS="${WARMUP_STEPS:-500}"
BATCH_SIZE="${BATCH_SIZE:-16}"
ACCUM="${ACCUM:-8}"
PRETRAIN_EPOCHS="${PRETRAIN_EPOCHS:-300}"
EARLY_STOP="${EARLY_STOP:-60}"

# ─────────────────────────────────────────────────────────────────────────────
# WandB
# ─────────────────────────────────────────────────────────────────────────────
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-llama-domain-dna"
WANDB_RUN_NAME="llama-domain-dna-k${SUBSET_K}-r${INPUT_RATIO}-${SLURM_JOB_ID}"

OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

# Reuse the same tokenizer as the baseline (same CpGs — must match)
TOKENIZER_PATH="${REPO}/tokenizer_llama_domain21k"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "METHYLLAMA DOMAIN PRETRAINING — BMFM-DNA INIT"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data:    ${DATA}"
echo "CpGs:    ${SUBSET_K}, input_ratio=${INPUT_RATIO}"
echo "Model:   ${NUM_LAYERS}L × ${HIDDEN_SIZE}D × ${NUM_HEADS}H"
echo "DNA emb: ${CPG_EMB_NPY}"
echo "Train:   lr=${LR}, batch=${BATCH_SIZE}×2GPUs×${ACCUM}accum=$(( BATCH_SIZE * 2 * ACCUM )) eff"
echo "Tokenizer: ${TOKENIZER_PATH}"
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
# Pretraining with BMFM-DNA CpG embedding initialization
# ─────────────────────────────────────────────────────────────────────────────
python -m bmfm_methylation.llama.pretrain_llama \
    data_path="${DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    +cpg_embeddings_npy="${CPG_EMB_NPY}" \
    +cpg_embeddings_ids="${CPG_EMB_IDS}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset=false \
    data_module.max_length=$(python3 -c "import math; print(int(${SUBSET_K} * ${INPUT_RATIO}) + 1)") \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=8 \
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
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Domain pretraining (DNA init) finished: $(date)"
echo "Checkpoint: ${OUTDIR}/checkpoints/"
echo "============================================================"
