#!/bin/bash -l
#SBATCH --job-name=pilot-infonce
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --array=0-3
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript runs (authors' cluster): --partition=salmon --gres=gpu:l40s:1

set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# Contrastive-weight pilot, step 2: pretraining, one array task per arm (Supplementary Note 2).
#
#   task  λ      seed
#   0     0      42         reconstruction only (two views)
#   1     0.01   42
#   2     0.05   42         the weight of the reported model
#   3     0.2    42
#
# Every arm runs WCEDLlamaAblationModule (objective_pilot/wced_llama_recon_only.py): two-view reconstruction
# + λ · InfoNCE; at λ > 0 identical to the reported objective (objective_pilot/test_recon_only.py, check 7).
# Same architecture, genomic-rank RoPE, optimiser, per-GPU batch 32 (31 InfoNCE negatives, as in the reported run),
# data subset, views and data order. One 48 GB GPU per arm with gradient checkpointing (ABLATION_GRAD_CKPT=1),
# which gives identical loss and gradients (test_recon_only.py, check 9); all arms must run on the same GPU type.
# Pilot budget (not the reported one): effective batch 64 (32 × accumulation 2), 20 epochs over 9,000 profiles
# (about 2,800 optimiser steps), 200 warm-up steps, cosine over the 20 epochs. About 35 h per arm on an L40S.
#
#   mkdir -p logs && sbatch objective_pilot/run_objective_pilot.sh          (from the repository root)
#   SMOKE=1 sbatch --time=01:00:00 objective_pilot/run_objective_pilot.sh   # all arms, a few batches
#   RESUME_CHECKPOINT=<armdir>/checkpoints/last.ckpt sbatch --array=<k> objective_pilot/run_objective_pilot.sh
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS=(0.0 0.01 0.05 0.2)
SEEDS=(42 42 42 42)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
LAM="${LAMBDAS[$TASK]}"
SEED="${SEEDS[$TASK]}"
SMOKE="${SMOKE:-0}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
PILOT_DATA="${PILOT_DATA:-${REPO}/outputs/objective_pilot/pilot_data/pilot_9k_seed2026.h5ad}"
PROBE_IDS_CSV="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"
GENOMIC_RANK_NPY="${REPO}/resources/cpg_genomic_rank.npy"
TOKENIZER_PATH="${REPO}/tokenizer"

# Pilot budget
BATCH_SIZE=32
ACCUM=2
PRETRAIN_EPOCHS=20
WARMUP_STEPS=200
export ANALYSIS_EPOCHS="${ANALYSIS_EPOCHS:-0,1,4,9,14,19}"   # epoch 19 = the checkpoint evaluated
export STOP_AFTER_EPOCH="-1"
export ABLATION_MODULE=sweep
export ABLATION_GRAD_CKPT="${ABLATION_GRAD_CKPT:-1}"
RESUME_CHECKPOINT="${RESUME_CHECKPOINT:-}"
LIMITS=()
WANDB_ON=true
ROOT="${REPO}/outputs/objective_pilot/pilot"

if [ "${SMOKE}" = "1" ]; then
    BATCH_SIZE="${SMOKE_BATCH:-4}"
    export ANALYSIS_EPOCHS="0"
    export STOP_AFTER_EPOCH="1"
    export LIMIT_TEST_BATCHES=5
    LIMITS=(limit_train_batches=20 limit_val_batches=5)
    WANDB_ON=false
    ROOT="${REPO}/outputs/objective_pilot/pilot_smoke"
fi

ARM="lam${LAM}_seed${SEED}"
OUTDIR="${ROOT}/${ARM}"
mkdir -p "${OUTDIR}"
[ -f "${PILOT_DATA}" ]       || { echo "ERROR: pilot subset not found: ${PILOT_DATA} (run create_pilot_subset.sh)"; exit 1; }
[ -f "${GENOMIC_RANK_NPY}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK_NPY}"; exit 1; }
if [ -n "${RESUME_CHECKPOINT}" ] && [ ! -f "${RESUME_CHECKPOINT}" ]; then
    echo "ERROR: resume checkpoint not found: ${RESUME_CHECKPOINT}"; exit 1
fi

echo "============================================================"
echo "INFONCE PILOT  task=${TASK}  arm=${ARM}  λ=${LAM}  seed=${SEED}  SMOKE=${SMOKE}  batch=${BATCH_SIZE}  grad_ckpt=${ABLATION_GRAD_CKPT}"
echo "Host: $(hostname) | $(date)"
echo "Data: ${PILOT_DATA}"
echo "Output: ${OUTDIR}"
echo "============================================================"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
python -c "import torch; assert torch.cuda.is_available(), 'CUDA not available'; print('GPU:', torch.cuda.get_device_name(0), '| torch', torch.__version__)"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK:-16}

python -m objective_pilot.test_recon_only

python -m objective_pilot.pretrain_recon_only \
    data_path="${PILOT_DATA}" \
    probe_ids_csv="${PROBE_IDS_CSV}" \
    tokenizer_path="${TOKENIZER_PATH}" \
    output_directory="${OUTDIR}" \
    pretraining_mode=wced \
    data_module.subset_k=49156 \
    data_module.batch_size="${BATCH_SIZE}" \
    data_module.num_workers=14 \
    data_module.bmfm_style=false \
    model.hidden_size=256 \
    model.num_hidden_layers=6 \
    model.num_attention_heads=4 \
    model.intermediate_size=512 \
    model.rope_theta=10000.0 \
    model.n_sin_basis=48 \
    model.basis_scale=2.0 \
    trainer.learning_rate=3e-4 \
    trainer.weight_decay=0.01 \
    trainer.warmup_steps="${WARMUP_STEPS}" \
    wced_input_ratio=0.5 \
    wced_age_weight=0.0 \
    wced_contrastive=true \
    wced_contrastive_weight="${LAM}" \
    wced_contrastive_temp=0.1 \
    wced_normalize_loss=true \
    wced_decoder_dropout=0.1 \
    wced_genomic_rank_path="${GENOMIC_RANK_NPY}" \
    pretrain_epochs="${PRETRAIN_EPOCHS}" \
    accumulate_grad_batches="${ACCUM}" \
    early_stop_patience=100 \
    gradient_clip_val=1.0 \
    precision="16-mixed" \
    diag_check_every=5 \
    seed.seed_value="${SEED}" \
    track_wandb.enabled="${WANDB_ON}" \
    track_wandb.project="pretrain-llama-infonce-pilot" \
    track_wandb.entity="${WANDB_ENTITY:-}" \
    track_wandb.name="pilot-${ARM}-${SLURM_ARRAY_JOB_ID:-local}" \
    ${LIMITS[@]+"${LIMITS[@]}"} \
    ${RESUME_CHECKPOINT:+"resume_checkpoint='${RESUME_CHECKPOINT}'"}

echo "Arm ${ARM} finished: $(date) — checkpoints in ${OUTDIR}/checkpoints/analysis/"
