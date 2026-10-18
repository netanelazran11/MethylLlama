#!/bin/bash -l
#SBATCH --job-name=methyllama-random-init-testeval
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Standalone test-eval for the MethylLlama random-init baseline (Experiment A1 re-run),
# which hit the 48h wall (job 45683236, crashed at epoch 187, best
# val_medae=3.804) before its own auto test-eval step could run.
#
# Finds the best val_medae checkpoint under that run's output dir and
# evaluates it on the same fixed test set used by the k-fold benchmark.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
DATA="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
KFOLD_DIR="${REPO}/outputs/kfold_splits"
GENOMIC_RANK_FT_NPY="${REPO}/resources/cpg_genomic_rank_finetune.npy"
TOKENIZER_PATH="${REPO}/tokenizer"
OUTROOT="${REPO}/outputs/finetune-llama-small"

FOLD="${FOLD:-0}"
FOLD_TRAIN_NPY="${KFOLD_DIR}/fold_${FOLD}_train.npy"
FOLD_VAL_NPY="${KFOLD_DIR}/fold_${FOLD}_val.npy"

WANDB_ENTITY="${WANDB_ENTITY:-}"
WANDB_PROJECT="finetune-llama-small"

VOCAB_SIZE=49161
HIDDEN_SIZE=256
NUM_LAYERS=6
INTERMEDIATE_SIZE=512
NUM_HEADS=4
N_SIN_BASIS=48

methyllama_env   # environment setup, defined in configs/paths.sh
cd "${REPO}"
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# Pick the run dir that actually has val_medae checkpoints (newest if several)
OUTDIR=""
for CAND in $(find "${OUTROOT}" -maxdepth 1 -type d -name "methyllama-random-init-fold${FOLD}-ep*" 2>/dev/null | sort); do
    if find "${CAND}/checkpoints" -name "epoch=*-val_medae=*.ckpt" 2>/dev/null | grep -q .; then
        OUTDIR="${CAND}"
    fi
done
echo ">> OUTDIR='${OUTDIR}'"
[ -n "${OUTDIR}" ] || { echo "ERROR: no random-init run dir with val_medae checkpoints found"; exit 1; }

BEST_CKPT=$(find "${OUTDIR}/checkpoints" -name "epoch=*-val_medae=*.ckpt" 2>/dev/null \
    | sed -E 's/.*val_medae=([0-9.]+)\.ckpt/\1 &/' | sort -n | head -1 | cut -d' ' -f2-)
[ -n "${BEST_CKPT}" ] || { echo "ERROR: no val_medae checkpoint found"; exit 1; }
echo ">> BEST_CKPT='${BEST_CKPT}'"

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
    "+eval_checkpoint='${BEST_CKPT}'" \
    data_module.subset_k=49156 \
    data_module.fixed_subset_seed=42 \
    data_module.max_length=21369 \
    data_module.batch_size=32 \
    data_module.num_workers=8 \
    data_module.filter_age_outliers=true \
    "data_module.duplicate_pairs_csv='${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv'" \
    "+data_module.fold_train_ids_npy='${FOLD_TRAIN_NPY}'" \
    "+data_module.fold_val_ids_npy='${FOLD_VAL_NPY}'" \
    wced_input_ratio=1.0 \
    wced_genomic_rank_path="${GENOMIC_RANK_FT_NPY}" \
    finetune.head_hidden_size=128 \
    finetune.head_dropout=0.1 \
    finetune.pooling=cls \
    finetune.loss_type=huber \
    precision="16-mixed" \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="methyllama-random-init-fold${FOLD}-testeval"

echo "Test eval done."
