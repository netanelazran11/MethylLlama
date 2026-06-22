#!/usr/bin/env bash
#SBATCH --job-name=eval-llama-ckpt
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/eval_llama_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/eval_llama_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"

# Best checkpoint from V5 21k run (job 45031621, run 3rfmxeol)
# epoch=203: best val MedAE=3.5000yr  ← PRIMARY METRIC
CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-45031621/checkpoints/epoch=203-val_medae=3.5000.ckpt}"

OUTDIR="${REPO}/outputs/baselines/eval_methylllama_v5_21k_ep203"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "MethylLlama V5 21k — Checkpoint Evaluation"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data: ${DATA}"
echo "Output: ${OUTDIR}"
echo "============================================================"

python scripts/llama/eval_llama_checkpoint.py \
    --checkpoint  "${CHECKPOINT}" \
    --h5ad        "${DATA}" \
    --tokenizer   "${TOKENIZER}" \
    --outdir      "${OUTDIR}" \
    --subset_k    49156 \
    --seed        42 \
    --batch_size  32 \
    --filter_age_outliers

echo "============================================================"
echo "Done: $(date)"
echo "Results: ${OUTDIR}/eval_summary.txt"
echo "============================================================"
