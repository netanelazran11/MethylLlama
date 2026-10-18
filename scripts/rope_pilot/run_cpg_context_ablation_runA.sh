#!/bin/bash -l
#SBATCH --job-name=cpg-context-ablationA
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Genomic RoPE ablation — contextualized CpG extraction for RUN A (genomic).
# feed_position_ids=true (default): matches how Run A was actually trained.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
CKPT="${REPO}/outputs/pretrain-ablation-rope/ropeA-genomic-5k-45740643/checkpoints/epoch=60-recon=0.0906-pcc=0.9357.ckpt"
DATA="${REPO}/outputs/ablation_rope/pretrain_subset_5000.h5ad"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank.npy"
OUTDIR="${REPO}/outputs/ablation_rope/cpg_context_A"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

echo "Checkpoint: ${CKPT}"
[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found"; exit 1; }

python scripts/rope_pilot/extract_contextual_cpg.py \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --feed_position_ids true \
    --max_samples 5000 \
    --batch_size 16 \
    --outdir "${OUTDIR}"

echo "DONE: $(date) -> ${OUTDIR}/"
