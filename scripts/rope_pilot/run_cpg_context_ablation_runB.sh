#!/bin/bash -l
#SBATCH --job-name=cpg-context-ablationB
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=4:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Genomic RoPE ablation — contextualized CpG extraction for RUN B (no genomic).
#
# feed_position_ids=false: --genomic_rank is still used to fix a consistent
# CpG order across samples (needed so "sequence position p = same CpG for
# every sample" holds, required for correct averaging), but the resulting
# position_ids are discarded before calling the encoder. Run B never saw
# genomic-rank position_ids during training (position_ids=None -> sequential
# fallback) -- feeding it real genomic ranks now would hand it
# out-of-distribution position values it never learned to interpret, which
# would invalidate the comparison. See extract_contextual_cpg.py's
# --feed_position_ids docstring for the full reasoning.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
CKPT="${REPO}/outputs/pretrain-ablation-rope/ropeB-nogenomic-5k-45763963/checkpoints/epoch=72-recon=0.1585-pcc=0.9017.ckpt"
DATA="${REPO}/outputs/ablation_rope/pretrain_subset_5000.h5ad"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank.npy"
OUTDIR="${REPO}/outputs/ablation_rope/cpg_context_B"

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
    --feed_position_ids false \
    --max_samples 5000 \
    --batch_size 16 \
    --outdir "${OUTDIR}"

echo "DONE: $(date) -> ${OUTDIR}/"
