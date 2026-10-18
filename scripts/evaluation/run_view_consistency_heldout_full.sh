#!/bin/bash -l
#SBATCH --job-name=view-heldout-full
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=8:00:00
#SBATCH --array=0-6
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1
# ─────────────────────────────────────────────────────────────────────────────
# FINAL cross-view consistency for Fig. 2a/b: the whole held-out partition of the
# pretraining corpus (16,912 profiles), overlap and disjoint views, 3 seeds each, plus one
# negative control (pattern_only: measured-CpG pattern kept, values replaced by per-CpG means).
# Seven array tasks, one (condition, seed) each, about 2 h per task.
# Inference only. Writes to results/pretrained_cls/view_consistency_heldout_full/.
#
# Usage:   sbatch scripts/evaluation/run_view_consistency_heldout_full.sh
# Then:    python scripts/evaluation/view_consistency_heldout_full.py --merge
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

CONDS=(overlap overlap overlap disjoint disjoint disjoint pattern_only)
SEEDS=(0 1 2 0 1 2 0)
COND="${CONDS[$SLURM_ARRAY_TASK_ID]}"
SEED="${SEEDS[$SLURM_ARRAY_TASK_ID]}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
# released checkpoint (scripts/download_checkpoint.sh); set CKPT to use your own pretraining run
CKPT="${CKPT:-${REPO}/checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt}"
DATA="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank.npy"
OUTDIR="${REPO}/results/pretrained_cls/view_consistency_heldout_full"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[ -f "${CKPT}" ]         || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f "${DATA}" ]         || { echo "ERROR: pretrain h5ad not found: ${DATA}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }
[ -d "${TOKENIZER}" ]    || { echo "ERROR: tokenizer not found: ${TOKENIZER}"; exit 1; }

echo "============================================================"
echo "Cross-view consistency, full held-out partition"
echo "Array ${SLURM_ARRAY_JOB_ID} task ${SLURM_ARRAY_TASK_ID}: condition=${COND} seed=${SEED}"
echo "Host: $(hostname)  Time: $(date)"
echo "============================================================"

python scripts/evaluation/view_consistency_heldout_full.py \
    --checkpoint "${CKPT}" --data "${DATA}" --split test \
    --tokenizer "${TOKENIZER}" --genomic_rank "${GENOMIC_RANK}" \
    --condition "${COND}" --seed "${SEED}" --n_samples 0 \
    --outdir "${OUTDIR}"

echo "DONE task ${SLURM_ARRAY_TASK_ID}: $(date)"
