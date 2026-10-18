#!/bin/bash -l
#SBATCH --job-name=view-design-heldout
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1
# ─────────────────────────────────────────────────────────────────────────────
# Cross-view consistency on the HELD-OUT partition of the pretraining corpus
# (overlap vs disjoint views, 5 seeds). Inference only; same checkpoint and
# same analysis code as view_design_eval.py, different profiles.
# Writes to results/pretrained_cls/view_design_heldout/ -- the AltuMAge
# results in view_design/ are not touched.
#
# Usage: sbatch scripts/evaluation/run_view_design_heldout.sh
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
# released checkpoint (scripts/download_checkpoint.sh); set CKPT to use your own pretraining run
CKPT="${CKPT:-${REPO}/checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt}"
DATA="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank.npy"
OUTDIR="${REPO}/results/pretrained_cls/view_design_heldout"
N_SAMPLES="${N_SAMPLES:-2000}"
N_SEEDS="${N_SEEDS:-5}"

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
echo "View-design evaluation on HELD-OUT pretraining profiles"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "Profiles: ${N_SAMPLES}  Seeds: ${N_SEEDS}"
echo "============================================================"

python scripts/evaluation/view_design_eval_heldout.py \
    --checkpoint "${CKPT}" --data "${DATA}" --split test \
    --tokenizer "${TOKENIZER}" --genomic_rank "${GENOMIC_RANK}" \
    --n_samples "${N_SAMPLES}" --n_seeds "${N_SEEDS}" \
    --outdir "${OUTDIR}"

echo "DONE: $(date) -> ${OUTDIR}/"
