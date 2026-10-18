#!/bin/bash -l
#SBATCH --job-name=ablation-rope-subset
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Genomic RoPE ablation — Step 0: build the fixed 5,000-sample pretrain subset
# used by BOTH Run A (genomic RoPE) and Run B (no genomic RoPE). CPU-only,
# read-only on the source h5ad, writes a new file — does not touch anything
# in the existing pipeline.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
SOURCE="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

python scripts/rope_pilot/create_pretrain_subset_h5ad.py \
    --source "${SOURCE}" \
    --n_samples 5000 \
    --seed 42 \
    --outdir outputs/ablation_rope

echo "DONE: $(date)"
