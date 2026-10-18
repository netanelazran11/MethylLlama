#!/bin/bash -l
#SBATCH --job-name=methylgpt-convert-kfold
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.

set -euo pipefail

# MethylGPT comparison, step 1: the downstream h5ad as MethylGPT parquet files for the five
# folds (train_fold{0..4}, valid_fold{0..4}) and the shared test set, from the same split
# files as MethylLlama (outputs/kfold_splits/, written by scripts/data/create_kfold_splits.py).
# Run once, from the repository root, before the five fine-tuning jobs.

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
cd "${REPO}"
methylgpt_env

python scripts/methylgpt/convert_h5ad_to_parquet_21k_kfold.py "$@"
