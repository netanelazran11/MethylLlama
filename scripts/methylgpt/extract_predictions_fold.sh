#!/bin/bash -l
#SBATCH --job-name=methylgpt-extract-predictions
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=01:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

set -euo pipefail

# MethylGPT comparison, step 3: per-profile test predictions of one fold's selected checkpoint
# (lowest validation MedAE), written to outputs/bootstrap_predictions/methylgpt/.
#   sbatch --export=ALL,FOLD=0 scripts/methylgpt/extract_predictions_fold.sh     (FOLD=0..4)
# Then: python scripts/methylgpt/validate_kfold_predictions.py

FOLD="${FOLD:?ERROR: set FOLD=0..4}"
REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
cd "${REPO}"
methylgpt_env
export PYTHONUNBUFFERED=1

python -u scripts/methylgpt/extract_kfold_predictions.py --fold "${FOLD}"
