#!/bin/bash -l
#SBATCH --job-name=methylgpt-finetune
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=72:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=goldfish --gres=gpu:h200:1

set -euo pipefail

# MethylGPT comparison, step 2: fine-tune one fold from the released MethylGPT checkpoint.
#   sbatch --export=ALL,FOLD=0 scripts/methylgpt/finetune_fold.sh     (FOLD=0..4)
# The configuration is configs/train_methylgpt_21k_altumage_fold${FOLD}.yml: encoder frozen,
# age head trained, batch 1, up to 200 epochs, checkpoints ranked by validation MedAE.
# Training logs to W&B (project methylGPT_medium_21k_altumage_fold${FOLD}); log in with
# `wandb login` or export WANDB_API_KEY in your shell, never in this file.

FOLD="${FOLD:?ERROR: set FOLD=0..4}"
REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
cd "${REPO}"
methylgpt_env
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True

python scripts/methylgpt/finetuning/finetuning_age_main.py \
    --args_json scripts/methylgpt/configs/args.json \
    --train_yml scripts/methylgpt/configs/train_methylgpt_21k_altumage_fold${FOLD}.yml
