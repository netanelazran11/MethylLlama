#!/bin/bash -l
#SBATCH --job-name=pilot-subset
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=02:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.

# ─────────────────────────────────────────────────────────────────────────────
# Contrastive-weight pilot, step 1: the fixed pilot subset.
# Builds outputs/objective_pilot/pilot_data/pilot_9k_seed2026.h5ad (9,000 train / 1,000 valid / 1,000 test,
# drawn from the train / valid / held-out partitions of the reported pretraining run). Read-only on the source;
# refuses to overwrite. CPU work (loads the full corpus h5ad).
#   mkdir -p logs && sbatch objective_pilot/create_pilot_subset.sh        (from the repository root)
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail
REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

SOURCE="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
echo "Host: $(hostname) | $(date)"
python -m objective_pilot.create_pilot_subset --source "${SOURCE}" --outdir outputs/objective_pilot/pilot_data
