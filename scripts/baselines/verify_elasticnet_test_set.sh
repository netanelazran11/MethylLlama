#!/bin/bash -l
#SBATCH --job-name=verify-elasticnet-test-set
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=00:10:00

#SBATCH --output=logs/verify_elasticnet_test_set_%j.out
#SBATCH --error=logs/verify_elasticnet_test_set_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=catfish --gres=gpu:l4:1

# ─────────────────────────────────────────────────────────────────────────────
# One-off check: does elasticnet_age.py's live re-derived test split match the
# official fixed test_ids.npy exactly (same GSM IDs, not just same count)?
# CPU-only, backed-mode h5ad read (never loads the methylation matrix) -- runs
# as a tiny sbatch job because the login node's memory cap OOM-killed it twice.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
mkdir -p "${REPO}/logs"
cd "${REPO}"

methyllama_env   # environment setup, defined in configs/paths.sh

echo "============================================================"
echo "Verify ElasticNet test-set membership vs. official test_ids.npy"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"

python scripts/baselines/verify_elasticnet_test_set.py

echo "============================================================"
echo "DONE: $(date)"
echo "============================================================"
