#!/usr/bin/env bash
#SBATCH --job-name=elasticnet-age-baseline
#SBATCH --output=logs/elasticnet_%j.out
#SBATCH --error=logs/elasticnet_%j.err
#SBATCH --time=03:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --gres=gpu:1
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=catfish --gres=gpu:l4:1

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
cd "$REPO"
mkdir -p logs
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

DATA="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
DUP_CSV="${REPO}/dataset_fingerprint_outputs/duplicate_pairs.csv"
OUTDIR="${REPO}/outputs/baselines/elasticnet/elasticnetcv-${SLURM_JOB_ID:-local}"

# ── Fail loudly before burning a job allocation on a missing/moved path ──────
[ -f "${DATA}" ]    || { echo "ERROR: h5ad not found: ${DATA}"; exit 1; }
[ -f "${DUP_CSV}" ] || { echo "ERROR: duplicate_pairs_csv not found: ${DUP_CSV}"; exit 1; }

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "ElasticNet Age Baseline — sklearn ElasticNetCV, single run, standard settings"
echo "Data:            ${DATA}"
echo "Duplicate pairs: ${DUP_CSV}"
echo "Out:             ${OUTDIR}"
echo "============================================================"

python scripts/baselines/elasticnet_age.py \
    --h5ad     "${DATA}" \
    --outdir   "${OUTDIR}" \
    --duplicate_pairs_csv "${DUP_CSV}" \
    --filter_age_outliers

echo "============================================================"
echo "Done: $(date)"
echo "Results: ${OUTDIR}/elasticnet_results.json"
echo "============================================================"
