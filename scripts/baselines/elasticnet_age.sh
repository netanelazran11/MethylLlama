#!/usr/bin/env bash
#SBATCH --job-name=elasticnet-age-baseline
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/elasticnet_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/elasticnet_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=8
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
ALPHA="${ALPHA:-0.01}"
L1_RATIO="${L1_RATIO:-0.5}"
OUTDIR="${REPO}/outputs/baselines/elasticnet/alpha${ALPHA}_l1ratio${L1_RATIO}"

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "ElasticNet Age Baseline"
echo "alpha=${ALPHA} | l1_ratio=${L1_RATIO}"
echo "Data:  ${DATA}"
echo "Out:   ${OUTDIR}"
echo "============================================================"

python scripts/baselines/elasticnet_age.py \
    --h5ad     "${DATA}" \
    --outdir   "${OUTDIR}" \
    --alpha    "${ALPHA}" \
    --l1_ratio "${L1_RATIO}"

echo "============================================================"
echo "Done: $(date)"
echo "Results: ${OUTDIR}/elasticnet_results.json"
echo "============================================================"
