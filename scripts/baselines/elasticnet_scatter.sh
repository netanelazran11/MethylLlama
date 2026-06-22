#!/usr/bin/env bash
#SBATCH --job-name=elasticnet-scatter
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/elasticnet_scatter_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/elasticnet_scatter_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
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
OUTDIR="${REPO}/outputs/baselines/elasticnet/alpha${ALPHA}_l1ratio${L1_RATIO}/figures"

mkdir -p "${OUTDIR}"

echo "============================================================"
echo "ElasticNet Scatter + Error Analysis"
echo "alpha=${ALPHA} | l1_ratio=${L1_RATIO}"
echo "Figures: ${OUTDIR}"
echo "============================================================"

python scripts/baselines/elasticnet_scatter.py \
    --h5ad     "${DATA}" \
    --outdir   "${OUTDIR}" \
    --alpha    "${ALPHA}" \
    --l1_ratio "${L1_RATIO}"

echo "============================================================"
echo "Done: $(date)"
echo "Figures saved to: ${OUTDIR}"
echo "============================================================"
