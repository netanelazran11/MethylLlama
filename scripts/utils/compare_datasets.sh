#!/usr/bin/env bash
#SBATCH --job-name=compare_datasets
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/compare_datasets_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/compare_datasets_%j.err
#SBATCH --time=01:00:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=4
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"

mkdir -p logs

source bmfm_methyl_env/bin/activate

OUTDIR="$REPO/dataset_comparison_outputs"
mkdir -p "$OUTDIR"

echo "=============================="
echo "Dataset Comparison"
echo "Started: $(date)"
echo "=============================="

python scripts/utils/compare_datasets.py \
    --outdir "$OUTDIR"

echo ""
echo "=============================="
echo "Done: $(date)"
echo "Output → $OUTDIR"
echo "=============================="
