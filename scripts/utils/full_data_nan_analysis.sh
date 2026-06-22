#!/usr/bin/env bash
#SBATCH --job-name=full_nan_analysis
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/full_nan_analysis_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/full_nan_analysis_%j.err
#SBATCH --time=02:00:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1

set -euo pipefail

REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"

mkdir -p logs

source bmfm_methyl_env/bin/activate

OUTDIR="$REPO/full_nan_analysis_outputs"
mkdir -p "$OUTDIR"

echo "=============================="
echo "Full NaN Analysis (49k matrix)"
echo "Started: $(date)"
echo "=============================="

CPG_CSV="/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_49k/cpg_mapping/probe_ids_type3.csv"

python scripts/utils/full_data_nan_analysis.py \
    --cpg_csv  "$CPG_CSV" \
    --outdir   "$OUTDIR"

echo ""
echo "=============================="
echo "Done: $(date)"
echo "Output → $OUTDIR"
echo "=============================="
