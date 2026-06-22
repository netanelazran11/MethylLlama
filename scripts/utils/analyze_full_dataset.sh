#!/usr/bin/env bash
#SBATCH --job-name=analyze_full_dataset
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/analyze_full_dataset_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/analyze_full_dataset_%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=128G
#SBATCH --cpus-per-task=8
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1

set -euo pipefail
REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate

python scripts/utils/analyze_full_dataset.py \
    --outdir "$REPO/dataset_fingerprint_outputs"
