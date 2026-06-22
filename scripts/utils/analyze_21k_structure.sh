#!/usr/bin/env bash
#SBATCH --job-name=analyze_21k
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/analyze_21k_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/analyze_21k_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=64G
#SBATCH --cpus-per-task=4
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate

python scripts/utils/analyze_21k_structure.py \
    --outdir "$REPO/dataset_fingerprint_outputs"
