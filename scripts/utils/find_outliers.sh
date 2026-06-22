#!/usr/bin/env bash
#SBATCH --job-name=find_outliers
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/find_outliers_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/find_outliers_%j.err
#SBATCH --time=00:20:00
#SBATCH --mem=32G
#SBATCH --cpus-per-task=2
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate

python scripts/utils/find_outliers.py \
    --outdir "$REPO/dataset_fingerprint_outputs"
