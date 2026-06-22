#!/usr/bin/env bash
#SBATCH --job-name=inspect_cpg
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/inspect_cpg_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/inspect_cpg_%j.err
#SBATCH --time=00:10:00
#SBATCH --mem=8G
#SBATCH --cpus-per-task=1
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1

set -euo pipefail
REPO=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate
python3 scripts/utils/inspect_cpg_mapping.py
