#!/bin/bash -l
#SBATCH --job-name=remove-outliers
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=0:20:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/remove_outliers_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/remove_outliers_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "============================================================"
echo " Remove Outlier Samples"
echo " Time: $(date)"
echo "============================================================"

python scripts/utils/remove_outliers.py

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
