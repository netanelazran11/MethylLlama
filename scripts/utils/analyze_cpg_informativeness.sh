#!/bin/bash -l
#SBATCH --job-name=analyze-cpg-info
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/analyze_cpg_informativeness_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/analyze_cpg_informativeness_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

echo "============================================================"
echo " CpG Informativeness Analysis"
echo " Time: $(date)"
echo "============================================================"

python scripts/utils/analyze_cpg_informativeness.py

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
