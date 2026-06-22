#!/bin/bash -l
#SBATCH --job-name=validate-ft
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:20:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/validate_ft_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/validate_ft_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo " Fine-tuning pipeline validation"
echo " Time: $(date)"
echo "============================================================"

python3 scripts/utils/validate_finetune_pipeline.py

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
