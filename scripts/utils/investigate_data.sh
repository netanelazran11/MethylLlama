#!/bin/bash -l
#SBATCH --job-name=investigate-data
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/investigate_data_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/investigate_data_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo " Dataset Investigation — AltumAge 21k Original + Clean 21k Fine-tune"
echo " Time: $(date)"
echo "============================================================"

python3 scripts/utils/investigate_data.py

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
