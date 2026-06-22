#!/bin/bash
#SBATCH --job-name=extract-demo
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/extract_demo_%j.log

# Detect repo root from the script's own location so this works from any clone
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO="$(dirname "${SCRIPT_DIR}")"
INPUT="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
OUTPUT="${REPO}/methylllama_demo_120samples.h5ad"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "=== Extracting demo samples ==="
echo "REPO:   ${REPO}"
echo "INPUT:  ${INPUT}"
echo "OUTPUT: ${OUTPUT}"
python data_prep/extract_demo_samples.py \
    --input  "${INPUT}" \
    --output "${OUTPUT}" \
    --n_samples 120 \
    --seed 42

echo "Done: $(date)"
echo ""
echo "Copy to local machine:"
echo "  rsync -av netanel.azran@moriah:${OUTPUT} ~/Downloads/"
