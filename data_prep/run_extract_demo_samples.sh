#!/bin/bash
#SBATCH --job-name=extract-demo
#SBATCH --gres=gpu:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=logs/extract_demo_%j.log
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
INPUT="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
OUTPUT="${REPO}/methylllama_demo_500samples.h5ad"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh

echo "=== Extracting demo samples ==="
echo "REPO:   ${REPO}"
echo "INPUT:  ${INPUT}"
echo "OUTPUT: ${OUTPUT}"
python data_prep/extract_demo_samples.py \
    --input  "${INPUT}" \
    --output "${OUTPUT}" \
    --n_samples 500 \
    --seed 42

echo "Done: $(date)"
echo ""
echo "Copy to local machine:"
echo "  rsync -av <user>@<cluster>:${OUTPUT} ~/Downloads/"
