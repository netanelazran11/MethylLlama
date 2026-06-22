#!/bin/bash
#SBATCH --job-name=remerge-smoking
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/remerge_smoking_%j.log

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo"
TOKENIZER="${REPO}/tokenizer"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "=== Re-merge GSE50660 + GSE42861 with re-stratified splits ==="
python data_prep/merge_smoking_datasets.py \
    --inputs \
        "${DATA_DIR}/smoking_data.h5ad" \
        "${DATA_DIR}/gse42861_smoking.h5ad" \
    --tokenizer_path "${TOKENIZER}" \
    --output "${DATA_DIR}/smoking_combined_aligned.h5ad" \
    --restratify

echo "Done: $(date)"
