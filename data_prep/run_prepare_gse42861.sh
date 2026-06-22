#!/bin/bash
#SBATCH --job-name=prepare-gse42861
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=128G
#SBATCH --time=4:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/data_prep/gse42861_%j.log

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo"
TOKENIZER="${REPO}/tokenizer"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "=== Step 1: Download GSE42861 files ==="
mkdir -p "${DATA_DIR}"

if [ ! -f "${DATA_DIR}/GSE42861_series_matrix.txt.gz" ]; then
    wget -P "${DATA_DIR}" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42861/matrix/GSE42861_series_matrix.txt.gz"
fi

if [ ! -f "${DATA_DIR}/GSE42861_processed_methylation_matrix.txt.gz" ]; then
    wget -P "${DATA_DIR}" \
        "https://ftp.ncbi.nlm.nih.gov/geo/series/GSE42nnn/GSE42861/suppl/GSE42861_processed_methylation_matrix.txt.gz"
fi

echo "=== Step 2: Build GSE42861 h5ad ==="
python data_prep/prepare_gse42861.py --output_dir "${DATA_DIR}"

echo "=== Step 3: Merge GSE50660 + GSE42861 into combined aligned h5ad ==="
python data_prep/merge_smoking_datasets.py \
    --inputs \
        "${DATA_DIR}/smoking_data.h5ad" \
        "${DATA_DIR}/gse42861_smoking.h5ad" \
    --tokenizer_path "${TOKENIZER}" \
    --output "${DATA_DIR}/smoking_combined_aligned.h5ad"

echo "Done: $(date)"
