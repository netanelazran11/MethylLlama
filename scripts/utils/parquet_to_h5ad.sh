#!/bin/bash -l
#SBATCH --job-name=parquet-to-h5ad
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/parquet_to_h5ad_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/parquet_to_h5ad_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

export DATA_DIR="/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_49k"
export OUT="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"

echo "============================================================"
echo " parquet → h5ad converter"
echo " IN:  ${DATA_DIR}"
echo " OUT: ${OUT}"
echo " Time: $(date)"
echo "============================================================"

python3 scripts/utils/parquet_to_h5ad.py

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
