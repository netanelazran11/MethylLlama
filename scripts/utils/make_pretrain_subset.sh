#!/bin/bash -l
#SBATCH --job-name=make-smoke-subset
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/make_subset_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/make_subset_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
OUT_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_smoke"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

mkdir -p logs_llama-smoke "${OUT_DIR}"

python scripts/utils/make_pretrain_subset.py \
    --h5ad      "${H5AD}" \
    --out_dir   "${OUT_DIR}" \
    --n_samples 2000

echo "Done. Output in ${OUT_DIR}"
