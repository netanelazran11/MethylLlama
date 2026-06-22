#!/bin/bash
#SBATCH --job-name=merge-multitask
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/data_prep/merge_multitask_%j.log

cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
source bmfm_methyl_env/bin/activate

AGE_H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"
SMOKING_H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_data.h5ad"
TOKENIZER="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/tokenizer"
OUTPUT_MERGED="/sci/labs/benjamin.yakir/netanel.azran/data/merged/multitask_data.h5ad"
OUTPUT_ALIGNED="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_data_aligned.h5ad"

python data_prep/merge_multitask_data.py \
    --age_h5ad "${AGE_H5AD}" \
    --smoking_h5ad "${SMOKING_H5AD}" \
    --tokenizer_path "${TOKENIZER}" \
    --output_merged "${OUTPUT_MERGED}" \
    --output_aligned "${OUTPUT_ALIGNED}"
