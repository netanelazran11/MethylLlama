#!/bin/bash -l
#SBATCH --job-name=test-dna-init
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/test_dna_init_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/test_dna_init_%j.err

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

python scripts/utils/test_dna_init.py
