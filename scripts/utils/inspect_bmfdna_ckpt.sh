#!/bin/bash -l
#SBATCH --job-name=inspect-ckpt
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --mem=32G
#SBATCH --time=0:10:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/inspect_ckpt_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/inspect_ckpt_%j.err

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

python scripts/utils/inspect_bmfdna_ckpt.py
