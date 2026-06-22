#!/bin/bash -l
#SBATCH --job-name=create-valid-split
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:0
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=0:30:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/create_valid_split_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/create_valid_split_%j.err

cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
source bmfm_methyl_env/bin/activate

python scripts/utils/create_valid_split.py
