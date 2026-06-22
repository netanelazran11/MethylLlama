#!/bin/bash -l
#SBATCH --job-name=inspect-ft-data
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=16G
#SBATCH --time=0:10:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/inspect_ft_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/inspect_ft_%j.err

cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
source bmfm_methyl_env/bin/activate
python3 scripts/utils/inspect_finetune_data.py
