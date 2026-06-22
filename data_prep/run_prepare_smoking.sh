#!/bin/bash
#SBATCH --job-name=prepare-smoking
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=64G
#SBATCH --time=2:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/data_prep/prepare_smoking_%j.log

cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
source bmfm_methyl_env/bin/activate
python data_prep/prepare_geo_smoking.py --output_dir /sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo
