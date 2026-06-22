#!/bin/bash -l
#SBATCH --job-name=download-bmfdna
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/download-bmfdna_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-domain/download-bmfdna_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate

export HF_HOME="/sci/labs/benjamin.yakir/netanel.azran/data/hf_cache"

echo "Downloading BMFM-DNA last.ckpt: $(date)"

python -c "
from huggingface_hub import hf_hub_download
p = hf_hub_download(
    'ibm-research/biomed.dna.ref.modernbert.113m.v1',
    'last.ckpt',
    cache_dir='/sci/labs/benjamin.yakir/netanel.azran/data/hf_cache',
)
print('Downloaded to:', p)
"

echo "Done: $(date)"
