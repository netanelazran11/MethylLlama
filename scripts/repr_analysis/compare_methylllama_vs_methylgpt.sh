#!/usr/bin/env bash
#SBATCH --job-name=compare-llama-gpt
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/compare_llama_gpt_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs/compare_llama_gpt_%j.err
#SBATCH --time=00:30:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=2
#SBATCH --partition=catfish
#SBATCH --gres=gpu:l4:1

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "$REPO"
mkdir -p logs
source bmfm_methyl_env/bin/activate

OUTDIR="${REPO}/outputs/comparisons/methylllama_vs_methylgpt"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "MethylLlama (WCED) vs MethylGPT — Comparison Figures"
echo "Run 1: 3rfmxeol (MethylLlama)"
echo "Run 2: xzrw1qwr (MethylGPT)"
echo "Output: ${OUTDIR}"
echo "============================================================"

python scripts/repr_analysis/thesis_finetune_comparison_figures.py \
    --run1 "https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/finetune-llama-small/runs/3rfmxeol" \
    --run2 "https://wandb.ai/netanelazran11-hebrew-university-of-jerusalem/methylGPT_medium_21k_altumage/runs/xzrw1qwr" \
    --label1 "MethylLlama (WCED)" \
    --label2 "MethylGPT" \
    --outdir "${OUTDIR}"

echo "============================================================"
echo "Done: $(date)"
echo "Figures saved to: ${OUTDIR}"
echo "============================================================"
