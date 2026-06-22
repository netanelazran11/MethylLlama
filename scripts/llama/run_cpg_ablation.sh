#!/bin/bash -l
#SBATCH --job-name=cpg-ablation
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_cpg_ablation.sh  —  CpG Ablation Test (Distributed Attention Proof)
#
# Tests whether MethylLlama depends on the most-attended CpG sites.
# Masks top-k attention CpGs vs the same number of random CpGs at inference.
# If top-k removal ≈ random removal → model uses all CpGs equally (distributed).
#
# Requires: cpg_attention.npy from a previous run_attention_analysis.sh job.
#
# Usage:
#   sbatch scripts/llama/run_cpg_ablation.sh
#
#   # Override attention path:
#   ATTN_NPY=/path/to/cpg_attention.npy sbatch scripts/llama/run_cpg_ablation.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

FINETUNE_CKPT="${FINETUNE_CKPT:-${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt}"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/repr_analysis/cpg_ablation_${SLURM_JOB_ID}"

# cpg_attention.npy from a previous attention_analysis run — SET THIS
# Example: outputs/repr_analysis/attention_44XXXXXX/cpg_attention.npy
ATTN_NPY="${ATTN_NPY:-}"

echo "============================================================"
echo " CpG Ablation Test — Distributed Representation Proof"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Fine-tune ckpt : ${FINETUNE_CKPT}"
echo " Attention npy  : ${ATTN_NPY:-<not set>}"
echo " Data (19k)     : ${DATA}"
echo " Outdir         : ${OUTDIR}"
echo "============================================================"

# ── Validate inputs ──────────────────────────────────────────────────────────
if [ ! -f "${FINETUNE_CKPT}" ]; then
    echo "ERROR: fine-tune checkpoint not found: ${FINETUNE_CKPT}"
    exit 1
fi

if [ ! -f "${DATA}" ]; then
    echo "ERROR: data file not found: ${DATA}"
    exit 1
fi

if [ -z "${ATTN_NPY}" ] || [ ! -f "${ATTN_NPY}" ]; then
    echo "ERROR: cpg_attention.npy not found."
    echo "  First run attention_analysis.py to generate cpg_attention.npy, then set:"
    echo "    ATTN_NPY=/path/to/cpg_attention.npy sbatch $0"
    echo ""
    echo "  Latest attention run:"
    ls -td "${REPO}/outputs/repr_analysis/attention_"*/ 2>/dev/null | head -3 || true
    exit 1
fi

echo ""
python scripts/repr_analysis/cpg_ablation.py \
    --finetune_checkpoint  "${FINETUNE_CKPT}"  \
    --attention_npy        "${ATTN_NPY}"        \
    --data                 "${DATA}"            \
    --tokenizer            "${TOKENIZER}"       \
    --outdir               "${OUTDIR}"          \
    --batch_size           32                   \
    --device               cuda                 \
    --top_k_sizes          10 100 1000          \
    --n_random_seeds       3

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   ablation_results.csv          R² / MAE / MedAE per condition"
echo "   figures/cpg_ablation.png      bar chart: top-k vs random"
echo ""
echo " KEY RESULT: if top-k drop ≈ random drop → DISTRIBUTED representation"
echo "             if top-k drop >> random drop → SPARSE (specific biomarkers)"
echo "============================================================"
