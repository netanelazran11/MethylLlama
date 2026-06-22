#!/bin/bash -l
#SBATCH --job-name=robustness-missing-data
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
# run_robustness.sh  —  Figure 4f equivalent
#
# Evaluates age prediction robustness under increasing levels of missing CpG data.
# Systematically masks 0–90% of CpG inputs at inference time, measures MedAE,
# and compares MethylLlama against ElasticNet and Ridge baselines.
#
# Expected result:
#   MethylLlama degrades gradually (uses redundant signals across many CpGs)
#   ElasticNet degrades sharply    (relies on specific sites)
#
# Usage:
#   sbatch scripts/llama/run_robustness.sh
#
#   # Override fine-tune checkpoint:
#   FINETUNE_CKPT=/path/to/other.ckpt sbatch scripts/llama/run_robustness.sh
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
# Paths — update FINETUNE_CKPT to your best fine-tuned checkpoint
# ─────────────────────────────────────────────────────────────────────────────

FINETUNE_CKPT="${FINETUNE_CKPT:-${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt}"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/repr_analysis/robustness_${SLURM_JOB_ID}"

echo "============================================================"
echo " Age Prediction Robustness to Missing Data  (Fig 4f)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Fine-tune ckpt : ${FINETUNE_CKPT}"
echo " Data (19k)     : ${DATA}"
echo " Tokenizer      : ${TOKENIZER}"
echo " Outdir         : ${OUTDIR}"
echo "============================================================"

if [ -z "${FINETUNE_CKPT}" ] || [ ! -f "${FINETUNE_CKPT}" ]; then
    echo "ERROR: fine-tune checkpoint not found."
    echo "  Set FINETUNE_CKPT env var to the path of your best .ckpt file."
    echo "  Example:"
    echo "    FINETUNE_CKPT=/path/to/epoch=117-val_mae=6.3071.ckpt sbatch $0"
    exit 1
fi

if [ ! -f "${DATA}" ]; then
    echo "ERROR: finetune h5ad not found: ${DATA}"
    exit 1
fi

echo ""
python scripts/repr_analysis/robustness.py \
    --checkpoint    "${FINETUNE_CKPT}"  \
    --data          "${DATA}"           \
    --tokenizer     "${TOKENIZER}"      \
    --outdir        "${OUTDIR}"         \
    --batch_size    32                  \
    --device        cuda                \
    --mask_levels   0 10 20 30 40 50 60 70 80 90 \
    --age_col       age                 \
    --split_col     split

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   robustness_results.csv"
echo "   figures/robustness_missing_data.png"
echo "============================================================"
