#!/bin/bash -l
#SBATCH --job-name=age-predictions
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

CKPT="${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
EXT_META="${REPO}/data/pretrain_metadata.csv.gz"
OUTDIR="${REPO}/outputs/repr_analysis/age_predictions_${SLURM_JOB_ID}"

echo "============================================================"
echo " Extract full model age predictions"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo " Checkpoint: ${CKPT}"
echo " Output    : ${OUTDIR}"
echo "============================================================"

python scripts/repr_analysis/extract_age_predictions.py \
    --checkpoint  "${CKPT}"      \
    --data        "${DATA}"      \
    --tokenizer   "${TOKENIZER}" \
    --outdir      "${OUTDIR}"    \
    --batch_size  64             \
    --device      cuda           \
    --metadata    "${EXT_META}"  \
    --metadata_id_col GSM_ID

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Sync with:"
echo "   rsync -av netanel.azran@moriah:${OUTDIR}/age_predictions.csv \\"
echo "     /Users/netanelazran/Projects/BMFM-RNA_thesis/methyl/figures/figure4/"
echo "============================================================"
