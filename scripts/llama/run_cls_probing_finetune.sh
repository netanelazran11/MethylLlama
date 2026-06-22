#!/bin/bash -l
#SBATCH --job-name=cls-probe-ft
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
# run_cls_probing_finetune.sh
#
# Extracts CLS embeddings from the FINE-TUNED MethylLlama (v5 best checkpoint)
# on the 19k finetune dataset and runs the full probing analysis:
#   - UMAP colored by tissue / age / sex / disease
#   - Age probe: linear + MLP (R², MAE, MedAE)
#   - Classification probe: tissue / sex / disease
#   - Within-tissue age probe
#
# Compare results to run_cls_probing.sh (pretrained, 169k) to see how
# fine-tuning reshapes the representation space.
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

# ── Paths ──────────────────────────────────────────────────────────────────
FINETUNE_CKPT="${FINETUNE_CKPT:-${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt}"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/repr_analysis/cls_probing_finetune_${SLURM_JOB_ID}"
METADATA="${REPO}/data/pretrain_metadata.csv.gz"

echo "============================================================"
echo " MethylLlama — CLS Probing (Fine-tuned model, 19k)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Checkpoint : ${FINETUNE_CKPT}"
echo " Data       : ${DATA}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

if [ ! -f "${FINETUNE_CKPT}" ]; then
    echo "ERROR: checkpoint not found: ${FINETUNE_CKPT}"
    exit 1
fi

METADATA_ARG=""
if [ -f "${METADATA}" ]; then
    METADATA_ARG="--metadata ${METADATA} --metadata_id_col GSM_ID"
fi

python scripts/repr_analysis/cls_probing_analysis.py \
    --checkpoint    "${FINETUNE_CKPT}"  \
    --ckpt_type     finetune            \
    --data          "${DATA}"           \
    --tokenizer     "${TOKENIZER}"      \
    --outdir        "${OUTDIR}"         \
    --batch_size    32                  \
    --n_pca         50                  \
    --n_neighbors   15                  \
    --label_cols    tissue sex disease dataset \
    --age_col       age                 \
    --split_col     split               \
    --min_tissue_samples 30             \
    ${METADATA_ARG}

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   embeddings_cls.npy       fine-tuned CLS [N, 256]"
echo "   probing_results.csv"
echo "   figures/umap_*.png"
echo "   figures/age_scatter_*.png"
echo "   figures/probing_summary.png"
echo "   report.txt"
echo "============================================================"
