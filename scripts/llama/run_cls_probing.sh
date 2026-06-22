#!/bin/bash -l
#SBATCH --job-name=cls-probing
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=4:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-8}"

# ── Paths ──────────────────────────────────────────────────────────────────
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/repr_analysis/cls_probing_${SLURM_JOB_ID}"

# Pre-computed embeddings — 169k pretrain run
CLS_NPY="${CLS_NPY:-${REPO}/outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_cls.npy}"
MEAN_NPY="${MEAN_NPY:-${REPO}/outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_mean.npy}"
RAND_NPY="${RAND_NPY:-}"  # no random embeddings for 169k run

# External metadata for tissue/sex/disease labels
METADATA="${REPO}/data/pretrain_metadata.csv.gz"

echo "============================================================"
echo " MethylLlama — CLS Probing Analysis"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Checkpoint : ${PRETRAIN_CKPT}"
echo " Data       : ${DATA}"
echo " Outdir     : ${OUTDIR}"
echo " CLS npy    : ${CLS_NPY:-<will extract>}"
echo "============================================================"

# Build optional args for pre-computed embeddings
PRECOMP_ARGS=""
if [ -n "${CLS_NPY}" ] && [ -f "${CLS_NPY}" ] && [ -n "${MEAN_NPY}" ] && [ -f "${MEAN_NPY}" ]; then
    PRECOMP_ARGS="--cls_embeddings ${CLS_NPY} --mean_embeddings ${MEAN_NPY}"
    echo "  Using pre-computed CLS + mean embeddings"
    if [ -n "${RAND_NPY}" ] && [ -f "${RAND_NPY}" ]; then
        PRECOMP_ARGS="${PRECOMP_ARGS} --random_embeddings ${RAND_NPY}"
        echo "  Using pre-computed random embeddings"
    fi
fi

METADATA_ARG=""
if [ -f "${METADATA}" ]; then
    METADATA_ARG="--metadata ${METADATA} --metadata_id_col GSM_ID"
    echo "  Using metadata: ${METADATA}"
else
    echo "  WARNING: metadata not found at ${METADATA} — tissue/sex/disease will be skipped"
fi

python scripts/repr_analysis/cls_probing_analysis.py \
    --checkpoint    "${PRETRAIN_CKPT}" \
    --ckpt_type     pretrain           \
    --data          "${DATA}"          \
    --tokenizer     "${TOKENIZER}"     \
    --outdir        "${OUTDIR}"        \
    --batch_size    32                 \
    --n_pca         50                 \
    --n_neighbors   15                 \
    --label_cols    tissue sex disease dataset \
    --age_col       age                \
    --split_col     split              \
    --min_tissue_samples 50            \
    --compare_random                   \
    ${PRECOMP_ARGS}                    \
    ${METADATA_ARG}

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   embeddings_cls.npy       pretrained CLS [N, 256]"
echo "   embeddings_random_cls.npy random-init CLS [N, 256]"
echo "   probing_results.csv      all probe scores"
echo "   figures/umap_*.png       UMAP panels (tissue/sex/disease/age)"
echo "   figures/age_scatter_*.png true vs predicted age"
echo "   figures/probing_summary.png R² / MAE comparison"
echo "   report.txt               full text report"
echo "============================================================"
