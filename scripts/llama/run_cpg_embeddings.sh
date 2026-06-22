#!/bin/bash -l
#SBATCH --job-name=cpg-emb-umap
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=2:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_cpg_embeddings.sh  —  Figure 2 equivalent
#
# Extracts the 49k CpG site embedding lookup table from the WCED pretrain
# checkpoint and visualises it as a UMAP coloured by:
#   b  — CpG island relation  (Island / Shore / Shelf / OpenSea)
#   c  — Enhancer region      (Yes / No)
#   d  — Chromosomal location (Autosome / Sex chromosome)
#
# Requires: cpg_annotations_tokenizer49k.tsv produced by align_cpg_manifest.py
#
# Usage:
#   sbatch scripts/llama/run_cpg_embeddings.sh
#
#   # Override checkpoint:
#   PRETRAIN_CKPT=/path/to/other.ckpt sbatch scripts/llama/run_cpg_embeddings.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

PRETRAIN_CKPT="${PRETRAIN_CKPT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
MANIFEST="${REPO}/outputs/cpg_manifest/cpg_annotations_tokenizer49k.tsv"
OUTDIR="${REPO}/outputs/repr_analysis/cpg_embeddings_${SLURM_JOB_ID}"

echo "============================================================"
echo " MethylLlama-Small — CpG Embedding UMAP  (Fig 2)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Pretrain ckpt : ${PRETRAIN_CKPT}"
echo " Tokenizer     : ${TOKENIZER}"
echo " Manifest      : ${MANIFEST}"
echo " Outdir        : ${OUTDIR}"
echo "============================================================"

if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}"
    exit 1
fi

MANIFEST_ARG=""
if [ -f "${MANIFEST}" ]; then
    MANIFEST_ARG="--manifest ${MANIFEST}"
else
    echo "WARNING: manifest not found at ${MANIFEST} — running without annotation colours"
    echo "  Run align_cpg_manifest.sh first to generate the manifest."
fi

echo ""
python scripts/repr_analysis/cpg_embeddings.py \
    --checkpoint    "${PRETRAIN_CKPT}"  \
    --tokenizer     "${TOKENIZER}"      \
    --outdir        "${OUTDIR}"         \
    --n_pca         50                  \
    --n_neighbors   15                  \
    --min_dist      0.1                 \
    ${MANIFEST_ARG}

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   cpg_embeddings.npy    [49156 × 256]"
echo "   cpg_umap.csv          with island/enhancer/chr annotations"
echo "   figures/umap_cpg_*.png"
echo "============================================================"
