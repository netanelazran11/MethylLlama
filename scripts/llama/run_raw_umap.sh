#!/bin/bash -l
#SBATCH --job-name=raw-methylation-umap
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=3:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_raw_umap.sh  —  Figure 3 (d-f) comparison baseline
#
# Computes PCA + UMAP directly on the RAW methylation data matrix (no model).
# Produces the "before" half of MethylGPT Fig 3:
#   d  — raw UMAP coloured by tissue   (less distinct than model UMAP)
#   e  — raw UMAP coloured by dataset  (stronger batch effects visible)
#   f  — raw UMAP coloured by sex      (weaker separation than model)
#
# No GPU required — pure sklearn/scanpy PCA + UMAP on 30k subsampled cells.
# 64G RAM needed to hold the subsampled 30k × 49k float32 matrix (~5.6 GB).
#
# Usage:
#   sbatch scripts/llama/run_raw_umap.sh
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

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
METADATA="${REPO}/data/pretrain_metadata.csv.gz"
OUTDIR="${REPO}/outputs/repr_analysis/raw_umap_${SLURM_JOB_ID}"

N_SAMPLES="${N_SAMPLES:-30000}"
N_PCA="${N_PCA:-50}"
N_NEIGHBORS="${N_NEIGHBORS:-15}"

echo "============================================================"
echo " Raw Methylation UMAP  (Fig 3 comparison baseline)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Data (169k)   : ${DATA}"
echo " Metadata      : ${METADATA}"
echo " Outdir        : ${OUTDIR}"
echo " n_samples     : ${N_SAMPLES}  n_pca: ${N_PCA}  n_neighbors: ${N_NEIGHBORS}"
echo "============================================================"

if [ ! -f "${DATA}" ]; then
    echo "ERROR: pretrain h5ad not found: ${DATA}"
    exit 1
fi

METADATA_ARG=""
if [ -f "${METADATA}" ]; then
    METADATA_ARG="--metadata ${METADATA} --metadata_id_col GSM_ID"
else
    echo "WARNING: metadata not found at ${METADATA} — running without labels"
fi

echo ""
python scripts/repr_analysis/raw_umap.py \
    --data          "${DATA}"       \
    --outdir        "${OUTDIR}"     \
    --n_samples     "${N_SAMPLES}"  \
    --n_pca         "${N_PCA}"      \
    --n_neighbors   "${N_NEIGHBORS}" \
    --label_cols    tissue sex dataset \
    ${METADATA_ARG}

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   raw_umap_coords.npy   [${N_SAMPLES} × 2]"
echo "   raw_umap.csv          with tissue/sex/dataset labels"
echo "   figures/umap_raw_*.png"
echo "============================================================"
