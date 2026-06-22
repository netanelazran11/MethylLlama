#!/bin/bash -l
#SBATCH --job-name=extract-pretrain-cls
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=30:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/extract_pretrain_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/extract_pretrain_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# extract_pretrain_embeddings.sh
#
# Extracts CLS bottleneck (pooler_output) from the WCED pretrain checkpoint
# for ALL ~169k samples in the pretrain h5ad, then runs full representation
# analysis (UMAP × tissue/sex/disease/dataset, clustering metrics, age probe).
#
# Key difference from extract_sample_embeddings.sh:
#   • Uses the PRETRAIN h5ad (169k samples × 49k CpGs) — not the finetune 19k
#   • Joins sample metadata from data/pretrain_metadata.csv.gz (compiled from
#     MethylGPT DSV files: tissue, sex, age, disease, dataset per GSM_ID)
#   • 64G RAM + 4h time limit for 169k samples
#
# Usage:
#   sbatch scripts/llama/extract_pretrain_embeddings.sh
#
#   # Override checkpoint:
#   PRETRAIN_CKPT=/path/to/other.ckpt sbatch scripts/llama/extract_pretrain_embeddings.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

# WCED pretrain checkpoint — epoch=98, best val_loss=0.0059
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"

# Full pretrain h5ad: 169,120 samples × 49,156 CpGs (no tissue/age metadata in obs)
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"

# External metadata compiled from MethylGPT DSV files (186k unique GSM_IDs)
# Columns: GSM_ID, tissue, sex, age, disease, dataset, PATIENT_ID, ...
METADATA="${REPO}/data/pretrain_metadata.csv.gz"

TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/repr_analysis/pretrain_cls_169k_${SLURM_JOB_ID}"

BATCH_SIZE="${BATCH_SIZE:-64}"
N_PCA="${N_PCA:-50}"
N_NEIGHBORS="${N_NEIGHBORS:-15}"

echo "============================================================"
echo " MethylLlama-Small — Pretrain CLS Representation Analysis"
echo " (Full 169k pretrain dataset)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Pretrain ckpt : ${PRETRAIN_CKPT}"
echo " Data (169k)   : ${DATA}"
echo " Metadata      : ${METADATA}"
echo " Tokenizer     : ${TOKENIZER}"
echo " Outdir        : ${OUTDIR}"
echo " batch_size    : ${BATCH_SIZE}   n_pca: ${N_PCA}   n_neighbors: ${N_NEIGHBORS}"
echo "============================================================"

# Sanity checks
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}"
    exit 1
fi
if [ ! -f "${DATA}" ]; then
    echo "ERROR: pretrain h5ad not found: ${DATA}"
    exit 1
fi
if [ ! -f "${METADATA}" ]; then
    echo "ERROR: metadata file not found: ${METADATA}"
    echo "  It should be at ${METADATA}"
    echo "  (committed in data/pretrain_metadata.csv.gz)"
    exit 1
fi

echo ""
python scripts/utils/extract_sample_embeddings.py \
    --checkpoint        "${PRETRAIN_CKPT}"      \
    --ckpt_type         pretrain                \
    --data              "${DATA}"               \
    --tokenizer         "${TOKENIZER}"          \
    --metadata          "${METADATA}"           \
    --metadata_id_col   GSM_ID                  \
    --outdir            "${OUTDIR}"             \
    --batch_size        "${BATCH_SIZE}"         \
    --compare_random                            \
    --n_pca             "${N_PCA}"              \
    --n_neighbors       "${N_NEIGHBORS}"        \
    --label_cols        tissue sex disease dataset \
    --age_col           age

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   embeddings_cls.npy        [169120, 256]  pretrained CLS"
echo "   embeddings_mean.npy       [169120, 256]  mean-pool"
echo "   embeddings_random_cls.npy [169120, 256]  random-init ablation"
echo "   adata.h5ad                AnnData with UMAP + tissue/sex/age/disease"
echo "   figures/                  all PNGs"
echo "   report.txt                clustering + probe metrics"
echo "============================================================"
