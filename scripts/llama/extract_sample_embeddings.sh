#!/bin/bash -l
#SBATCH --job-name=extract-cls
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:30:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/extract_cls_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/extract_cls_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# MethylLlama-Small — CLS bottleneck extraction + representation analysis
#
# Runs the full pipeline:
#   • Extract pretrained CLS (pooler_output) from WCED epoch=98 checkpoint
#   • Extract mean-pool embeddings from the same encoder for comparison
#   • Optionally extract random-init CLS as ablation baseline
#   • UMAP (PCA-50 → neighbors-15 → UMAP-2D + Leiden)
#   • Clustering metrics: ASW, ARI, NMI, DBI, CHI  vs. tissue/gender/dataset
#   • Age linear probe: Ridge R², PCC, MAE
#   • Save all PNGs + adata.h5ad + report.txt
#
# Usage:
#   sbatch scripts/llama/extract_sample_embeddings.sh
#
#   # Override pretrain checkpoint:
#   PRETRAIN_CKPT=/path/to/other.ckpt sbatch scripts/llama/extract_sample_embeddings.sh
#
#   # Also analyse fine-tuned checkpoint (set after v4b finishes):
#   FINETUNE_CKPT=/path/to/finetune_best.ckpt sbatch scripts/llama/extract_sample_embeddings.sh
# ─────────────────────────────────────────────────────────────────────────────

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
# Checkpoints
# ─────────────────────────────────────────────────────────────────────────────

# WCED pretrain — MethylLlama-Small, epoch=98, best val_loss=0.0059
PRETRAIN_CKPT="${PRETRAIN_CKPT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"

# Fine-tuned — set FINETUNE_CKPT env var to enable fine-tune analysis
# e.g. FINETUNE_CKPT=outputs/finetune-llama-small/llama-small-ft-v4b-.../checkpoints/epoch=117-val_mae=6.3071.ckpt
FINETUNE_CKPT="${FINETUNE_CKPT:-}"

# ─────────────────────────────────────────────────────────────────────────────
# Data
# ─────────────────────────────────────────────────────────────────────────────
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"

# ─────────────────────────────────────────────────────────────────────────────
# Settings
# ─────────────────────────────────────────────────────────────────────────────
BATCH_SIZE="${BATCH_SIZE:-64}"
N_PCA="${N_PCA:-50}"
N_NEIGHBORS="${N_NEIGHBORS:-15}"
OUTROOT="${REPO}/outputs/repr_analysis"

echo "============================================================"
echo " MethylLlama-Small — CLS Representation Analysis"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Pretrain ckpt : ${PRETRAIN_CKPT}"
echo " Finetune ckpt : ${FINETUNE_CKPT:-<not set>}"
echo " Data          : ${DATA}"
echo " Tokenizer     : ${TOKENIZER}"
echo " batch_size    : ${BATCH_SIZE}   n_pca: ${N_PCA}   n_neighbors: ${N_NEIGHBORS}"
echo "============================================================"

# Sanity check: pretrain checkpoint must exist
if [ ! -f "${PRETRAIN_CKPT}" ]; then
    echo "ERROR: pretrain checkpoint not found: ${PRETRAIN_CKPT}"
    echo "  Override with: PRETRAIN_CKPT=/path/to/ckpt sbatch ..."
    exit 1
fi

# ─────────────────────────────────────────────────────────────────────────────
# Run 1: Pretrained CLS  (main analysis)
# ─────────────────────────────────────────────────────────────────────────────
PRETRAIN_OUTDIR="${OUTROOT}/pretrain_cls_${SLURM_JOB_ID}"

echo ""
echo "── [1/2] Pretrained CLS extraction ──────────────────────────────────"

python scripts/utils/extract_sample_embeddings.py \
    --checkpoint   "${PRETRAIN_CKPT}"  \
    --ckpt_type    pretrain            \
    --data         "${DATA}"           \
    --tokenizer    "${TOKENIZER}"      \
    --outdir       "${PRETRAIN_OUTDIR}" \
    --batch_size   "${BATCH_SIZE}"     \
    --compare_random                   \
    --n_pca        "${N_PCA}"          \
    --n_neighbors  "${N_NEIGHBORS}"    \
    --label_cols   tissue_type gender dataset \
    --age_col      age                 \
    --split_col    split

echo ""
echo " Pretrained analysis done → ${PRETRAIN_OUTDIR}"

# ─────────────────────────────────────────────────────────────────────────────
# Run 2: Fine-tuned CLS  (if checkpoint is provided)
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "── [2/2] Fine-tuned CLS extraction ──────────────────────────────────"

if [ -n "${FINETUNE_CKPT}" ] && [ -f "${FINETUNE_CKPT}" ]; then
    FINETUNE_OUTDIR="${OUTROOT}/finetune_cls_${SLURM_JOB_ID}"

    python scripts/utils/extract_sample_embeddings.py \
        --checkpoint   "${FINETUNE_CKPT}"   \
        --ckpt_type    finetune             \
        --data         "${DATA}"            \
        --tokenizer    "${TOKENIZER}"       \
        --outdir       "${FINETUNE_OUTDIR}" \
        --batch_size   "${BATCH_SIZE}"      \
        --n_pca        "${N_PCA}"           \
        --n_neighbors  "${N_NEIGHBORS}"     \
        --label_cols   tissue_type gender dataset \
        --age_col      age                  \
        --split_col    split

    echo " Fine-tuned analysis done → ${FINETUNE_OUTDIR}"
else
    echo " FINETUNE_CKPT not set or file not found — skipping."
    echo " To run: FINETUNE_CKPT=/path/to/best.ckpt sbatch scripts/llama/extract_sample_embeddings.sh"
fi

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs:"
echo "   ${PRETRAIN_OUTDIR}/"
echo "     embeddings_cls.npy      pretrained CLS  [N, 256]"
echo "     embeddings_mean.npy     mean-pool       [N, 256]"
echo "     embeddings_random_cls.npy  random-init  [N, 256]"
echo "     adata.h5ad              AnnData with UMAP + metadata"
echo "     figures/                all PNGs"
echo "     report.txt              clustering + probe metrics"
echo "============================================================"
