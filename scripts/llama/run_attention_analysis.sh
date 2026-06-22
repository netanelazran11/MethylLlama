#!/bin/bash -l
#SBATCH --job-name=attention-analysis
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_attention_analysis.sh  —  Figure 5 equivalent
#
# Extracts attention weights from the final transformer layer across all 19k
# fine-tuning test samples, then:
#   1. Computes per-CpG attention scores (column-sum, head-averaged)
#   2. Groups samples by age: young (<20), mid (20-60), old (>60)
#   3. Two-sided t-test per CpG, Benjamini-Hochberg FDR correction
#   4. Volcano plot: log2FC vs -log10(FDR)  [Fig 5a/b style]
#   5. Heatmap of top differentially attended CpGs across age groups  [Fig 5c]
#
# Usage:
#   sbatch scripts/llama/run_attention_analysis.sh
#
#   # Override fine-tune checkpoint:
#   FINETUNE_CKPT=/path/to/other.ckpt sbatch scripts/llama/run_attention_analysis.sh
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
OUTDIR="${REPO}/outputs/repr_analysis/attention_${SLURM_JOB_ID}"

MANIFEST="${REPO}/outputs/cpg_manifest/cpg_annotations_finetune19k.tsv"

# Last transformer layer (0-indexed); MethylLlama-Small has 4 layers → layer 3
LAYER="${LAYER:-3}"
BATCH_SIZE="${BATCH_SIZE:-1}"    # [B,H,L,L] for L=19k needs ~6GB even at B=1

echo "============================================================"
echo " CpG Attention Weight Analysis  (Fig 5)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Fine-tune ckpt : ${FINETUNE_CKPT}"
echo " Data (19k)     : ${DATA}"
echo " Tokenizer      : ${TOKENIZER}"
echo " Target layer   : ${LAYER}  (0-indexed)"
echo " Batch size     : ${BATCH_SIZE}"
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

MANIFEST_ARG=""
if [ -f "${MANIFEST}" ]; then
    MANIFEST_ARG="--manifest ${MANIFEST}"
else
    echo "WARNING: CpG manifest not found at ${MANIFEST}"
    echo "  Run align_cpg_manifest.sh first for genomic annotation labels."
fi

echo ""
METADATA="${REPO}/data/pretrain_metadata.csv.gz"
METADATA_ARG=""
if [ -f "${METADATA}" ]; then
    METADATA_ARG="--metadata ${METADATA} --metadata_id_col GSM_ID"
fi

python scripts/repr_analysis/attention_analysis.py \
    --checkpoint    "${FINETUNE_CKPT}"  \
    --ckpt_type     finetune            \
    --data          "${DATA}"           \
    --tokenizer     "${TOKENIZER}"      \
    --outdir        "${OUTDIR}"         \
    --layer         "${LAYER}"          \
    --batch_size    "${BATCH_SIZE}"     \
    --max_samples   -1                  \
    --device        cuda                \
    --age_col       age                 \
    --fdr_thresh    0.05                \
    --lfc_thresh    0.585               \
    ${MANIFEST_ARG}                     \
    ${METADATA_ARG}

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   cpg_attention.npy           [n_test × n_cpg]  per-sample attention"
echo "   differential_attention.csv  CpG-level stats (log2FC, FDR, sig flag)"
echo "   attention_report.txt        summary stats"
echo "   figures/attention_heatmap.png"
echo "   figures/attention_volcano.png"
echo "   figures/top_cpg_heatmap.png"
echo "============================================================"
