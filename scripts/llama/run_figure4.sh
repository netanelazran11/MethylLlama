#!/bin/bash -l
#SBATCH --job-name=figure4-age-pca
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=2:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_figure4.sh — Before vs After fine-tuning: PCA by age
#
# ┌─────────────────┬──────────────────────────────┬──────────────────────┐
# │                 │ MODEL                         │ DATA                 │
# ├─────────────────┼──────────────────────────────┼──────────────────────┤
# │ BEFORE fine-tune│ WCED pretrained checkpoint    │ 19k finetune h5ad    │
# │ (already done)  │ (trained on 169k pretrain)    │ (inference only)     │
# │                 │ cls_probing_44905909           │                      │
# ├─────────────────┼──────────────────────────────┼──────────────────────┤
# │ AFTER fine-tune │ MethylationAgeRegressorLlama  │ same 19k finetune    │
# │ (Step 1 here)   │ epoch=127, MedAE=3.5625yr     │ h5ad — same file     │
# │                 │ encoder unfrozen at epoch 10  │                      │
# └─────────────────┴──────────────────────────────┴──────────────────────┘
#
# Step 1: Extract fine-tuned CLS embeddings (GPU needed)
# Step 2: Align both embedding sets by sample ID + make PCA figure (CPU)
#
# NOTE: The two embedding sets may have different N (different samples
# survived their respective data loading). figure4_age_pca.py aligns
# them by sample ID intersection — this is safe and correct.
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

# BEFORE fine-tuning — already extracted in job 44905909
PRETRAINED_BASE="${REPO}/outputs/repr_analysis/cls_probing_44905909"
PRETRAINED_NPY="${PRETRAINED_BASE}/embeddings_cls.npy"
PRETRAINED_META="${PRETRAINED_BASE}/metadata.csv"

# AFTER fine-tuning — will be extracted in Step 1
FINETUNE_CKPT="${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
EXT_META="${REPO}/data/pretrain_metadata.csv.gz"

# Reuse previous extraction dir if embeddings already exist (e.g. from job 44944545)
PREV_EXTRACT="${REPO}/outputs/repr_analysis/finetune_extract_44944545"
if [ -f "${PREV_EXTRACT}/embeddings_cls.npy" ]; then
    EXTRACT_OUTDIR="${PREV_EXTRACT}"
    echo "  Reusing existing extraction from: ${EXTRACT_OUTDIR}"
else
    EXTRACT_OUTDIR="${REPO}/outputs/repr_analysis/finetune_extract_${SLURM_JOB_ID}"
fi
FINETUNED_NPY="${EXTRACT_OUTDIR}/embeddings_cls.npy"
FINETUNED_META="${EXTRACT_OUTDIR}/metadata.csv"

FIGURE_OUTDIR="${REPO}/outputs/repr_analysis/figure4_${SLURM_JOB_ID}"

echo "============================================================"
echo " Figure 4: CLS Space Before vs After Fine-tuning"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " BEFORE FT npy  : ${PRETRAINED_NPY}"
echo " BEFORE FT meta : ${PRETRAINED_META}"
echo " Fine-tune ckpt : ${FINETUNE_CKPT}"
echo " Data (both)    : ${DATA}"
echo " AFTER FT outdir: ${EXTRACT_OUTDIR}"
echo " Figure outdir  : ${FIGURE_OUTDIR}"
echo "============================================================"

# Validate
for f in "${PRETRAINED_NPY}" "${PRETRAINED_META}" "${FINETUNE_CKPT}" "${DATA}" "${TOKENIZER}"; do
    if [ ! -e "${f}" ]; then
        echo "ERROR: required file not found: ${f}"
        exit 1
    fi
done

# ─────────────────────────────────────────────────────────────────────────────
# STEP 1: Extract fine-tuned CLS embeddings
#   Model : epoch=127 (MethylationAgeRegressorLlama, encoder updated from epoch 10)
#   Data  : same 19k finetune h5ad as was used for cls_probing_44905909
#   Output: finetune_extract_JOBID/embeddings_cls.npy + metadata.csv
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 1: Extracting fine-tuned CLS embeddings"
echo "         Model : epoch=127 fine-tuned checkpoint"
echo "         Data  : ${DATA}"
echo "============================================================"
mkdir -p "${EXTRACT_OUTDIR}"

# If embeddings already extracted (e.g. from a previous failed run), skip re-extraction
PRECOMP_ARGS=""
if [ -f "${FINETUNED_NPY}" ]; then
    echo "  Found pre-computed embeddings — skipping GPU extraction"
    PRECOMP_ARGS="--cls_embeddings ${FINETUNED_NPY}"
else
    echo "  No pre-computed embeddings — will extract from checkpoint"
fi

python scripts/repr_analysis/cls_probing_analysis.py \
    --checkpoint      "${FINETUNE_CKPT}"   \
    --ckpt_type       finetune             \
    --data            "${DATA}"            \
    --tokenizer       "${TOKENIZER}"       \
    --metadata        "${EXT_META}"        \
    --metadata_id_col GSM_ID              \
    --outdir          "${EXTRACT_OUTDIR}"  \
    --batch_size      64                   \
    --device          cuda                 \
    --skip_probing                         \
    --label_cols      tissue sex dataset   \
    --age_col         age                  \
    --split_col       split                \
    ${PRECOMP_ARGS}

if [ ! -f "${FINETUNED_NPY}" ]; then
    echo "ERROR: fine-tuned embeddings not created: ${FINETUNED_NPY}"
    exit 1
fi
if [ ! -f "${FINETUNED_META}" ]; then
    echo "ERROR: fine-tuned metadata not created: ${FINETUNED_META}"
    exit 1
fi

echo ""
echo " STEP 1 done:"
echo "   ${FINETUNED_NPY}"
echo "   ${FINETUNED_META}"

# ─────────────────────────────────────────────────────────────────────────────
# STEP 2: Generate Figure 4
#   Aligns both embedding sets by sample ID intersection
#   PCA separately on each embedding, then plots side by side
# ─────────────────────────────────────────────────────────────────────────────
echo ""
echo "============================================================"
echo " STEP 2: Generating Figure 4 (PCA colored by age + tissue)"
echo "         Aligning by sample ID — safe regardless of row count"
echo "============================================================"

python scripts/repr_analysis/figure4_age_pca.py \
    --pretrained_npy   "${PRETRAINED_NPY}"   \
    --pretrained_meta  "${PRETRAINED_META}"  \
    --finetuned_npy    "${FINETUNED_NPY}"    \
    --finetuned_meta   "${FINETUNED_META}"   \
    --ext_metadata     "${EXT_META}"         \
    --ext_id_col       GSM_ID               \
    --age_col          age                  \
    --outdir           "${FIGURE_OUTDIR}"   \
    --dpi              200

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo ""
echo " Fine-tuned embeddings : ${EXTRACT_OUTDIR}/"
echo " Figure outputs        : ${FIGURE_OUTDIR}/figures/"
echo "   figure4_age_pca.png     2×2: before/after FT × age/tissue"
echo "   figure4_age_pca.pdf"
echo "   pretrained_age.png      individual panel"
echo "   pretrained_tissue.png   individual panel"
echo "   finetuned_age.png       individual panel"
echo "   finetuned_tissue.png    individual panel"
echo "============================================================"
