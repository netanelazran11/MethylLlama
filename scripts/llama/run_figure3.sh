#!/bin/bash -l
#SBATCH --job-name=figure3-comparison
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --mem=64G
#SBATCH --time=3:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# run_figure3.sh  —  Embedding space comparison: CLS vs Raw methylation
#
# Generates a publication-quality 6-panel UMAP figure:
#   Top row    : MethylLlama CLS embedding (256D, WCED pretrained)
#   Bottom row : Raw DNA methylation (PCA-50 → UMAP)
#   Columns    : tissue  |  dataset (batch)  |  sex
#
# Uses 19k finetune embeddings from cls_probing_44905909.
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-16}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths
# ─────────────────────────────────────────────────────────────────────────────

BASE_19K="${REPO}/outputs/repr_analysis/cls_probing_44905909"
CLS_NPY="${BASE_19K}/embeddings_cls.npy"
RANDOM_NPY="${BASE_19K}/embeddings_random_cls.npy"
METADATA="${BASE_19K}/metadata.csv"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
EXT_META="${REPO}/data/pretrain_metadata.csv.gz"

OUTDIR="${REPO}/outputs/repr_analysis/figure3_${SLURM_JOB_ID}"

echo "============================================================"
echo " Figure 3: MethylLlama CLS vs Raw Methylation (UMAP)"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " CLS npy    : ${CLS_NPY}"
echo " Random npy : ${RANDOM_NPY}"
echo " Raw h5ad   : ${DATA}"
echo " Metadata   : ${METADATA}"
echo " Ext meta   : ${EXT_META}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

# ── Validate required files ───────────────────────────────────────────────────
for f in "${CLS_NPY}" "${METADATA}" "${DATA}"; do
    if [ ! -f "${f}" ]; then
        echo "ERROR: required file not found: ${f}"
        exit 1
    fi
done

EXT_ARG=""
if [ -f "${EXT_META}" ]; then
    EXT_ARG="--ext_metadata ${EXT_META} --ext_id_col GSM_ID"
    echo "  External metadata found — will join tissue/sex labels"
else
    echo "  WARNING: external metadata not found at ${EXT_META}"
fi

echo ""
RANDOM_ARG=""
if [ -f "${RANDOM_NPY}" ]; then
    RANDOM_ARG="--random_npy ${RANDOM_NPY}"
    echo "  Random embeddings found — will include as third row"
else
    echo "  WARNING: random embeddings not found — figure will have 2 rows only"
fi

python scripts/repr_analysis/figure3_comparison.py \
    --cls_npy      "${CLS_NPY}"      \
    ${RANDOM_ARG}                    \
    --data         "${DATA}"         \
    --metadata_csv "${METADATA}"     \
    ${EXT_ARG}                       \
    --outdir       "${OUTDIR}"       \
    --n_pca        50                \
    --n_neighbors  15                \
    --min_dist     0.1               \
    --seed         42                \
    --max_nan_frac 0.2               \
    --dpi          200

echo ""
echo "============================================================"
echo " ALL DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   figure3_cls_vs_raw.png     6-panel publication figure (PNG)"
echo "   figure3_cls_vs_raw.pdf     6-panel publication figure (PDF)"
echo "   panels/                    individual panel PNGs"
echo "   metadata_aligned.csv       sample metadata used for coloring"
echo "============================================================"
