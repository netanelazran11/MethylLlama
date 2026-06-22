#!/bin/bash -l
#SBATCH --job-name=align-cpg
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=0:30:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/align_cpg_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/align_cpg_%j.err

# ─────────────────────────────────────────────────────────────────────────────
# align_cpg_manifest.sh
#
# Validates that the tokenizer 49k CpG IDs and h5ad 19k CpG IDs are covered
# by the Illumina HM450K and/or EPIC manifest.
#
# Produces:
#   outputs/cpg_manifest/alignment_report.txt
#   outputs/cpg_manifest/cpg_annotations_tokenizer49k.tsv   ← use in Fig 2 UMAP
#   outputs/cpg_manifest/cpg_annotations_finetune19k.tsv
#
# Usage:
#   # Use default manifest paths (set below):
#   sbatch scripts/llama/align_cpg_manifest.sh
#
#   # Override manifest locations:
#   MANIFEST_450K=/path/to/HM450.csv MANIFEST_EPIC=/path/to/EPIC.csv \
#     sbatch scripts/llama/align_cpg_manifest.sh
#
#   # If you only have one manifest:
#   MANIFEST_EPIC=/path/to/EPIC.csv sbatch scripts/llama/align_cpg_manifest.sh
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"
mkdir -p "${LOGDIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

# ─────────────────────────────────────────────────────────────────────────────
# Paths — adjust MANIFEST_450K and/or MANIFEST_EPIC to your manifest locations
# ─────────────────────────────────────────────────────────────────────────────
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
OUTDIR="${REPO}/outputs/cpg_manifest"

# Set these to wherever your manifest files live on the cluster.
# Leave empty ("") to skip a manifest (e.g. if you only have EPIC).
MANIFEST_450K="${MANIFEST_450K:-/sci/labs/benjamin.yakir/netanel.azran/data/manifests/HM450.hg38.manifest.tsv}"
MANIFEST_EPIC="${MANIFEST_EPIC:-}"

echo "============================================================"
echo " CpG Manifest Alignment"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Tokenizer : ${TOKENIZER}"
echo " h5ad      : ${DATA}"
echo " HM450K    : ${MANIFEST_450K:-<not set>}"
echo " EPIC      : ${MANIFEST_EPIC:-<not set>}"
echo " Outdir    : ${OUTDIR}"
echo "============================================================"

# Build the --manifests argument dynamically
MANIFEST_ARGS=""
if [ -n "${MANIFEST_450K}" ] && [ -f "${MANIFEST_450K}" ]; then
    MANIFEST_ARGS="${MANIFEST_ARGS} ${MANIFEST_450K}"
    echo "Using HM450K manifest: ${MANIFEST_450K}"
elif [ -n "${MANIFEST_450K}" ]; then
    echo "WARNING: MANIFEST_450K set but file not found: ${MANIFEST_450K}"
fi

if [ -n "${MANIFEST_EPIC}" ] && [ -f "${MANIFEST_EPIC}" ]; then
    MANIFEST_ARGS="${MANIFEST_ARGS} ${MANIFEST_EPIC}"
    echo "Using EPIC manifest  : ${MANIFEST_EPIC}"
elif [ -n "${MANIFEST_EPIC}" ]; then
    echo "WARNING: MANIFEST_EPIC set but file not found: ${MANIFEST_EPIC}"
fi

if [ -z "${MANIFEST_ARGS}" ]; then
    echo ""
    echo "ERROR: No manifest files found."
    echo "  Set MANIFEST_450K and/or MANIFEST_EPIC environment variables."
    echo "  Example:"
    echo "    MANIFEST_450K=/path/to/HumanMethylation450_15017482_v1-2.csv \\"
    echo "    MANIFEST_EPIC=/path/to/MethylationEPIC_v-1-0_B5.csv \\"
    echo "    sbatch scripts/llama/align_cpg_manifest.sh"
    exit 1
fi

echo ""
python scripts/utils/align_cpg_manifest.py \
    --tokenizer  "${TOKENIZER}"  \
    --data       "${DATA}"       \
    --manifests  ${MANIFEST_ARGS} \
    --outdir     "${OUTDIR}"

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Results in: ${OUTDIR}/"
echo "   alignment_report.txt"
echo "   cpg_annotations_tokenizer49k.tsv  ← pass as --cpg_manifest to"
echo "                                        extract_sample_embeddings.py"
echo "   cpg_annotations_finetune19k.tsv"
echo "============================================================"
