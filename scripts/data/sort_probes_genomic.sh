#!/bin/bash -l
#SBATCH --job-name=sort-cpg-genomic
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:20:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=catfish --gres=gpu:l4:1

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
mkdir -p "${REPO}/logs_llama-wced"
cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

MANIFEST="${METHYLLAMA_DATA}/manifests/HM450.hg38.manifest.tsv"
PROBE_IDS="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"
OUTDIR="${REPO}/resources"

echo "============================================================"
echo " CpG Genomic Sort"
echo " Job : ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "============================================================"
echo " Manifest  : ${MANIFEST}"
echo " Probe IDs : ${PROBE_IDS}"
echo " Outdir    : ${OUTDIR}"
echo "============================================================"

python scripts/data/sort_probes_genomic.py \
    --manifest  "${MANIFEST}" \
    --probe_ids "${PROBE_IDS}" \
    --outdir    "${OUTDIR}"

echo ""
echo "============================================================"
echo " DONE: $(date)"
echo " Key output: ${OUTDIR}/cpg_genomic_rank.npy"
echo " Use as: wced_genomic_rank_path=${OUTDIR}/cpg_genomic_rank.npy"
echo "============================================================"
