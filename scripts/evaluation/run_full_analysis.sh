#!/bin/bash -l
#SBATCH --job-name=methyllama-repr-full
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=4:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# Full MethylLlama pretrain (ep85) representation analysis — new-architecture folder.
# Runs: (1) extract CLS + CpG-embedding matrices  →  answers Q1-Q4
#       (2) visualize_cls        (Bonus 1: PCA/UMAP colored by age/tissue)
#       (3) analyze_cpg_embeddings (Bonus 2: genomic locality in CpG embeddings)
#       (4) two_view_consistency (Bonus 3: contrastive-quality check, GPU)
# All outputs → results/pretrained_cls/
# ─────────────────────────────────────────────────────────────────────────────

set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false

# released checkpoint (scripts/download_checkpoint.sh); set CKPT to use your own pretraining run
CKPT="${CKPT:-${REPO}/checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt}"
DATA="${DATA:-${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad}"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank_finetune.npy"
OUTDIR="${REPO}/results/pretrained_cls"
D="scripts/evaluation"

echo "=== [1/2] extract CLS + CpG embedding matrices (Q1-Q4) ==="
python ${D}/extract_pretrain_cls.py \
    --checkpoint "${CKPT}" --data "${DATA}" --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" --outdir "${OUTDIR}" \
    --age_col age --label_cols tissue_type sex dataset --split_col split

echo "=== [2/2] visualize CLS (Bonus 1: PCA/UMAP) ==="
python ${D}/visualize_cls.py --dir "${OUTDIR}" \
    --color_cols age tissue_type sex dataset

# Two further steps of the original pipeline are not part of this release: the CpG-embedding
# locality analysis (not reported) and the two-view consistency on 2,000 downstream profiles
# (superseded by run_view_consistency_heldout_full.sh on the held-out pretraining partition).

echo "============================================================"
echo " FULL ANALYSIS DONE: $(date)"
echo " Outputs → ${OUTDIR}/"
echo "   report.txt / analysis_summary.json        (Q1-Q4)"
echo "   cls_pca_panels.png / cls_umap_panels.png   (Bonus 1)"
echo "============================================================"
