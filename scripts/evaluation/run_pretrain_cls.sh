#!/bin/bash -l
#SBATCH --job-name=methyllama-pretrain-cls
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=3:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1

# ─────────────────────────────────────────────────────────────────────────────
# MethylLlama pretrain (ep85) CLS extraction + representation analysis.
# Genomic-RoPE correct: collator uses genomic_rank_path, encoder gets position_ids.
# Extracts: per-sample CLS matrix, mean-pool matrix, CpG embedding matrix, metadata.
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

echo "============================================================"
echo " MethylLlama pretrain CLS analysis | job ${SLURM_JOB_ID:-local} | $(date)"
echo " Checkpoint : ${CKPT}"
echo " Data       : ${DATA}"
echo " Genomic rank: ${GENOMIC_RANK}"
echo " Outdir     : ${OUTDIR}"
echo "============================================================"

[ -f "${CKPT}" ] || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }

python scripts/evaluation/extract_pretrain_cls.py \
    --checkpoint   "${CKPT}" \
    --data         "${DATA}" \
    --tokenizer    "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --outdir       "${OUTDIR}" \
    --batch_size   32 \
    --age_col      age \
    --label_cols   tissue_type sex dataset \
    --split_col    split

echo "============================================================"
echo " DONE: $(date) — outputs in ${OUTDIR}/"
echo "============================================================"
