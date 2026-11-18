#!/bin/bash -l
#SBATCH --job-name=pilot-eval
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=96G
#SBATCH --time=12:00:00
#SBATCH --array=0-4
#SBATCH --output=logs/%x_%A_%a.out
#SBATCH --error=logs/%x_%A_%a.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript runs (authors' cluster): --partition=salmon --gres=gpu:l40s:1

set -euo pipefail
# ─────────────────────────────────────────────────────────────────────────────
# Contrastive-weight pilot, step 3: the evaluation pipelines of the manuscript, on each arm.
#   tasks 0–3  pilot arms λ 0 / 0.01 / 0.05 / 0.2, checkpoint of epoch EVAL_EPOCH (19 = the last)
#   task  4    released pretrained checkpoint (epoch 85) — pipeline check
#
# Per task, into outputs/objective_pilot/pilot_eval/<arm>/ (each step skipped if already done):
#   1. cross-view retrieval on the full held-out partition (16,912), overlapping and disjoint views, seed 0
#      (scripts/evaluation/view_consistency_heldout_full.py; also saves the view embeddings for
#      hard_negative_eval.py)
#   2. reconstruction of withheld CpGs, 5,000 held-out profiles (objective_pilot/reconstruction_withheld_subset.py)
#   3. frozen CLS on the downstream collection: geometry and age / tissue / sex / study probes
#      (scripts/evaluation/extract_pretrain_cls.py)
# Then hard_negative_eval.py per arm and summarize_pilot.py once (objective_pilot/README.md).
#
#   mkdir -p logs && sbatch objective_pilot/evaluate_pilot.sh            (from the repository root)
# ─────────────────────────────────────────────────────────────────────────────

LAMBDAS=(0.0 0.01 0.05 0.2)
TASK="${SLURM_ARRAY_TASK_ID:?run as an array job}"
EVAL_EPOCH="${EVAL_EPOCH:-19}"

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
PRETRAIN_H5AD="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
DOWNSTREAM_H5AD="${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
TOKENIZER="${REPO}/tokenizer"
RANK_PRETRAIN="${REPO}/resources/cpg_genomic_rank.npy"
RANK_FINETUNE="${REPO}/resources/cpg_genomic_rank_finetune.npy"
PILOT_ROOT="${REPO}/outputs/objective_pilot/pilot"
EVAL_ROOT="${REPO}/outputs/objective_pilot/pilot_eval"

if [ "${TASK}" = "4" ]; then
    ARM="published_ep85"
    CKPT="${REPO}/checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt"
else
    ARM="lam${LAMBDAS[$TASK]}_seed42"
    CKPT="${PILOT_ROOT}/${ARM}/checkpoints/analysis/epoch=${EVAL_EPOCH}-analysis.ckpt"
    ARM="${ARM}_ep${EVAL_EPOCH}"
fi
OUT="${EVAL_ROOT}/${ARM}"
mkdir -p "${OUT}"
for f in "${CKPT}" "${PRETRAIN_H5AD}" "${DOWNSTREAM_H5AD}" "${RANK_PRETRAIN}" "${RANK_FINETUNE}"; do
    [ -f "$f" ] || { echo "ERROR: not found: $f"; exit 1; }
done
echo "PILOT EVAL task=${TASK} arm=${ARM} | $(hostname) | $(date)"
echo "Checkpoint: ${CKPT}"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${REPO}/scripts/evaluation:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1
D=scripts/evaluation

# 1. cross-view retrieval (held-out, full)
for COND in overlap disjoint; do
    if [ -f "${OUT}/view_consistency/per_run/${COND}_seed0.json" ]; then echo "skip view ${COND}"; continue; fi
    python ${D}/view_consistency_heldout_full.py \
        --checkpoint "${CKPT}" --data "${PRETRAIN_H5AD}" --split test \
        --tokenizer "${TOKENIZER}" --genomic_rank "${RANK_PRETRAIN}" \
        --condition "${COND}" --seed 0 --n_samples 0 --outdir "${OUT}/view_consistency"
done

# 2. reconstruction of withheld CpGs (5,000 held-out profiles)
if [ ! -f "${OUT}/reconstruction/reconstruction_withheld_summary.json" ]; then
    python objective_pilot/reconstruction_withheld_subset.py \
        --checkpoint "${CKPT}" --data "${PRETRAIN_H5AD}" --tokenizer "${TOKENIZER}" \
        --genomic_rank "${RANK_PRETRAIN}" --split test --max_samples 5000 --outdir "${OUT}/reconstruction"
else echo "skip reconstruction"; fi

# 3. frozen CLS probes on the downstream collection
if [ ! -f "${OUT}/frozen_probes/analysis_summary.json" ]; then
    python ${D}/extract_pretrain_cls.py \
        --checkpoint "${CKPT}" --data "${DOWNSTREAM_H5AD}" --tokenizer "${TOKENIZER}" \
        --genomic_rank "${RANK_FINETUNE}" --outdir "${OUT}/frozen_probes" --batch_size 32 \
        --age_col age --label_cols tissue_type sex dataset --split_col split
else echo "skip frozen probes"; fi

echo "Done ${ARM}: $(date)"
