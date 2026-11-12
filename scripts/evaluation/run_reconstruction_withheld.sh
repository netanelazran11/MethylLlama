#!/bin/bash -l
#SBATCH --job-name=recon-withheld
#SBATCH --gres=gpu:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err
# SLURM partition is site-specific: add one if your cluster requires it.
# Resources used for the manuscript run (authors' cluster): --partition=salmon --gres=gpu:l40s:1
# ─────────────────────────────────────────────────────────────────────────────
# Withheld-CpG reconstruction on ALL 16,912 held-out pretraining profiles, with Pearson r and the
# observed-vs-predicted histogram over ALL withheld positions (Fig. 2c, Supplementary Fig. S1c).
# Inference only. Writes to results/pretrained_cls/reconstruction_withheld/; then run
#   python scripts/evaluation/reconstruction_paired_bootstrap.py
# for the paired bootstrap intervals (CPU, about a minute).
#
# Usage:   sbatch scripts/evaluation/run_reconstruction_withheld.sh
# Smoke test first (about 10 min): sbatch --export=ALL,SMOKE=1 scripts/evaluation/run_reconstruction_withheld.sh
#          -> first 320 profiles only, written to a separate *_smoke folder.
# ─────────────────────────────────────────────────────────────────────────────
set -euo pipefail

REPO="${METHYLLAMA_REPO:-${SLURM_SUBMIT_DIR:-$(pwd)}}"
source "${REPO}/configs/paths.sh" || { echo "ERROR: create configs/paths.sh from configs/paths.example.sh and run from the repository root"; exit 1; }
# released checkpoint (scripts/download_checkpoint.sh); set CKPT to use your own pretraining run
CKPT="${CKPT:-${REPO}/checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt}"
DATA="${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
TOKENIZER="${REPO}/tokenizer"
GENOMIC_RANK="${REPO}/resources/cpg_genomic_rank.npy"
OUTDIR="${REPO}/results/pretrained_cls/reconstruction_withheld"
EXPECTED_N=16912
SMOKE="${SMOKE:-0}"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

for f in "${CKPT}" "${DATA}" "${GENOMIC_RANK}"; do
    [ -f "${f}" ] || { echo "ERROR: not found: ${f}"; exit 1; }
done
[ -d "${TOKENIZER}" ] || { echo "ERROR: tokenizer not found: ${TOKENIZER}"; exit 1; }

SCRIPT="scripts/evaluation/reconstruction_withheld_eval.py"
if [ "${SMOKE}" = "1" ]; then
    # smoke: first 320 profiles only, own folder, no population check -- verifies the code path, not results
    OUTDIR="${OUTDIR}_smoke"
    python - <<PY
import runpy, sys
from bmfm_methylation.shared import data_module as dm
_orig = dm.MethylationDataset.__len__
dm.MethylationDataset.__len__ = lambda self: min(320, _orig(self))
sys.argv = ["${SCRIPT}", "--checkpoint", "${CKPT}", "--data", "${DATA}", "--tokenizer", "${TOKENIZER}",
            "--genomic_rank", "${GENOMIC_RANK}", "--split", "test", "--expected_n", "0", "--outdir", "${OUTDIR}"]
runpy.run_path("${SCRIPT}", run_name="__main__")
PY
    echo "SMOKE DONE: $(date)  -> ${OUTDIR}"
    exit 0
fi

echo "============================================================"
echo "Withheld-CpG reconstruction, ALL held-out profiles (expected ${EXPECTED_N})"
echo "Job: ${SLURM_JOB_ID:-local}  Host: $(hostname)  Time: $(date)"
echo "Output: ${OUTDIR}"
echo "============================================================"

python "${SCRIPT}" \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --split test \
    --expected_n "${EXPECTED_N}" \
    --outdir "${OUTDIR}"

python scripts/evaluation/reconstruction_paired_bootstrap.py --dir "${OUTDIR}"

echo "DONE: $(date)"
