#!/bin/bash -l
#SBATCH --job-name=recon-withheld-eval
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
# Canonical withheld-CpG reconstruction eval (Figure 2C/2D + Supp S3).
# Replaces the legacy reconstruction_baselines run, which used the wrong
# checkpoint, input_ratio=1.0 (no withheld CpGs), and no genomic position_ids.
#
# Also prints the pretraining h5ad's total row count + split sizes, to close
# the manuscript's "verify 169,120" todo in the same job.
#
# Usage: sbatch scripts/evaluation/run_reconstruction_withheld.sh
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
SPLIT="${SPLIT:-test}"
MAX_SAMPLES="${MAX_SAMPLES:-5000}"

cd "${REPO}"
methyllama_env   # environment setup, defined in configs/paths.sh
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"
export TOKENIZERS_PARALLELISM=false
export PYTHONUNBUFFERED=1

[ -f "${CKPT}" ]         || { echo "ERROR: checkpoint not found: ${CKPT}"; exit 1; }
[ -f "${DATA}" ]         || { echo "ERROR: pretrain h5ad not found: ${DATA}"; exit 1; }
[ -f "${GENOMIC_RANK}" ] || { echo "ERROR: genomic rank not found: ${GENOMIC_RANK}"; exit 1; }

echo "============================================================"
echo "Withheld-CpG reconstruction eval (canonical ep85 checkpoint)"
echo "Job: ${SLURM_JOB_ID}  Host: $(hostname)  Time: $(date)"
echo "Split: ${SPLIT}  Max samples: ${MAX_SAMPLES}"
echo "============================================================"

# ── Manuscript verification: total pretraining corpus size + split counts ────
python - <<'PY'
import anndata
a = anndata.read_h5ad(
    "${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad",
    backed="r",
)
print(f"PRETRAIN H5AD VERIFICATION: n_obs={a.n_obs:,}  n_vars={a.n_vars:,}")
if "split" in a.obs.columns:
    print("  split counts:", a.obs["split"].value_counts().to_dict())
else:
    print("  NOTE: no 'split' column; columns =", list(a.obs.columns))
PY

python scripts/evaluation/reconstruction_withheld_eval.py \
    --checkpoint "${CKPT}" \
    --data "${DATA}" \
    --tokenizer "${TOKENIZER}" \
    --genomic_rank "${GENOMIC_RANK}" \
    --split "${SPLIT}" \
    --max_samples "${MAX_SAMPLES}" \
    --outdir "${OUTDIR}"

echo "DONE: $(date)"
