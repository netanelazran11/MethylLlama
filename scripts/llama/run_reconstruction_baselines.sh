#!/bin/bash -l
#SBATCH --job-name=recon-baselines
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=02:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Experiment B: Reconstruction baselines for WCED decoder
#
# Answers the question: does the WCED decoder actually USE the CLS embedding,
# or does it reconstruct methylation from population-level statistics alone?
#
# Three baselines are computed in one pass:
#   B1 — Per-CpG training-mean prediction (trivial lower bound)
#   B3 — Shuffled CLS: real embeddings, wrong samples (CLS connectivity test)
#   B4 — Random Gaussian CLS (decoder ignores input test)
#
# If model MSE ≈ B3 MSE → WCED shortcut confirmed.
# If model MSE << B3 MSE → CLS carries real sample-specific signal.
#
# Runtime: ~20 min for 10k samples on H200.
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# WCED pretrained checkpoint (epoch=98, val_loss=0.0059)
CHECKPOINT="${CHECKPOINT:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"

TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

BATCH_SIZE="${BATCH_SIZE:-64}"
OUTDIR="${REPO}/outputs/repr_analysis/reconstruction_baselines_${SLURM_JOB_ID}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "RECONSTRUCTION BASELINES (Experiment B)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Checkpoint: ${CHECKPOINT}"
echo "Data: ${DATA}"
echo "Output: ${OUTDIR}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate

export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

python scripts/repr_analysis/reconstruction_baselines.py \
    --checkpoint  "${CHECKPOINT}" \
    --data        "${DATA}" \
    --tokenizer   "${TOKENIZER_PATH}" \
    --outdir      "${OUTDIR}" \
    --batch_size  "${BATCH_SIZE}" \
    --device      cuda

echo "============================================================"
echo "Reconstruction baselines done: $(date)"
echo "Results: ${OUTDIR}/reconstruction_baselines.json"
echo "============================================================"

# Print summary directly in the SLURM log
python -c "
import json, sys
p = '${OUTDIR}/reconstruction_baselines.json'
d = json.load(open(p))
print('=== RESULT SUMMARY ===')
for k in ['model_mse','b1_mse','b3_mse','b4_mse']:
    print(f'{k:20s}: mean={d[k][\"mean\"]:.6f}  median={d[k][\"median\"]:.6f}')
for k in [x for x in d if x.startswith('ratio')]:
    print(f'{k:30s}: {d[k]:.4f}')
"
