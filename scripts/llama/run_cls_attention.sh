#!/bin/bash -l
#SBATCH --job-name=cls-attention
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=04:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

# ─────────────────────────────────────────────────────────────────────────────
# Experiment D1: CLS-to-CpG attention analysis
#
# Extracts the attention weights from the CLS query row (position 0) to all
# CpG key positions.  This is the CORRECT way to measure CpG selectivity,
# unlike the existing column-sum approach which averages over all 19k query
# rows and makes every head look near-uniform.
#
# Compares pretrained vs fine-tuned checkpoints to see whether fine-tuning
# for age prediction causes the model to attend more selectively to
# age-informative CpGs.
#
# Output:
#   cls_attention_summary.json  — per-(layer, head) entropy/gini/top-k metrics
#   cls_attn_pretrained.npy     — (n_samples, 4_layers, 4_heads, n_cpg)
#   cls_attn_finetuned.npy      — same for fine-tuned checkpoint
#   top_cpg_indices.npz         — top-10 CpG indices per head
#
# Runtime: ~1-2 hrs for 2000 samples (CLS recomputation per layer).
# ─────────────────────────────────────────────────────────────────────────────

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="${REPO}/logs_llama-wced"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# WCED pretrained checkpoint
PRETRAINED="${PRETRAINED:-${REPO}/outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt}"

# Best fine-tuned checkpoint (MedAE=3.5625yr)
FINETUNED="${FINETUNED:-${REPO}/outputs/finetune-llama-small/llama-small-ft-v5-cls-huber-ep300-wu500-scratch-44895876/checkpoints/epoch=127-val_medae=3.5625.ckpt}"

TOKENIZER_PATH="${REPO}/tokenizer_llama_pretrain49k"

BATCH_SIZE="${BATCH_SIZE:-16}"
MAX_SAMPLES="${MAX_SAMPLES:-2000}"
OUTDIR="${REPO}/outputs/repr_analysis/cls_attention_${SLURM_JOB_ID}"

mkdir -p "${LOGDIR}" "${OUTDIR}"

echo "============================================================"
echo "CLS ATTENTION ANALYSIS (Experiment D1)"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Pretrained:  ${PRETRAINED}"
echo "Fine-tuned:  ${FINETUNED}"
echo "Max samples: ${MAX_SAMPLES}"
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

python scripts/repr_analysis/cls_attention_analysis.py \
    --pretrained  "${PRETRAINED}" \
    --finetuned   "${FINETUNED}" \
    --data        "${DATA}" \
    --tokenizer   "${TOKENIZER_PATH}" \
    --outdir      "${OUTDIR}" \
    --batch_size  "${BATCH_SIZE}" \
    --max_samples "${MAX_SAMPLES}" \
    --device      cuda

echo "============================================================"
echo "CLS attention analysis done: $(date)"
echo "Results: ${OUTDIR}/cls_attention_summary.json"
echo "============================================================"

# Sync results back to local machine:
# rsync -av netanel.azran@moriah:${OUTDIR}/ \
#   /Users/netanelazran/Projects/BMFM-RNA_thesis/methyl/figures/cls_attention_JOBID/
