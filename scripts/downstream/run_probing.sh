#!/bin/bash -l
#SBATCH --job-name=downstream-probing
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=12:00:00
#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

# Usage:
#   sbatch run_probing.sh                        → WCED + random, age task
#   INIT_TYPE=wced_pretrained sbatch run_probing.sh → WCED only
#   INIT_TYPE=random_init     sbatch run_probing.sh → random only
#   TASK=smoking sbatch run_probing.sh           → smoking task

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
CHECKPOINT="${REPO}/outputs/pretrain-wced-bmfm/wced-contrastive-k8000-w0.1-44206138/pretrain/checkpoints/epoch=epoch=190-val_loss=validation/loss=0.1264.ckpt"
DATA_AGE="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"
DATA_SMOKING="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/smoking_combined_aligned.h5ad"
DATA_MULTITASK="/sci/labs/benjamin.yakir/netanel.azran/data/smoking_geo/multitask_data.h5ad"

TASK="${TASK:-age}"
INIT_TYPE="${INIT_TYPE:-both}"
N_STEPS="${N_STEPS:-3000}"

if [ "${TASK}" = "age" ]; then
    DATA="${DATA_AGE}"
else
    DATA="${DATA_SMOKING}"
fi

OUTDIR="${REPO}/outputs/downstream/probing"
mkdir -p "${REPO}/logs" "${OUTDIR}"

source /etc/profile.d/modules.sh 2>/dev/null || true
module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

cd "${REPO}"
source bmfm_methyl_env/bin/activate
export TOKENIZERS_PARALLELISM=false
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

echo "============================================================"
echo "TASK C-1 — Data Efficiency (task=${TASK}, init=${INIT_TYPE})"
echo "Fixed steps: ${N_STEPS} per run (normalized labels for age)"
echo "============================================================"

python -m bmfm_methylation.downstream.probing.data_efficiency \
    --checkpoint_path "${CHECKPOINT}" \
    --data_path "${DATA}" \
    --task "${TASK}" \
    --init_type "${INIT_TYPE}" \
    --output_dir "${OUTDIR}/data_efficiency_${TASK}_${INIT_TYPE}" \
    --n_steps "${N_STEPS}" \
    --n_seeds 3

echo "Done data efficiency: $(date)"

# Only run embedding analysis when doing the full age run
if [ "${TASK}" = "age" ] && [ "${INIT_TYPE}" = "both" ]; then
    echo "============================================================"
    echo "TASK C-2 — Embedding Analysis (UMAP + linear probes)"
    echo "============================================================"
    if [ -f "${DATA_MULTITASK}" ]; then
        python -m bmfm_methylation.downstream.probing.embedding_analysis \
            --checkpoint_path "${CHECKPOINT}" \
            --data_path "${DATA_MULTITASK}" \
            --output_dir "${OUTDIR}/embeddings_multitask" \
            --label_cols age smoking_status sex \
            --compare_random_init
    else
        python -m bmfm_methylation.downstream.probing.embedding_analysis \
            --checkpoint_path "${CHECKPOINT}" \
            --data_path "${DATA_AGE}" \
            --output_dir "${OUTDIR}/embeddings_age" \
            --label_cols age sex \
            --compare_random_init
    fi
    echo "Done embeddings: $(date)"
fi

echo "All done: $(date)"
