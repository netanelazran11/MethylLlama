#!/bin/bash -l
#SBATCH --job-name=eval-embeddings
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=1:00:00
#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/eval_embeddings_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/eval_embeddings_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"
source bmfm_methyl_env/bin/activate
export PYTHONPATH="${REPO}:${PYTHONPATH:-}"

CHECKPOINT="${CHECKPOINT:-???}"        # required: path to .ckpt
CKPT_TYPE="${CKPT_TYPE:-pretrain}"    # pretrain or finetune
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean.h5ad"
TOKENIZER="${REPO}/tokenizer_llama_pretrain49k"
TARGET="${TARGET:-tissue_type}"       # or: gender, dataset
OUTDIR="${REPO}/outputs/eval_embeddings/${CKPT_TYPE}_${TARGET}_${SLURM_JOB_ID}"

echo "============================================================"
echo " MethylLlama Embedding Evaluation"
echo " Checkpoint : ${CHECKPOINT}"
echo " Type       : ${CKPT_TYPE}"
echo " Target     : ${TARGET}"
echo " Output     : ${OUTDIR}"
echo "============================================================"

python scripts/utils/eval_embeddings.py \
    --checkpoint "${CHECKPOINT}" \
    --ckpt_type  "${CKPT_TYPE}" \
    --data       "${DATA}" \
    --tokenizer  "${TOKENIZER}" \
    --outdir     "${OUTDIR}" \
    --target_col "${TARGET}" \
    --batch_size 32

echo "============================================================"
echo " DONE: $(date)"
echo "============================================================"
