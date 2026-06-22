#!/bin/bash
# submit_repr_analysis.sh
# Submit all 4 representation analysis jobs at once.
# Run from: /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl

set -uo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
cd "${REPO}"

FINETUNE_CKPT="${REPO}/outputs/finetune-llama-small/llama-small-ft-v4b-huber-ep300-wu500-scratch-44770333/checkpoints/epoch=121-val_medae=3.6497.ckpt"

echo "Submitting representation analysis jobs..."
echo "Fine-tune ckpt: ${FINETUNE_CKPT}"
echo ""

JOB1=$(sbatch --parsable scripts/llama/run_cpg_embeddings.sh)
echo "  [1] run_cpg_embeddings   → job ${JOB1}"

JOB2=$(sbatch --parsable scripts/llama/run_raw_umap.sh)
echo "  [2] run_raw_umap         → job ${JOB2}"

JOB3=$(sbatch --parsable --export=ALL,FINETUNE_CKPT="${FINETUNE_CKPT}" scripts/llama/run_robustness.sh)
echo "  [3] run_robustness       → job ${JOB3}"

JOB4=$(sbatch --parsable --export=ALL,FINETUNE_CKPT="${FINETUNE_CKPT}" scripts/llama/run_attention_analysis.sh)
echo "  [4] run_attention_analysis → job ${JOB4}"

echo ""
echo "All submitted. Monitor with:"
echo "  squeue -u \$(whoami)"
echo "  tail -f logs_llama-wced/run_robustness_${JOB3}.out"
