#!/bin/bash -l
#SBATCH --job-name=pretrain-fixed2k-bmfm
#SBATCH --partition=goldfish
#SBATCH --gres=gpu:h200:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G
#SBATCH --time=50:00:00

#SBATCH --output=logs/%x_%j.out
#SBATCH --error=logs/%x_%j.err

set -euo pipefail

# -------------------------
# Paths
# -------------------------
REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
LOGDIR="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs"

DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_8k_h5ad/methylgpt_8k_altumage_combined.h5ad"

# Combine style: "add" (standard) or "multiply" (scGPT style)
COMBINE_STYLE="${COMBINE_STYLE:-add}"

# CpG subset settings (FIXED subset - same CpGs for all samples)
# Use ALL 8k CpGs to train all CpG embeddings properly
SUBSET_K="${SUBSET_K:-8000}"
FIXED_SUBSET="true"
FIXED_SUBSET_SEED="42"

# W&B naming
WANDB_ENTITY="netanelazran11-hebrew-university-of-jerusalem"
WANDB_PROJECT="pretrain-full8k-bmfm-rna-methylation"
WANDB_RUN_NAME="${COMBINE_STYLE}-full8k-${SLURM_JOB_ID}"

# Output directory (unique per run to avoid overwriting checkpoints)
OUTROOT="${REPO}/outputs/${WANDB_PROJECT}"
OUTDIR="${OUTROOT}/${WANDB_RUN_NAME}"

mkdir -p "${LOGDIR}"
mkdir -p "${OUTDIR}"

echo "============================================================"
echo "METHYLATION PRETRAINING (FIXED SUBSET)"
echo "============================================================"
echo "Job started: $(date)"
echo "Host: $(hostname)"
echo "JobID: ${SLURM_JOB_ID}"
echo "Node(s): ${SLURM_NODELIST}"
echo "============================================================"
echo "CpG Subset:  FIXED ${SUBSET_K} CpGs (same for all samples)"
echo "Seed:        ${FIXED_SUBSET_SEED}"
echo "Combine style: ${COMBINE_STYLE}"
if [ "${COMBINE_STYLE}" = "multiply" ]; then
    echo "Architecture: h = CpG_embed * β_value (scGPT style)"
else
    echo "Architecture: h = CpG_embed + β_embed (standard BMFM)"
fi
echo "============================================================"
echo "W&B project: ${WANDB_PROJECT}"
echo "W&B run:     ${WANDB_RUN_NAME}"
echo "Data:        ${DATA}"
echo "Output dir:  ${OUTDIR}"
echo "============================================================"

# -------------------------
# Modules (CUDA/NVCC)
# -------------------------
# Initialize module system (required for non-login shells)
source /etc/profile.d/modules.sh 2>/dev/null || source /usr/share/modules/init/bash 2>/dev/null || true

module purge
module load spack/all
module load cuda/12.3.2-gcc-5bv3kyh

# -------------------------
# Env
# -------------------------
cd "${REPO}"
source bmfm_methyl_env/bin/activate

# Perf / stability knobs
export TOKENIZERS_PARALLELISM=false
export PYTORCH_ALLOC_CONF="expandable_segments:True"
export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}

# Tensor cores utilization
python - <<'PY'
import torch
torch.set_float32_matmul_precision("medium")
print("torch:", torch.__version__, "cuda:", torch.version.cuda)
print("matmul_precision:", torch.get_float32_matmul_precision())
PY

# -------------------------
# Run (all params from config files)
# -------------------------
python -m bmfm_methylation.mlm.pretrain_mlm \
    data_path="${DATA}" \
    output_directory="${OUTDIR}" \
    combine_style="${COMBINE_STYLE}" \
    data_module.subset_k="${SUBSET_K}" \
    data_module.fixed_subset="${FIXED_SUBSET}" \
    data_module.fixed_subset_seed="${FIXED_SUBSET_SEED}" \
    data_module.max_length=$((SUBSET_K + 2)) \
    track_wandb.enabled=true \
    track_wandb.project="${WANDB_PROJECT}" \
    track_wandb.entity="${WANDB_ENTITY}" \
    track_wandb.name="${WANDB_RUN_NAME}"

echo "============================================================"
echo "Job finished: $(date)"
echo "============================================================"
