# MethylLlama — local paths and environment.
#
# Copy this file to configs/paths.sh (ignored by git), edit the values, and run
# every script from the repository root. Each SLURM/bash script sources it.

# Directory that holds the input matrices, laid out as:
#   ${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad
#   ${METHYLLAMA_DATA}/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv
#   ${METHYLLAMA_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad
#   ${METHYLLAMA_DATA}/manifests/HM450.hg38.manifest.tsv
export METHYLLAMA_DATA="${METHYLLAMA_DATA:-/path/to/data}"

# MethylGPT comparison (scripts/methylgpt/; see its README). METHYLGPT_DATA holds the MethylGPT
# input list (cpg_ids/probe_ids_type3.csv) and receives the fold parquet files; the configs in
# scripts/methylgpt/configs/ expect it at data/methylgpt. METHYLGPT_DIR is only used by
# scripts/data/compare_splits_21k.py and verify_21k_comparison.py.
export METHYLGPT_DATA="${METHYLGPT_DATA:-data/methylgpt}"
export METHYLGPT_DIR="${METHYLGPT_DIR:-/path/to/MethylGPT}"

# Weights & Biases. Leave the entity empty to use your default account, or set
# WANDB_MODE=offline to log locally only.
export WANDB_ENTITY="${WANDB_ENTITY:-}"

# Environment setup run by every script before Python starts. Replace the body
# with whatever your system needs. The example is what the authors' SLURM cluster used.
methyllama_env() {
    # source /etc/profile.d/modules.sh 2>/dev/null || true
    # module purge
    # module load spack/all
    # module load cuda/12.3.2-gcc-5bv3kyh
    # shellcheck disable=SC1091
    source "${METHYLLAMA_VENV:-${REPO:-.}/.venv}/bin/activate"
}

# Environment for scripts/methylgpt/ (a separate environment with the methylgpt package installed).
methylgpt_env() {
    # shellcheck disable=SC1091
    source "${METHYLGPT_VENV:-/path/to/methylgpt_venv}/bin/activate"
}
