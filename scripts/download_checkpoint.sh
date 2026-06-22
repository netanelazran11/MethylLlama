#!/usr/bin/env bash
# Download MethylLlama pretrained checkpoint from HuggingFace Hub.
# Usage: bash scripts/download_checkpoint.sh [output_dir]

set -euo pipefail

REPO_ID="netanelazran1/MethylLlama"
FILENAME="pretrain_llama_epoch98_val0.0059.ckpt"
OUTPUT_DIR="${1:-checkpoints}"

mkdir -p "${OUTPUT_DIR}"

echo "Downloading ${FILENAME} from ${REPO_ID} ..."
python - <<EOF
from huggingface_hub import hf_hub_download
path = hf_hub_download(
    repo_id="${REPO_ID}",
    filename="${FILENAME}",
    local_dir="${OUTPUT_DIR}",
)
print(f"Saved to: {path}")
EOF

echo "Done. Checkpoint at: ${OUTPUT_DIR}/${FILENAME}"
