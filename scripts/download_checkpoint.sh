#!/bin/bash
# Download the released MethylLlama checkpoints from Hugging Face into checkpoints/.
#
#   bash scripts/download_checkpoint.sh            # pretrained encoder and the five age models
#   bash scripts/download_checkpoint.sh pretrained # pretrained encoder only
#
# Layout (same as on Hugging Face):
#   checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt
#   checkpoints/finetuned_age/fold{0..4}.ckpt
#
# Every file is checked against SHA256SUMS of the Hugging Face repository (the checksums of the
# files written by the original runs).
set -euo pipefail
REPO_ID="netanelazran1/MethylLlama"
WHAT="${1:-all}"
OUT="${2:-checkpoints}"
FILES=("pretrained/methyllama_pretrained_epoch85.ckpt")
if [ "${WHAT}" = "all" ]; then
    for k in 0 1 2 3 4; do FILES+=("finetuned_age/fold${k}.ckpt"); done
fi
python - "${REPO_ID}" "${OUT}" "${FILES[@]}" <<'PY'
import sys
from huggingface_hub import hf_hub_download
repo, out, files = sys.argv[1], sys.argv[2], sys.argv[3:]
import hashlib
sums = {}
for line in open(hf_hub_download(repo_id=repo, filename="SHA256SUMS", local_dir=out)):
    digest, name = line.split()
    sums[name.removeprefix("./")] = digest
for f in files:
    path = hf_hub_download(repo_id=repo, filename=f, local_dir=out)
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    if h.hexdigest() != sums[f]:
        sys.exit(f"checksum mismatch: {path}")
    print("downloaded, checksum OK", path)
PY
