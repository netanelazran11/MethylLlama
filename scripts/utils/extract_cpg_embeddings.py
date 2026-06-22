"""
One-time script: extract BMFM-DNA embeddings for all AltumAge CpG sites.

For each CpG in probe_ids_type3_21k.csv:
  1. Look up genomic coordinates in HM450 hg38 manifest
  2. Extract ±2048bp DNA window from hg38.fa
  3. Run BMFM-DNA → hidden state at the CpG token position → 768-dim embedding

Output: cpg_embeddings_bmfdna_21k.npy  shape [21368, 768]
        cpg_ids_order.txt              CpG IDs in same row order as embeddings

Usage (via SLURM):
    sbatch scripts/llama/extract_cpg_embeddings.sh
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from pathlib import Path

# ── Paths ─────────────────────────────────────────────────────────────────────
BASE       = "/sci/labs/benjamin.yakir/netanel.azran"
PROBE_CSV  = f"{BASE}/data/data_methyl_21k_h5ad/probe_ids_type3_21k.csv"
MANIFEST   = f"{BASE}/data/manifests/HM450.hg38.manifest.tsv"
GENOME_FA  = f"{BASE}/data/genomes/hg38/hg38.fa"
HF_CACHE   = f"{BASE}/data/hf_cache"
OUT_DIR    = f"{BASE}/data/cpg_embeddings"
OUT_NPY    = f"{OUT_DIR}/cpg_embeddings_bmfdna_21k.npy"
OUT_IDS    = f"{OUT_DIR}/cpg_ids_order.txt"

MODEL_ID   = "ibm-research/biomed.dna.ref.modernbert.113m.v1"
WINDOW     = 2048     # ±2048 bp around CpG position (~956 tokens, fits within model max 1024)
BATCH_SIZE = 64       # sequences per forward pass
# ─────────────────────────────────────────────────────────────────────────────

os.environ["HF_HOME"] = HF_CACHE
os.makedirs(OUT_DIR, exist_ok=True)

print("=" * 60)
print("BMFM-DNA CpG Embedding Extraction")
print("=" * 60)

# ── 1. Load CpG IDs ───────────────────────────────────────────────────────────
print(f"\n[1] Loading CpG IDs from {PROBE_CSV}")
probe_df = pd.read_csv(PROBE_CSV)
cpg_ids = probe_df["illumina_probe_id"].tolist()
print(f"    {len(cpg_ids)} CpG IDs loaded")

# ── 2. Load manifest → coordinates ───────────────────────────────────────────
print(f"\n[2] Loading manifest from {MANIFEST}")
manifest = pd.read_csv(MANIFEST, sep="\t", low_memory=False)
print(f"    Manifest columns: {list(manifest.columns[:8])}")

# Identify key columns (different manifests use different names)
id_col  = "probeID"   if "probeID"  in manifest.columns else manifest.columns[0]
chr_col = "CpG_chrm"  if "CpG_chrm" in manifest.columns else "CHR"
pos_col = "CpG_beg"   if "CpG_beg"  in manifest.columns else "MAPINFO"

manifest = manifest.set_index(id_col)
print(f"    Using columns: id={id_col}, chr={chr_col}, pos={pos_col}")
print(f"    Manifest probes: {len(manifest)}")

# ── 3. Load BMFM-DNA ──────────────────────────────────────────────────────────
# SCModernBertModel is the base encoder (no MLM head) — exactly what we need.
# PreTrainedTokenizerFast loads BMFM-DNA's own k-mer BPE tokenizer for DNA sequences.
# NOTE: this is BMFM-DNA's tokenizer for DNA text, completely separate from
#       the methylation tokenizer (cg_id → integer) used by MethylLlama.
#
# Loading strategy (SCModernBert is NOT registered with HF AutoModel/AutoConfig):
#   1. Find HF snapshot dir → has config.json + tokenizer files
#   2. Load tokenizer from snapshot dir
#   3. Instantiate model from config, then load weights from local last.ckpt
print(f"\n[3] Loading BMFM-DNA from local files")
from bmfm_targets.models.predictive.scmodernbert.modeling_scmodernbert import SCModernBertModel
from transformers import PreTrainedTokenizerFast

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"    Device: {device}")

# Find snapshot directory in HF cache.
# hf_hub_download uses HF_CACHE/models--... (no hub/ subdir),
# while huggingface-cli uses HF_CACHE/hub/models--... — check both.
model_cache_name = "models--" + MODEL_ID.replace("/", "--")
_candidates = [
    Path(HF_CACHE) / model_cache_name / "snapshots",
    Path(HF_CACHE) / "hub" / model_cache_name / "snapshots",
]
snapshots_dir = next((p for p in _candidates if p.exists()), None)
if snapshots_dir is None:
    raise FileNotFoundError(f"HF snapshot dir not found in {HF_CACHE}")
snapshot_dirs = [p for p in snapshots_dir.iterdir() if p.is_dir()]
if not snapshot_dirs:
    raise FileNotFoundError(f"No snapshots found in {snapshots_dir}")
# Prefer the snapshot that contains last.ckpt
snapshot_dir = next((p for p in snapshot_dirs if (p / "last.ckpt").exists()), snapshot_dirs[0])
print(f"    Snapshot dir: {snapshot_dir}")

# Load DNA tokenizer bundled in bmfm_targets package (dna_chunks BPE tokenizer)
# The HF repo only contains config.json — tokenizer is shipped with the package.
import bmfm_targets as _bmfm
_dna_tok_dir = Path(_bmfm.__file__).parent / "tests/resources/tokenizers/dna_chunks"
if not (_dna_tok_dir / "tokenizer.json").exists():
    raise FileNotFoundError(f"dna_chunks tokenizer not found at {_dna_tok_dir}")
tokenizer = PreTrainedTokenizerFast.from_pretrained(str(_dna_tok_dir))
if tokenizer.pad_token is None:
    tokenizer.add_special_tokens({"pad_token": "[PAD]"})
print(f"    Tokenizer loaded from {_dna_tok_dir}")
print(f"    vocab_size={tokenizer.vocab_size}")

# Load checkpoint first — config may be embedded in hyper_parameters
LOCAL_CKPT = snapshot_dir / "last.ckpt"
print(f"    Loading checkpoint from {LOCAL_CKPT}")
if not LOCAL_CKPT.exists():
    raise FileNotFoundError(f"Checkpoint not found: {LOCAL_CKPT}")

import json
ckpt = torch.load(str(LOCAL_CKPT), map_location="cpu", weights_only=False)
print(f"    Checkpoint top-level keys: {list(ckpt.keys())}")

# Extract state dict
raw_sd = ckpt["state_dict"] if "state_dict" in ckpt else ckpt

# Build model config — prefer checkpoint hyper_parameters over config.json
# (config.json on HF is only 112 bytes and may have wrong architecture)
config = None
if "hyper_parameters" in ckpt:
    hp = ckpt["hyper_parameters"]
    print(f"    hyper_parameters keys: {list(hp.keys()) if isinstance(hp, dict) else type(hp)}")
    for _key in ["model_config", "config", "cfg"]:
        if isinstance(hp, dict) and _key in hp:
            _c = hp[_key]
            if hasattr(_c, "hidden_size"):   # it's already a config object
                config = _c
                print(f"    Config from checkpoint[hyper_parameters][{_key}]")
                break

if config is None:
    with open(snapshot_dir / "config.json") as _f:
        _cfg_dict = json.load(_f)
    print(f"    config.json contents: {_cfg_dict}")
    _cfg_dict.setdefault("fields", [])
    config = SCModernBertModel.config_class.from_dict(_cfg_dict)
    print(f"    Config from config.json")

model = SCModernBertModel(config)
n_init = sum(p.numel() for p in model.parameters())
print(f"    Model initialized: {n_init/1e6:.1f}M params")

# Strip key prefix — try longest match first
model_keys = set(dict(model.named_parameters()).keys())
sd = dict(raw_sd)
for _prefix in ["model.scmodernbert.", "scmodernbert.", "model."]:
    _stripped = {k[len(_prefix):]: v for k, v in sd.items() if k.startswith(_prefix)}
    if _stripped and any(k in model_keys for k in list(_stripped.keys())[:10]):
        print(f"    Stripping state_dict prefix: '{_prefix}'")
        sd = _stripped
        break

missing, unexpected = model.load_state_dict(sd, strict=False)
if missing:
    print(f"    WARNING: {len(missing)} missing keys (first 5: {missing[:5]})")
if unexpected:
    print(f"    WARNING: {len(unexpected)} unexpected keys (first 5: {unexpected[:5]})")

model = model.to(device).eval()
n_params = sum(p.numel() for p in model.parameters())
print(f"    Model loaded: {n_params/1e6:.1f}M params")

# ── 4. Load genome ────────────────────────────────────────────────────────────
print(f"\n[4] Opening genome {GENOME_FA}")
from pyfaidx import Fasta
genome = Fasta(GENOME_FA)
print(f"    Chromosomes available: {len(genome.keys())}")

# ── 5. Extract embeddings ─────────────────────────────────────────────────────
print(f"\n[5] Extracting embeddings (window=±{WINDOW}bp → ~{WINDOW//2}tokens, batch={BATCH_SIZE})")

embeddings = np.zeros((len(cpg_ids), 768), dtype=np.float32)
missing = []

def get_dna_sequence(cpg_id):
    """Get DNA window around CpG site. Returns (seq, cpg_char_pos) or None."""
    if cpg_id not in manifest.index:
        return None
    row = manifest.loc[cpg_id]
    chrom = str(row[chr_col])
    pos   = int(row[pos_col])

    # Ensure chromosome name matches genome (chr1 vs 1)
    if chrom not in genome and f"chr{chrom}" in genome:
        chrom = f"chr{chrom}"
    if chrom not in genome:
        return None

    chrom_len = len(genome[chrom])
    start = max(0, pos - WINDOW)
    end   = min(chrom_len, pos + WINDOW)
    seq   = str(genome[chrom][start:end]).upper()
    cpg_char_pos = pos - start   # character offset of CpG within the window
    return seq, cpg_char_pos

_embed_batch_debug = True   # print shapes on first call only

def embed_batch(sequences, cpg_char_positions):
    """Run BMFM-DNA → hidden state at CpG token position for each sequence.

    Uses offset_mapping to find which token covers the CpG site character
    position, then extracts last_hidden_state at that token index.
    Falls back to the center token if offset mapping fails.
    """
    global _embed_batch_debug
    inputs = tokenizer(
        sequences,
        return_tensors="pt",
        padding=True,
        truncation=True,
        max_length=1024,
        return_offsets_mapping=True,
    )
    offset_mapping = inputs.pop("offset_mapping")   # [B, L, 2] — not a model input

    inputs = {k: v for k, v in inputs.items() if k != "token_type_ids"}
    inputs_gpu = {k: v.to(device) for k, v in inputs.items()}
    inputs_gpu["input_ids"] = inputs_gpu["input_ids"].unsqueeze(1)  # [B,1,L]

    with torch.no_grad():
        outputs = model(**inputs_gpu)

    # last_hidden_state: [B, 1, L, H] or [B, L, H]
    hidden = outputs.last_hidden_state
    if hidden.dim() == 4:
        hidden = hidden.squeeze(1)   # → [B, L, H]
    hidden = hidden.cpu().float()    # move to CPU for indexing

    if _embed_batch_debug:
        print(f"    [DEBUG] last_hidden_state shape (after squeeze): {hidden.shape}")
        print(f"    [DEBUG] offset_mapping shape: {offset_mapping.shape}")
        _embed_batch_debug = False

    B = hidden.shape[0]
    result = np.zeros((B, hidden.shape[2]), dtype=np.float32)

    for b in range(B):
        cpg_pos = cpg_char_positions[b]
        offsets = offset_mapping[b]   # [L, 2]

        # Find the token whose span covers cpg_pos
        tok_idx = None
        for t, (start_c, end_c) in enumerate(offsets.tolist()):
            if start_c <= cpg_pos < end_c:
                tok_idx = t
                break

        if tok_idx is None:
            # Fallback: token closest to cpg_pos by midpoint
            midpoints = [(s + e) / 2 for s, e in offsets.tolist()]
            tok_idx = int(np.argmin([abs(m - cpg_pos) for m in midpoints]))

        result[b] = hidden[b, tok_idx].numpy()

    return result

batch_seqs      = []
batch_cpg_pos   = []
batch_idxs      = []

for i, cpg_id in enumerate(cpg_ids):
    result = get_dna_sequence(cpg_id)
    if result is None or len(result[0]) < 10:
        missing.append(cpg_id)
        batch_seqs.append("ACGT" * 10)   # dummy — will be zeroed out below
        batch_cpg_pos.append(20)          # dummy position (middle of dummy seq)
        batch_idxs.append((i, False))
    else:
        seq, cpg_char_pos = result
        batch_seqs.append(seq)
        batch_cpg_pos.append(cpg_char_pos)
        batch_idxs.append((i, True))

    if len(batch_seqs) == BATCH_SIZE or i == len(cpg_ids) - 1:
        embs = embed_batch(batch_seqs, batch_cpg_pos)
        for (idx, valid), emb in zip(batch_idxs, embs):
            if valid:
                embeddings[idx] = emb
        batch_seqs    = []
        batch_cpg_pos = []
        batch_idxs    = []

        if (i + 1) % 1000 == 0 or i == len(cpg_ids) - 1:
            print(f"    {i+1}/{len(cpg_ids)} CpGs processed, {len(missing)} missing")

print(f"\n    Done. Missing/skipped: {len(missing)} CpGs")
if missing:
    print(f"    First 5 missing: {missing[:5]}")

# ── 6. Save ───────────────────────────────────────────────────────────────────
print(f"\n[6] Saving embeddings to {OUT_NPY}")
np.save(OUT_NPY, embeddings)
with open(OUT_IDS, "w") as f:
    f.write("\n".join(cpg_ids))

print(f"    Saved: {embeddings.shape} float32 ({embeddings.nbytes/1e6:.1f} MB)")
print(f"    CpG ID order saved to {OUT_IDS}")
print("\n" + "=" * 60)
print("Extraction complete.")
print("=" * 60)
