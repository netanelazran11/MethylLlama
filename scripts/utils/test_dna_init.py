"""
Quick test: verify that init_cpg_embeddings_from_dna works correctly.

Checks:
  1. Embeddings are loaded from npy + ids files
  2. Correct number of CpGs initialized (expect ~21368)
  3. Spot-check: a known CpG's embedding actually changed from random init
  4. Special tokens (rows 0-4) were NOT overwritten
  5. Embedding norms are finite (no NaN/Inf)

Usage (on cluster):
    cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
    source bmfm_methyl_env/bin/activate
    python scripts/utils/test_dna_init.py
"""

import sys
import os
import numpy as np
import torch

REPO = "/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
sys.path.insert(0, REPO)

CPG_EMB_NPY = "/sci/labs/benjamin.yakir/netanel.azran/data/cpg_embeddings/cpg_embeddings_bmfdna_21k.npy"
CPG_EMB_IDS = "/sci/labs/benjamin.yakir/netanel.azran/data/cpg_embeddings/cpg_ids_order.txt"
TOKENIZER_PATH = f"{REPO}/tokenizer_llama_domain21k"

print("=" * 60)
print("TEST: init_cpg_embeddings_from_dna")
print("=" * 60)

# ── 1. Load embeddings file
print("\n[1] Loading BMFM-DNA embeddings...")
emb = np.load(CPG_EMB_NPY)
cpg_names = open(CPG_EMB_IDS).read().splitlines()
print(f"    npy shape: {emb.shape}  (expect [21368, 768])")
print(f"    ids count: {len(cpg_names)}  (expect 21368)")
assert emb.shape == (len(cpg_names), 768), "Shape mismatch!"
print("    OK")

# ── 2. Build model + tokenizer
print("\n[2] Building MethylLlamaModel + tokenizer...")
from bmfm_methylation.llama.model import MethylLlamaConfig, MethylLlamaModel, init_cpg_embeddings_from_dna
from bmfm_targets.tokenization import MultiFieldTokenizer

tokenizer = MultiFieldTokenizer.from_pretrained(TOKENIZER_PATH)
vocab_size = len(tokenizer.tokenizers["cpg_sites"].get_vocab())
print(f"    Tokenizer vocab size: {vocab_size}")

cfg = MethylLlamaConfig(
    vocab_size=vocab_size,
    hidden_size=768,
    num_hidden_layers=8,
    num_attention_heads=12,
    intermediate_size=2048,
)
model = MethylLlamaModel(cfg)
print(f"    Model params: {sum(p.numel() for p in model.parameters())/1e6:.1f}M")

# Save a copy of original weights for comparison
weight_before = model.embeddings.cpg_sites_embeddings.weight.data.clone()
# Save special token rows (0-4) for later check
special_before = weight_before[:5].clone()

# ── 3. Run init
print("\n[3] Running init_cpg_embeddings_from_dna...")
n_init = init_cpg_embeddings_from_dna(
    model=model,
    tokenizer=tokenizer,
    npy_path=CPG_EMB_NPY,
    ids_path=CPG_EMB_IDS,
)
print(f"    CpGs initialized: {n_init}  (expect ~21368)")
assert n_init > 21000, f"Too few CpGs initialized: {n_init}"
print("    OK")

weight_after = model.embeddings.cpg_sites_embeddings.weight.data

# ── 4. Check special tokens not overwritten
print("\n[4] Checking special tokens (rows 0-4) not overwritten...")
special_after = weight_after[:5]
assert torch.allclose(special_before, special_after), "Special tokens were modified!"
print("    OK — special tokens unchanged")

# ── 5. Check that some rows actually changed
print("\n[5] Checking that embedding rows actually changed...")
diff = (weight_after - weight_before).abs().sum(dim=1)  # [vocab_size]
n_changed = (diff > 1e-6).sum().item()
print(f"    Rows changed: {n_changed}  (expect ~{n_init})")
assert n_changed >= n_init * 0.99, f"Too few rows changed: {n_changed}"
print("    OK")

# ── 6. Spot-check: first CpG in ids file
print("\n[6] Spot-check first CpG...")
cpg_name = cpg_names[0]
vocab = tokenizer.tokenizers["cpg_sites"].get_vocab()
token_id = vocab.get(cpg_name)
print(f"    CpG name: {cpg_name}  → token_id: {token_id}")
if token_id is not None and token_id >= 5:
    expected = torch.tensor(emb[0], dtype=weight_after.dtype)
    actual   = weight_after[token_id]
    max_err  = (expected - actual).abs().max().item()
    print(f"    Max abs error: {max_err:.2e}  (expect <1e-5)")
    assert max_err < 1e-5, f"Spot-check failed: max_err={max_err}"
    print("    OK")
else:
    print("    SKIP (not in vocab or special token)")

# ── 7. Check for NaN/Inf
print("\n[7] Checking for NaN/Inf in initialized rows...")
initialized_weights = weight_after[5:5 + n_init]
assert torch.isfinite(initialized_weights).all(), "NaN or Inf in initialized embeddings!"
print("    OK — all finite")

# ── 8. Summary stats
print("\n[8] Embedding statistics (initialized rows)...")
norms = initialized_weights.norm(dim=1)
print(f"    Norm: mean={norms.mean():.3f}, std={norms.std():.3f}, min={norms.min():.3f}, max={norms.max():.3f}")

print("\n" + "=" * 60)
print(f"ALL TESTS PASSED — {n_init}/{len(cpg_names)} CpGs initialized")
print("=" * 60)
