"""
Post-process BMFM-DNA CpG embeddings with PCA whitening.

Problem: raw embeddings have cosine similarity ~1.000 (anisotropy — all vectors
point in nearly the same direction). This weakens the DNA init signal.

Fix: PCA whitening
  1. Subtract the mean vector  → removes dominant shared direction
  2. Project onto principal components
  3. Divide each PC by its std  → unit variance in every direction
  4. (Optional) re-scale to match original norm magnitude

Output: cpg_embeddings_bmfdna_21k_whitened.npy  — same shape [21368, 768]
        Same row order as the original file (cpg_ids_order.txt unchanged).

No GPU / model / genome needed — pure numpy + sklearn.
"""

import numpy as np
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

BASE    = "/sci/labs/benjamin.yakir/netanel.azran/data/cpg_embeddings"
IN_NPY  = f"{BASE}/cpg_embeddings_bmfdna_21k.npy"
OUT_NPY = f"{BASE}/cpg_embeddings_bmfdna_21k_whitened.npy"

# ── Load ──────────────────────────────────────────────────────────────────────
print("Loading embeddings...")
emb = np.load(IN_NPY).astype(np.float32)   # [21368, 768]
print(f"  Shape: {emb.shape}")

norms_before = np.linalg.norm(emb, axis=1)
emb_norm_before = normalize(emb, norm="l2")
rng = np.random.default_rng(42)
idx = rng.choice(len(emb), size=2000, replace=False)
sims_before = (emb_norm_before[idx] @ emb_norm_before[idx].T)
upper_before = sims_before[np.triu_indices(2000, k=1)]
print(f"\nBefore whitening:")
print(f"  L2 norm      — mean: {norms_before.mean():.4f}  std: {norms_before.std():.4f}")
print(f"  Cosine sim   — mean: {upper_before.mean():.4f}  std: {upper_before.std():.4f}")

# ── PCA whitening ─────────────────────────────────────────────────────────────
# Keep all 768 components so no information is discarded.
# whiten=True divides each PC by its std → unit variance per direction.
print("\nRunning PCA whitening (768 components)...")
pca = PCA(n_components=768, whiten=True, random_state=42)
emb_white = pca.fit_transform(emb).astype(np.float32)   # [21368, 768]

explained = pca.explained_variance_ratio_
print(f"  Variance captured: {explained.sum()*100:.1f}%")
print(f"  PC1: {explained[0]*100:.2f}%  PC2: {explained[1]*100:.2f}%  (was 33.5% / 11.0% before)")

# ── Re-scale to original norm magnitude ──────────────────────────────────────
# Whitening makes norms ~1. Re-scale to original mean norm (~7.62) so the
# embedding table entries stay in the same magnitude range as before.
# This avoids disrupting the cpg_scale initialisation in MethylLlama.
target_norm = norms_before.mean()
current_norm = np.linalg.norm(emb_white, axis=1).mean()
emb_white = emb_white * (target_norm / current_norm)
print(f"\nRe-scaled norms to match original magnitude (~{target_norm:.2f})")

# ── Verify ────────────────────────────────────────────────────────────────────
norms_after = np.linalg.norm(emb_white, axis=1)
emb_norm_after = normalize(emb_white, norm="l2")
sims_after = (emb_norm_after[idx] @ emb_norm_after[idx].T)
upper_after = sims_after[np.triu_indices(2000, k=1)]

print(f"\nAfter whitening:")
print(f"  L2 norm      — mean: {norms_after.mean():.4f}  std: {norms_after.std():.4f}")
print(f"  Cosine sim   — mean: {upper_after.mean():.4f}  std: {upper_after.std():.4f}")
print(f"  (Good target: cosine sim mean ~0.0, std ~0.05)")

# ── Save ──────────────────────────────────────────────────────────────────────
np.save(OUT_NPY, emb_white)
print(f"\nSaved: {OUT_NPY}")
print(f"  Shape: {emb_white.shape}  dtype: {emb_white.dtype}")
print(f"  Size:  {emb_white.nbytes / 1e6:.1f} MB")
print("\nDone. Update CPG_EMB_NPY in pretrain_llama_domain_dna.sh to point to the new file.")
