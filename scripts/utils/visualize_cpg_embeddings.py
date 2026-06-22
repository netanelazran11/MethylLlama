"""
Visualize BMFM-DNA CpG embeddings to understand their structure.

Shows:
  1. Basic statistics
  2. PCA colored by chromosome — does DNA context capture genomic location?
  3. Cosine similarity distribution — how distinct are CpG environments?
  4. Nearest neighbors example — do nearby CpGs cluster together?

Usage:
    python scripts/utils/visualize_cpg_embeddings.py
"""

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.decomposition import PCA
from sklearn.preprocessing import normalize

BASE      = "/sci/labs/benjamin.yakir/netanel.azran"
EMB_NPY   = f"{BASE}/data/cpg_embeddings/cpg_embeddings_bmfdna_21k.npy"
IDS_TXT   = f"{BASE}/data/cpg_embeddings/cpg_ids_order.txt"
MANIFEST  = f"{BASE}/data/manifests/HM450.hg38.manifest.tsv"
REPO      = "/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
OUT_DIR   = f"{REPO}/docs/images"
Path(OUT_DIR).mkdir(parents=True, exist_ok=True)

print("Loading embeddings...")
emb  = np.load(EMB_NPY)               # [21368, 768]
ids  = open(IDS_TXT).read().splitlines()
print(f"  Shape: {emb.shape}")

# ── 1. Basic stats ────────────────────────────────────────────────────────────
norms = np.linalg.norm(emb, axis=1)
print(f"\n[1] Basic statistics:")
print(f"  L2 norm   — mean: {norms.mean():.3f}  std: {norms.std():.3f}")
print(f"  Values    — mean: {emb.mean():.4f}  std: {emb.std():.4f}")
print(f"  Range     — min:  {emb.min():.4f}  max: {emb.max():.4f}")

# ── 2. Load chromosome labels ─────────────────────────────────────────────────
print("\nLoading manifest for chromosome labels...")
manifest = pd.read_csv(MANIFEST, sep="\t", low_memory=False, usecols=["probeID", "CpG_chrm"])
manifest = manifest.set_index("probeID")
chroms = []
for cpg in ids:
    if cpg in manifest.index:
        c = str(manifest.loc[cpg, "CpG_chrm"])
        chroms.append(c if c.startswith("chr") else f"chr{c}")
    else:
        chroms.append("unknown")
chroms = np.array(chroms)

# ── 3. PCA ────────────────────────────────────────────────────────────────────
print("\n[2] Running PCA...")
pca = PCA(n_components=2, random_state=42)
emb_2d = pca.fit_transform(emb)
var = pca.explained_variance_ratio_
print(f"  PC1: {var[0]*100:.1f}%  PC2: {var[1]*100:.1f}%")

# Color by chromosome (autosomes only for clarity)
main_chroms = [f"chr{i}" for i in range(1, 23)]
cmap = plt.cm.get_cmap("tab20", 22)

fig, ax = plt.subplots(figsize=(10, 8))
other_mask = ~np.isin(chroms, main_chroms)
ax.scatter(emb_2d[other_mask, 0], emb_2d[other_mask, 1],
           c="lightgray", s=1, alpha=0.3, label="other")
for i, chrom in enumerate(main_chroms):
    mask = chroms == chrom
    if mask.sum() == 0:
        continue
    ax.scatter(emb_2d[mask, 0], emb_2d[mask, 1],
               c=[cmap(i)], s=1, alpha=0.5, label=chrom)

ax.set_xlabel(f"PC1 ({var[0]*100:.1f}% var)")
ax.set_ylabel(f"PC2 ({var[1]*100:.1f}% var)")
ax.set_title("PCA of BMFM-DNA CpG Embeddings (colored by chromosome)")
lgnd = ax.legend(loc="upper right", markerscale=5, fontsize=6, ncol=2)
plt.tight_layout()
out = f"{OUT_DIR}/pca_by_chrom.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")

# ── 4. Cosine similarity distribution ────────────────────────────────────────
print("\n[3] Cosine similarity distribution (random 2000 pairs)...")
emb_norm = normalize(emb, norm="l2")
rng = np.random.default_rng(42)
idx = rng.choice(len(emb), size=2000, replace=False)
sims = (emb_norm[idx] @ emb_norm[idx].T)
# upper triangle only
upper = sims[np.triu_indices(2000, k=1)]

fig, ax = plt.subplots(figsize=(7, 4))
ax.hist(upper, bins=80, color="steelblue", edgecolor="none", alpha=0.8)
ax.axvline(upper.mean(), color="red", linestyle="--", label=f"mean={upper.mean():.3f}")
ax.set_xlabel("Cosine similarity")
ax.set_ylabel("Count")
ax.set_title("Pairwise cosine similarity of BMFM-DNA CpG embeddings\n(2000 random CpGs)")
ax.legend()
plt.tight_layout()
out = f"{OUT_DIR}/cosine_sim_dist.png"
plt.savefig(out, dpi=150)
plt.close()
print(f"  Saved: {out}")
print(f"  Mean similarity: {upper.mean():.4f}  (0=orthogonal, 1=identical)")
print(f"  Std:             {upper.std():.4f}")

# ── 5. Nearest neighbors example ─────────────────────────────────────────────
print("\n[4] Nearest neighbor examples:")
sample_idx = rng.choice(len(emb), size=5, replace=False)
for si in sample_idx:
    sims_row = emb_norm[si] @ emb_norm.T
    top5 = np.argsort(-sims_row)[1:6]
    cpg = ids[si]
    chrom = chroms[si]
    neighbors = [(ids[j], chroms[j], f"{sims_row[j]:.3f}") for j in top5]
    print(f"  {cpg} ({chrom}) → {neighbors}")

print(f"\nAll plots saved to {OUT_DIR}/")
