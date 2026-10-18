"""
Bonus 1 — CLS visualization (PCA + UMAP) colored by age / tissue / sex / dataset.

Pure post-processing on the saved matrices from extract_pretrain_cls.py — no GPU,
no checkpoint. Answers "does the CLS space cluster by biology?" visually.

Usage:
  python scripts/evaluation/visualize_cls.py --dir results/pretrained_cls
"""

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--dir", default="results/pretrained_cls")
    p.add_argument("--emb", default="embeddings_cls.npy")
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--color_cols", nargs="+", default=["age", "tissue_type", "sex", "dataset"])
    return p.parse_args()


def embed_2d(X):
    Xs = StandardScaler().fit_transform(X)
    pca50 = PCA(n_components=min(50, X.shape[1])).fit_transform(Xs)
    pca2 = pca50[:, :2]
    try:
        import umap
        um = umap.UMAP(n_neighbors=15, min_dist=0.1, random_state=0).fit_transform(pca50)
    except Exception as e:
        print(f"[umap unavailable: {e}] — using PCA for both panels")
        um = pca2
    return pca2, um


def plot(coords, meta, col, ax, title):
    if col not in meta:
        ax.set_visible(False)
        return
    vals = meta[col]
    if np.issubdtype(pd.to_numeric(vals, errors="coerce").dtype, np.number) and pd.to_numeric(vals, errors="coerce").notna().mean() > 0.9:
        c = pd.to_numeric(vals, errors="coerce").values
        sc = ax.scatter(coords[:, 0], coords[:, 1], c=c, s=4, cmap="viridis", alpha=0.6)
        plt.colorbar(sc, ax=ax, fraction=0.046)
    else:
        cats = pd.Series(vals.astype(str))
        top = cats.value_counts().head(12).index
        for cat in top:
            m = (cats == cat).values
            ax.scatter(coords[m, 0], coords[m, 1], s=4, alpha=0.6, label=str(cat)[:16])
        ax.legend(markerscale=3, fontsize=6, loc="best", framealpha=0.7)
    ax.set_title(title, fontsize=10)
    ax.set_xticks([]); ax.set_yticks([])


def main():
    a = parse_args()
    d = Path(a.dir)
    X = np.load(d / a.emb).astype(np.float64)
    meta = pd.read_csv(d / "metadata.csv")
    print(f"Loaded {X.shape} embeddings, {len(meta)} metadata rows")
    pca2, um = embed_2d(X)
    np.save(d / "cls_pca_coords.npy", pca2)
    np.save(d / "cls_umap_coords.npy", um)

    for name, coords in [("PCA", pca2), ("UMAP", um)]:
        cols = [c for c in a.color_cols if c in meta]
        fig, axes = plt.subplots(1, len(cols), figsize=(5 * len(cols), 4.5))
        if len(cols) == 1:
            axes = [axes]
        for ax, col in zip(axes, cols):
            plot(coords, meta, col, ax, f"{name} — {col}")
        fig.suptitle(f"MethylLlama pretrain CLS — {name}", fontsize=12, y=1.02)
        fig.tight_layout()
        out = d / f"cls_{name.lower()}_panels.png"
        fig.savefig(out, dpi=150, bbox_inches="tight")
        print(f"saved {out}")


if __name__ == "__main__":
    main()
