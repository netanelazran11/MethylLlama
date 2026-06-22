#!/usr/bin/env python3
"""
raw_umap.py  —  Figure 3 (d-f) equivalent
==========================================
Compute PCA + UMAP on the RAW methylation data matrix (no model involved).
Used as the COMPARISON baseline showing what the data looks like before
model processing — the "before" half of the MethylGPT Fig 3 comparison.

Replicates MethylGPT Figure 3 (d-f):
  d  — raw UMAP coloured by tissue   (less distinct than model)
  e  — raw UMAP coloured by batch/dataset  (stronger batch effects)
  f  — raw UMAP coloured by sex       (weaker separation than model)

Memory strategy for 169k × 49k matrix (33 GB):
  • Subsample n_samples (default 30k) for UMAP visualisation
  • IncrementalPCA in mini-batches if needed
  • NaN → column mean imputation before PCA

Usage:
  python scripts/repr_analysis/raw_umap.py \\
      --data     /path/to/methylgpt_pretrain_type3.h5ad \\
      --metadata data/pretrain_metadata.csv.gz \\
      --outdir   outputs/repr_analysis/raw_umap \\
      --n_samples 30000
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
sc.settings.verbosity = 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="Raw methylation UMAP (Fig 3 comparison baseline)")
    p.add_argument("--data",        required=True,  help="h5ad file (pretrain 169k)")
    p.add_argument("--metadata",    default=None,   help="pretrain_metadata.csv.gz (GSM_ID → tissue/sex/...)")
    p.add_argument("--metadata_id_col", default="GSM_ID")
    p.add_argument("--outdir",      default="outputs/repr_analysis/raw_umap")
    p.add_argument("--n_samples",   type=int, default=30000,
                   help="Subsample for UMAP (default 30k; use -1 for all)")
    p.add_argument("--n_pca",       type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--seed",        type=int, default=42)
    p.add_argument("--label_cols",  nargs="+", default=["tissue", "sex", "dataset"])
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Data loading (h5py fallback for old-format h5ad)
# ─────────────────────────────────────────────────────────────────────────────

def load_h5ad_matrix(h5ad_path: str):
    """
    Load X matrix and obs_names from h5ad.
    Falls back to h5py if anndata version mismatch (as in pretrain h5ad).
    Returns (X [N, M] float32, obs_names list[str]).
    """
    import anndata
    try:
        adata = anndata.read_h5ad(h5ad_path, backed="r")
        if adata.obs.shape[0] != adata.X.shape[0]:
            raise ValueError("obs rows != X rows")
        log.info(f"Loaded via anndata: {adata.shape}")
        X = adata.X[:]  # load into memory
        obs_names = list(adata.obs_names)
        return X.astype(np.float32), obs_names
    except Exception as e:
        log.warning(f"sc.read_h5ad failed ({e}). Falling back to h5py.")

    import h5py
    with h5py.File(h5ad_path, "r") as f:
        X = f["X"][:]
        obs_names = [s.decode() if isinstance(s, bytes) else s
                     for s in f["obs"]["_index"][:]]
    log.info(f"h5py fallback: X={X.shape}  obs={len(obs_names)}")
    return X.astype(np.float32), obs_names


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def impute_nan(X: np.ndarray) -> np.ndarray:
    """Column-mean imputation for NaN values (in-place)."""
    log.info("Imputing NaN values (column means) ...")
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        inds = np.where(nan_mask)
        X[inds] = col_means[inds[1]]
        log.info(f"  Imputed {nan_mask.sum():,} NaN entries")
    return X


def subsample(X: np.ndarray, obs_names: list, n: int, seed: int):
    """Random subsample of rows."""
    rng = np.random.default_rng(seed)
    idx = rng.choice(len(obs_names), size=min(n, len(obs_names)), replace=False)
    idx = np.sort(idx)
    log.info(f"Subsampled {len(idx):,} / {X.shape[0]:,} samples")
    return X[idx], [obs_names[i] for i in idx], idx


# ─────────────────────────────────────────────────────────────────────────────
# Metadata joining
# ─────────────────────────────────────────────────────────────────────────────

def load_metadata(meta_path: str, id_col: str, obs_names: list) -> pd.DataFrame:
    log.info(f"Loading metadata: {meta_path}")
    meta = pd.read_csv(meta_path)
    meta = meta.drop_duplicates(subset=id_col).set_index(id_col)
    df = pd.DataFrame(index=obs_names).join(meta, how="left")
    n_matched = df.notna().any(axis=1).sum()
    log.info(f"  Matched {n_matched:,} / {len(obs_names):,} samples ({100*n_matched/len(obs_names):.1f}%)")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# PCA + UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_pca_umap(X: np.ndarray, n_pca: int, n_neighbors: int) -> tuple[np.ndarray, np.ndarray]:
    """StandardScaler → PCA → neighbors → UMAP. Returns (pca_coords, umap_coords)."""
    from sklearn.preprocessing import StandardScaler

    log.info("StandardScaler ...")
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    log.info(f"PCA-{n_pca} ...")
    adata = sc.AnnData(X=X_scaled.astype(np.float32))
    n_pca_eff = min(n_pca, X.shape[1] - 1, X.shape[0] - 1)
    sc.tl.pca(adata, n_comps=n_pca_eff)

    log.info(f"Neighbors (k={n_neighbors}) + UMAP ...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(adata)

    return adata.obsm["X_pca"].astype(np.float32), adata.obsm["X_umap"].astype(np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_PALETTE = {
    "Whole Blood": "#E64B35", "Brain": "#4DBBD5", "Cells": "#00A087",
    "Liver": "#3C5488",       "Breast": "#F39B7F", "Skin": "#8491B4",
    "Lung": "#91D1C2",        "Colon": "#DC0000",  "Prostate": "#7E6148",
    "Other": "#B09C85",
}
SEX_PALETTE = {"M": "#4DBBD5", "F": "#E64B35", "Other": "#ADB6B6"}


def _get_palette(col: str, values):
    if col == "tissue":
        cats = sorted(set(str(v) for v in values if str(v) not in ("nan", "None")))
        palette = {}
        base = TISSUE_PALETTE.copy()
        cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
        for i, c in enumerate(cats):
            palette[c] = base.get(c, matplotlib.colors.to_hex(cmap(i)))
        return palette
    if col == "sex":
        return SEX_PALETTE
    cats = sorted(set(str(v) for v in values if str(v) not in ("nan", "None")))
    cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
    return {c: matplotlib.colors.to_hex(cmap(i)) for i, c in enumerate(cats)}


def _scatter_cat(ax, coords, labels, palette, title, s=2, alpha=0.4):
    cats = list(dict.fromkeys(str(l) for l in labels))
    for cat in cats:
        if cat in ("nan", "None"):
            continue
        mask = np.array([str(l) == cat for l in labels])
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=s, alpha=alpha, c=palette.get(cat, "#888888"),
                   label=cat, rasterized=True)
    n_cats = sum(1 for c in cats if c not in ("nan", "None"))
    if n_cats <= 20:
        ax.legend(markerscale=5, fontsize=7, framealpha=0.4,
                  loc="best", ncol=max(1, n_cats // 10))
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _scatter_density(ax, coords, title):
    """Batch density as a 2D hexbin (Fig 3b/e style)."""
    hb = ax.hexbin(coords[:, 0], coords[:, 1], gridsize=60, cmap="YlOrRd",
                   mincnt=1, linewidths=0.1)
    plt.colorbar(hb, ax=ax, label="Sample count", fraction=0.046, pad=0.04)
    ax.set_title(title, fontsize=10, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=8); ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def plot_raw_umaps(coords, meta_df, label_cols, outdir: Path):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Individual plots
    for col in label_cols:
        if col not in meta_df.columns:
            log.warning(f"  Column '{col}' not in metadata — skipping")
            continue
        labels = meta_df[col].astype(str).tolist()
        palette = _get_palette(col, labels)
        fig, ax = plt.subplots(figsize=(6.5, 5.5))
        _scatter_cat(ax, coords, labels, palette, f"Raw methylation | {col}")
        plt.tight_layout()
        out = fig_dir / f"umap_raw_{col}.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        log.info(f"  {out.name}")

    # Density / batch plot
    fig, ax = plt.subplots(figsize=(6, 5))
    _scatter_density(ax, coords, "Raw methylation | sample density (batch proxy)")
    plt.tight_layout()
    out = fig_dir / "umap_raw_density.png"
    plt.savefig(out, dpi=160, bbox_inches="tight")
    plt.close()
    log.info(f"  {out.name}")

    # 3-panel combined (mirrors MethylGPT Fig 3 d-f)
    available_cols = [c for c in label_cols if c in meta_df.columns]
    n = len(available_cols)
    if n >= 2:
        fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8))
        if n == 1:
            axes = [axes]
        for ax, col in zip(axes, available_cols):
            labels = meta_df[col].astype(str).tolist()
            palette = _get_palette(col, labels)
            _scatter_cat(ax, coords, labels, palette, f"Raw | {col}")
        plt.suptitle("Raw DNA Methylation Sample Space", fontsize=12, y=1.01)
        plt.tight_layout()
        out = fig_dir / "umap_raw_combined.png"
        plt.savefig(out, dpi=160, bbox_inches="tight")
        plt.close()
        log.info(f"  {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load data
    log.info("[1/4] Loading methylation matrix ...")
    X, obs_names = load_h5ad_matrix(args.data)
    log.info(f"  Loaded X: {X.shape}  obs: {len(obs_names):,}")

    # 2. Subsample
    if args.n_samples > 0 and args.n_samples < len(obs_names):
        X, obs_names, _ = subsample(X, obs_names, args.n_samples, args.seed)

    # 3. Impute NaN
    X = impute_nan(X)

    # 4. Load metadata
    meta_df = pd.DataFrame(index=obs_names)
    if args.metadata:
        log.info("[2/4] Loading metadata ...")
        meta_df = load_metadata(args.metadata, args.metadata_id_col, obs_names)

    # 5. PCA + UMAP
    log.info("[3/4] Running PCA + UMAP ...")
    pca_coords, umap_coords = run_pca_umap(X, args.n_pca, args.n_neighbors)

    # Save coordinates
    result_df = pd.DataFrame({
        "obs_name": obs_names,
        "PC1": pca_coords[:, 0], "PC2": pca_coords[:, 1],
        "UMAP1": umap_coords[:, 0], "UMAP2": umap_coords[:, 1],
    })
    for col in args.label_cols:
        if col in meta_df.columns:
            result_df[col] = meta_df[col].values
    result_df.to_csv(outdir / "raw_umap.csv", index=False)
    np.save(outdir / "raw_pca_coords.npy", pca_coords)
    np.save(outdir / "raw_umap_coords.npy", umap_coords)
    log.info(f"  Saved raw_umap.csv  ({len(result_df):,} samples)")

    # 6. Plots
    log.info("[4/4] Generating plots ...")
    plot_raw_umaps(umap_coords, meta_df, args.label_cols, outdir)

    log.info(f"\nDone. Outputs → {outdir}/")
    log.info(f"  raw_umap_coords.npy   [{umap_coords.shape[0]:,} × 2]")
    log.info(f"  figures/umap_raw_*.png")


if __name__ == "__main__":
    main()
