#!/usr/bin/env python3
"""
figure3_comparison.py
=====================
Publication-quality 6-panel UMAP figure comparing:
  Top row    : MethylLlama CLS embedding sample space
  Bottom row : Raw DNA methylation sample space

Panels:
  a / d  —  colored by tissue type
  b / e  —  colored by dataset (batch proxy)
  c / f  —  colored by sex

Both spaces use the SAME samples.

Usage:
  python scripts/repr_analysis/figure3_comparison.py \\
      --cls_npy      outputs/repr_analysis/cls_probing_44905909/embeddings_cls.npy \\
      --data         /path/to/finetuning_19608.h5ad \\
      --metadata_csv outputs/repr_analysis/cls_probing_44905909/metadata.csv \\
      --ext_metadata data/pretrain_metadata.csv.gz \\
      --ext_id_col   GSM_ID \\
      --outdir       outputs/repr_analysis/figure3
"""

import argparse
import logging
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np
import pandas as pd
import scanpy as sc
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
sc.settings.verbosity = 0


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cls_npy",      required=True,
                   help="Pre-computed CLS embeddings .npy [N, 256]")
    p.add_argument("--random_npy",   default=None,
                   help="Random-init CLS embeddings .npy [N, 256] (ablation)")
    p.add_argument("--data",         required=True,
                   help="h5ad file (finetune 19k) for raw methylation matrix")
    p.add_argument("--metadata_csv", required=True,
                   help="metadata.csv saved by cls_probing (obs aligned to cls_npy)")
    p.add_argument("--ext_metadata", default=None,
                   help="External metadata CSV.gz with tissue/sex columns")
    p.add_argument("--ext_id_col",   default="GSM_ID")
    p.add_argument("--outdir",       default="outputs/repr_analysis/figure3")
    p.add_argument("--n_pca",        type=int, default=50)
    p.add_argument("--n_neighbors",  type=int, default=15)
    p.add_argument("--min_dist",     type=float, default=0.1)
    p.add_argument("--seed",         type=int, default=42)
    p.add_argument("--max_nan_frac", type=float, default=0.2,
                   help="Drop CpG columns with more than this fraction NaN")
    p.add_argument("--dpi",          type=int, default=200)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Palettes
# ─────────────────────────────────────────────────────────────────────────────

TISSUE_COLORS = {
    # ── major categories (high N → need most distinct colors) ─────────────────
    "Whole Blood":        "#E64B35",   # vivid red
    "Brain":              "#4DBBD5",   # sky blue
    "Other":              "#AAAAAA",   # neutral gray
    "Cells":              "#9B59B6",   # purple
    "Breast":             "#F39B7F",   # salmon
    "Lung":               "#91D1C2",   # mint green
    "Colon":              "#C0392B",   # dark red
    "Liver":              "#3C5488",   # navy
    "Prostate":           "#7E6148",   # brown
    "Skin":               "#8491B4",   # slate blue
    "Testis":             "#27AE60",   # emerald
    "Ovary":              "#FF69B4",   # hot pink
    "Stomach":            "#E67E22",   # orange
    "Muscle":             "#00A087",   # teal
    "Kidney":             "#F4A460",   # sandy
    "Esophagus":          "#808000",   # olive
    "Pancreas":           "#F1C40F",   # gold
    "Adipose":            "#B09C85",   # warm beige
    "Bladder":            "#FA8072",   # coral
    "Uterus":             "#C39BD3",   # light purple
    "Cervix":             "#DDA0DD",   # plum
    "Thyroid":            "#5B2C6F",   # deep violet
    "Adrenal Gland":      "#D35400",   # burnt orange
    "Nerve":              "#F7DC6F",   # pale yellow
    "Small Intestine":    "#7DCEA0",   # sage
    "Heart":              "#922B21",   # crimson
    "Minor Salivary Gland": "#708090", # slate gray
    "Artery":             "#FF4500",   # orange-red
    "Pituitary":          "#98FB98",   # mint
    "Fallopian Tube":     "#FF91A4",   # rose
    "Spleen":             "#6B8E23",   # olive green
    "Vagina":             "#FFDAB9",   # peach
    # alias
    "Blood":              "#E64B35",
}

SEX_COLORS   = {"M": "#4DBBD5", "F": "#E64B35", "m": "#4DBBD5", "f": "#E64B35",
                "male": "#4DBBD5", "female": "#E64B35", "Other": "#AAAAAA"}

PANEL_LABELS = list("abcdef")


# ─────────────────────────────────────────────────────────────────────────────
# Data loading
# ─────────────────────────────────────────────────────────────────────────────

def load_data(args) -> dict:
    """Load CLS embeddings, raw methylation matrix, and metadata."""

    # ── CLS embeddings ────────────────────────────────────────────────────────
    log.info(f"Loading CLS embeddings: {args.cls_npy}")
    cls = np.load(args.cls_npy).astype(np.float32)
    log.info(f"  CLS shape: {cls.shape}")

    # ── Metadata (aligned to cls rows) ────────────────────────────────────────
    log.info(f"Loading metadata: {args.metadata_csv}")
    meta = pd.read_csv(args.metadata_csv, index_col=0)
    log.info(f"  Metadata: {meta.shape}  columns: {list(meta.columns)}")

    if len(meta) != len(cls):
        raise ValueError(f"Metadata rows ({len(meta)}) ≠ CLS rows ({cls.shape[0]})")

    # Join external metadata for tissue / sex
    if args.ext_metadata and Path(args.ext_metadata).exists():
        log.info(f"  Joining external metadata: {args.ext_metadata}")
        ext = pd.read_csv(args.ext_metadata)
        ext = ext.drop_duplicates(subset=args.ext_id_col).set_index(args.ext_id_col)
        join_cols = [c for c in ["tissue", "sex", "disease"] if c in ext.columns]
        meta = meta.join(ext[join_cols], how="left")
        for col in join_cols:
            n = meta[col].notna().sum()
            log.info(f"    {col}: {n:,} / {len(meta):,} matched")

    sample_ids = list(meta.index)

    # ── Raw methylation matrix ────────────────────────────────────────────────
    log.info(f"Loading raw methylation: {args.data}")
    X_raw, obs_names_raw = _load_h5ad_matrix(args.data)
    log.info(f"  Raw shape: {X_raw.shape}  obs: {len(obs_names_raw):,}")

    # Align raw matrix to same samples as CLS (same order)
    log.info("  Aligning raw matrix to CLS sample order ...")
    obs_index = {name: i for i, name in enumerate(obs_names_raw)}
    common = [sid for sid in sample_ids if sid in obs_index]
    log.info(f"  Common samples: {len(common):,} / {len(sample_ids):,}")

    if len(common) < len(sample_ids) * 0.5:
        raise ValueError(
            f"Too few common samples ({len(common)}). Check that obs_names in h5ad match metadata index."
        )

    cls_mask  = np.array([sid in obs_index for sid in sample_ids])
    raw_idx   = np.array([obs_index[sid] for sid in sample_ids if sid in obs_index])

    cls_aligned  = cls[cls_mask]
    raw_aligned  = X_raw[raw_idx]
    meta_aligned = meta[cls_mask].reset_index(drop=False)

    log.info(f"  Aligned: {cls_aligned.shape[0]:,} samples")

    # ── Random embeddings (optional) ─────────────────────────────────────────
    random_aligned = None
    if args.random_npy and Path(args.random_npy).exists():
        log.info(f"Loading random CLS embeddings: {args.random_npy}")
        rnd = np.load(args.random_npy).astype(np.float32)
        log.info(f"  shape: {rnd.shape}")
        if rnd.shape[0] != len(meta):
            raise ValueError(f"random_npy rows ({rnd.shape[0]}) ≠ metadata rows ({len(meta)})")
        random_aligned = rnd[cls_mask]
        log.info(f"  Random CLS aligned: {random_aligned.shape}")

    return {
        "cls":    cls_aligned,
        "random": random_aligned,
        "raw":    raw_aligned,
        "meta":   meta_aligned,
        "n":      cls_aligned.shape[0],
    }


def _load_h5ad_matrix(path: str):
    import anndata
    try:
        adata = anndata.read_h5ad(path, backed="r")
        X = adata.X[:]
        obs_names = list(adata.obs_names)
        log.info(f"  anndata read: {adata.shape}")
        return X.astype(np.float32), obs_names
    except Exception as e:
        log.warning(f"  anndata failed ({e}), trying h5py ...")

    import h5py
    with h5py.File(path, "r") as f:
        X = f["X"][:]
        obs_names = [s.decode() if isinstance(s, bytes) else s
                     for s in f["obs"]["_index"][:]]
    log.info(f"  h5py read: {X.shape}")
    return X.astype(np.float32), obs_names


# ─────────────────────────────────────────────────────────────────────────────
# Preprocessing
# ─────────────────────────────────────────────────────────────────────────────

def preprocess_raw_methylation(X: np.ndarray, max_nan_frac: float = 0.2) -> np.ndarray:
    """Drop high-NaN CpGs → impute remaining → standardize."""
    log.info(f"Preprocessing raw methylation {X.shape} ...")

    # Drop columns with too many NaN
    nan_frac = np.isnan(X).mean(axis=0)
    keep = nan_frac <= max_nan_frac
    X = X[:, keep]
    log.info(f"  Kept {keep.sum():,} / {len(keep):,} CpGs (max NaN frac={max_nan_frac})")

    # Column-mean imputation
    col_means = np.nanmean(X, axis=0)
    nan_mask = np.isnan(X)
    if nan_mask.any():
        inds = np.where(nan_mask)
        X[inds] = col_means[inds[1]]
        log.info(f"  Imputed {nan_mask.sum():,} NaN entries")

    # Standardize
    log.info("  StandardScaler ...")
    X = StandardScaler().fit_transform(X).astype(np.float32)
    return X


# ─────────────────────────────────────────────────────────────────────────────
# UMAP computation
# ─────────────────────────────────────────────────────────────────────────────

def compute_umap(X: np.ndarray, name: str, n_pca: int,
                 n_neighbors: int, min_dist: float, seed: int,
                 run_pca: bool = True) -> np.ndarray:
    """PCA (optional) → neighbors → UMAP. Returns [N, 2] UMAP coords."""
    log.info(f"[{name}] Computing UMAP (n={X.shape[0]:,}, d={X.shape[1]}) ...")
    adata = sc.AnnData(X=X.astype(np.float32))

    if run_pca:
        n_pca_eff = min(n_pca, X.shape[1] - 1, X.shape[0] - 1)
        log.info(f"  PCA-{n_pca_eff} ...")
        sc.tl.pca(adata, n_comps=n_pca_eff)
        use_rep = "X_pca"
    else:
        use_rep = "X"

    log.info(f"  Neighbors (k={n_neighbors}, metric=cosine) ...")
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep=use_rep,
                    metric="cosine", random_state=seed)

    log.info("  UMAP ...")
    sc.tl.umap(adata, min_dist=min_dist, random_state=seed)

    coords = adata.obsm["X_umap"].astype(np.float32)
    log.info(f"  [{name}] UMAP done → {coords.shape}")
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

def _tissue_palette(values):
    cats = sorted(set(str(v) for v in values if str(v) not in ("nan", "None", "unknown")))
    cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
    palette = {}
    for i, c in enumerate(cats):
        palette[c] = TISSUE_COLORS.get(c, matplotlib.colors.to_hex(cmap(i)))
    return palette


def _dataset_palette(values):
    cats = sorted(set(str(v) for v in values if str(v) not in ("nan", "None")))
    cmap = plt.cm.get_cmap("tab20", max(len(cats), 1))
    return {c: matplotlib.colors.to_hex(cmap(i)) for i, c in enumerate(cats)}


def _scatter(ax, coords: np.ndarray, labels, palette: dict,
             title: str, s: float = 2.5, alpha: float = 0.45,
             max_legend: int = 25):
    """Scatter plot one panel."""
    cats = list(dict.fromkeys(str(l) for l in labels))
    plotted = []
    for cat in cats:
        if cat in ("nan", "None", "unknown"):
            continue
        mask = np.array([str(l) == cat for l in labels])
        color = palette.get(cat, "#AAAAAA")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=s, alpha=alpha, c=color, linewidths=0,
                   rasterized=True)
        plotted.append((cat, color))

    ax.set_title(title, fontsize=11, fontweight="bold", pad=6)
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)
    for spine in ax.spines.values():
        spine.set_linewidth(0.5)

    # Legend (tissue names on side)
    if 0 < len(plotted) <= max_legend:
        handles = [mpatches.Patch(color=c, label=l) for l, c in plotted]
        ncol = 1 if len(plotted) <= 12 else 2
        ax.legend(handles=handles, fontsize=7, markerscale=1,
                  loc="lower right", framealpha=0.6,
                  handlelength=1.2, handleheight=1.0,
                  ncol=ncol, borderpad=0.5, labelspacing=0.3)


def plot_embedding_panels(cls_coords: np.ndarray, raw_coords: np.ndarray,
                          meta: pd.DataFrame, outdir: Path, dpi: int = 200,
                          random_coords: np.ndarray = None):
    """Create the full comparison figure (2 or 3 rows × 2 cols: tissue | sex)."""
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    # Resolve labels
    tissue_labels = (meta["tissue"].fillna("unknown").tolist()
                     if "tissue" in meta.columns else ["unknown"] * len(meta))
    sex_labels    = (meta["sex"].fillna("unknown").tolist()
                     if "sex" in meta.columns else ["unknown"] * len(meta))

    tissue_pal = _tissue_palette(tissue_labels)
    sex_pal    = {**SEX_COLORS, **{k: "#AAAAAA" for k in set(sex_labels)
                                   if k not in SEX_COLORS}}

    # Build rows: pretrained CLS | (optional) random CLS | raw methylation
    rows = [
        ("MethylLlama CLS embedding space",  cls_coords),
    ]
    if random_coords is not None:
        rows.append(("Random-init CLS (baseline)",        random_coords))
    rows.append(("Raw DNA methylation space",             raw_coords))

    n_rows = len(rows)
    fig_h  = 5.5 * n_rows
    fig = plt.figure(figsize=(14, fig_h))
    fig.patch.set_facecolor("white")

    top_margin    = 0.97
    bottom_margin = 0.03
    row_gap       = 0.06
    usable        = top_margin - bottom_margin
    row_h         = (usable - row_gap * (n_rows - 1)) / n_rows

    for r_idx, (row_title, coords) in enumerate(rows):
        top    = top_margin - r_idx * (row_h + row_gap)
        bottom = top - row_h
        label_top = top + 0.005

        # Row label
        fig.text(0.5, label_top, row_title,
                 ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color="#1E3A5F")
        fig.add_artist(plt.Line2D([0.04, 0.96], [label_top - 0.005, label_top - 0.005],
                                  transform=fig.transFigure,
                                  color="#1E3A5F", linewidth=1.0, alpha=0.4))

        inner_top    = label_top - 0.022
        inner_bottom = bottom
        inner_h      = inner_top - inner_bottom
        panel_h      = inner_h
        gap_w        = 0.04

        # Tissue panel (left)
        ax_t = fig.add_axes([0.04, inner_bottom, 0.44, panel_h])
        panel_letter = chr(ord('a') + r_idx * 2)
        _scatter(ax_t, coords, tissue_labels, tissue_pal,
                 f"{panel_letter}  |  Tissue type")

        # Sex panel (right)
        ax_s = fig.add_axes([0.52, inner_bottom, 0.44, panel_h])
        panel_letter = chr(ord('a') + r_idx * 2 + 1)
        _scatter(ax_s, coords, sex_labels, sex_pal,
                 f"{panel_letter}  |  Sex", max_legend=5)

    # Save PNG + PDF
    for ext in ["png", "pdf"]:
        out = fig_dir / f"figure3_comparison.{ext}"
        fig.savefig(out, dpi=dpi if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        log.info(f"  Saved → {out}")
    plt.close()

    # ── Individual panels ─────────────────────────────────────────────────────
    _save_individual(cls_coords, raw_coords, tissue_labels, sex_labels,
                     tissue_pal, sex_pal, fig_dir, dpi, random_coords)


def _save_individual(cls_c, raw_c, tissue_l, sex_l,
                     t_pal, s_pal, fig_dir, dpi, random_c=None):
    configs = [
        ("cls_tissue",    cls_c,    tissue_l, t_pal, "MethylLlama CLS | Tissue"),
        ("cls_sex",       cls_c,    sex_l,    s_pal, "MethylLlama CLS | Sex"),
        ("raw_tissue",    raw_c,    tissue_l, t_pal, "Raw Methylation | Tissue"),
        ("raw_sex",       raw_c,    sex_l,    s_pal, "Raw Methylation | Sex"),
    ]
    if random_c is not None:
        configs += [
            ("random_tissue", random_c, tissue_l, t_pal, "Random-init CLS | Tissue"),
            ("random_sex",    random_c, sex_l,    s_pal, "Random-init CLS | Sex"),
        ]
    for fname, coords, labels, palette, title in configs:
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter(ax, coords, labels, palette, title, s=3, alpha=0.5)
        plt.tight_layout()
        fig.savefig(fig_dir / f"{fname}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Load + align all data
    log.info("=" * 60)
    log.info(" Embedding Space Comparison: MethylLlama CLS vs Raw Methylation")
    log.info("=" * 60)
    data = load_data(args)

    cls_emb    = data["cls"]     # [N, 256]
    random_emb = data["random"]  # [N, 256] or None
    raw_mat    = data["raw"]     # [N, n_cpg]
    meta       = data["meta"]

    log.info(f"\nAligned dataset: {data['n']:,} samples")
    log.info(f"  CLS   : {cls_emb.shape}")
    log.info(f"  Random: {random_emb.shape if random_emb is not None else 'not loaded'}")
    log.info(f"  Raw   : {raw_mat.shape}")

    # 2. Preprocess raw methylation
    log.info("\n[2/4] Preprocessing raw methylation ...")
    raw_clean = preprocess_raw_methylation(raw_mat, max_nan_frac=args.max_nan_frac)

    # 3. Compute UMAPs
    n_umaps = 3 if random_emb is not None else 2
    log.info(f"\n[3/4] Computing {n_umaps} UMAPs ...")

    cls_umap = compute_umap(cls_emb, name="CLS",
                            n_pca=args.n_pca, n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist, seed=args.seed,
                            run_pca=False)   # 256D — skip PCA

    random_umap = None
    if random_emb is not None:
        random_umap = compute_umap(random_emb, name="Random",
                                   n_pca=args.n_pca, n_neighbors=args.n_neighbors,
                                   min_dist=args.min_dist, seed=args.seed,
                                   run_pca=False)   # same 256D — skip PCA

    raw_umap = compute_umap(raw_clean, name="Raw",
                            n_pca=args.n_pca, n_neighbors=args.n_neighbors,
                            min_dist=args.min_dist, seed=args.seed,
                            run_pca=True)    # high-dim — PCA first

    # Save coords
    np.save(outdir / "cls_umap_coords.npy", cls_umap)
    np.save(outdir / "raw_umap_coords.npy", raw_umap)
    if random_umap is not None:
        np.save(outdir / "random_umap_coords.npy", random_umap)
    meta.to_csv(outdir / "aligned_metadata.csv", index=False)
    log.info(f"  Saved UMAP coordinates and metadata")

    # 4. Plot
    log.info("\n[4/4] Generating figure ...")
    plot_embedding_panels(cls_umap, raw_umap, meta, outdir,
                          dpi=args.dpi, random_coords=random_umap)

    log.info("\n" + "=" * 60)
    log.info(f" DONE  →  {outdir}/figures/figure3_comparison.png")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
