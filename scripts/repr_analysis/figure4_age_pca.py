#!/usr/bin/env python3
"""
figure4_age_pca.py
==================
PCA visualization: CLS embedding space colored by age — before vs after fine-tuning.

EXACT DATA / MODEL MAPPING
---------------------------
BEFORE fine-tuning:
  Model    : WCED pretrained checkpoint  (trained on 169k pretrain data)
  Data     : 19k finetune h5ad           (inference only — model never trained on this)
  Embedding: --pretrained_npy            (row-aligned to --pretrained_meta)

AFTER fine-tuning:
  Model    : MethylationAgeRegressorLlama (encoder unfrozen at epoch 10, updated 127 epochs)
  Data     : same 19k finetune h5ad      (same samples, same h5ad file)
  Embedding: --finetuned_npy             (row-aligned to --finetuned_meta)

The two embedding files likely have DIFFERENT row counts / orderings.
This script aligns them by SAMPLE ID (index of metadata CSVs) — safe and correct.

Output: 2-row × 2-col figure
  Row 1: Pretrained CLS — a: age (continuous)  b: tissue
  Row 2: Fine-tuned CLS — c: age (continuous)  d: tissue

Usage:
  python scripts/repr_analysis/figure4_age_pca.py \\
      --pretrained_npy   outputs/repr_analysis/cls_probing_44905909/embeddings_cls.npy \\
      --pretrained_meta  outputs/repr_analysis/cls_probing_44905909/metadata.csv \\
      --finetuned_npy    outputs/repr_analysis/cls_probing_finetune_JOBID/embeddings_cls.npy \\
      --finetuned_meta   outputs/repr_analysis/cls_probing_finetune_JOBID/metadata.csv \\
      --ext_metadata     data/pretrain_metadata.csv.gz \\
      --outdir           outputs/repr_analysis/figure4
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
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO,
                    format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)


TISSUE_COLORS = {
    "Whole Blood":          "#E64B35",
    "Brain":                "#4DBBD5",
    "Other":                "#AAAAAA",
    "Cells":                "#9B59B6",
    "Breast":               "#F39B7F",
    "Lung":                 "#91D1C2",
    "Colon":                "#C0392B",
    "Liver":                "#3C5488",
    "Prostate":             "#7E6148",
    "Skin":                 "#8491B4",
    "Testis":               "#27AE60",
    "Ovary":                "#FF69B4",
    "Stomach":              "#E67E22",
    "Muscle":               "#00A087",
    "Kidney":               "#F4A460",
    "Esophagus":            "#808000",
    "Pancreas":             "#F1C40F",
    "Adipose":              "#B09C85",
    "Bladder":              "#FA8072",
    "Uterus":               "#C39BD3",
    "Cervix":               "#DDA0DD",
    "Thyroid":              "#5B2C6F",
    "Adrenal Gland":        "#D35400",
    "Nerve":                "#F7DC6F",
    "Small Intestine":      "#7DCEA0",
    "Heart":                "#922B21",
    "Minor Salivary Gland": "#708090",
    "Artery":               "#FF4500",
    "Pituitary":            "#98FB98",
    "Fallopian Tube":       "#FF91A4",
    "Spleen":               "#6B8E23",
    "Vagina":               "#FFDAB9",
    "Blood":                "#E64B35",
}


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────
def parse_args():
    p = argparse.ArgumentParser()
    # Pretrained embeddings + their own metadata
    p.add_argument("--pretrained_npy",  required=True,
                   help="CLS from WCED pretrained model [N_pre, 256]")
    p.add_argument("--pretrained_meta", required=True,
                   help="metadata.csv for pretrained_npy (index = sample IDs)")
    # Fine-tuned embeddings + their own metadata
    p.add_argument("--finetuned_npy",   required=True,
                   help="CLS from fine-tuned model [N_ft, 256]")
    p.add_argument("--finetuned_meta",  required=True,
                   help="metadata.csv for finetuned_npy (index = sample IDs)")
    # Labels
    p.add_argument("--ext_metadata",    default=None,
                   help="External metadata CSV.gz for tissue labels (joined on sample ID)")
    p.add_argument("--ext_id_col",      default="GSM_ID")
    p.add_argument("--age_col",         default="age")
    # Output
    p.add_argument("--outdir",          default="outputs/repr_analysis/figure4")
    p.add_argument("--dpi",             type=int, default=200)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Load one embedding set with its metadata → DataFrame indexed by sample ID
# ─────────────────────────────────────────────────────────────────────────────
def _load_one(npy_path: str, meta_path: str, label: str) -> tuple:
    log.info(f"\n[{label}]")
    log.info(f"  npy  : {npy_path}")
    log.info(f"  meta : {meta_path}")

    emb  = np.load(npy_path).astype(np.float32)
    meta = pd.read_csv(meta_path, index_col=0)

    log.info(f"  emb shape : {emb.shape}")
    log.info(f"  meta shape: {meta.shape}  columns: {list(meta.columns)}")
    log.info(f"  index sample: {list(meta.index[:3])}")

    if emb.shape[0] != len(meta):
        raise ValueError(
            f"[{label}] embedding rows ({emb.shape[0]}) ≠ metadata rows ({len(meta)}). "
            f"The npy and metadata.csv must come from the SAME extraction run."
        )

    return emb, meta


# ─────────────────────────────────────────────────────────────────────────────
# Join tissue labels from external metadata
# ─────────────────────────────────────────────────────────────────────────────
def _join_tissue(meta: pd.DataFrame, ext_meta_path: str, ext_id_col: str) -> pd.DataFrame:
    if "tissue" in meta.columns:
        return meta
    if not ext_meta_path or not Path(ext_meta_path).exists():
        log.warning("  No ext_metadata — tissue labels unavailable")
        meta["tissue"] = "unknown"
        return meta
    ext = pd.read_csv(ext_meta_path)
    ext = ext.drop_duplicates(subset=ext_id_col).set_index(ext_id_col)
    cols = [c for c in ["tissue", "sex"] if c in ext.columns]
    meta = meta.join(ext[cols], how="left")
    n = meta["tissue"].notna().sum() if "tissue" in meta.columns else 0
    log.info(f"  tissue labels joined: {n:,}/{len(meta):,}")
    return meta


# ─────────────────────────────────────────────────────────────────────────────
# Align two embedding sets by common sample IDs
# ─────────────────────────────────────────────────────────────────────────────
def align_by_sample_id(pre_emb, pre_meta, ft_emb, ft_meta):
    pre_ids = set(pre_meta.index)
    ft_ids  = set(ft_meta.index)
    common  = sorted(pre_ids & ft_ids)   # sorted for reproducibility

    log.info(f"\nAlignment by sample ID:")
    log.info(f"  Pretrained samples : {len(pre_ids):,}")
    log.info(f"  Fine-tuned samples : {len(ft_ids):,}")
    log.info(f"  Common samples     : {len(common):,}")
    log.info(f"  Only in pretrained : {len(pre_ids - ft_ids):,}")
    log.info(f"  Only in fine-tuned : {len(ft_ids - pre_ids):,}")

    if len(common) < 100:
        raise ValueError(
            f"Only {len(common)} common samples between pretrained and fine-tuned metadata. "
            f"Check that both runs used the same h5ad file."
        )

    # Build positional index maps
    pre_id2row = {sid: i for i, sid in enumerate(pre_meta.index)}
    ft_id2row  = {sid: i for i, sid in enumerate(ft_meta.index)}

    pre_idx = np.array([pre_id2row[sid] for sid in common])
    ft_idx  = np.array([ft_id2row[sid]  for sid in common])

    pre_emb_aligned  = pre_emb[pre_idx]
    ft_emb_aligned   = ft_emb[ft_idx]
    pre_meta_aligned = pre_meta.iloc[pre_idx].copy()

    log.info(f"  Aligned shapes — pretrained: {pre_emb_aligned.shape}  fine-tuned: {ft_emb_aligned.shape}")
    return pre_emb_aligned, ft_emb_aligned, pre_meta_aligned


# ─────────────────────────────────────────────────────────────────────────────
# PCA
# ─────────────────────────────────────────────────────────────────────────────
def run_pca(emb: np.ndarray, name: str) -> tuple:
    log.info(f"[{name}] PCA {emb.shape} → 2D ...")
    X = StandardScaler().fit_transform(emb)
    pca = PCA(n_components=2, random_state=42)
    coords = pca.fit_transform(X).astype(np.float32)
    var = pca.explained_variance_ratio_
    log.info(f"  PC1={var[0]*100:.1f}%  PC2={var[1]*100:.1f}%  total={sum(var)*100:.1f}%")
    return coords, var


# ─────────────────────────────────────────────────────────────────────────────
# Plot helpers
# ─────────────────────────────────────────────────────────────────────────────
def _style_ax(ax):
    ax.set_facecolor("#F7F7F7")
    ax.grid(True, color="white", linewidth=0.8, alpha=1.0, zorder=0)
    for sp in ax.spines.values():
        sp.set_linewidth(0.6)
        sp.set_color("#AAAAAA")
    ax.tick_params(left=False, bottom=False, labelleft=False, labelbottom=False)


def _scatter_age(ax, coords, ages, var, title):
    _style_ax(ax)
    valid = ~np.isnan(ages)
    sc = ax.scatter(coords[valid, 0], coords[valid, 1],
                    c=ages[valid], cmap="coolwarm", vmin=0, vmax=100,
                    s=9, alpha=0.70, linewidths=0, rasterized=True, zorder=2)
    if (~valid).sum() > 0:
        ax.scatter(coords[~valid, 0], coords[~valid, 1],
                   c="#CCCCCC", s=4, alpha=0.25, linewidths=0, rasterized=True, zorder=1)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel(f"PC1 ({var[0]*100:.1f}%)", fontsize=9)
    ax.set_ylabel(f"PC2 ({var[1]*100:.1f}%)", fontsize=9)
    return sc


def _scatter_tissue(ax, coords, tissue_labels, title):
    _style_ax(ax)
    cats = [t for t in dict.fromkeys(tissue_labels)
            if t not in ("unknown", "nan", "None", float("nan"))]
    for cat in cats:
        mask  = np.array([str(t) == str(cat) for t in tissue_labels])
        color = TISSUE_COLORS.get(cat, "#AAAAAA")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   c=color, s=9, alpha=0.70, linewidths=0, rasterized=True, zorder=2)
    ax.set_title(title, fontsize=11, fontweight="bold", pad=5)
    ax.set_xlabel("PC1", fontsize=9)
    ax.set_ylabel("PC2", fontsize=9)
    handles = [mpatches.Patch(color=TISSUE_COLORS.get(c, "#AAAAAA"), label=c)
               for c in cats if c in TISSUE_COLORS]
    if handles:
        ncol = 1 if len(handles) <= 12 else 2
        ax.legend(handles=handles, fontsize=6.5, loc="lower right",
                  framealpha=0.75, ncol=ncol, handlelength=1.2,
                  borderpad=0.4, labelspacing=0.25, edgecolor="#CCCCCC")


# ─────────────────────────────────────────────────────────────────────────────
# Main figure
# ─────────────────────────────────────────────────────────────────────────────
def make_figure(pre_coords, pre_var, ft_coords, ft_var,
                ages, tissue_labels, outdir: Path, dpi: int):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    fig = plt.figure(figsize=(14, 11))
    fig.patch.set_facecolor("white")

    rows = [
        ("Pretrained CLS  (before fine-tuning)", pre_coords, pre_var),
        ("Fine-tuned CLS  (after fine-tuning)",  ft_coords,  ft_var),
    ]

    top_m, bot_m, gap = 0.96, 0.08, 0.07
    row_h = (top_m - bot_m - gap) / 2

    cbar_ax = fig.add_axes([0.47, bot_m, 0.015, top_m - bot_m])
    age_sc  = None

    for r, (row_title, coords, var) in enumerate(rows):
        top    = top_m - r * (row_h + gap)
        bottom = top - row_h
        inner  = top - 0.025
        h      = inner - bottom

        fig.text(0.5, top + 0.005, row_title,
                 ha="center", va="bottom", fontsize=12, fontweight="bold",
                 color="#1E3A5F")
        fig.add_artist(plt.Line2D([0.04, 0.96], [top, top],
                                  transform=fig.transFigure,
                                  color="#1E3A5F", linewidth=1.0, alpha=0.4))

        ax_age = fig.add_axes([0.05, bottom, 0.39, h])
        letter = chr(ord('a') + r * 2)
        sc = _scatter_age(ax_age, coords, ages, var,
                          f"{letter}  |  by Age (years)")
        if r == 0:
            age_sc = sc

        ax_tis = fig.add_axes([0.52, bottom, 0.44, h])
        letter = chr(ord('a') + r * 2 + 1)
        _scatter_tissue(ax_tis, coords, tissue_labels,
                        f"{letter}  |  by Tissue")

    if age_sc is not None:
        cbar = fig.colorbar(age_sc, cax=cbar_ax, ticks=[0, 25, 50, 75, 100])
        cbar.set_label("Age (years)", fontsize=9)
        cbar.ax.tick_params(labelsize=8)
        cbar.ax.set_yticklabels(["0", "25", "50", "75", "100"])

    for ext in ["png", "pdf"]:
        out = fig_dir / f"figure4_age_pca.{ext}"
        fig.savefig(out, dpi=dpi if ext == "png" else 72,
                    bbox_inches="tight", facecolor="white")
        log.info(f"  Saved → {out}")
    plt.close()

    # Individual panels
    for fname, coords, var, mode in [
        ("pretrained_age",    pre_coords, pre_var, "age"),
        ("pretrained_tissue", pre_coords, pre_var, "tissue"),
        ("finetuned_age",     ft_coords,  ft_var,  "age"),
        ("finetuned_tissue",  ft_coords,  ft_var,  "tissue"),
    ]:
        fig2, ax2 = plt.subplots(figsize=(7, 6))
        if mode == "age":
            sc2 = _scatter_age(ax2, coords, ages, var, fname.replace("_", " ").title())
            plt.colorbar(sc2, ax=ax2, label="Age (years)")
        else:
            _scatter_tissue(ax2, coords, tissue_labels, fname.replace("_", " ").title())
        plt.tight_layout()
        fig2.savefig(fig_dir / f"{fname}.png", dpi=dpi, bbox_inches="tight")
        plt.close()


# ─────────────────────────────────────────────────────────────────────────────
# Entry point
# ─────────────────────────────────────────────────────────────────────────────
def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    log.info("=" * 60)
    log.info(" Figure 4: CLS Space — Before vs After Fine-tuning")
    log.info("=" * 60)
    log.info(f" BEFORE FT: {args.pretrained_npy}")
    log.info(f"            meta: {args.pretrained_meta}")
    log.info(f" AFTER  FT: {args.finetuned_npy}")
    log.info(f"            meta: {args.finetuned_meta}")
    log.info("=" * 60)

    # 1. Load each embedding with its OWN metadata
    pre_emb, pre_meta = _load_one(args.pretrained_npy, args.pretrained_meta, "PRETRAINED")
    ft_emb,  ft_meta  = _load_one(args.finetuned_npy,  args.finetuned_meta,  "FINE-TUNED")

    # 2. Join tissue labels
    pre_meta = _join_tissue(pre_meta, args.ext_metadata, args.ext_id_col)
    ft_meta  = _join_tissue(ft_meta,  args.ext_metadata, args.ext_id_col)

    # 3. Align by common sample IDs — safe regardless of row order or count
    pre_emb, ft_emb, meta_aligned = align_by_sample_id(pre_emb, pre_meta, ft_emb, ft_meta)

    # 4. Extract labels from aligned metadata
    ages = pd.to_numeric(
        meta_aligned[args.age_col] if args.age_col in meta_aligned.columns
        else pd.Series([float("nan")] * len(meta_aligned)),
        errors="coerce"
    ).values
    tissue_labels = (meta_aligned["tissue"].fillna("unknown").tolist()
                     if "tissue" in meta_aligned.columns
                     else ["unknown"] * len(meta_aligned))

    n_age = (~np.isnan(ages)).sum()
    log.info(f"\nFinal dataset: {len(meta_aligned):,} aligned samples")
    log.info(f"  Age labels   : {n_age:,} valid")
    log.info(f"  Tissue labels: {sum(1 for t in tissue_labels if t != 'unknown'):,} valid")

    # 5. PCA — independent for each embedding
    log.info("\n[2/3] Running PCA ...")
    pre_coords, pre_var = run_pca(pre_emb,  "Pretrained")
    ft_coords,  ft_var  = run_pca(ft_emb,   "Fine-tuned")

    np.save(outdir / "pretrained_pca_coords.npy", pre_coords)
    np.save(outdir / "finetuned_pca_coords.npy",  ft_coords)
    meta_aligned.to_csv(outdir / "aligned_metadata.csv")

    # 6. Figure
    log.info("\n[3/3] Generating figure ...")
    make_figure(pre_coords, pre_var, ft_coords, ft_var,
                ages, tissue_labels, outdir, args.dpi)

    log.info("\n" + "=" * 60)
    log.info(f" DONE → {outdir}/figures/figure4_age_pca.png")
    log.info("=" * 60)


if __name__ == "__main__":
    main()
