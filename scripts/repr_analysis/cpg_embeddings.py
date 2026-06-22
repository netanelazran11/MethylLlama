#!/usr/bin/env python3
"""
cpg_embeddings.py  —  Figure 2 equivalent
==========================================
Extract the learned CpG site embedding table from the WCED pretrain checkpoint
and visualise it in 2-D (PCA → UMAP), coloured by genomic annotations.

Replicates MethylGPT Figure 2 (b-d):
  b  — UMAP coloured by CpG island relation  (Island / Shore / Shelf / OpenSea)
  c  — UMAP coloured by enhancer region       (Yes / No)
  d  — UMAP coloured by chromosomal location  (Autosome / Sex chromosome)

What is extracted:
  encoder.embeddings.cpg_sites_embeddings.weight  [49156, 256]
  — the learned lookup table that maps each CpG token → 256-dim vector.
  This is a STATIC table (not context-dependent), reflecting what the model
  learned about each CpG's identity during WCED pretraining.

Usage:
  python scripts/repr_analysis/cpg_embeddings.py \\
      --checkpoint outputs/pretrain-llama-wced/.../epoch=98-val_loss=0.0059.ckpt \\
      --tokenizer  tokenizer_llama_pretrain49k \\
      --manifest   outputs/cpg_manifest/cpg_annotations_tokenizer49k.tsv \\
      --outdir     outputs/repr_analysis/cpg_embeddings
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
import torch

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
sc.settings.verbosity = 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="CpG embedding table UMAP (Fig 2)")
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--tokenizer",  required=True,
                   help="tokenizer_llama_pretrain49k directory")
    p.add_argument("--manifest",   default=None,
                   help="cpg_annotations_tokenizer49k.tsv from align_cpg_manifest.py. "
                        "If omitted, only saves embeddings (no annotation coloring).")
    p.add_argument("--outdir",     default="outputs/repr_analysis/cpg_embeddings")
    p.add_argument("--n_pca",      type=int, default=50)
    p.add_argument("--n_neighbors",type=int, default=15)
    p.add_argument("--min_dist",   type=float, default=0.1)
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_cpg_embeddings(checkpoint_path: str, tokenizer_path: str):
    """
    Load WCED pretrain checkpoint and extract the CpG embedding lookup table.

    Returns:
        emb_matrix  np.ndarray [49156, 256]
        cpg_ids     list[str]   in vocabulary order
    """
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    log.info(f"Loading checkpoint: {checkpoint_path}")
    module = load_wced_llama_checkpoint(checkpoint_path)
    encoder = module.encoder

    # CpG site embedding lookup table  [vocab_size, hidden_size]
    emb_weight = encoder.embeddings.cpg_sites_embeddings.weight.detach().cpu().numpy()
    log.info(f"CpG embedding table: {emb_weight.shape}  (vocab × hidden)")

    # Map embedding rows back to CpG IDs via vocab.txt (one token per line, index = id)
    import os
    vocab_file = os.path.join(tokenizer_path, "tokenizers", "cpg_sites", "vocab.txt")
    with open(vocab_file) as f:
        id2token = {i: line.strip() for i, line in enumerate(f)}

    cpg_ids = []
    for i in range(emb_weight.shape[0]):
        token = id2token.get(i, f"<unk_{i}>")
        cpg_ids.append(token)

    # Filter to CpG tokens only (cg/ch prefix)
    cpg_mask = np.array([t.startswith(("cg", "ch")) for t in cpg_ids])
    emb_matrix = emb_weight[cpg_mask]
    cpg_ids_filtered = [cpg_ids[i] for i in range(len(cpg_ids)) if cpg_mask[i]]

    log.info(f"CpG tokens: {emb_matrix.shape[0]:,} / {len(cpg_ids):,} vocab entries")
    return emb_matrix, cpg_ids_filtered


# ─────────────────────────────────────────────────────────────────────────────
# Annotation loading
# ─────────────────────────────────────────────────────────────────────────────

def load_manifest(manifest_path: str, cpg_ids: list[str]) -> pd.DataFrame:
    """Load annotation TSV from align_cpg_manifest.py, aligned to cpg_ids order."""
    log.info(f"Loading manifest: {manifest_path}")
    df = pd.read_csv(manifest_path, sep="\t")
    df = df.set_index("cpg_id").reindex(cpg_ids)

    # Fill missing annotations
    if "island_relation" in df.columns:
        df["island_relation"] = df["island_relation"].fillna("OpenSea")
    if "is_enhancer" in df.columns:
        df["is_enhancer"] = df["is_enhancer"].fillna(False)
        df["enhancer"] = df["is_enhancer"].map({True: "Enhancer", False: "Non-enhancer"})
    if "chr_group" in df.columns:
        df["chr_group"] = df["chr_group"].fillna("Other")

    log.info(f"Manifest loaded: {len(df):,} sites, columns: {list(df.columns)}")
    return df


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_umap(emb_matrix: np.ndarray, n_pca: int, n_neighbors: int, min_dist: float):
    """PCA → neighbors → UMAP on CpG embedding matrix."""
    log.info(f"Running PCA-{n_pca} → UMAP (n_neighbors={n_neighbors}, min_dist={min_dist})")
    adata = sc.AnnData(X=emb_matrix.astype(np.float32))
    n_pca_eff = min(n_pca, emb_matrix.shape[1] - 1, emb_matrix.shape[0] - 1)
    sc.tl.pca(adata, n_comps=n_pca_eff)
    sc.pp.neighbors(adata, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(adata, min_dist=min_dist)
    coords = adata.obsm["X_umap"]
    log.info(f"UMAP done: {coords.shape}")
    return coords


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

# Colours matching MethylGPT paper palette
ISLAND_PALETTE = {
    "Island":  "#E64B35",
    "Shore":   "#4DBBD5",
    "Shelf":   "#00A087",
    "OpenSea": "#3C5488",
}
ENHANCER_PALETTE = {
    "Non-enhancer": "#ADB6B6",
    "Enhancer":     "#F39B7F",
}
CHR_PALETTE = {
    "Autosome":        "#4DBBD5",
    "Sex (X/Y)":       "#E64B35",
    "Other":           "#ADB6B6",
}


def _scatter(ax, coords, labels, palette, title, s=1, alpha=0.3):
    cats = list(dict.fromkeys(labels))  # preserve order, deduplicate
    for cat in cats:
        mask = np.array(labels) == cat
        color = palette.get(cat, "#888888")
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=s, alpha=alpha, c=color, label=cat, rasterized=True)
    ax.legend(markerscale=6, fontsize=8, framealpha=0.4,
              loc="upper right", ncol=1)
    ax.set_title(title, fontsize=11, fontweight="bold")
    ax.set_xlabel("UMAP 1", fontsize=9)
    ax.set_ylabel("UMAP 2", fontsize=9)
    ax.set_xticks([]); ax.set_yticks([])


def plot_cpg_umaps(coords, annot_df, outdir: Path):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(parents=True, exist_ok=True)

    panels = []

    # Panel b: island relation
    if "island_relation" in annot_df.columns:
        panels.append(("island_relation", ISLAND_PALETTE, "CpG Island Relation"))

    # Panel c: enhancer
    if "enhancer" in annot_df.columns:
        panels.append(("enhancer", ENHANCER_PALETTE, "Enhancer Region"))

    # Panel d: chromosome group
    if "chr_group" in annot_df.columns:
        panels.append(("chr_group", CHR_PALETTE, "Chromosomal Location"))

    if not panels:
        log.warning("No annotation columns found — skipping annotation plots")
        return

    # Individual panels
    for col, palette, title in panels:
        labels = annot_df[col].astype(str).tolist()
        fig, ax = plt.subplots(figsize=(6, 5))
        _scatter(ax, coords, labels, palette, title)
        plt.tight_layout()
        out = fig_dir / f"umap_cpg_{col}.png"
        plt.savefig(out, dpi=180, bbox_inches="tight")
        plt.close()
        log.info(f"  Saved {out.name}")

    # Combined 3-panel figure (replicates Fig 2 layout)
    n = len(panels)
    fig, axes = plt.subplots(1, n, figsize=(5.5 * n, 4.8))
    if n == 1:
        axes = [axes]
    for ax, (col, palette, title) in zip(axes, panels):
        labels = annot_df[col].astype(str).tolist()
        _scatter(ax, coords, labels, palette, title)
    plt.suptitle("MethylLlama-Small — Contextualized CpG Embedding Space",
                 fontsize=12, y=1.01)
    plt.tight_layout()
    out = fig_dir / "umap_cpg_combined.png"
    plt.savefig(out, dpi=180, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved {out.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # 1. Extract CpG embedding table
    log.info("[1/4] Extracting CpG embedding table from checkpoint ...")
    emb_matrix, cpg_ids = extract_cpg_embeddings(args.checkpoint, args.tokenizer)
    np.save(outdir / "cpg_embeddings.npy", emb_matrix)
    pd.Series(cpg_ids, name="cpg_id").to_csv(outdir / "cpg_ids.txt", index=False, header=False)
    log.info(f"  Saved cpg_embeddings.npy  {emb_matrix.shape}")

    # 2. Load annotations
    annot_df = pd.DataFrame(index=cpg_ids)
    if args.manifest:
        log.info("[2/4] Loading manifest annotations ...")
        annot_df = load_manifest(args.manifest, cpg_ids)
    else:
        log.info("[2/4] No manifest provided — skipping annotations")

    # 3. UMAP
    log.info("[3/4] Running UMAP ...")
    coords = run_umap(emb_matrix, args.n_pca, args.n_neighbors, args.min_dist)
    np.save(outdir / "cpg_umap_coords.npy", coords)

    # Save with annotations
    result_df = pd.DataFrame({"cpg_id": cpg_ids, "UMAP1": coords[:, 0], "UMAP2": coords[:, 1]})
    for col in ["island_relation", "enhancer", "chr_group", "chr"]:
        if col in annot_df.columns:
            result_df[col] = annot_df[col].values
    result_df.to_csv(outdir / "cpg_umap.csv", index=False)
    log.info(f"  Saved cpg_umap.csv")

    # 4. Plots
    log.info("[4/4] Generating plots ...")
    if not annot_df.empty:
        plot_cpg_umaps(coords, annot_df, outdir)

    # Summary
    log.info(f"\nDone. Outputs → {outdir}/")
    log.info(f"  cpg_embeddings.npy       [{emb_matrix.shape[0]:,} × {emb_matrix.shape[1]}]")
    log.info(f"  cpg_umap_coords.npy      [{coords.shape[0]:,} × 2]")
    log.info(f"  cpg_umap.csv             with annotations")
    log.info(f"  figures/umap_cpg_*.png   UMAP panels")


if __name__ == "__main__":
    main()
