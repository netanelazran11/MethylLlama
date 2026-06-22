#!/usr/bin/env python3
"""
avg_bio_eval.py
===============
Compute scIB-style Avg_bio metric on MethylLlama embeddings.

Avg_bio = mean(NMI, ARI, ASW)
  - NMI / ARI: compare unsupervised Leiden clusters vs ground-truth labels
  - ASW: silhouette width using ground-truth labels in embedding space

Compares:
  pretrained_cls   WCED CLS bottleneck (frozen)
  pretrained_mean  mean-pool over CpG tokens
  random_cls       same architecture, random weights (ablation)

Usage:
  python scripts/repr_analysis/avg_bio_eval.py \
      --cls_npy      outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_cls.npy \
      --mean_npy     outputs/repr_analysis/pretrain_cls_169k_44892802/embeddings_mean.npy \
      --random_npy   outputs/repr_analysis/cls_probing_44905909/embeddings_random_cls.npy \
      --metadata_csv outputs/repr_analysis/cls_probing_44931911/metadata.csv \
      --label_col    tissue \
      --outdir       outputs/repr_analysis/avg_bio
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
from sklearn.metrics import (
    adjusted_rand_score,
    normalized_mutual_info_score,
    silhouette_score,
)

logging.basicConfig(level=logging.INFO, format="%(asctime)s  %(levelname)-8s  %(message)s",
                    datefmt="%H:%M:%S")
log = logging.getLogger(__name__)
sc.settings.verbosity = 0


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--cls_npy",      required=True)
    p.add_argument("--mean_npy",     default=None)
    p.add_argument("--random_npy",   default=None)
    p.add_argument("--metadata_csv", required=True,
                   help="metadata.csv saved by cls_probing_analysis (rows aligned to embeddings)")
    p.add_argument("--label_col",    default="tissue",
                   help="Column in metadata to use as biological label")
    p.add_argument("--ext_metadata", default=None,
                   help="External metadata CSV/CSV.gz to join on index (e.g. pretrain_metadata.csv.gz)")
    p.add_argument("--ext_id_col",   default="GSM_ID",
                   help="Column in ext_metadata that matches metadata index (sample IDs)")
    p.add_argument("--min_samples",  type=int, default=10,
                   help="Drop labels with fewer than this many samples")
    p.add_argument("--n_pca",        type=int, default=50)
    p.add_argument("--leiden_res",   type=float, default=0.6)
    p.add_argument("--outdir",       default="outputs/repr_analysis/avg_bio")
    return p.parse_args()


def compute_avg_bio(name: str, emb: np.ndarray, labels: np.ndarray,
                    n_pca: int, leiden_res: float) -> dict:
    log.info(f"  [{name}] building AnnData ({emb.shape[0]:,} samples × {emb.shape[1]}D) ...")
    adata = sc.AnnData(X=emb.astype(np.float32))
    adata.obs["label"] = labels

    # PCA → neighbors → clustering
    sc.pp.pca(adata, n_comps=min(n_pca, emb.shape[1] - 1))
    sc.pp.neighbors(adata, use_rep="X_pca", n_neighbors=15)
    try:
        sc.tl.leiden(adata, resolution=leiden_res, key_added="leiden", random_state=0)
        cluster_key = "leiden"
    except ImportError:
        log.warning("leidenalg not installed — falling back to louvain")
        sc.tl.louvain(adata, resolution=leiden_res, key_added="louvain",
                      random_state=0, flavor="igraph")
        cluster_key = "louvain"

    clusters = adata.obs[cluster_key].values
    labs     = adata.obs["label"].values

    nmi = normalized_mutual_info_score(labs, clusters)
    ari = adjusted_rand_score(labs, clusters)

    vc = pd.Series(labs).value_counts()
    if vc.size > 1 and vc.min() >= 2:
        asw = float(silhouette_score(emb, labs, sample_size=min(10000, len(labs)), random_state=42))
    else:
        asw = float("nan")

    avg_bio = float(np.nanmean([nmi, ari, asw]))

    result = {"embedding": name, "NMI": round(nmi, 4), "ARI": round(ari, 4),
              "ASW": round(asw, 4), "Avg_bio": round(avg_bio, 4)}
    log.info(f"    NMI={nmi:.4f}  ARI={ari:.4f}  ASW={asw:.4f}  → Avg_bio={avg_bio:.4f}")
    return result


def plot_results(results_df: pd.DataFrame, outdir: Path):
    metrics = ["NMI", "ARI", "ASW", "Avg_bio"]
    colors  = {"pretrained_cls": "#1D6FA5", "pretrained_mean": "#117A65", "random_cls": "#B03A2E"}

    fig, axes = plt.subplots(1, len(metrics), figsize=(14, 5), sharey=False)
    fig.suptitle("scIB Avg_bio Metrics: MethylLlama vs Baselines", fontsize=14, fontweight="bold")

    for ax, metric in zip(axes, metrics):
        vals  = results_df[metric].values
        names = results_df["embedding"].values
        cols  = [colors.get(n, "#888888") for n in names]
        bars  = ax.bar(names, vals, color=cols, edgecolor="white", linewidth=0.5)
        for bar, val in zip(bars, vals):
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.005,
                    f"{val:.3f}", ha="center", va="bottom", fontsize=10, fontweight="bold")
        ax.set_title(metric, fontsize=12, fontweight="bold")
        ax.set_ylim(0, min(1.05, max(vals) * 1.25 + 0.05))
        ax.set_xticks(range(len(names)))
        ax.set_xticklabels(names, rotation=20, ha="right", fontsize=9)
        ax.spines[["top", "right"]].set_visible(False)
        if metric == "Avg_bio":
            ax.set_facecolor("#F0F8FF")

    plt.tight_layout()
    out = outdir / "figures" / "avg_bio_comparison.png"
    out.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close()
    log.info(f"  Saved → {out}")


def main():
    args = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── Load metadata ────────────────────────────────────────────────────────
    log.info(f"Loading metadata: {args.metadata_csv}")
    meta = pd.read_csv(args.metadata_csv, index_col=0)
    log.info(f"  {len(meta):,} rows, columns: {list(meta.columns)}")

    # Join external metadata if label_col is missing
    if args.label_col not in meta.columns:
        if args.ext_metadata and Path(args.ext_metadata).exists():
            log.info(f"  '{args.label_col}' not in metadata — joining from {args.ext_metadata}")
            ext = pd.read_csv(args.ext_metadata)
            ext = ext.drop_duplicates(subset=args.ext_id_col).set_index(args.ext_id_col)
            if args.label_col not in ext.columns:
                raise ValueError(f"'{args.label_col}' not found in ext_metadata columns: {list(ext.columns)}")
            meta = meta.join(ext[[args.label_col]], how="left")
            n_matched = meta[args.label_col].notna().sum()
            log.info(f"  Matched {n_matched:,} / {len(meta):,} samples with tissue labels")
        else:
            raise ValueError(
                f"--label_col '{args.label_col}' not in metadata columns: {list(meta.columns)}\n"
                f"Pass --ext_metadata to join tissue labels from an external file."
            )

    labels_raw = meta[args.label_col].fillna("unknown").values

    # ── Load embeddings ──────────────────────────────────────────────────────
    embedding_paths = {"pretrained_cls": args.cls_npy}
    if args.mean_npy and Path(args.mean_npy).exists():
        embedding_paths["pretrained_mean"] = args.mean_npy
    if args.random_npy and Path(args.random_npy).exists():
        embedding_paths["random_cls"] = args.random_npy

    embeddings = {}
    for name, path in embedding_paths.items():
        log.info(f"Loading {name}: {path}")
        emb = np.load(path)
        log.info(f"  shape: {emb.shape}")
        if emb.shape[0] != len(meta):
            raise ValueError(f"{name}: embedding rows ({emb.shape[0]}) ≠ metadata rows ({len(meta)})")
        embeddings[name] = emb

    # ── Filter labels with too few samples ───────────────────────────────────
    vc = pd.Series(labels_raw).value_counts()
    valid_labels = set(vc[vc >= args.min_samples].index)
    mask = np.array([l in valid_labels for l in labels_raw])
    log.info(f"Keeping {mask.sum():,} / {len(mask):,} samples ({len(valid_labels)} {args.label_col} classes, min={args.min_samples})")

    labels = labels_raw[mask]
    embeddings = {k: v[mask] for k, v in embeddings.items()}

    # ── Compute metrics ───────────────────────────────────────────────────────
    log.info("\n" + "="*60)
    log.info(f" Computing Avg_bio (label={args.label_col})")
    log.info("="*60)

    results = []
    for name, emb in embeddings.items():
        row = compute_avg_bio(name, emb, labels, args.n_pca, args.leiden_res)
        results.append(row)

    results_df = pd.DataFrame(results)

    # ── Save ─────────────────────────────────────────────────────────────────
    csv_path = outdir / "avg_bio_results.csv"
    results_df.to_csv(csv_path, index=False)
    log.info(f"\nResults saved → {csv_path}")

    print("\n" + "="*60)
    print(" RESULTS")
    print("="*60)
    print(results_df.to_string(index=False))
    print("="*60)

    plot_results(results_df, outdir)

    log.info("\nDone.")


if __name__ == "__main__":
    main()
