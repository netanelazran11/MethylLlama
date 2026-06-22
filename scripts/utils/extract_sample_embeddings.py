#!/usr/bin/env python3
"""
extract_sample_embeddings.py
============================
Full CLS-bottleneck extraction and representation-analysis pipeline for
MethylLlama-Small WCED pretrain checkpoints.

What this script does:
  1.  Load WCEDLlamaModule (pretrain) or MethylationAgeRegressorLlama (finetune)
  2.  Extract per-sample embeddings for ALL samples in the h5ad:
        CLS  — pooler_output: Linear(token_0) + Tanh  (WCED-supervised bottleneck)
        Mean — average over all non-CLS CpG tokens    (comparison baseline)
  3.  Optionally extract random-init CLS for ablation (quantify pretraining gain)
  4.  Build AnnData (X=CLS, obsm["X_mean"]=mean-pool, obs=all metadata)
  5.  Run UMAP pipeline per embedding: PCA-50 → neighbors → UMAP-2D + Leiden
  6.  Compute clustering quality vs. tissue / gender / dataset labels:
        ASW (silhouette), ARI, NMI, Davies-Bouldin, Calinski-Harabasz
  7.  Age linear probe: Ridge regression R², PCC, MAE, MedAE
  8.  Save all figures (UMAP per label + age, age scatter, metrics bar chart)
  9.  Write a summary report.txt

Outputs (<outdir>/):
  embeddings_cls.npy          [N, 256]  pretrained CLS
  embeddings_mean.npy         [N, 256]  pretrained mean-pool
  embeddings_random_cls.npy   [N, 256]  random-init CLS  (if --compare_random)
  metadata.csv                per-sample labels
  adata.h5ad                  full AnnData (save once, analyse many times)
  figures/                    all PNG plots
  report.txt                  clustering + probe metrics summary

Usage:
  # Main use case — pretrained CLS:
  python scripts/utils/extract_sample_embeddings.py \\
      --checkpoint outputs/pretrain-llama-wced/llama-small-all49k-r0.5-w0.0-44450919/checkpoints/epoch=98-val_loss=0.0059.ckpt \\
      --ckpt_type  pretrain \\
      --data       /path/to/finetuning_19608_clean_stratified_no_outliers.h5ad \\
      --tokenizer  tokenizer_llama_pretrain49k \\
      --outdir     outputs/repr_analysis/pretrain_cls \\
      --compare_random

  # Fine-tuned checkpoint for comparison:
  python scripts/utils/extract_sample_embeddings.py \\
      --checkpoint outputs/finetune-llama-small/.../best.ckpt \\
      --ckpt_type  finetune \\
      --outdir     outputs/repr_analysis/finetune_cls
"""

import argparse
import logging
import os
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc
import torch
from scipy.stats import pearsonr
from sklearn.cluster import KMeans
from sklearn.linear_model import RidgeCV
from sklearn.metrics import (
    adjusted_rand_score,
    calinski_harabasz_score,
    davies_bouldin_score,
    normalized_mutual_info_score,
    silhouette_score,
)
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger(__name__)
sc.settings.verbosity = 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(
        description="Extract MethylLlama CLS embeddings and run full repr-analysis"
    )
    p.add_argument("--checkpoint",  required=True,  help="Path to .ckpt file")
    p.add_argument("--ckpt_type",   default="pretrain",
                   choices=["pretrain", "finetune"],
                   help="pretrain=WCEDLlamaModule  finetune=MethylationAgeRegressorLlama")
    p.add_argument("--data",        required=True,  help="Path to finetuning .h5ad")
    p.add_argument("--tokenizer",   required=True,  help="Path to tokenizer_llama_pretrain49k/")
    p.add_argument("--outdir",      default="outputs/repr_analysis/cls")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",
                   default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compare_random", action="store_true",
                   help="Also extract random-init CLS to quantify pretraining gain")
    p.add_argument("--n_pca",       type=int, default=50,
                   help="PCA components before UMAP (default 50)")
    p.add_argument("--n_neighbors", type=int, default=15,
                   help="k-NN neighbors for UMAP graph (default 15)")
    p.add_argument("--label_cols",  nargs="+",
                   default=["tissue_type", "gender", "dataset"],
                   help="obs columns used for clustering metrics and UMAP coloring")
    p.add_argument("--age_col",     default="age",
                   help="obs column containing continuous age values")
    p.add_argument("--split_col",   default="split",
                   help="obs column with train/valid/test labels")
    # External metadata (for pretrain h5ad that has no tissue/age/sex in obs)
    p.add_argument("--metadata",    default=None,
                   help="External metadata CSV/CSV.gz to join by obs_names. "
                        "Used when the h5ad only has sample IDs (e.g. pretrain 169k).")
    p.add_argument("--metadata_id_col", default="GSM_ID",
                   help="Column in --metadata that matches the h5ad obs_names (default: GSM_ID)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Checkpoint loading
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path: str, ckpt_type: str):
    """
    Load the MethylLlamaModel encoder from either checkpoint type.
    Returns (encoder, parent_module).
    """
    if ckpt_type == "pretrain":
        from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
        module = load_wced_llama_checkpoint(checkpoint_path)
    else:
        from bmfm_methylation.llama.finetune_llama import MethylationAgeRegressorLlama
        module = MethylationAgeRegressorLlama.load_from_checkpoint(
            checkpoint_path, map_location="cpu"
        )
    encoder = module.encoder
    encoder.eval()
    n_params = sum(p.numel() for p in encoder.parameters())
    logger.info(
        f"  Encoder loaded: {encoder.config.num_hidden_layers}L × "
        f"{encoder.config.hidden_size}D × {encoder.config.num_attention_heads}H  "
        f"({n_params/1e6:.1f}M params)"
    )
    return encoder, module


def build_random_encoder(reference_encoder):
    """
    Same architecture as the reference encoder but randomly initialised weights.
    Used as ablation baseline: what does pretraining add vs. random features?
    """
    from bmfm_methylation.llama.model import MethylLlamaModel
    rand_enc = MethylLlamaModel(reference_encoder.config)
    rand_enc.eval()
    logger.info("  Random-init encoder built (same architecture, untrained weights)")
    return rand_enc


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

@torch.no_grad()
def extract_embeddings(
    encoder,
    data_path: str,
    tokenizer_path: str,
    batch_size: int,
    device: str,
):
    """
    Extract CLS (pooler_output) and mean-pool embeddings for every sample.

    Both arrays are in h5ad obs order (DataLoader shuffle=False).
    CLS  = token-0 → Linear(256→256) + Tanh (the WCED bottleneck).
    Mean = average of all non-CLS CpG token hidden states.
    """
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    encoder = encoder.to(device)

    tokenizer = MultiFieldTokenizer.from_pretrained(tokenizer_path)

    # split=None → load all samples in h5ad row order
    dataset  = MethylationDataset(h5ad_path=data_path, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites
    logger.info(f"  Dataset: {len(dataset)} samples × {len(cpg_sites)} CpGs")

    # input_ratio=1.0 → all CpGs fed as encoder input, nothing held out
    collator = WCEDCollator(
        tokenizer=tokenizer,
        cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites),
        input_ratio=1.0,
        contrastive=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        collate_fn=collator,
        shuffle=False,
        num_workers=4,
        pin_memory=(device == "cuda"),
    )

    cls_list, mean_list = [], []
    for i, batch in enumerate(loader):
        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)

        # Dual-field input: [B, 2, L]  (field-0=cpg_ids, field-1=betas)
        input_ids = torch.stack([cpg_ids.float(), beta_values], dim=1)
        out = encoder(input_ids=input_ids, attention_mask=attn_mask)

        # CLS: MethylLlamaPooler applies Linear + Tanh to token position 0
        cls_emb = out.pooler_output.cpu().float()                   # [B, D]

        # Mean: average over CpG token positions (skip CLS at pos-0)
        hidden   = out.last_hidden_state[:, 1:, :].cpu().float()    # [B, L-1, D]
        mask_1d  = attn_mask[:, 1:].cpu().float().unsqueeze(-1)     # [B, L-1, 1]
        mean_emb = (hidden * mask_1d).sum(1) / mask_1d.sum(1).clamp(min=1)  # [B, D]

        cls_list.append(cls_emb.numpy())
        mean_list.append(mean_emb.numpy())

        if (i + 1) % 20 == 0:
            logger.info(f"    batch {i+1}/{len(loader)}")

    cls_embs  = np.concatenate(cls_list,  axis=0)   # [N, D]
    mean_embs = np.concatenate(mean_list, axis=0)   # [N, D]
    logger.info(f"  Done — CLS shape: {cls_embs.shape}  Mean shape: {mean_embs.shape}")
    return cls_embs, mean_embs


# ─────────────────────────────────────────────────────────────────────────────
# AnnData construction
# ─────────────────────────────────────────────────────────────────────────────

def load_external_metadata(
    metadata_path: str,
    id_col: str,
    obs_names: pd.Index,
) -> pd.DataFrame:
    """
    Load an external metadata CSV/CSV.gz and left-join it onto the h5ad obs_names.

    Returns a DataFrame indexed like obs_names (same order, same length).
    Samples not found in metadata get NaN for all metadata columns.
    """
    logger.info(f"  Loading external metadata: {metadata_path}")
    meta = pd.read_csv(metadata_path)
    logger.info(f"  Metadata shape: {meta.shape}  id_col='{id_col}'")

    if id_col not in meta.columns:
        raise ValueError(
            f"--metadata_id_col '{id_col}' not found in metadata columns: {list(meta.columns)}"
        )

    meta = meta.drop_duplicates(subset=id_col).set_index(id_col)
    obs_df = pd.DataFrame(index=obs_names)
    obs_df = obs_df.join(meta, how="left")

    n_matched = obs_df.notna().any(axis=1).sum()
    logger.info(
        f"  Matched {n_matched:,} / {len(obs_names):,} samples "
        f"({100*n_matched/len(obs_names):.1f}%) to metadata"
    )
    if n_matched == 0:
        logger.warning(
            "  !! No samples matched. Check that obs_names are GSM IDs "
            "and that --metadata_id_col is correct."
        )
    return obs_df


def make_adata(
    cls_embs:  np.ndarray,
    mean_embs: np.ndarray,
    ref_adata: sc.AnnData,
    label_cols: list,
    age_col: str,
    split_col: str,
    external_meta: pd.DataFrame | None = None,
) -> sc.AnnData:
    """
    Build output AnnData.
    X = CLS embeddings [N, D].
    obsm["X_cls"]  = CLS (same as X).
    obsm["X_mean"] = mean-pool.
    obs = metadata from h5ad obs OR external_meta (if provided).
    """
    adata = sc.AnnData(X=cls_embs.copy().astype(np.float32))
    adata.obs_names = ref_adata.obs_names.copy()
    adata.obsm["X_cls"]  = cls_embs.copy().astype(np.float32)
    adata.obsm["X_mean"] = mean_embs.copy().astype(np.float32)

    want = list(dict.fromkeys([age_col, split_col] + label_cols))

    if external_meta is not None:
        # External metadata takes priority; fall back to h5ad obs for any missing cols
        for col in want:
            if col in external_meta.columns:
                adata.obs[col] = external_meta[col].values
            elif col in ref_adata.obs.columns:
                adata.obs[col] = ref_adata.obs[col].values
    else:
        for col in want:
            if col in ref_adata.obs.columns:
                adata.obs[col] = ref_adata.obs[col].values

    # Ensure age is numeric
    if age_col in adata.obs.columns:
        adata.obs[age_col] = pd.to_numeric(adata.obs[age_col], errors="coerce")

    # Add age-decade categorical for UMAP coloring
    if age_col in adata.obs.columns:
        ages = adata.obs[age_col].astype(float)
        valid = ages.notna()
        adata.obs["age_decade"] = "unknown"
        adata.obs.loc[valid, "age_decade"] = (
            (ages[valid] // 10 * 10).astype(int).astype(str) + "s"
        )

    logger.info(
        f"  AnnData: {adata.n_obs} samples × {adata.n_vars} dims  "
        f"obs: {list(adata.obs.columns)}"
    )
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# UMAP pipeline
# ─────────────────────────────────────────────────────────────────────────────

def run_umap(
    adata: sc.AnnData,
    obsm_key: str,
    umap_out_key: str,
    leiden_out_key: str,
    n_pca: int,
    n_neighbors: int,
) -> sc.AnnData:
    """
    PCA → neighbors → UMAP → Leiden on adata.obsm[obsm_key].
    Stores:
      adata.obsm[umap_out_key]   = 2-D UMAP coordinates
      adata.obsm[pca_out_key]    = PCA coordinates
      adata.obs[leiden_out_key]  = Leiden cluster labels
    """
    logger.info(f"  [{obsm_key}] PCA-{n_pca} → neighbors({n_neighbors}) → UMAP → Leiden")

    work = sc.AnnData(X=adata.obsm[obsm_key].copy())
    work.obs = adata.obs.copy()

    n_pca_eff = min(n_pca, work.n_vars - 1, work.n_obs - 1)
    sc.tl.pca(work, n_comps=n_pca_eff, use_highly_variable=False)
    sc.pp.neighbors(work, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(work)
    sc.tl.leiden(work, resolution=0.5, key_added="leiden")

    adata.obsm[umap_out_key] = work.obsm["X_umap"].astype(np.float32)
    pca_out_key = umap_out_key.replace("X_umap_", "X_pca_")
    adata.obsm[pca_out_key]  = work.obsm["X_pca"].astype(np.float32)
    adata.obs[leiden_out_key] = work.obs["leiden"].values

    n_leiden = adata.obs[leiden_out_key].nunique()
    logger.info(f"    Leiden clusters: {n_leiden}")
    return adata


# ─────────────────────────────────────────────────────────────────────────────
# Clustering metrics
# ─────────────────────────────────────────────────────────────────────────────

def compute_clustering_metrics(
    embs: np.ndarray,
    labels_raw,
    tag: str,
) -> dict:
    """
    Compute ASW, ARI, NMI, DBI, CHI for embeddings vs. ground-truth labels.

    ASW  — silhouette score vs. true labels (no clustering needed).
    ARI, NMI — KMeans(k=n_classes) compared against true labels.
    DBI, CHI — KMeans(k=n_classes) cluster quality.
    """
    labels_str = np.array(labels_raw).astype(str)
    le = LabelEncoder()
    labels = le.fit_transform(labels_str)
    n_cls = len(le.classes_)

    if n_cls < 2:
        logger.warning(f"  [{tag}] Only {n_cls} class — skipping")
        return {}

    scaler = StandardScaler()
    X = scaler.fit_transform(embs)

    # ASW on true labels — no clustering required
    asw = silhouette_score(X, labels,
                           sample_size=min(5000, len(labels)), random_state=42)

    # KMeans with k = n_true_classes
    km = KMeans(n_clusters=n_cls, random_state=42, n_init=10)
    km_labels = km.fit_predict(X)

    ari = adjusted_rand_score(labels, km_labels)
    nmi = normalized_mutual_info_score(labels, km_labels)
    dbi = davies_bouldin_score(X, km_labels)
    chi = calinski_harabasz_score(X, km_labels)

    result = dict(
        tag=tag, n_classes=n_cls,
        ASW=round(asw, 4),
        ARI=round(ari, 4),
        NMI=round(nmi, 4),
        DBI=round(dbi, 4),
        CHI=round(chi, 1),
        classes=list(le.classes_),
    )
    logger.info(
        f"  [{tag}]  ASW={asw:+.3f}  ARI={ari:.3f}  NMI={nmi:.3f}  "
        f"DBI={dbi:.3f}  CHI={chi:.0f}  (n_cls={n_cls})"
    )
    return result


# ─────────────────────────────────────────────────────────────────────────────
# Age linear probe
# ─────────────────────────────────────────────────────────────────────────────

def run_age_probe(
    embs: np.ndarray,
    ages: np.ndarray,
    train_mask: np.ndarray,
    test_mask: np.ndarray,
    tag: str,
):
    """
    Ridge regression probe: frozen embeddings → age.
    Returns (metrics_dict, y_true_test, y_pred_test).
    """
    valid_tr = train_mask & ~np.isnan(ages)
    valid_te = test_mask  & ~np.isnan(ages)

    if valid_tr.sum() < 10 or valid_te.sum() < 5:
        logger.warning(f"  [{tag}] Insufficient samples — skipping age probe")
        return {}, None, None

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embs[valid_tr])
    X_te = scaler.transform(embs[valid_te])
    y_tr = ages[valid_tr]
    y_te = ages[valid_te]

    ridge = RidgeCV(alphas=[0.01, 0.1, 1.0, 10.0, 100.0, 1000.0], cv=5)
    ridge.fit(X_tr, y_tr)
    y_pred = ridge.predict(X_te)

    ss_res = np.sum((y_te - y_pred) ** 2)
    ss_tot = np.sum((y_te - y_te.mean()) ** 2)
    r2  = float(1 - ss_res / (ss_tot + 1e-8))
    pcc = float(pearsonr(y_te, y_pred)[0])
    mae = float(np.abs(y_te - y_pred).mean())
    med = float(np.median(np.abs(y_te - y_pred)))

    result = dict(
        tag=tag, best_alpha=round(ridge.alpha_, 4),
        R2=round(r2, 4), PCC=round(pcc, 4),
        MAE_yr=round(mae, 2), MedAE_yr=round(med, 2),
        train_n=int(valid_tr.sum()), test_n=int(valid_te.sum()),
    )
    logger.info(
        f"  [{tag}]  R²={r2:.3f}  PCC={pcc:.3f}  "
        f"MAE={mae:.1f}yr  MedAE={med:.1f}yr  α={ridge.alpha_:.1e}"
    )
    return result, y_te, y_pred


# ─────────────────────────────────────────────────────────────────────────────
# Plotting
# ─────────────────────────────────────────────────────────────────────────────

_TAB20 = plt.cm.get_cmap("tab20", 20)


def _scatter_categorical(ax, coords, values, title):
    cats = sorted(np.unique(values.astype(str)))
    palette = plt.cm.get_cmap("tab20", max(len(cats), 1))
    for i, cat in enumerate(cats):
        mask = values.astype(str) == cat
        ax.scatter(coords[mask, 0], coords[mask, 1],
                   s=3, alpha=0.55, c=[palette(i)], label=cat, rasterized=True)
    if len(cats) <= 20:
        ax.legend(markerscale=5, fontsize=6, loc="best",
                  framealpha=0.4, ncol=max(1, len(cats) // 10))
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def _scatter_continuous(ax, coords, values, title, cmap="RdYlBu_r"):
    sc_obj = ax.scatter(coords[:, 0], coords[:, 1],
                        s=3, alpha=0.55, c=values.astype(float),
                        cmap=cmap, rasterized=True)
    plt.colorbar(sc_obj, ax=ax, fraction=0.04, pad=0.01)
    ax.set_title(title, fontsize=10)
    ax.set_xlabel("UMAP 1", fontsize=8)
    ax.set_ylabel("UMAP 2", fontsize=8)
    ax.set_xticks([]); ax.set_yticks([])


def plot_umaps(
    adata: sc.AnnData,
    umap_obsm_key: str,
    label_cols: list,
    age_col: str,
    outdir: Path,
    tag: str,
):
    """Save one UMAP PNG per metadata column, plus age-continuous and age-decade."""
    fig_dir = outdir / "figures"
    fig_dir.mkdir(exist_ok=True)
    coords = adata.obsm[umap_obsm_key]

    cat_cols = [c for c in label_cols + ["split", "age_decade"] if c in adata.obs.columns]
    for col in cat_cols:
        vals = adata.obs[col].astype(str).values
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_categorical(ax, coords, vals, title=f"{tag}  |  {col}")
        plt.tight_layout()
        fname = fig_dir / f"umap_{tag}_{col}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"    {fname.name}")

    if age_col in adata.obs.columns:
        vals = adata.obs[age_col].values.astype(float)
        fig, ax = plt.subplots(figsize=(7, 6))
        _scatter_continuous(ax, coords, vals, title=f"{tag}  |  age (years)", cmap="RdYlBu_r")
        plt.tight_layout()
        fname = fig_dir / f"umap_{tag}_age.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight")
        plt.close()
        logger.info(f"    {fname.name}")


def plot_age_probe_scatter(
    y_true: np.ndarray, y_pred: np.ndarray,
    r2: float, pcc: float, mae: float, med_ae: float,
    outdir: Path, tag: str,
):
    """True vs. predicted age scatter from the Ridge probe."""
    fig_dir = outdir / "figures"
    fig_dir.mkdir(exist_ok=True)
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=7, alpha=0.5, c="steelblue", rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", linewidth=0.9, label="y=x")
    ax.set_xlabel("True age (years)", fontsize=10)
    ax.set_ylabel("Ridge predicted age (years)", fontsize=10)
    ax.set_title(
        f"{tag}\nR²={r2:.3f}  PCC={pcc:.3f}  MAE={mae:.1f}yr  MedAE={med_ae:.1f}yr",
        fontsize=10,
    )
    plt.tight_layout()
    fname = fig_dir / f"age_probe_{tag}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"    {fname.name}")


def plot_metrics_bar(all_metrics: list, outdir: Path):
    """
    Grouped bar chart: ASW / ARI / NMI per (embedding_type, label_col).
    Gives a quick visual of which embedding separates which label best.
    """
    fig_dir = outdir / "figures"
    fig_dir.mkdir(exist_ok=True)

    rows = []
    for m in all_metrics:
        if not m:
            continue
        for metric in ("ASW", "ARI", "NMI"):
            rows.append({"Embedding": m["tag"], "Metric": metric, "Score": m[metric]})
    if not rows:
        return
    df = pd.DataFrame(rows)

    tags = df["Embedding"].unique().tolist()
    metrics = ["ASW", "ARI", "NMI"]
    x = np.arange(len(tags))
    width = 0.25
    colors = ["#4C72B0", "#DD8452", "#55A868"]

    fig, ax = plt.subplots(figsize=(max(8, len(tags) * 1.8), 5))
    for i, (metric, color) in enumerate(zip(metrics, colors)):
        vals = [
            df[(df["Embedding"] == t) & (df["Metric"] == metric)]["Score"].values[0]
            if len(df[(df["Embedding"] == t) & (df["Metric"] == metric)]) > 0 else 0
            for t in tags
        ]
        ax.bar(x + i * width, vals, width, label=metric, color=color, alpha=0.85)

    ax.axhline(0, color="gray", linestyle="--", linewidth=0.7)
    ax.set_xticks(x + width)
    ax.set_xticklabels(tags, rotation=30, ha="right", fontsize=8)
    ax.set_ylabel("Score (↑ better for all three)")
    ax.set_title("Clustering Quality: ASW / ARI / NMI by Embedding × Label")
    ax.legend(fontsize=9)
    plt.tight_layout()
    fname = fig_dir / "clustering_metrics.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info(f"    {fname.name}")


def plot_embedding_grid(
    embeddings_dict: dict,
    adata: sc.AnnData,
    umap_keys: dict,
    age_col: str,
    outdir: Path,
):
    """
    Multi-panel comparison grid: rows=embedding type, cols=age/tissue/split.
    Gives one big overview figure for the paper / thesis.
    """
    fig_dir = outdir / "figures"
    fig_dir.mkdir(exist_ok=True)

    col_defs = []
    if age_col in adata.obs.columns:
        col_defs.append((age_col,        False, "RdYlBu_r"))
    for c in ["tissue_type", "dataset", "split"]:
        if c in adata.obs.columns:
            col_defs.append((c, True, None))
    if not col_defs:
        return

    row_labels = list(umap_keys.keys())
    n_rows = len(row_labels)
    n_cols = len(col_defs)
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(4 * n_cols, 3.5 * n_rows))
    if n_rows == 1:
        axes = axes[np.newaxis, :]
    if n_cols == 1:
        axes = axes[:, np.newaxis]

    for r, emb_tag in enumerate(row_labels):
        umap_key = umap_keys[emb_tag]
        coords   = adata.obsm[umap_key]
        for c, (col, is_cat, cmap) in enumerate(col_defs):
            ax = axes[r, c]
            vals = adata.obs[col].values
            if is_cat:
                _scatter_categorical(ax, coords, vals,
                                     title=f"{emb_tag} | {col}" if r == 0 else "")
            else:
                _scatter_continuous(ax, coords, vals.astype(float),
                                    title=f"{emb_tag} | {col}" if r == 0 else "",
                                    cmap=cmap)
            if c == 0:
                ax.set_ylabel(emb_tag, fontsize=9, fontweight="bold")

    plt.suptitle("MethylLlama-Small — CLS Representation Analysis", fontsize=12, y=1.01)
    plt.tight_layout()
    fname = fig_dir / "overview_grid.png"
    plt.savefig(fname, dpi=130, bbox_inches="tight")
    plt.close()
    logger.info(f"    {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(
    outdir: Path,
    args,
    clustering_results: list,
    age_probe_results: list,
):
    SEP = "─" * 68
    lines = [
        "=" * 68,
        "  MethylLlama-Small — CLS Representation Analysis  ",
        "=" * 68,
        f"  Checkpoint : {args.checkpoint}",
        f"  Type       : {args.ckpt_type}",
        f"  Data       : {args.data}",
        f"  Device     : {args.device}",
        f"  Random CLS : {args.compare_random}",
        "",
        SEP,
        "  Clustering Quality  (↑ASW ↑ARI ↑NMI  ↓DBI ↑CHI)",
        SEP,
    ]

    for m in clustering_results:
        if not m:
            continue
        lines += [
            f"",
            f"  [{m['tag']}]  n_classes={m['n_classes']}",
            f"    ASW (silhouette, ↑)  = {m['ASW']:+.4f}",
            f"    ARI (adj. rand,  ↑)  = {m['ARI']:+.4f}",
            f"    NMI (norm. MI,   ↑)  = {m['NMI']:+.4f}",
            f"    DBI (Davies-B,   ↓)  = {m['DBI']:.4f}",
            f"    CHI (Calinski,   ↑)  = {m['CHI']:.1f}",
            f"    Classes: {', '.join(m['classes'][:10])}"
            + (" ..." if len(m['classes']) > 10 else ""),
        ]

    lines += ["", SEP, "  Age Linear Probe (Ridge Regression)", SEP]
    for r in age_probe_results:
        if not r:
            continue
        lines += [
            f"",
            f"  [{r['tag']}]  train={r['train_n']}  test={r['test_n']}",
            f"    R²     = {r['R2']:.4f}",
            f"    PCC    = {r['PCC']:.4f}",
            f"    MAE    = {r['MAE_yr']:.2f} years",
            f"    MedAE  = {r['MedAE_yr']:.2f} years",
            f"    Ridge α= {r['best_alpha']:.1e}",
        ]

    lines += ["", "=" * 68]
    report = "\n".join(lines)
    print("\n" + report)
    (outdir / "report.txt").write_text(report)
    logger.info(f"  Report → {outdir / 'report.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    logger.info("=" * 68)
    logger.info("  MethylLlama-Small — CLS Extraction & Representation Analysis")
    logger.info("=" * 68)
    logger.info(f"  Checkpoint : {args.checkpoint}")
    logger.info(f"  Type       : {args.ckpt_type}")
    logger.info(f"  Data       : {args.data}")

    # ── 1. Load encoder ──────────────────────────────────────────────────────
    logger.info("\n[1/9] Loading encoder ...")
    encoder, _ = load_encoder(args.checkpoint, args.ckpt_type)

    # ── 2. Extract pretrained embeddings ─────────────────────────────────────
    logger.info("\n[2/9] Extracting pretrained CLS & mean embeddings ...")
    cls_embs, mean_embs = extract_embeddings(
        encoder, args.data, args.tokenizer, args.batch_size, args.device
    )
    np.save(outdir / "embeddings_cls.npy",  cls_embs)
    np.save(outdir / "embeddings_mean.npy", mean_embs)

    # ── 3. Optional random-init ablation ─────────────────────────────────────
    rand_cls = None
    if args.compare_random:
        logger.info("\n[3/9] Extracting random-init CLS (ablation) ...")
        rand_enc = build_random_encoder(encoder)
        rand_cls, _ = extract_embeddings(
            rand_enc, args.data, args.tokenizer, args.batch_size, args.device
        )
        np.save(outdir / "embeddings_random_cls.npy", rand_cls)
    else:
        logger.info("\n[3/9] Skipping random-init ablation (use --compare_random to enable)")

    # ── 4. Load metadata & build AnnData ─────────────────────────────────────
    logger.info("\n[4/9] Loading h5ad metadata & building AnnData ...")
    ref_adata = sc.read_h5ad(args.data)
    logger.info(f"  Available obs: {list(ref_adata.obs.columns)}")

    external_meta = None
    if args.metadata:
        external_meta = load_external_metadata(
            args.metadata, args.metadata_id_col, ref_adata.obs_names
        )

    adata = make_adata(
        cls_embs, mean_embs, ref_adata,
        args.label_cols, args.age_col, args.split_col,
        external_meta=external_meta,
    )
    if rand_cls is not None:
        adata.obsm["X_random_cls"] = rand_cls.astype(np.float32)

    # ── 5. UMAP pipelines ────────────────────────────────────────────────────
    logger.info("\n[5/9] Running UMAP pipelines ...")
    adata = run_umap(adata, "X_cls",  "X_umap_cls",  "leiden_cls",  args.n_pca, args.n_neighbors)
    adata = run_umap(adata, "X_mean", "X_umap_mean", "leiden_mean", args.n_pca, args.n_neighbors)
    if rand_cls is not None:
        adata = run_umap(adata, "X_random_cls", "X_umap_random", "leiden_random", args.n_pca, args.n_neighbors)

    # ── 6. Save AnnData ───────────────────────────────────────────────────────
    logger.info("\n[6/9] Saving AnnData ...")
    adata.write_h5ad(outdir / "adata.h5ad")
    adata.obs.to_csv(outdir / "metadata.csv")
    logger.info(f"  adata.h5ad  metadata.csv  → {outdir}")

    # ── 7. Clustering metrics ─────────────────────────────────────────────────
    logger.info("\n[7/9] Computing clustering metrics ...")
    available = [c for c in args.label_cols if c in adata.obs.columns]
    if not available:
        logger.warning("  No label columns found in obs — skipping clustering metrics")
    all_cluster_metrics = []
    for col in available:
        vals = adata.obs[col].astype(str).values
        if len(np.unique(vals)) < 2:
            continue
        logger.info(f"  Label: {col} ({len(np.unique(vals))} classes)")
        all_cluster_metrics.append(
            compute_clustering_metrics(cls_embs,  vals, tag=f"CLS/{col}")
        )
        all_cluster_metrics.append(
            compute_clustering_metrics(mean_embs, vals, tag=f"Mean/{col}")
        )
        if rand_cls is not None:
            all_cluster_metrics.append(
                compute_clustering_metrics(rand_cls, vals, tag=f"Random/{col}")
            )

    # ── 8. Age linear probe ───────────────────────────────────────────────────
    logger.info("\n[8/9] Running age linear probes ...")
    all_age_results = []
    if args.age_col in adata.obs.columns:
        ages  = adata.obs[args.age_col].values.astype(float)
        if args.split_col in adata.obs.columns:
            split = adata.obs[args.split_col].values
            train_mask = np.isin(split, ["train", "valid"])
            test_mask  = split == "test"
        else:
            # No split column (pretrain data): random 80/20 on samples with valid age
            rng = np.random.default_rng(42)
            valid_idx = np.where(~np.isnan(ages))[0]
            n_test = max(1, int(0.2 * len(valid_idx)))
            test_idx = set(rng.choice(valid_idx, size=n_test, replace=False).tolist())
            train_mask = np.array([i not in test_idx for i in range(len(ages))])
            test_mask  = np.array([i in test_idx     for i in range(len(ages))])
            logger.info(
                f"  No split_col found — using random 80/20 split "
                f"(train={train_mask.sum()}, test={test_mask.sum()})"
            )

        probe_targets = [
            (cls_embs,  "pretrained_cls"),
            (mean_embs, "pretrained_mean"),
        ]
        if rand_cls is not None:
            probe_targets.append((rand_cls, "random_cls"))

        for emb_arr, tag in probe_targets:
            res, y_te, y_pred = run_age_probe(emb_arr, ages, train_mask, test_mask, tag)
            if res:
                all_age_results.append(res)
                plot_age_probe_scatter(
                    y_te, y_pred,
                    res["R2"], res["PCC"], res["MAE_yr"], res["MedAE_yr"],
                    outdir, tag,
                )
    else:
        logger.warning(f"  '{args.age_col}' not in obs — skipping age probe")

    # ── 9. Plots ─────────────────────────────────────────────────────────────
    logger.info("\n[9/9] Generating plots ...")
    all_label_cols_for_plot = args.label_cols + ["age_decade"]

    logger.info("  UMAP — pretrained CLS")
    plot_umaps(adata, "X_umap_cls",  all_label_cols_for_plot, args.age_col, outdir, "pretrained_cls")

    logger.info("  UMAP — pretrained Mean")
    plot_umaps(adata, "X_umap_mean", all_label_cols_for_plot, args.age_col, outdir, "pretrained_mean")

    if rand_cls is not None:
        logger.info("  UMAP — random-init CLS")
        plot_umaps(adata, "X_umap_random", all_label_cols_for_plot, args.age_col, outdir, "random_cls")

    logger.info("  Metrics bar chart")
    plot_metrics_bar(all_cluster_metrics, outdir)

    # Overview grid (CLS vs mean, rows × cols)
    umap_keys = {"Pretrained CLS": "X_umap_cls", "Pretrained Mean": "X_umap_mean"}
    if rand_cls is not None:
        umap_keys["Random CLS"] = "X_umap_random"
    logger.info("  Overview grid")
    plot_embedding_grid({}, adata, umap_keys, args.age_col, outdir)

    # Report
    write_report(outdir, args, all_cluster_metrics, all_age_results)

    logger.info("\nAll done.")
    logger.info(f"Outputs → {outdir}/")
    logger.info(f"  embeddings_cls.npy    CLS 256-dim [N, 256]")
    logger.info(f"  adata.h5ad            AnnData with UMAP + metadata")
    logger.info(f"  figures/              all plots")
    logger.info(f"  report.txt            metrics summary")


if __name__ == "__main__":
    main()
