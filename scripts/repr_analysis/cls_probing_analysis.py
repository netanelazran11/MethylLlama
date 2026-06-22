#!/usr/bin/env python3
"""
cls_probing_analysis.py
=======================
Analyze the CLS representation learned by the pretrained MethylLlama model.

Steps
-----
1. Extract pretrained CLS + mean-pool + (optionally) random-init CLS embeddings
2. UMAP coloured by tissue / age / sex / disease / dataset
3. Age probing on frozen CLS  (MLP + linear probe — R², MAE, MedAE, PCC)
4. Classification probing     (tissue / sex / disease — acc, F1-macro, AUROC)
5. Within-tissue age probe    (top tissues with enough samples)

All analyses compare:
  • pretrained_cls   — the WCED-supervised CLS bottleneck
  • pretrained_mean  — mean-pool over CpG tokens (no bottleneck)
  • random_cls       — same architecture, random weights (ablation)

Output (<outdir>/)
------------------
  embeddings_cls.npy         pretrained CLS  [N, 256]
  embeddings_mean.npy        pretrained mean-pool [N, 256]
  embeddings_random_cls.npy  random-init CLS (if --compare_random)
  metadata.csv
  probing_results.csv        all probe scores per embedding × task
  figures/umap_*.png
  figures/age_scatter_*.png
  figures/probing_summary.png
  report.txt
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
from scipy.stats import pearsonr
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, roc_auc_score
from sklearn.neural_network import MLPRegressor, MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import LabelEncoder, StandardScaler
from torch.utils.data import DataLoader

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s  %(levelname)-8s  %(message)s",
    datefmt="%H:%M:%S",
)
log = logging.getLogger(__name__)
sc.settings.verbosity = 1


# ─────────────────────────────────────────────────────────────────────────────
# CLI
# ─────────────────────────────────────────────────────────────────────────────

def parse_args():
    p = argparse.ArgumentParser(description="MethylLlama CLS probing analysis")
    p.add_argument("--checkpoint",  required=True)
    p.add_argument("--ckpt_type",   default="pretrain", choices=["pretrain", "finetune"])
    p.add_argument("--data",        required=True,  help="h5ad with metadata (finetune 19k)")
    p.add_argument("--tokenizer",   required=True)
    p.add_argument("--outdir",      default="outputs/repr_analysis/cls_probing")
    p.add_argument("--batch_size",  type=int, default=32)
    p.add_argument("--device",      default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--compare_random", action="store_true",
                   help="Also extract random-init CLS as ablation baseline")
    # Skip extraction if embeddings already exist
    p.add_argument("--cls_embeddings",    default=None,
                   help="Path to pre-computed embeddings_cls.npy  (skips extraction)")
    p.add_argument("--mean_embeddings",   default=None,
                   help="Path to pre-computed embeddings_mean.npy (skips extraction)")
    p.add_argument("--random_embeddings", default=None,
                   help="Path to pre-computed embeddings_random_cls.npy")
    p.add_argument("--n_pca",       type=int, default=50)
    p.add_argument("--n_neighbors", type=int, default=15)
    p.add_argument("--label_cols",  nargs="+", default=["tissue", "sex", "disease", "dataset"])
    p.add_argument("--age_col",     default="age")
    p.add_argument("--split_col",   default="split")
    p.add_argument("--min_tissue_samples", type=int, default=50)
    p.add_argument("--metadata",          default=None,
                   help="External metadata CSV/CSV.gz joined on obs_names (e.g. pretrain_metadata.csv.gz)")
    p.add_argument("--metadata_id_col",   default="GSM_ID",
                   help="Column in --metadata that matches h5ad obs_names")
    p.add_argument("--skip_probing",      action="store_true",
                   help="Only run UMAP, skip age/classification probing (use for pretrain data)")
    return p.parse_args()


# ─────────────────────────────────────────────────────────────────────────────
# Encoder loading & embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def load_encoder(checkpoint_path: str, ckpt_type: str):
    if ckpt_type == "pretrain":
        from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
        module = load_wced_llama_checkpoint(checkpoint_path)
    else:
        from bmfm_methylation.llama.finetune_llama import load_finetune_llama_checkpoint
        module = load_finetune_llama_checkpoint(checkpoint_path)
    encoder = module.encoder
    encoder.eval()
    log.info(f"Encoder: {encoder.config.num_hidden_layers}L × {encoder.config.hidden_size}D")
    return encoder


def build_random_encoder(ref_encoder):
    from bmfm_methylation.llama.model import MethylLlamaModel
    enc = MethylLlamaModel(ref_encoder.config)
    enc.eval()
    log.info("Random-init encoder built")
    return enc


@torch.no_grad()
def extract_embeddings(encoder, data_path, tokenizer_path, batch_size, device):
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator

    encoder = encoder.to(device)
    tokenizer = MultiFieldTokenizer.from_pretrained(tokenizer_path)
    dataset   = MethylationDataset(h5ad_path=data_path, split=None, normalize_age=False)
    cpg_sites = dataset.cpg_sites
    log.info(f"  Dataset: {len(dataset)} samples × {len(cpg_sites)} CpGs")

    collator = WCEDCollator(
        tokenizer=tokenizer, cpg_sites=cpg_sites,
        vocab_size=len(cpg_sites), input_ratio=1.0, contrastive=False,
    )
    loader = DataLoader(dataset, batch_size=batch_size, collate_fn=collator,
                        shuffle=False, num_workers=0,
                        pin_memory=(device == "cuda"))

    cls_list, mean_list = [], []
    for i, batch in enumerate(loader):
        cpg_ids     = batch["cpg_ids"].to(device)
        beta_values = batch["beta_values"].to(device)
        attn_mask   = batch["attention_mask"].to(device)
        input_ids   = torch.stack([cpg_ids.float(), beta_values], dim=1)
        out         = encoder(input_ids=input_ids, attention_mask=attn_mask)

        cls_emb  = out.pooler_output.cpu().float()
        hidden   = out.last_hidden_state[:, 1:, :].cpu().float()
        mask_1d  = attn_mask[:, 1:].cpu().float().unsqueeze(-1)
        mean_emb = (hidden * mask_1d).sum(1) / mask_1d.sum(1).clamp(min=1)

        cls_list.append(cls_emb.numpy())
        mean_list.append(mean_emb.numpy())
        if (i + 1) % 50 == 0:
            log.info(f"    batch {i+1}/{len(loader)}")

    cls_embs  = np.concatenate(cls_list)
    mean_embs = np.concatenate(mean_list)
    log.info(f"  Done — CLS {cls_embs.shape}  Mean {mean_embs.shape}")
    return cls_embs, mean_embs


# ─────────────────────────────────────────────────────────────────────────────
# Metadata helpers
# ─────────────────────────────────────────────────────────────────────────────

def _read_h5ad_obs(data_path):
    """Read obs names and obs columns safely, falling back to h5py if anndata fails."""
    try:
        import h5py
        with h5py.File(data_path, "r") as f:
            obs_grp = f["obs"]
            if "_index" in obs_grp:
                raw = obs_grp["_index"][:]
            elif "__categories" in obs_grp:
                raw = list(obs_grp.keys())
                raw = obs_grp[raw[0]][:]
            else:
                raw = np.arange(f["X"].shape[0] if "X" in f else f["raw/X"].shape[0])
            obs_names = [s.decode() if isinstance(s, bytes) else str(s) for s in raw]
            obs_cols = {}
            for col in obs_grp.keys():
                if col.startswith("_"):
                    continue
                try:
                    vals = obs_grp[col][:]
                    if hasattr(obs_grp[col], "attrs") and "categories" in obs_grp[col].attrs:
                        cats = obs_grp[col].attrs["categories"]
                        vals = [cats[i] if i < len(cats) else None for i in vals]
                    obs_cols[col] = [v.decode() if isinstance(v, bytes) else v for v in vals]
                except Exception:
                    pass
        return obs_names, obs_cols
    except Exception as e:
        log.warning(f"h5py obs read failed ({e}), falling back to anndata backed mode")
        adata = sc.read_h5ad(data_path, backed="r")
        return list(adata.obs_names), {c: adata.obs[c].values.tolist() for c in adata.obs.columns}


def load_metadata(data_path, label_cols, age_col, split_col,
                  metadata_path=None, metadata_id_col="GSM_ID") -> pd.DataFrame:
    obs_names, obs_cols = _read_h5ad_obs(data_path)
    want  = list({age_col, split_col} | set(label_cols))
    meta  = pd.DataFrame(index=obs_names)
    for col in want:
        if col in obs_cols:
            meta[col] = obs_cols[col]

    # Join external metadata (e.g. tissue/sex/disease from pretrain_metadata.csv.gz)
    if metadata_path:
        log.info(f"  Joining external metadata: {metadata_path}")
        ext = pd.read_csv(metadata_path)
        if metadata_id_col not in ext.columns:
            raise ValueError(f"--metadata_id_col '{metadata_id_col}' not in {list(ext.columns)}")
        ext = ext.drop_duplicates(subset=metadata_id_col).set_index(metadata_id_col)
        new_cols = [c for c in ext.columns if c not in meta.columns]
        meta = meta.join(ext[new_cols], how="left")
        n_matched = meta.notna().any(axis=1).sum()
        log.info(f"  Matched {n_matched:,} / {len(meta):,} samples to external metadata")

    if age_col in meta.columns:
        meta[age_col] = pd.to_numeric(meta[age_col], errors="coerce")
    log.info(f"  Metadata columns: {list(meta.columns)}  n={len(meta)}")
    return meta


def get_split_masks(meta, split_col):
    if split_col in meta.columns:
        s = meta[split_col].values
        return np.isin(s, ["train", "valid"]), s == "test"
    rng = np.random.default_rng(42)
    idx = np.arange(len(meta))
    test_idx = set(rng.choice(idx, size=max(1, int(0.2 * len(idx))), replace=False))
    train_mask = np.array([i not in test_idx for i in idx])
    log.info(f"  No split_col — random 80/20: train={train_mask.sum()} test={(~train_mask).sum()}")
    return train_mask, ~train_mask


# ─────────────────────────────────────────────────────────────────────────────
# Age probing
# ─────────────────────────────────────────────────────────────────────────────

def _age_metrics(y_true, y_pred, tag) -> dict:
    r2  = float(1 - np.sum((y_true - y_pred)**2) / (np.sum((y_true - y_true.mean())**2) + 1e-8))
    pcc = float(pearsonr(y_true, y_pred)[0])
    mae = float(np.abs(y_true - y_pred).mean())
    med = float(np.median(np.abs(y_true - y_pred)))
    log.info(f"  [{tag}]  R²={r2:.3f}  PCC={pcc:.3f}  MAE={mae:.1f}yr  MedAE={med:.1f}yr")
    return dict(tag=tag, task="age", R2=round(r2,4), PCC=round(pcc,4),
                MAE_yr=round(mae,2), MedAE_yr=round(med,2))


def run_age_probing(embs, meta, train_mask, test_mask, age_col, tag):
    if age_col not in meta.columns:
        return []
    ages     = meta[age_col].values.astype(float)
    valid_tr = train_mask & ~np.isnan(ages)
    valid_te = test_mask  & ~np.isnan(ages)
    if valid_tr.sum() < 20 or valid_te.sum() < 5:
        log.warning(f"  [{tag}] Not enough age samples")
        return []

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embs[valid_tr])
    X_te = scaler.transform(embs[valid_te])
    y_tr, y_te = ages[valid_tr], ages[valid_te]

    results, scatter_data = [], {}

    # Linear probe
    from sklearn.linear_model import LinearRegression
    lr = LinearRegression().fit(X_tr, y_tr)
    p  = lr.predict(X_te)
    results.append(_age_metrics(y_te, p, f"{tag}/linear_probe"))
    scatter_data["linear_probe"] = (y_te, p)

    # MLP probe
    mlp = MLPRegressor(hidden_layer_sizes=(128, 64), max_iter=500,
                       random_state=42, early_stopping=True, n_iter_no_change=15,
                       learning_rate_init=1e-3)
    mlp.fit(X_tr, y_tr)
    p = mlp.predict(X_te)
    results.append(_age_metrics(y_te, p, f"{tag}/mlp_probe"))
    scatter_data["mlp_probe"] = (y_te, p)

    return results, scatter_data


# ─────────────────────────────────────────────────────────────────────────────
# Classification probing
# ─────────────────────────────────────────────────────────────────────────────

def run_classification_probing(embs, meta, label_col, train_mask, test_mask, tag) -> dict:
    if label_col not in meta.columns:
        return {}
    labels = meta[label_col].astype(str).values
    valid  = labels != "nan"
    tr = train_mask & valid
    te = test_mask  & valid
    if tr.sum() < 10 or te.sum() < 5:
        return {}

    le = LabelEncoder()
    le.fit(labels[valid])   # fit on all valid labels so test-only classes don't crash
    y_tr = le.transform(labels[tr])
    y_te = le.transform(labels[te])
    n_cls = len(le.classes_)
    if n_cls < 2:
        return {}

    scaler = StandardScaler()
    X_tr = scaler.fit_transform(embs[tr])
    X_te = scaler.transform(embs[te])

    # Logistic probe
    clf = LogisticRegression(max_iter=1000, C=1.0, class_weight="balanced",
                              solver="lbfgs", random_state=42)
    clf.fit(X_tr, y_tr)
    y_pred = clf.predict(X_te)
    y_prob = clf.predict_proba(X_te)

    acc = float(accuracy_score(y_te, y_pred))
    f1  = float(f1_score(y_te, y_pred, average="macro", zero_division=0))
    try:
        auc = float(roc_auc_score(y_te, y_prob[:, 1] if n_cls == 2 else y_prob,
                                   multi_class="ovr" if n_cls > 2 else "raise",
                                   average="macro"))
    except Exception:
        auc = float("nan")

    log.info(f"  [{tag}/{label_col}]  acc={acc:.3f}  F1={f1:.3f}  AUROC={auc:.3f}  ({n_cls} classes)")
    return dict(tag=f"{tag}/{label_col}", task=f"classify_{label_col}",
                accuracy=round(acc,4), F1_macro=round(f1,4), AUROC=round(auc,4),
                n_classes=n_cls)


# ─────────────────────────────────────────────────────────────────────────────
# Within-tissue age probe
# ─────────────────────────────────────────────────────────────────────────────

def run_within_tissue_age(embs, meta, train_mask, test_mask,
                           age_col, tissue_col, tag, min_samples=50):
    if age_col not in meta.columns or tissue_col not in meta.columns:
        return []
    from sklearn.linear_model import LinearRegression
    ages    = meta[age_col].values.astype(float)
    tissues = meta[tissue_col].astype(str).values
    results = []
    for tissue in sorted(np.unique(tissues)):
        t_mask = tissues == tissue
        tr = train_mask & t_mask & ~np.isnan(ages)
        te = test_mask  & t_mask & ~np.isnan(ages)
        if tr.sum() < min_samples // 2 or te.sum() < 10:
            continue
        scaler = StandardScaler()
        X_tr = scaler.fit_transform(embs[tr])
        X_te = scaler.transform(embs[te])
        lr = LinearRegression().fit(X_tr, ages[tr])
        p  = lr.predict(X_te)
        r  = _age_metrics(ages[te], p, f"{tag}/{tissue}")
        r["tissue"] = tissue
        results.append(r)
    return results


# ─────────────────────────────────────────────────────────────────────────────
# UMAP
# ─────────────────────────────────────────────────────────────────────────────

def run_and_plot_umap(embs, meta, label_cols, age_col, outdir, tag, n_pca, n_neighbors):
    fig_dir = outdir / "figures"
    fig_dir.mkdir(exist_ok=True)

    work = sc.AnnData(X=embs.astype(np.float32))
    for col in list(label_cols) + [age_col]:
        if col in meta.columns:
            work.obs[col] = meta[col].values

    n_pca_eff = min(n_pca, embs.shape[1] - 1, embs.shape[0] - 1)
    sc.tl.pca(work, n_comps=n_pca_eff)
    sc.pp.neighbors(work, n_neighbors=n_neighbors, use_rep="X_pca")
    sc.tl.umap(work)
    coords = work.obsm["X_umap"]

    for col in label_cols:
        if col not in work.obs.columns:
            continue
        vals = work.obs[col].astype(str).values
        cats = sorted(np.unique(vals))
        palette = plt.cm.get_cmap("tab20", max(len(cats), 1))
        fig, ax = plt.subplots(figsize=(7, 6))
        for i, cat in enumerate(cats):
            mask = vals == cat
            ax.scatter(coords[mask, 0], coords[mask, 1],
                       s=3, alpha=0.5, c=[palette(i)], label=cat, rasterized=True)
        if len(cats) <= 20:
            ax.legend(markerscale=5, fontsize=6, ncol=max(1, len(cats)//10))
        ax.set_title(f"{tag}  |  {col}", fontsize=10)
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fname = fig_dir / f"umap_{tag}_{col}.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"    {fname.name}")

    if age_col in work.obs.columns:
        vals = pd.to_numeric(work.obs[age_col], errors="coerce").values
        valid = ~np.isnan(vals)
        fig, ax = plt.subplots(figsize=(7, 6))
        sc_obj = ax.scatter(coords[valid, 0], coords[valid, 1], s=3, alpha=0.5,
                            c=vals[valid], cmap="RdYlBu_r", rasterized=True)
        plt.colorbar(sc_obj, ax=ax, fraction=0.04)
        ax.set_title(f"{tag}  |  age (years)")
        ax.set_xlabel("UMAP 1"); ax.set_ylabel("UMAP 2")
        ax.set_xticks([]); ax.set_yticks([])
        plt.tight_layout()
        fname = fig_dir / f"umap_{tag}_age.png"
        plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
        log.info(f"    {fname.name}")


def plot_age_scatter(y_true, y_pred, metrics_dict, outdir, suffix):
    fig_dir = outdir / "figures"
    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_true, y_pred, s=5, alpha=0.4, c="steelblue", rasterized=True)
    lo = min(y_true.min(), y_pred.min()) - 2
    hi = max(y_true.max(), y_pred.max()) + 2
    ax.plot([lo, hi], [lo, hi], "k--", lw=0.9)
    r2  = metrics_dict.get("R2", float("nan"))
    mae = metrics_dict.get("MAE_yr", float("nan"))
    med = metrics_dict.get("MedAE_yr", float("nan"))
    ax.set_title(f"{suffix}\nR²={r2:.3f}  MAE={mae:.1f}yr  MedAE={med:.1f}yr")
    ax.set_xlabel("True age (years)"); ax.set_ylabel("Predicted age (years)")
    plt.tight_layout()
    fname = fig_dir / f"age_scatter_{suffix.replace('/', '_')}.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"    {fname.name}")


def plot_probing_summary(results_df, outdir):
    fig_dir = outdir / "figures"
    age_df = results_df[results_df["task"] == "age"].copy()
    if age_df.empty:
        return
    age_df = age_df.sort_values("R2", ascending=False).head(20)

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, max(4, len(age_df) * 0.4)))
    ax1.barh(age_df["tag"], age_df["R2"].astype(float), color="steelblue", alpha=0.85)
    ax1.set_xlabel("R²  (↑ better)"); ax1.set_title("Age probe — R²")
    ax1.axvline(0, color="gray", lw=0.7, ls="--")

    ax2.barh(age_df["tag"], age_df["MAE_yr"].astype(float), color="tomato", alpha=0.85)
    ax2.set_xlabel("MAE (years)  (↓ better)"); ax2.set_title("Age probe — MAE")

    plt.suptitle("CLS Probing: Age Prediction Across Embeddings", fontsize=11)
    plt.tight_layout()
    fname = fig_dir / "probing_summary.png"
    plt.savefig(fname, dpi=150, bbox_inches="tight"); plt.close()
    log.info(f"    {fname.name}")


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────

def write_report(outdir, args, all_results):
    age_rows = [r for r in all_results if r.get("task") == "age" and "tissue" not in r]
    clf_rows = [r for r in all_results if str(r.get("task","")).startswith("classify")]
    tis_rows = [r for r in all_results if "tissue" in r]

    SEP = "─" * 70
    lines = [
        "=" * 70, "  MethylLlama-Small — CLS Probing Analysis", "=" * 70,
        f"  Checkpoint : {args.checkpoint}",
        f"  Data       : {args.data}",
        "",
        SEP, "  Age Probing (R²  PCC  MAE  MedAE)", SEP,
    ]
    for r in sorted(age_rows, key=lambda x: -float(x.get("R2", -999))):
        lines.append(
            f"  {r['tag']:<45s}  R²={float(r['R2']):+.3f}  "
            f"MAE={float(r['MAE_yr']):.1f}yr  MedAE={float(r['MedAE_yr']):.1f}yr"
        )

    lines += ["", SEP, "  Classification Probing (acc  F1-macro  AUROC)", SEP]
    for r in clf_rows:
        lines.append(
            f"  {r['tag']:<45s}  acc={float(r['accuracy']):.3f}  "
            f"F1={float(r['F1_macro']):.3f}  AUROC={float(r['AUROC']):.3f}"
        )

    if tis_rows:
        lines += ["", SEP, "  Within-Tissue Age Probe (Ridge)", SEP]
        for r in sorted(tis_rows, key=lambda x: -float(x.get("R2", -999))):
            lines.append(
                f"  {r['tag']:<50s}  R²={float(r['R2']):+.3f}  MAE={float(r['MAE_yr']):.1f}yr"
            )

    lines += ["", "=" * 70]
    report = "\n".join(lines)
    print("\n" + report)
    (outdir / "report.txt").write_text(report)
    log.info(f"Report → {outdir / 'report.txt'}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    args   = parse_args()
    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "figures").mkdir(exist_ok=True)

    log.info("=" * 70)
    log.info("  MethylLlama CLS Probing Analysis")
    log.info("=" * 70)

    # 1 & 2. Embeddings — load from disk if already computed, else extract
    if args.cls_embeddings and args.mean_embeddings:
        log.info("\n[1/7] Loading pre-computed embeddings from disk ...")
        cls_embs  = np.load(args.cls_embeddings)
        mean_embs = np.load(args.mean_embeddings)
        log.info(f"  CLS  {cls_embs.shape}  loaded from {args.cls_embeddings}")
        log.info(f"  Mean {mean_embs.shape}  loaded from {args.mean_embeddings}")
        np.save(outdir / "embeddings_cls.npy",  cls_embs)
        np.save(outdir / "embeddings_mean.npy", mean_embs)
    else:
        log.info("\n[1/7] Loading encoder ...")
        encoder = load_encoder(args.checkpoint, args.ckpt_type)
        log.info("\n[2/7] Extracting pretrained CLS + mean embeddings ...")
        cls_embs, mean_embs = extract_embeddings(
            encoder, args.data, args.tokenizer, args.batch_size, args.device
        )
        np.save(outdir / "embeddings_cls.npy",  cls_embs)
        np.save(outdir / "embeddings_mean.npy", mean_embs)

    rand_cls = None
    if args.random_embeddings:
        log.info("\n[2b] Loading pre-computed random-init CLS from disk ...")
        rand_cls = np.load(args.random_embeddings)
        log.info(f"  Random {rand_cls.shape}  loaded from {args.random_embeddings}")
        np.save(outdir / "embeddings_random_cls.npy", rand_cls)
    elif args.compare_random:
        log.info("\n[2b] Extracting random-init CLS (ablation) ...")
        encoder = load_encoder(args.checkpoint, args.ckpt_type)
        rand_enc = build_random_encoder(encoder)
        rand_cls, _ = extract_embeddings(
            rand_enc, args.data, args.tokenizer, args.batch_size, args.device
        )
        np.save(outdir / "embeddings_random_cls.npy", rand_cls)

    # 3. Metadata
    log.info("\n[3/7] Loading metadata ...")
    meta = load_metadata(args.data, args.label_cols, args.age_col, args.split_col,
                         metadata_path=args.metadata, metadata_id_col=args.metadata_id_col)
    meta.to_csv(outdir / "metadata.csv")
    train_mask, test_mask = get_split_masks(meta, args.split_col)
    log.info(f"  Train={train_mask.sum()}  Test={test_mask.sum()}")

    # 4. UMAP
    log.info("\n[4/7] Running UMAPs ...")
    label_cols = [c for c in args.label_cols if c in meta.columns]
    probe_targets = [("pretrained_cls", cls_embs), ("pretrained_mean", mean_embs)]
    if rand_cls is not None:
        probe_targets.append(("random_cls", rand_cls))

    for tag, embs in probe_targets:
        run_and_plot_umap(embs, meta, label_cols, args.age_col,
                          outdir, tag, args.n_pca, args.n_neighbors)

    if args.skip_probing:
        log.info("\n[5-7/7] Skipping probing (--skip_probing set — pretrain data)")
        log.info(f"\nAll done → {outdir}/")
        return

    # 5. Age probing
    log.info("\n[5/7] Age probing ...")
    all_results = []
    for tag, embs in probe_targets:
        res = run_age_probing(embs, meta, train_mask, test_mask, args.age_col, tag)
        if res:
            age_results, scatter_data = res
            all_results.extend(age_results)
            for probe_name, (yt, yp) in scatter_data.items():
                m = next((r for r in age_results if r["tag"].endswith(f"/{probe_name}")), {})
                if m:
                    plot_age_scatter(yt, yp, m, outdir, f"{tag}_{probe_name}")

    # 6. Classification probing
    log.info("\n[6/7] Classification probing ...")
    for label_col in [c for c in ["tissue", "sex", "disease"] if c in meta.columns]:
        for tag, embs in probe_targets:
            r = run_classification_probing(embs, meta, label_col, train_mask, test_mask, tag)
            if r:
                all_results.append(r)

    # 7. Within-tissue age probe
    log.info("\n[7/7] Within-tissue age probing ...")
    tissue_col = next((c for c in ["tissue", "tissue_type"] if c in meta.columns), None)
    if tissue_col:
        for tag, embs in [("pretrained_cls", cls_embs)]:  # main embedding only
            tissue_results = run_within_tissue_age(
                embs, meta, train_mask, test_mask,
                args.age_col, tissue_col, tag, args.min_tissue_samples
            )
            all_results.extend(tissue_results)

    # Save
    clean = [r for r in all_results if isinstance(r, dict)]
    results_df = pd.DataFrame(clean)
    results_df.to_csv(outdir / "probing_results.csv", index=False)
    log.info(f"  probing_results.csv  ({len(results_df)} rows)")

    plot_probing_summary(results_df, outdir)
    write_report(outdir, args, clean)

    log.info(f"\nAll done → {outdir}/")


if __name__ == "__main__":
    main()
