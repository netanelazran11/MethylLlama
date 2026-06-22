#!/usr/bin/env python3
"""
Task C-2 — Contrastive Representation Analysis.

Extracts CLS embeddings from the pretrained WCED encoder (no fine-tuning)
and evaluates whether the learned representations already encode biological
structure without any task-specific supervision.

Analyses:
  1. UMAP visualization colored by age / smoking / sex / tissue
  2. Linear probe: logistic regression on frozen CLS → accuracy per task
  3. KNN probe: k-nearest-neighbor accuracy (k=5) from CLS distances
  4. Clustering: K-means silhouette score by label groups
  5. t-SNE (optional, slower)

Usage:
  python -m bmfm_methylation.downstream.probing.embedding_analysis \
      --checkpoint_path /path/to/wced.ckpt \
      --data_path /path/to/data.h5ad \
      --output_dir ./outputs/downstream/probing/embeddings \
      --label_cols age smoking_status sex
"""

import argparse
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import torch

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

SUBSET_K = 4000
BATCH_SIZE = 64


# ─────────────────────────────────────────────────────────────────────────────
# Embedding extraction
# ─────────────────────────────────────────────────────────────────────────────

def extract_embeddings(encoder, h5ad_path, subset_k=SUBSET_K, device=None, split=None):
    """
    Run the full dataset through the encoder (frozen) and return:
      embeddings: [N, hidden_size] numpy array
      metadata:   pandas DataFrame with obs columns
    """
    import torch.utils.data
    from bmfm_methylation.shared.data_module import _read_h5ad_robust

    if device is None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    adata = _read_h5ad_robust(h5ad_path)
    if split is not None and "split" in adata.obs.columns:
        adata = adata[adata.obs["split"] == split].copy()

    X = adata.X if not hasattr(adata.X, "toarray") else adata.X.toarray()
    n_cpg = X.shape[1]

    try:
        cpg_vocab = np.array(adata.var["cpg_id"].values, dtype=np.int64)
    except (KeyError, AttributeError):
        cpg_vocab = np.arange(5, n_cpg + 5, dtype=np.int64)

    # Fixed CpG subset for reproducibility
    rng = np.random.default_rng(42)
    fixed_idx = rng.choice(n_cpg, min(subset_k, n_cpg), replace=False)
    fixed_idx.sort()

    encoder = encoder.to(device)
    encoder.eval()

    all_embeddings = []

    with torch.no_grad():
        for start in range(0, len(X), BATCH_SIZE):
            batch_X = X[start:start + BATCH_SIZE]
            batch_size = batch_X.shape[0]

            # Sample the fixed CpG subset (skip NaN per sample)
            cpg_ids_batch = []
            beta_batch = []
            attn_batch = []
            max_len = 0

            for i in range(batch_size):
                row = batch_X[i]
                valid = ~np.isnan(row[fixed_idx])
                chosen = fixed_idx[valid]
                k = len(chosen)
                cpg_ids_batch.append(cpg_vocab[chosen].astype(np.float32))
                beta_batch.append(row[chosen].astype(np.float32))
                attn_batch.append(np.ones(k, dtype=np.float32))
                max_len = max(max_len, k)

            # Pad to max_len
            cpg_t = torch.zeros(batch_size, max_len, device=device)
            beta_t = torch.zeros(batch_size, max_len, device=device)
            attn_t = torch.zeros(batch_size, max_len, device=device)
            for i in range(batch_size):
                n = len(cpg_ids_batch[i])
                cpg_t[i, :n] = torch.from_numpy(cpg_ids_batch[i])
                beta_t[i, :n] = torch.from_numpy(beta_batch[i])
                attn_t[i, :n] = 1.0

            input_ids = torch.stack([cpg_t, beta_t], dim=1)  # [B, 2, seq]
            out = encoder(input_ids, attention_mask=attn_t)
            cls = out.pooler_output  # [B, hidden]
            all_embeddings.append(cls.cpu().numpy())

            if (start // BATCH_SIZE) % 10 == 0:
                logger.info(f"  Embedded {min(start + BATCH_SIZE, len(X))}/{len(X)} samples")

    embeddings = np.concatenate(all_embeddings, axis=0)
    metadata = adata.obs.copy()
    logger.info(f"Extracted embeddings: {embeddings.shape}")
    return embeddings, metadata


# ─────────────────────────────────────────────────────────────────────────────
# Probing analyses
# ─────────────────────────────────────────────────────────────────────────────

def linear_probe(embeddings, labels, task_name, test_frac=0.2, seed=42):
    """Logistic/ridge regression on frozen embeddings. Returns accuracy or R²."""
    from sklearn.model_selection import train_test_split
    from sklearn.preprocessing import StandardScaler

    valid = labels >= 0 if labels.dtype in [np.int64, np.int32] else ~np.isnan(labels)
    X = embeddings[valid]
    y = labels[valid]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if labels.dtype in [np.int64, np.int32]:
        from sklearn.linear_model import LogisticRegression
        from sklearn.metrics import accuracy_score, f1_score
        clf = LogisticRegression(max_iter=1000, random_state=seed)
        clf.fit(X_tr, y_tr)
        preds = clf.predict(X_te)
        acc = accuracy_score(y_te, preds)
        f1 = f1_score(y_te, preds, average="macro", zero_division=0)
        logger.info(f"  Linear probe [{task_name}]: acc={acc:.4f}, macro-F1={f1:.4f}")
        return {"task": task_name, "type": "classification", "accuracy": acc, "f1_macro": f1}
    else:
        from sklearn.linear_model import Ridge
        from sklearn.metrics import r2_score, mean_absolute_error
        reg = Ridge()
        reg.fit(X_tr, y_tr)
        preds = reg.predict(X_te)
        r2 = r2_score(y_te, preds)
        mae = mean_absolute_error(y_te, preds)
        logger.info(f"  Linear probe [{task_name}]: R²={r2:.4f}, MAE={mae:.2f}")
        return {"task": task_name, "type": "regression", "r2": r2, "mae": mae}


def knn_probe(embeddings, labels, task_name, k=5, test_frac=0.2, seed=42):
    """KNN accuracy from CLS distance (no training)."""
    from sklearn.model_selection import train_test_split
    from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
    from sklearn.preprocessing import StandardScaler

    valid = labels >= 0 if labels.dtype in [np.int64, np.int32] else ~np.isnan(labels)
    X = embeddings[valid]
    y = labels[valid]

    X_tr, X_te, y_tr, y_te = train_test_split(X, y, test_size=test_frac, random_state=seed)
    scaler = StandardScaler()
    X_tr = scaler.fit_transform(X_tr)
    X_te = scaler.transform(X_te)

    if labels.dtype in [np.int64, np.int32]:
        from sklearn.metrics import accuracy_score
        clf = KNeighborsClassifier(n_neighbors=k)
        clf.fit(X_tr, y_tr)
        acc = accuracy_score(y_te, clf.predict(X_te))
        logger.info(f"  KNN probe [{task_name}]: acc={acc:.4f}")
        return {"task": task_name, "type": "knn_classification", "accuracy": acc, "k": k}
    else:
        from sklearn.metrics import r2_score
        reg = KNeighborsRegressor(n_neighbors=k)
        reg.fit(X_tr, y_tr)
        r2 = r2_score(y_te, reg.predict(X_te))
        logger.info(f"  KNN probe [{task_name}]: R²={r2:.4f}")
        return {"task": task_name, "type": "knn_regression", "r2": r2, "k": k}


def umap_plot(embeddings, metadata, label_cols, output_dir, random_init_embeddings=None):
    """UMAP projection colored by each label_col. Side-by-side WCED vs random if provided."""
    try:
        import umap
        import matplotlib.pyplot as plt
        import matplotlib.cm as cm
    except ImportError:
        logger.warning("umap-learn or matplotlib not installed. pip install umap-learn matplotlib")
        return

    logger.info("Running UMAP...")
    from sklearn.preprocessing import StandardScaler

    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)
    reducer = umap.UMAP(n_components=2, random_state=42, n_neighbors=30, min_dist=0.1)
    proj = reducer.fit_transform(emb_scaled)

    proj_rand = None
    if random_init_embeddings is not None:
        emb_rand_scaled = scaler.transform(random_init_embeddings)
        proj_rand = reducer.transform(emb_rand_scaled)

    for col in label_cols:
        if col not in metadata.columns:
            continue

        fig_cols = 2 if proj_rand is not None else 1
        fig, axes = plt.subplots(1, fig_cols, figsize=(7 * fig_cols, 6))
        if fig_cols == 1:
            axes = [axes]

        for ax, (title, p) in zip(axes, [("WCED pretrained", proj), ("Random init", proj_rand)]):
            if p is None:
                continue
            vals = metadata[col].values
            try:
                # Try numeric coloring (e.g., age)
                numeric = pd.to_numeric(vals, errors="coerce")
                valid = ~np.isnan(numeric)
                sc = ax.scatter(p[valid, 0], p[valid, 1], c=numeric[valid],
                                cmap="viridis", s=3, alpha=0.6)
                plt.colorbar(sc, ax=ax, label=col)
            except Exception:
                # Categorical coloring
                categories = sorted(set(str(v) for v in vals))
                cat_map = {c: i for i, c in enumerate(categories)}
                colors = [cat_map.get(str(v), -1) for v in vals]
                cmap = cm.get_cmap("tab10", len(categories))
                sc = ax.scatter(p[:, 0], p[:, 1], c=colors, cmap=cmap, s=3, alpha=0.6,
                                vmin=0, vmax=len(categories) - 1)
                handles = [plt.Line2D([0], [0], marker="o", color="w",
                                      markerfacecolor=cmap(cat_map[c]), markersize=6, label=c)
                           for c in categories]
                ax.legend(handles=handles, loc="best", fontsize=7, markerscale=2)

            ax.set_title(f"{title} — colored by {col}", fontsize=11)
            ax.set_xlabel("UMAP 1")
            ax.set_ylabel("UMAP 2")
            ax.set_aspect("equal")

        fig.tight_layout()
        fig_path = output_dir / f"umap_{col}.png"
        fig.savefig(fig_path, dpi=150, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"  UMAP saved: {fig_path}")


def silhouette_analysis(embeddings, metadata, label_cols):
    """K-means silhouette score for each categorical label."""
    from sklearn.metrics import silhouette_score
    from sklearn.preprocessing import LabelEncoder, StandardScaler

    results = []
    scaler = StandardScaler()
    emb_scaled = scaler.fit_transform(embeddings)

    for col in label_cols:
        if col not in metadata.columns:
            continue
        vals = metadata[col].values
        try:
            # Must be categorical
            le = LabelEncoder()
            encoded = le.fit_transform([str(v) for v in vals])
            if len(np.unique(encoded)) < 2:
                continue
            score = silhouette_score(emb_scaled, encoded, sample_size=min(5000, len(emb_scaled)), random_state=42)
            logger.info(f"  Silhouette [{col}]: {score:.4f}")
            results.append({"label": col, "silhouette": score})
        except Exception as e:
            logger.warning(f"  Silhouette [{col}] failed: {e}")

    return results


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    import torch
    import torch.serialization
    _orig = torch.load
    def _p(*a, **kw):
        kw["weights_only"] = False
        return _orig(*a, **kw)
    torch.load = _p
    torch.serialization.load = _p

    parser = argparse.ArgumentParser()
    parser.add_argument("--checkpoint_path", required=True)
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--output_dir", default="./outputs/downstream/probing/embeddings")
    parser.add_argument("--label_cols", nargs="+", default=["age", "smoking_status", "sex"])
    parser.add_argument("--split", default=None, help="train/valid/test or None for all")
    parser.add_argument("--compare_random_init", action="store_true",
                        help="Also extract embeddings from random-init encoder for comparison")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    import sys
    sys.path.insert(0, str(Path(__file__).parent.parent.parent.parent))

    from bmfm_targets.config import SCBertConfig, TrainerConfig, FieldInfo
    from bmfm_methylation.wced.wced_module import WCEDTrainingModule
    from bmfm_methylation.shared.config import PretrainingConfig
    from bmfm_methylation.downstream.probing.data_efficiency import _build_encoder_config

    torch.serialization.add_safe_globals([SCBertConfig, TrainerConfig, FieldInfo])

    model_config = _build_encoder_config()
    model_config.checkpoint = None
    pt = WCEDTrainingModule.load_from_checkpoint(
        args.checkpoint_path,
        model_config=model_config,
        pretrain_config=PretrainingConfig(mode="wced"),
    )
    encoder = pt.encoder

    # ── Extract embeddings ────────────────────────────────────────────────────
    logger.info("Extracting WCED embeddings...")
    embeddings, metadata = extract_embeddings(
        encoder, args.data_path, subset_k=SUBSET_K, device=device, split=args.split,
    )
    np.save(output_dir / "embeddings_wced.npy", embeddings)
    metadata.to_csv(output_dir / "metadata.csv")

    rand_embeddings = None
    if args.compare_random_init:
        logger.info("Extracting random-init embeddings...")
        from bmfm_targets.models.predictive.scbert.modeling_scbert import SCBertModel
        rand_encoder = SCBertModel(_build_encoder_config())
        rand_embeddings, _ = extract_embeddings(
            rand_encoder, args.data_path, subset_k=SUBSET_K, device=device, split=args.split,
        )
        np.save(output_dir / "embeddings_random_init.npy", rand_embeddings)

    # ── Probing ───────────────────────────────────────────────────────────────
    probe_results = []
    label_arrays = {}

    SMOKING_MAP = {"current": 0, "former": 1, "never": 2}
    SEX_MAP = {"M": 0, "F": 1, "male": 0, "female": 1}

    for col in args.label_cols:
        if col not in metadata.columns:
            logger.warning(f"Column {col} not found in metadata, skipping")
            continue

        vals = metadata[col].values
        if col == "smoking_status":
            arr = np.array([SMOKING_MAP.get(str(v), -1) for v in vals], dtype=np.int64)
        elif col == "sex":
            arr = np.array([SEX_MAP.get(str(v), -1) for v in vals], dtype=np.int64)
        elif col == "age":
            import pandas as pd
            arr = pd.to_numeric(vals, errors="coerce").values.astype(np.float32)
        else:
            # Try to auto-detect type
            try:
                arr = vals.astype(np.float32)
            except (ValueError, TypeError):
                from sklearn.preprocessing import LabelEncoder
                le = LabelEncoder()
                arr = le.fit_transform([str(v) for v in vals]).astype(np.int64)

        label_arrays[col] = arr

        logger.info(f"\nProbing: {col}")
        res = linear_probe(embeddings, arr, col)
        probe_results.append(res)

        res_knn = knn_probe(embeddings, arr, col)
        probe_results.append(res_knn)

    probe_df = pd.DataFrame(probe_results)
    probe_df.to_csv(output_dir / "probe_results.csv", index=False)
    logger.info(f"\nProbing results saved to {output_dir / 'probe_results.csv'}")
    logger.info(probe_df.to_string())

    # ── Silhouette ────────────────────────────────────────────────────────────
    cat_cols = [c for c in args.label_cols if c in ["smoking_status", "sex"]]
    if cat_cols:
        sil_results = silhouette_analysis(embeddings, metadata, cat_cols)
        sil_df = pd.DataFrame(sil_results)
        sil_df.to_csv(output_dir / "silhouette_results.csv", index=False)

    # ── UMAP ─────────────────────────────────────────────────────────────────
    umap_plot(embeddings, metadata, args.label_cols, output_dir, rand_embeddings)

    logger.info(f"\nAll outputs saved to {output_dir}")


if __name__ == "__main__":
    main()
