#!/usr/bin/env python3
"""
Extract a representative demo subset from the 21k CpG h5ad.

Goal: a compact file that can be shared publicly with the repo so
      tutorial notebooks show real biology, not synthetic data.

Stratification:
  - Age bins (0-20, 20-30, 30-40, 40-50, 50-60, 60-70, 70-80, 80+): n_samples/8 per bin
  - Tissue diversity: cap n_samples/8 samples from any single tissue
  - Split: keeps train/valid/test labels intact
  - NaN rate: prefers samples with fewer missing CpGs (lower NaN fraction)

Output:
  methylllama_demo_500samples.h5ad  (or as specified by --output)

Usage on cluster:
  cd /path/to/MethylLlama
  python data_prep/extract_demo_samples.py \
      --input  /sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad \
      --output ./methylllama_demo_500samples.h5ad \
      --n_samples 500
"""

import argparse
import logging
from pathlib import Path

import anndata as ad
import numpy as np
import pandas as pd
import scanpy as sc

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)

AGE_BINS = [0, 20, 30, 40, 50, 60, 70, 80, 120]
AGE_BIN_LABELS = ["0-20", "20-30", "30-40", "40-50", "50-60", "60-70", "70-80", "80+"]


def load_h5ad(path: str) -> ad.AnnData:
    """Load h5ad with h5py fallback for old-format files."""
    try:
        adata = sc.read_h5ad(path)
        logger.info(f"Loaded: {adata.shape[0]} samples × {adata.shape[1]} CpGs")
        return adata
    except Exception as exc:
        logger.warning(f"sc.read_h5ad failed ({exc}), trying h5py fallback…")

    import h5py

    def _read_group(grp, n_rows):
        data = {}
        for key in grp.keys():
            item = grp[key]
            try:
                if isinstance(item, h5py.Dataset) and item.shape == (n_rows,):
                    arr = item[:]
                    if arr.dtype.kind in ("S", "O"):
                        arr = arr.astype(str)
                    data[key] = arr
                elif isinstance(item, h5py.Group) and "codes" in item and "categories" in item:
                    codes = item["codes"][:]
                    cats = item["categories"][:].astype(str)
                    data[key] = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], "")
            except Exception:
                pass
        return data

    with h5py.File(path, "r") as f:
        n_obs, n_var = f["X"].shape
        obs_data = _read_group(f["obs"], n_obs)
        obs_idx = obs_data.pop("_index", np.arange(n_obs).astype(str))
        var_data = _read_group(f["var"], n_var)
        var_idx = var_data.pop("_index", np.arange(n_var).astype(str))
        logger.info(f"h5py fallback: loading X {(n_obs, n_var)} ({n_obs * n_var * 4 / 1e9:.1f} GB)…")
        X = f["X"][:]

    adata = ad.AnnData(
        X=X,
        obs=pd.DataFrame(obs_data, index=obs_idx),
        var=pd.DataFrame(var_data, index=var_idx),
    )
    logger.info(f"h5py loaded: {adata.shape[0]} samples × {adata.shape[1]} CpGs")
    return adata


def nan_fraction(X: np.ndarray) -> np.ndarray:
    """Return per-sample NaN fraction (works for dense arrays)."""
    if hasattr(X, "toarray"):
        X = X.toarray()
    return np.isnan(X).mean(axis=1)


def stratified_sample(adata: ad.AnnData, n_samples: int, seed: int = 42) -> ad.AnnData:
    import math
    target_per_bin = math.ceil(n_samples / len(AGE_BIN_LABELS))
    max_per_tissue = max(math.ceil(n_samples / 8), 30)

    rng = np.random.default_rng(seed)
    obs = adata.obs.copy()

    # ── Require age column ────────────────────────────────────────────────────
    if "age" not in obs.columns:
        raise ValueError("h5ad obs has no 'age' column — cannot stratify")

    obs["age_num"] = pd.to_numeric(obs["age"], errors="coerce")
    obs = obs[obs["age_num"].notna()].copy()
    logger.info(f"Samples with valid age: {len(obs)}")

    # ── NaN rate per sample (prefer low-NaN) ─────────────────────────────────
    X_sub = adata[obs.index].X
    if hasattr(X_sub, "toarray"):
        X_sub = X_sub.toarray()
    obs["nan_frac"] = np.isnan(X_sub.astype(np.float32)).mean(axis=1)
    logger.info(f"NaN fraction — mean: {obs['nan_frac'].mean():.3f}, "
                f"median: {obs['nan_frac'].median():.3f}")

    # ── Age bins ──────────────────────────────────────────────────────────────
    obs["age_bin"] = pd.cut(obs["age_num"], bins=AGE_BINS, labels=AGE_BIN_LABELS, right=False)
    logger.info("Age bin distribution:\n" + str(obs["age_bin"].value_counts().sort_index()))

    # ── Tissue column (optional) ──────────────────────────────────────────────
    tissue_col = next((c for c in ["tissue", "tissue_type", "cell_type"] if c in obs.columns), None)
    if tissue_col:
        logger.info(f"Using '{tissue_col}' for tissue diversity (top 10):\n"
                    + str(obs[tissue_col].value_counts().head(10)))

    # ── Stratified sampling ───────────────────────────────────────────────────
    selected_idx: list[str] = []
    tissue_counts: dict[str, int] = {}

    for bin_label in AGE_BIN_LABELS:
        pool = obs[obs["age_bin"] == bin_label].copy()
        if pool.empty:
            continue
        # Sort by NaN fraction ascending (prefer clean samples)
        pool = pool.sort_values("nan_frac")
        # Tissue diversity: skip if a tissue is already capped
        chosen: list[str] = []
        for idx in pool.index:
            if len(chosen) >= target_per_bin:
                break
            tissue = obs.loc[idx, tissue_col] if tissue_col else "unknown"
            if tissue_counts.get(str(tissue), 0) >= max_per_tissue:
                continue
            chosen.append(idx)
            tissue_counts[str(tissue)] = tissue_counts.get(str(tissue), 0) + 1

        # If not enough from tissue-diversity pass, fill without constraint
        if len(chosen) < target_per_bin:
            remaining = pool.index.difference(chosen)
            fill = list(remaining[: target_per_bin - len(chosen)])
            chosen.extend(fill)

        logger.info(f"  Bin {bin_label}: selected {len(chosen)} samples")
        selected_idx.extend(chosen)

    # ── Deduplicate and cap at n_samples ─────────────────────────────────────
    seen: set = set()
    unique_idx = [i for i in selected_idx if not (i in seen or seen.add(i))]
    if len(unique_idx) < n_samples:
        logger.warning(f"Only {len(unique_idx)} samples found (target {n_samples}). "
                       "Dataset may be small for some age bins.")
    else:
        unique_idx = unique_idx[:n_samples]

    subset = adata[unique_idx].copy()
    # Copy computed columns into subset.obs so we can log stats before dropping them
    for col in ["age_num", "nan_frac", "age_bin"]:
        if col in obs.columns:
            subset.obs[col] = obs.loc[unique_idx, col].values

    logger.info(f"\nFinal subset: {subset.shape[0]} samples × {subset.shape[1]} CpGs")
    if tissue_col:
        logger.info("Tissue distribution:\n" + str(subset.obs[tissue_col].value_counts().head(15)))
    logger.info(f"Split distribution:\n{subset.obs.get('split', pd.Series(dtype=str)).value_counts()}")
    logger.info(f"Age: min={subset.obs['age_num'].min():.1f}, "
                f"max={subset.obs['age_num'].max():.1f}, "
                f"mean={subset.obs['age_num'].mean():.1f}")

    # Drop temporary columns before saving
    drop_cols = [c for c in ["age_num", "nan_frac", "age_bin"] if c in subset.obs.columns]
    subset.obs.drop(columns=drop_cols, inplace=True)

    return subset


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--input",
        default="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad",
        help="Path to full 21k CpG h5ad file",
    )
    parser.add_argument(
        "--output",
        default="methylllama_demo_120samples.h5ad",
        help="Output path for demo h5ad",
    )
    parser.add_argument("--n_samples", type=int, default=120)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    logger.info("=" * 60)
    logger.info("MethylLlama — Demo Dataset Extraction")
    logger.info("=" * 60)

    adata = load_h5ad(args.input)

    # ── Print all available obs columns for transparency ──────────────────────
    logger.info(f"\nAvailable obs columns: {list(adata.obs.columns)}")
    logger.info(f"Available var columns: {list(adata.var.columns)}")
    for col in ["age", "split", "tissue", "sex", "disease", "dataset"]:
        if col in adata.obs.columns:
            vc = adata.obs[col].value_counts()
            logger.info(f"\nColumn '{col}':\n{vc.head(10)}")

    subset = stratified_sample(adata, n_samples=args.n_samples, seed=args.seed)

    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    subset.write_h5ad(out)

    size_mb = out.stat().st_size / 1e6
    logger.info(f"\nSaved: {out}  ({size_mb:.1f} MB)")
    logger.info("Copy to local machine:")
    logger.info(f"  rsync -av netanel.azran@moriah:{out.resolve()} ~/Downloads/")


if __name__ == "__main__":
    main()
