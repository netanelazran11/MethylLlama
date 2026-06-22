"""
CpG Informativeness Analysis
==============================
Comprehensive analysis of all CpG sites and all samples in the fine-tuning h5ad.

Sections:
  1. Dataset overview — shape, splits, age/tissue distributions
  2. CpG-level variance — identify low-variance (uninformative) sites
  3. CpG-age Pearson correlation — on TRAINING split only (no leakage)
  4. Combined informativeness score — variance rank + age-corr rank
  5. Split quality — compare age/tissue distribution across train/val/test
  6. Sample-level analysis — per-sample mean beta, outliers
  7. Outputs:
       analysis_report.txt        — full text report
       cpg_variance.npy           — [N_cpg] variance per CpG (all samples)
       cpg_age_corr.npy           — [N_cpg] |Pearson r| with age (train only)
       cpg_informativeness.npy    — [N_cpg] combined rank score (higher = more informative)
       cpg_top5k_indices.npy      — indices of top-5000 most informative CpGs
       cpg_top10k_indices.npy     — indices of top-10000 most informative CpGs
       split_age_stats.txt        — age mean/std/range per split

Usage:
    python scripts/utils/analyze_cpg_informativeness.py
"""

import os
import sys
import time
import numpy as np
import pandas as pd
import scanpy as sc
from pathlib import Path
from scipy.stats import pearsonr

# ── Paths ─────────────────────────────────────────────────────────────────────
H5AD_PATH = (
    "/sci/labs/benjamin.yakir/netanel.azran/data/"
    "data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified.h5ad"
)
OUTDIR = Path(
    "/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/"
    "outputs/cpg_analysis"
)
OUTDIR.mkdir(parents=True, exist_ok=True)

SEP = "=" * 70
sep = "-" * 70

lines = []   # collect all output for the text report

def p(*args, **kwargs):
    """Print and collect for report."""
    msg = " ".join(str(a) for a in args)
    print(msg, **kwargs)
    lines.append(msg)

# ── 1. Load data ──────────────────────────────────────────────────────────────
p(SEP)
p("CpG INFORMATIVENESS ANALYSIS")
p(f"File: {H5AD_PATH}")
p(f"Time: {time.strftime('%Y-%m-%d %H:%M:%S')}")
p(SEP)

p("\n[1] Loading h5ad ...")
adata = sc.read_h5ad(H5AD_PATH)
N, C = adata.n_obs, adata.n_vars
p(f"    {N} samples × {C} CpGs")
p(f"    obs columns: {list(adata.obs.columns)}")
p(f"    X dtype: {adata.X.dtype}  sparse: {hasattr(adata.X, 'toarray')}")

# Dense matrix — confirmed zero NaN in this file
import scipy.sparse as sp
X = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)
p(f"    X shape: {X.shape}  NaN count: {np.isnan(X).sum()}")

# ── 2. Dataset overview ───────────────────────────────────────────────────────
p(f"\n{SEP}")
p("[2] Dataset overview")
p(sep)

# Split counts
if "split" in adata.obs.columns:
    split_counts = adata.obs["split"].value_counts()
    p("Split counts:")
    for s, n in split_counts.items():
        p(f"  {s:8s}: {n:5d} ({100*n/N:.1f}%)")

# Age distribution
if "age" in adata.obs.columns:
    age = adata.obs["age"].astype(float)
    p(f"\nAge (all samples, N={age.notna().sum()}):")
    p(f"  mean±std : {age.mean():.1f} ± {age.std():.1f} years")
    p(f"  range    : [{age.min():.0f}, {age.max():.0f}] years")
    p(f"  median   : {age.median():.1f} years")
    p(f"  NaN      : {age.isna().sum()}")

    # Age histogram
    bins = np.arange(0, 110, 10)
    hist, edges = np.histogram(age.dropna(), bins=bins)
    p("\n  Age histogram (10-year bins):")
    for i, count in enumerate(hist):
        bar = "█" * (count // 20)
        p(f"    {edges[i]:3.0f}-{edges[i+1]:3.0f}: {count:4d}  {bar}")

# Tissue distribution
if "tissue_type" in adata.obs.columns:
    tissue_counts = adata.obs["tissue_type"].value_counts()
    p(f"\nTissue types ({len(tissue_counts)} unique):")
    for t, n in tissue_counts.head(20).items():
        p(f"  {str(t):35s}: {n:5d} ({100*n/N:.1f}%)")
    if len(tissue_counts) > 20:
        p(f"  ... and {len(tissue_counts)-20} more")

# Dataset source
if "dataset" in adata.obs.columns:
    ds_counts = adata.obs["dataset"].value_counts()
    p(f"\nDataset sources ({len(ds_counts)} unique):")
    for d, n in ds_counts.head(10).items():
        p(f"  {str(d):40s}: {n:5d}")

# ── 3. CpG-level variance (all samples) ──────────────────────────────────────
p(f"\n{SEP}")
p("[3] CpG-level variance analysis (all samples, no labels)")
p(sep)

t0 = time.time()
cpg_mean = X.mean(axis=0)          # [C]
cpg_var  = X.var(axis=0)           # [C]
cpg_std  = np.sqrt(cpg_var)
p(f"    Computed in {time.time()-t0:.1f}s")

p(f"\nVariance distribution across {C} CpGs:")
for pct in [0, 1, 5, 10, 25, 50, 75, 90, 95, 99, 100]:
    v = np.percentile(cpg_var, pct)
    p(f"  p{pct:3d}: {v:.6f}")

# How many CpGs at each variance threshold
thresholds = [0.001, 0.005, 0.01, 0.02, 0.05]
p("\nCpGs above variance threshold:")
for thr in thresholds:
    n_above = (cpg_var > thr).sum()
    p(f"  var > {thr:.3f}: {n_above:5d} / {C} ({100*n_above/C:.1f}%)")

# Mean beta distribution
p(f"\nCpG mean beta distribution:")
for pct in [0, 5, 25, 50, 75, 95, 100]:
    p(f"  p{pct:3d}: {np.percentile(cpg_mean, pct):.4f}")

n_near_zero  = (cpg_mean < 0.05).sum()
n_near_one   = (cpg_mean > 0.95).sum()
n_near_half  = ((cpg_mean > 0.4) & (cpg_mean < 0.6)).sum()
p(f"\n  Near-zero mean (< 0.05): {n_near_zero} CpGs ({100*n_near_zero/C:.1f}%) — nearly always unmethylated")
p(f"  Near-one  mean (> 0.95): {n_near_one}  CpGs ({100*n_near_one/C:.1f}%) — nearly always methylated")
p(f"  Near-half mean (0.4-0.6): {n_near_half} CpGs ({100*n_near_half/C:.1f}%) — intermediate")

np.save(OUTDIR / "cpg_variance.npy", cpg_var)
np.save(OUTDIR / "cpg_mean.npy", cpg_mean)
p(f"\nSaved: cpg_variance.npy, cpg_mean.npy")

# ── 4. CpG-age Pearson correlation (TRAINING split only) ─────────────────────
p(f"\n{SEP}")
p("[4] CpG-age Pearson correlation — TRAINING SPLIT ONLY (no leakage)")
p(sep)

if "age" in adata.obs.columns and "split" in adata.obs.columns:
    train_mask  = (adata.obs["split"] == "train").values
    X_train     = X[train_mask]
    age_train   = adata.obs.loc[train_mask, "age"].astype(float).values

    # Drop samples with NaN age
    valid_age   = np.isfinite(age_train)
    X_train_v   = X_train[valid_age]
    age_train_v = age_train[valid_age]
    p(f"    Training samples: {train_mask.sum()} total, {valid_age.sum()} with valid age")

    t0 = time.time()
    # Vectorized Pearson r: corrcoef between each CpG column and age vector
    # Efficient: center once, compute dot product
    X_centered   = X_train_v - X_train_v.mean(axis=0, keepdims=True)   # [N, C]
    age_centered = age_train_v - age_train_v.mean()                     # [N]

    # Numerator: dot product of age with each CpG column
    num = (X_centered * age_centered[:, None]).sum(axis=0)              # [C]

    # Denominator: std of age × std of each CpG × N
    age_std  = np.sqrt((age_centered ** 2).sum())
    cpg_std_train = np.sqrt((X_centered ** 2).sum(axis=0)).clip(min=1e-8)
    cpg_age_corr = num / (age_std * cpg_std_train)                      # [C] Pearson r
    p(f"    Computed in {time.time()-t0:.1f}s")

    abs_corr = np.abs(cpg_age_corr)

    p(f"\n|Pearson r| with age (training split only):")
    for pct in [0, 50, 75, 90, 95, 99, 100]:
        p(f"  p{pct:3d}: {np.percentile(abs_corr, pct):.4f}")

    thresholds_r = [0.1, 0.2, 0.3, 0.4, 0.5]
    p("\nCpGs above |r| threshold:")
    for thr in thresholds_r:
        n_above = (abs_corr > thr).sum()
        p(f"  |r| > {thr:.1f}: {n_above:5d} / {C} ({100*n_above/C:.1f}%)")

    # Top correlated CpGs
    top_corr_idx = np.argsort(abs_corr)[::-1]
    p(f"\nTop 20 most age-correlated CpGs (train only):")
    cpg_names = list(adata.var_names)
    for i, idx in enumerate(top_corr_idx[:20]):
        p(f"  {i+1:2d}. {cpg_names[idx]:12s}  |r|={abs_corr[idx]:.4f}  "
          f"mean={cpg_mean[idx]:.3f}  var={cpg_var[idx]:.5f}")

    np.save(OUTDIR / "cpg_age_corr.npy", cpg_age_corr)
    p(f"\nSaved: cpg_age_corr.npy  (signed Pearson r, train only)")
else:
    abs_corr = None
    p("    Skipped — age or split column not found")

# ── 5. Combined informativeness score ────────────────────────────────────────
p(f"\n{SEP}")
p("[5] Combined informativeness score (variance rank + age-corr rank)")
p(sep)

# Rank each CpG by variance (all samples) and |age correlation| (train only)
# Both are equally weighted. Rank 0 = least informative, C-1 = most informative.
var_rank  = np.argsort(np.argsort(cpg_var))          # rank by variance

if abs_corr is not None:
    corr_rank = np.argsort(np.argsort(abs_corr))     # rank by |age corr|
    # Combined: equal weight
    combined  = 0.5 * var_rank + 0.5 * corr_rank
else:
    combined  = var_rank.astype(float)
    p("    Warning: age correlation not available — using variance only")

combined_sorted = np.argsort(combined)[::-1]   # most informative first

p(f"    Top 5k threshold score  : {combined[combined_sorted[4999]]:.1f} / {C}")
p(f"    Top 10k threshold score : {combined[combined_sorted[9999]]:.1f} / {C}")

# Save indices
top_5k  = combined_sorted[:5000]
top_10k = combined_sorted[:10000]
np.save(OUTDIR / "cpg_informativeness.npy",  combined)
np.save(OUTDIR / "cpg_top5k_indices.npy",    top_5k)
np.save(OUTDIR / "cpg_top10k_indices.npy",   top_10k)

p(f"\nSaved:")
p(f"  cpg_informativeness.npy  — combined score per CpG")
p(f"  cpg_top5k_indices.npy    — {len(top_5k)} most informative CpG indices")
p(f"  cpg_top10k_indices.npy   — {len(top_10k)} most informative CpG indices")

# Show variance and correlation stats for top subsets
for k, idx_set in [("top 5k", top_5k), ("top 10k", top_10k), ("all 19k", np.arange(C))]:
    mean_var  = cpg_var[idx_set].mean()
    mean_corr = abs_corr[idx_set].mean() if abs_corr is not None else float("nan")
    p(f"  {k:10s}: mean_var={mean_var:.5f}  mean_|r|={mean_corr:.4f}")

# ── 6. Split quality analysis ─────────────────────────────────────────────────
p(f"\n{SEP}")
p("[6] Split quality — age and tissue distribution per split")
p(sep)

if "split" in adata.obs.columns and "age" in adata.obs.columns:
    split_stats = []
    for s in ["train", "valid", "test"]:
        mask = (adata.obs["split"] == s).values
        if mask.sum() == 0:
            continue
        ages_s = adata.obs.loc[mask, "age"].astype(float).dropna()
        row = {
            "split": s,
            "n": mask.sum(),
            "age_mean": ages_s.mean(),
            "age_std":  ages_s.std(),
            "age_min":  ages_s.min(),
            "age_max":  ages_s.max(),
            "age_p25":  ages_s.quantile(0.25),
            "age_p50":  ages_s.quantile(0.50),
            "age_p75":  ages_s.quantile(0.75),
        }
        # Mean beta per split
        X_s = X[mask]
        row["beta_mean"] = float(X_s.mean())
        row["beta_std"]  = float(X_s.std())
        split_stats.append(row)

    p(f"{'Split':8s}  {'N':6s}  {'AgeMean':8s}  {'AgeStd':7s}  "
      f"{'AgeMin':7s}  {'AgeMax':7s}  {'BetaMean':9s}  {'BetaStd':8s}")
    p(sep)
    for r in split_stats:
        p(f"{r['split']:8s}  {r['n']:6d}  {r['age_mean']:8.2f}  "
          f"{r['age_std']:7.2f}  {r['age_min']:7.1f}  {r['age_max']:7.1f}  "
          f"{r['beta_mean']:9.4f}  {r['beta_std']:8.4f}")

    # Age distribution per split — histogram comparison
    p("\nAge histogram comparison (10-year bins):")
    bins = np.arange(0, 110, 10)
    header = f"{'Age':10s}"
    for r in split_stats:
        header += f"  {r['split']:>10s}"
    p(header)
    for i in range(len(bins)-1):
        row_str = f"{bins[i]:3.0f}-{bins[i+1]:3.0f}    "
        for r in split_stats:
            mask = (adata.obs["split"] == r["split"]).values
            ages_s = adata.obs.loc[mask, "age"].astype(float).dropna()
            count = ((ages_s >= bins[i]) & (ages_s < bins[i+1])).sum()
            pct = 100 * count / len(ages_s)
            row_str += f"  {count:4d} ({pct:4.1f}%)"
        p(row_str)

    # Save split stats
    df_stats = pd.DataFrame(split_stats)
    df_stats.to_csv(OUTDIR / "split_age_stats.csv", index=False)
    p(f"\nSaved: split_age_stats.csv")

# Tissue coverage per split
if "tissue_type" in adata.obs.columns and "split" in adata.obs.columns:
    p("\nTissue distribution per split (top 15 tissues):")
    all_tissues = adata.obs["tissue_type"].value_counts().index.tolist()[:15]
    header = f"{'Tissue':35s}"
    for s in ["train", "valid", "test"]:
        header += f"  {s:>8s}"
    p(header)
    p(sep)
    for t in all_tissues:
        row_str = f"{str(t):35s}"
        for s in ["train", "valid", "test"]:
            mask = (adata.obs["split"] == s).values
            n_s = mask.sum()
            n_t = ((adata.obs["split"] == s) & (adata.obs["tissue_type"] == t)).sum()
            pct = 100 * n_t / n_s if n_s > 0 else 0
            row_str += f"  {n_t:4d}({pct:4.1f}%)"
        p(row_str)

# ── 7. Sample-level analysis ──────────────────────────────────────────────────
p(f"\n{SEP}")
p("[7] Sample-level analysis — beta value distribution per sample")
p(sep)

sample_mean  = X.mean(axis=1)    # [N] mean beta per sample
sample_std   = X.std(axis=1)     # [N] std beta per sample
sample_zeros = (X == 0).mean(axis=1)   # [N] fraction of unmethylated CpGs

p(f"Per-sample mean beta distribution:")
for pct in [0, 5, 25, 50, 75, 95, 100]:
    p(f"  p{pct:3d}: {np.percentile(sample_mean, pct):.4f}")

p(f"\nPer-sample std beta distribution:")
for pct in [0, 25, 50, 75, 100]:
    p(f"  p{pct:3d}: {np.percentile(sample_std, pct):.4f}")

# Outlier samples (mean beta very far from population mean)
pop_mean  = sample_mean.mean()
pop_std   = sample_mean.std()
outlier_mask = np.abs(sample_mean - pop_mean) > 3 * pop_std
n_outliers = outlier_mask.sum()
p(f"\nSamples with mean beta > 3σ from population mean: {n_outliers}")
if n_outliers > 0:
    outlier_idx = np.where(outlier_mask)[0]
    p(f"  Showing first {min(20, n_outliers)} of {n_outliers}:")
    p("  Sample index | mean beta | split | age | tissue")
    for idx in outlier_idx[:20]:
        row = adata.obs.iloc[idx]
        split_val = row.get("split", "?")
        age_val = row.get("age", float("nan"))
        tissue_val = row.get("tissue_type", "?")
        p(f"  {idx:6d}       | {sample_mean[idx]:.4f}    | {split_val} | {age_val:.1f} | {tissue_val}")
    # Outlier split distribution
    if "split" in adata.obs.columns:
        outlier_splits = adata.obs.iloc[outlier_idx]["split"].value_counts()
        p(f"\n  Outlier split distribution: {dict(outlier_splits)}")

# ── 8. Summary and recommendations ───────────────────────────────────────────
p(f"\n{SEP}")
p("[8] SUMMARY AND RECOMMENDATIONS")
p(SEP)

# CpG filtering recommendation
low_var_count = (cpg_var < 0.001).sum()
p(f"\nCpG filtering:")
p(f"  {low_var_count} CpGs ({100*low_var_count/C:.1f}%) have variance < 0.001 — near constant across samples")
p(f"  These add noise to mean pooling without contributing signal")
p(f"  Recommendation: filter to top 10k by informativeness score")
if abs_corr is not None:
    top10k_mean_corr = abs_corr[top_10k].mean()
    all_mean_corr    = abs_corr.mean()
    p(f"  Top 10k mean |age corr| = {top10k_mean_corr:.4f}  vs  all CpGs = {all_mean_corr:.4f}")
    p(f"  Gain factor: {top10k_mean_corr/all_mean_corr:.2f}x more age-informative on average")

# Split quality
if "split" in adata.obs.columns:
    p(f"\nSplit quality:")
    for r in split_stats:
        p(f"  {r['split']:6s}: N={r['n']}, age={r['age_mean']:.1f}±{r['age_std']:.1f} yr")
    if len(split_stats) >= 3:
        val_  = next((r for r in split_stats if r["split"] == "valid"), None)
        test_ = next((r for r in split_stats if r["split"] == "test"),  None)
        if val_ and test_:
            age_diff = abs(val_["age_mean"] - test_["age_mean"])
            p(f"  Val vs Test age mean difference: {age_diff:.2f} yr")
            if age_diff > 2:
                p(f"  WARNING: val and test age distributions differ by {age_diff:.1f} yr")
                p(f"  Consider running create_stratified_split.py to rebalance")

p(f"\nOutput files in: {OUTDIR}")
p(f"  cpg_variance.npy        — variance per CpG (all {N} samples)")
p(f"  cpg_mean.npy            — mean beta per CpG")
if abs_corr is not None:
    p(f"  cpg_age_corr.npy       — Pearson r with age (training split only)")
p(f"  cpg_informativeness.npy — combined rank score")
p(f"  cpg_top5k_indices.npy   — top 5,000 informative CpG indices")
p(f"  cpg_top10k_indices.npy  — top 10,000 informative CpG indices")
p(f"  split_age_stats.csv     — age/beta stats per split")

p(f"\n{SEP}")
p(f"DONE — {time.strftime('%Y-%m-%d %H:%M:%S')}")
p(SEP)

# ── Write report ──────────────────────────────────────────────────────────────
report_path = OUTDIR / "analysis_report.txt"
with open(report_path, "w") as f:
    f.write("\n".join(lines))
print(f"\nReport saved: {report_path}")
