"""
Remove per-sample beta outliers from the stratified h5ad.

Outlier definition: sample mean beta > 3σ from population mean beta.
This catches two distinct groups identified in the analysis:
  - Brain cerebellum samples: mean_beta ~0.08-0.10 (hypomethylated)
  - Blood whole adolescents:  mean_beta ~0.47-0.49 (hypermethylated batch effect)

These 298 samples (2.8% of cohort) inflate MAE by ~1.8yr without contributing
useful age signal — the 1.81yr gap between test/mae and test/medae is caused
almost entirely by them.

Split distribution of outliers (from analysis):
  train: 201  valid: 30  test: 67

Output: finetuning_19608_clean_stratified_no_outliers.h5ad
  Same CpG sites, same split labels — outlier rows removed only.
"""

import numpy as np
import anndata as ad
import scipy.sparse as sp
from pathlib import Path

SRC = Path(
    "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean_stratified.h5ad"
)
DST = Path(
    "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean_stratified_no_outliers.h5ad"
)

print(f"Loading: {SRC}")
adata = ad.read_h5ad(SRC)
print(f"  {adata.n_obs} samples × {adata.n_vars} CpGs")

# Dense matrix
X = adata.X.toarray().astype(np.float32) if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)

# Compute per-sample mean beta and identify outliers
sample_mean = X.mean(axis=1)
pop_mean    = sample_mean.mean()
pop_std     = sample_mean.std()
threshold   = 3 * pop_std

outlier_mask = np.abs(sample_mean - pop_mean) > threshold
n_outliers   = outlier_mask.sum()

print(f"\nPopulation mean beta: {pop_mean:.4f} ± {pop_std:.4f}")
print(f"3σ threshold: [{pop_mean - threshold:.4f}, {pop_mean + threshold:.4f}]")
print(f"Outliers (>3σ): {n_outliers} samples ({100*n_outliers/adata.n_obs:.1f}%)")

# Per-split breakdown
print("\nOutliers per split:")
for s in ["train", "valid", "test"]:
    mask_split = (adata.obs["split"] == s).values
    n_out = (outlier_mask & mask_split).sum()
    n_tot = mask_split.sum()
    print(f"  {s:6s}: {n_out:3d} / {n_tot} ({100*n_out/n_tot:.1f}%)")

# Show what we're removing — tissue breakdown
print("\nTissue breakdown of removed samples:")
outlier_obs = adata.obs[outlier_mask]
tissue_counts = outlier_obs["tissue_type"].value_counts()
for tissue, count in tissue_counts.items():
    total_tissue = (adata.obs["tissue_type"] == tissue).sum()
    pct_removed = 100 * count / total_tissue
    print(f"  {str(tissue):35s}: {count:3d} / {total_tissue:4d} ({pct_removed:5.1f}% of tissue removed)")

# Age distribution of removed samples
removed_ages = outlier_obs["age"].astype(float).dropna()
print(f"\nAge distribution of removed samples:")
print(f"  mean={removed_ages.mean():.1f}  std={removed_ages.std():.1f}  "
      f"range=[{removed_ages.min():.0f}, {removed_ages.max():.0f}]")

# Mean beta distribution of removed samples
removed_betas = sample_mean[outlier_mask]
print(f"\nMean beta of removed samples:")
print(f"  min={removed_betas.min():.4f}  max={removed_betas.max():.4f}  "
      f"mean={removed_betas.mean():.4f}")
low  = (removed_betas < pop_mean - threshold).sum()
high = (removed_betas > pop_mean + threshold).sum()
print(f"  hypomethylated (below threshold): {low}")
print(f"  hypermethylated (above threshold): {high}")

# Show top 20 outliers by deviation
print("\nTop 20 outliers by deviation:")
deviations = np.abs(sample_mean - pop_mean)
top_idx = np.argsort(deviations)[::-1][:20]
print(f"  {'idx':>6}  {'mean_beta':>9}  {'sigma':>6}  {'split':>6}  {'age':>5}  tissue")
for idx in top_idx:
    row = adata.obs.iloc[idx]
    sigma = deviations[idx] / pop_std
    print(f"  {idx:6d}  {sample_mean[idx]:9.4f}  {sigma:6.1f}σ  "
          f"{str(row.get('split','?')):>6}  {row.get('age', float('nan')):5.1f}  "
          f"{row.get('tissue_type','?')}")

# Remove outliers
keep_mask = ~outlier_mask
adata_clean = adata[keep_mask].copy()

print(f"\nAfter removal: {adata_clean.n_obs} samples ({n_outliers} removed)")

# Verify split counts
print("\nNew split counts:")
for s in ["train", "valid", "test"]:
    n = (adata_clean.obs["split"] == s).sum()
    pct = 100 * n / adata_clean.n_obs
    print(f"  {s:6s}: {n} ({pct:.1f}%)")

# Verify age distribution still balanced
print("\nAge distribution per split (mean ± std):")
for s in ["train", "valid", "test"]:
    mask = (adata_clean.obs["split"] == s).values
    ages = adata_clean.obs.loc[mask, "age"].astype(float).dropna()
    print(f"  {s:6s}: mean={ages.mean():.1f}  std={ages.std():.1f}  "
          f"range=[{ages.min():.0f}, {ages.max():.0f}]  n={len(ages)}")

# Write
print(f"\nWriting: {DST}")
DST.parent.mkdir(parents=True, exist_ok=True)
adata_clean.write_h5ad(DST, compression="gzip")
print("Done.")
print(f"\nTo use: set DATA= to {DST} in your finetune script")
