"""
Create a stratified train/valid/test split for the fine-tuning h5ad.

Problem with the current split:
  - Inherited from MethylGPT parquet files — unknown how it was made
  - val/mae > test/mae in V3 results means the splits have different difficulty
  - Early stopping monitors val, so model stops at wrong point

Strategy — stratify by BOTH age group AND tissue_type:
  - Bin age into 10-year groups (0-10, 10-20, ..., 80+)
  - Within each (age_bin, tissue_type) stratum, split 68/12/20
  - If a stratum has <5 samples, assign all to train
  - Result: val and test have matched age and tissue distributions

Ratios: 68% train / 12% val / 20% test
  → Train: ~7,246  Val: ~1,279  Test: ~2,131

Output: same h5ad file path (new filename _stratified.h5ad)
  Only obs['split'] is changed — X matrix untouched.
"""

import anndata as ad
import numpy as np
import pandas as pd
from pathlib import Path

SRC = Path(
    "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean.h5ad"
)
DST = Path(
    "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean_stratified.h5ad"
)

VAL_FRAC  = 0.12   # 12% validation
TEST_FRAC = 0.20   # 20% test
SEED      = 42

print(f"Loading: {SRC}")
adata = ad.read_h5ad(SRC)
print(f"  {adata.n_obs} samples × {adata.n_vars} CpGs")
print(f"\nObs columns: {list(adata.obs.columns)}")
print(f"\nOriginal split:\n{adata.obs['split'].value_counts().to_string()}")

# ── Build stratification key ─────────────────────────────────────────────────

age = adata.obs["age"].values.astype(float)

# Age bin (10-year groups)
age_bin = np.floor(age / 10).astype(int)
age_bin = np.clip(age_bin, 0, 8)   # cap at 80+
age_bin_str = [f"age_{b*10:02d}" for b in age_bin]

# Tissue type (if available)
if "tissue_type" in adata.obs.columns:
    tissue = adata.obs["tissue_type"].astype(str).values
    print(f"\nTissue types ({len(np.unique(tissue))}): {np.unique(tissue)[:10]}")
    strat_key = [f"{a}__{t}" for a, t in zip(age_bin_str, tissue)]
else:
    print("\ntissue_type column not found — stratifying by age only")
    strat_key = age_bin_str

adata.obs["_strat_key"] = strat_key

# ── Assign splits per stratum ────────────────────────────────────────────────

rng        = np.random.default_rng(SEED)
new_split  = np.full(adata.n_obs, "train", dtype=object)

strat_series = pd.Series(strat_key)
strata       = strat_series.unique()

small_strata = 0
for stratum in strata:
    idx = np.where(strat_series == stratum)[0]
    n   = len(idx)

    if n < 5:
        # Too few samples to split — keep all in train
        small_strata += 1
        continue

    shuffled = rng.permutation(idx)
    n_test   = max(1, int(n * TEST_FRAC))
    n_val    = max(1, int(n * VAL_FRAC))

    test_idx  = shuffled[:n_test]
    val_idx   = shuffled[n_test:n_test + n_val]
    # rest → train (implicit)

    new_split[test_idx] = "test"
    new_split[val_idx]  = "valid"

print(f"\n  {small_strata} strata with <5 samples — all assigned to train")

# ── Apply and verify ─────────────────────────────────────────────────────────

adata.obs["split"] = pd.Categorical(new_split, categories=["train", "valid", "test"])
adata.obs.drop(columns=["_strat_key"], inplace=True)

counts = adata.obs["split"].value_counts()
print(f"\nNew split:\n{counts.to_string()}")
print(f"  Ratios: train={counts['train']/adata.n_obs:.1%}  "
      f"val={counts['valid']/adata.n_obs:.1%}  "
      f"test={counts['test']/adata.n_obs:.1%}")

# Check age distribution per split
print("\nAge distribution per split (mean ± std):")
for s in ["train", "valid", "test"]:
    mask = adata.obs["split"] == s
    ages = adata.obs.loc[mask, "age"].dropna()
    print(f"  {s:6s}: mean={ages.mean():.1f}  std={ages.std():.1f}  "
          f"range=[{ages.min():.0f}, {ages.max():.0f}]  n={len(ages)}")

# Check tissue coverage per split (if available)
if "tissue_type" in adata.obs.columns:
    print("\nTissue coverage per split (% of all tissue types present):")
    all_tissues = set(adata.obs["tissue_type"].unique())
    for s in ["train", "valid", "test"]:
        mask    = adata.obs["split"] == s
        tissues = set(adata.obs.loc[mask, "tissue_type"].unique())
        pct     = 100 * len(tissues) / len(all_tissues)
        print(f"  {s:6s}: {len(tissues)}/{len(all_tissues)} tissues = {pct:.0f}%")

# ── Write ─────────────────────────────────────────────────────────────────────

print(f"\nWriting: {DST}")
DST.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(DST, compression="gzip")
print("Done.")
print(f"\nTo use: set DATA= to {DST} in your finetune script")
