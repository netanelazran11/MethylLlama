"""
Add a proper 'valid' split to altumage_21k_combined.h5ad.

Current state:  'train' = 8724,  'test' = 2264  (val == test — bad for checkpoint selection)
After this:     'train' = 7416,  'valid' = 1308, 'test' = 2264  (proper 3-way split)

Only obs['split'] is modified — the methylation matrix is untouched.
A new file is written; the original is never overwritten.
"""

import anndata as ad
import numpy as np

DATA_DIR = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad"
SRC = f"{DATA_DIR}/altumage_21k_combined.h5ad"
DST = f"{DATA_DIR}/altumage_21k_3way.h5ad"

VAL_FRACTION = 0.15  # 15% of train → valid  (~1308 samples)
SEED = 42

adata = ad.read_h5ad(SRC)

print(f"Loaded: {adata.n_obs} samples")
print(f"Split counts before:\n{adata.obs['split'].value_counts().to_string()}\n")

# Identify train indices
train_mask = adata.obs["split"] == "train"
train_idx  = np.where(train_mask)[0]

# Sample 10% for validation (reproducible seed)
rng     = np.random.default_rng(SEED)
n_val   = int(len(train_idx) * VAL_FRACTION)
val_idx = rng.choice(train_idx, size=n_val, replace=False)

# Update split column (add 'valid' category first — column is Categorical)
split_col = adata.obs["split"].copy()
if hasattr(split_col, "cat"):
    split_col = split_col.cat.add_categories("valid")
split_col.iloc[val_idx] = "valid"
adata.obs["split"] = split_col

print(f"Split counts after:\n{adata.obs['split'].value_counts().to_string()}\n")
print(f"Writing to: {DST}")

adata.write_h5ad(DST)
print("Done.")
