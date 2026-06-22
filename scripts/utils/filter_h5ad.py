"""
One-time script: filter the 21k h5ad to remove non-aging tissues.

Removes placenta and sperm samples which have age=0 but do not reflect
biological aging. Keeps all cord blood and infant samples (real age=0).

Run once on the cluster:
    python scripts/filter_21k_h5ad.py

Output: altumage_21k_combined_filtered.h5ad
"""

import anndata as ad
import numpy as np

INPUT = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_combined.h5ad"
OUTPUT = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_combined_filtered.h5ad"

BAD_TISSUES = {"placenta", "sperm"}

print(f"Loading {INPUT} ...")
adata = ad.read_h5ad(INPUT)
print(f"Total samples before filter: {adata.n_obs}")

if "tissue_type" not in adata.obs.columns:
    raise ValueError("No 'tissue_type' column found. Check the obs column names:")

print(f"Tissue counts:\n{adata.obs['tissue_type'].value_counts()}")

mask = ~adata.obs["tissue_type"].isin(BAD_TISSUES)
removed = int((~mask).sum())
adata_filtered = adata[mask].copy()

print(f"\nRemoved {removed} samples ({BAD_TISSUES})")
print(f"Remaining samples: {adata_filtered.n_obs}")
print(f"Age range after filter: {adata_filtered.obs['age'].min():.1f} – {adata_filtered.obs['age'].max():.1f}")

adata_filtered.write_h5ad(OUTPUT)
print(f"\nSaved to {OUTPUT}")
