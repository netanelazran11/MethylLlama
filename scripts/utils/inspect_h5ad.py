#!/usr/bin/env python3
"""
Inspect obs columns, var_names, shape, and obsm keys of an AnnData h5ad file.

Usage:
  python scripts/utils/inspect_h5ad.py /path/to/data.h5ad
"""
import sys
import os
import anndata

path = sys.argv[1] if len(sys.argv) > 1 else (
    "/sci/labs/benjamin.yakir/netanel.azran/data/"
    "data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"
)

print(f"\nFile size: {os.path.getsize(path)/1e9:.2f} GB")
adata = anndata.read_h5ad(path, backed="r")

print(f"\n{'='*60}")
print(f" h5ad: {os.path.basename(path)}")
print(f"{'='*60}")
print(f" Shape: {adata.shape}  (n_obs={adata.n_obs:,}, n_vars={adata.n_vars:,})")

print(f"\n── obs columns ({len(adata.obs.columns)}) ──────────────────────────────")
for col in adata.obs.columns:
    s = adata.obs[col]
    dtype = str(s.dtype)
    n_unique = s.nunique()
    sample_vals = s.dropna().unique()[:5].tolist()
    print(f"  {col:<30s}  dtype={dtype:<12s}  nunique={n_unique:<6}  sample={sample_vals}")

print(f"\n── var_names sample (first 10) ─────────────────────────────────────")
print(f"  {list(adata.var_names[:10])}")

print(f"\n── var columns ({len(adata.var.columns)}) ──────────────────────────────")
for col in adata.var.columns:
    print(f"  {col}")

print(f"\n── obsm keys ───────────────────────────────────────────────────────")
print(f"  {list(adata.obsm.keys())}")

print(f"\n── uns keys ────────────────────────────────────────────────────────")
print(f"  {list(adata.uns.keys())}")
print(f"{'='*60}\n")
