#!/usr/bin/env python3
"""
Create a genomic rank array for the fine-tune CpG set (19k subset of 49k pretrain CpGs).

The pretrain genomic_rank.npy maps each DATA COLUMN INDEX in the pretrain h5ad
to its genomic rank (0-based, sorted by chromosome + position).

For fine-tuning, the h5ad has a different (smaller) set of CpGs.  WCEDCollator
requires len(genomic_rank) == len(cpg_sites) for the fine-tune dataset, so we
cannot reuse the pretrain array directly.

This script:
  1. Loads pretrain h5ad var names  → pretrain CpG name list (49,156 entries)
  2. Loads pretrain genomic_rank.npy → rank[i] = genomic rank of pretrain column i
  3. Loads fine-tune h5ad var names  → fine-tune CpG name list (19k entries)
  4. For each fine-tune CpG, looks up its pretrain column index → its genomic rank
  5. Saves cpg_genomic_rank_finetune.npy with len == n_finetune_cpgs

Usage (on cluster):
    python scripts/data/create_finetune_genomic_rank.py

Outputs:
    resources/cpg_genomic_rank_finetune.npy
"""

import sys
from pathlib import Path

import numpy as np

import os
DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh
REPO = os.environ.get("METHYLLAMA_REPO", ".")             # scripts run from the repository root
METHYLGPT = os.environ.get("METHYLGPT_DIR", "MethylGPT")  # MethylGPT pipeline (comparison only)

REPO = Path(__file__).resolve().parents[2]

PRETRAIN_H5AD      = Path(DATA + "/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad")
FINETUNE_H5AD      = Path(DATA + "/data_methyl_21k_h5ad/altumage_21k_3way.h5ad")
PRETRAIN_RANK_NPY  = REPO / "resources/cpg_genomic_rank.npy"
OUTPUT_NPY         = REPO / "resources/cpg_genomic_rank_finetune.npy"


def load_var_names(h5ad_path: Path):
    """Load CpG (var) names from an h5ad file without loading the full matrix."""
    try:
        import anndata as ad
        adata = ad.read_h5ad(str(h5ad_path), backed="r")
        names = list(adata.var_names)
        adata.file.close()
        return names
    except Exception:
        import h5py
        with h5py.File(str(h5ad_path), "r") as f:
            # anndata stores var index under var/_index or var/index
            if "var" in f:
                var_grp = f["var"]
                for key in ["_index", "index", "probe_id", "feature_name"]:
                    if key in var_grp:
                        return [x.decode() if isinstance(x, bytes) else x
                                for x in var_grp[key][:]]
        raise RuntimeError(f"Cannot read var names from {h5ad_path}")


def main():
    print(f"Pretrain h5ad:     {PRETRAIN_H5AD}")
    print(f"Fine-tune h5ad:    {FINETUNE_H5AD}")
    print(f"Pretrain rank npy: {PRETRAIN_RANK_NPY}")

    print("\nLoading pretrain CpG names...")
    pretrain_cpgs = load_var_names(PRETRAIN_H5AD)
    print(f"  {len(pretrain_cpgs)} CpGs in pretrain dataset")

    print("Loading pretrain genomic rank array...")
    pretrain_rank = np.load(str(PRETRAIN_RANK_NPY))
    assert len(pretrain_rank) == len(pretrain_cpgs), (
        f"Rank array length {len(pretrain_rank)} != pretrain CpGs {len(pretrain_cpgs)}"
    )
    print(f"  rank array shape: {pretrain_rank.shape}, dtype: {pretrain_rank.dtype}")

    # Build lookup: CpG name → pretrain column index
    pretrain_idx = {name: i for i, name in enumerate(pretrain_cpgs)}

    print("Loading fine-tune CpG names...")
    finetune_cpgs = load_var_names(FINETUNE_H5AD)
    print(f"  {len(finetune_cpgs)} CpGs in fine-tune dataset")

    print("Computing fine-tune genomic ranks...")
    missing = []
    finetune_rank = np.zeros(len(finetune_cpgs), dtype=pretrain_rank.dtype)
    for j, cpg in enumerate(finetune_cpgs):
        if cpg in pretrain_idx:
            finetune_rank[j] = pretrain_rank[pretrain_idx[cpg]]
        else:
            missing.append(cpg)
            finetune_rank[j] = j  # fallback: use column order

    if missing:
        print(f"  WARNING: {len(missing)} fine-tune CpGs not found in pretrain set.")
        print(f"  First 5 missing: {missing[:5]}")
        print(f"  These will use sequential positions as fallback.")
    else:
        print(f"  All {len(finetune_cpgs)} fine-tune CpGs found in pretrain set.")

    print(f"  Fine-tune rank: min={finetune_rank.min()}, max={finetune_rank.max()}")

    OUTPUT_NPY.parent.mkdir(parents=True, exist_ok=True)
    np.save(str(OUTPUT_NPY), finetune_rank)
    print(f"\nSaved: {OUTPUT_NPY}  shape={finetune_rank.shape}")
    print("Done.")


if __name__ == "__main__":
    main()
