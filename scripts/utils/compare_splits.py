"""
Compare train/val/test sample IDs between MethylGPT (parquet) and MethylLlama (h5ad).
Verifies both models used identical splits for fair comparison.
"""

import h5py
import pandas as pd
import pyarrow.parquet as pq

PARQUET_DIR = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/19k_data/finetuning_data"
H5AD_PATH   = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# ── MethylGPT (parquet) — read schema + first column only ────────────────────
print("Loading MethylGPT parquet files (schema inspect)...")
gpt = {}
for split in ("train", "valid", "test"):
    path = f"{PARQUET_DIR}/{split}.parquet"
    pf = pq.ParquetFile(path)
    col_names = pf.schema_arrow.names
    print(f"  {split} columns: {col_names[:8]}")
    # Read only the first column to get row count + index
    # Read only 'age' column (small) — avoid loading 'data' (full matrix)
    tbl = pf.read(columns=["age"])
    df  = tbl.to_pandas()
    ids = set(df.index.astype(str))
    gpt[split] = ids
    print(f"  MethylGPT  {split:5s}: {len(ids):5d} samples  (example: {list(ids)[:3]})")

# ── MethylLlama (h5ad) — read only obs via h5py, no matrix load ──────────────
print("\nLoading MethylLlama h5ad (obs only via h5py)...")
with h5py.File(H5AD_PATH, "r") as f:
    # obs index (sample IDs)
    obs_grp = f["obs"]
    print(f"  obs keys: {list(obs_grp.keys())[:10]}")
    # index is stored as _index or the first string dataset
    if "_index" in obs_grp:
        all_ids = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["_index"][:]]
    else:
        idx_key = list(obs_grp.keys())[0]
        all_ids = [x.decode() if isinstance(x, bytes) else x for x in obs_grp[idx_key][:]]
    # split column
    split_vals = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["split"]["codes"][:]]
    split_cats = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["split"]["categories"][:]]
    splits_decoded = [split_cats[c] for c in split_vals]

print(f"  Total samples: {len(all_ids)}")
from collections import Counter
print(f"  Split counts: {dict(Counter(splits_decoded))}")

llama = {}
for split in ("train", "valid", "test"):
    ids = set(sid for sid, sp in zip(all_ids, splits_decoded) if sp == split)
    llama[split] = ids
    print(f"  MethylLlama {split:5s}: {len(ids):5d} samples  (example: {list(ids)[:3]})")

# ── Map MethylGPT integer index → GSM ID via h5ad row order ──────────────────
print("\nMapping MethylGPT integer indices → GSM IDs...")
with h5py.File(H5AD_PATH, "r") as f:
    all_gsm = [x.decode() if isinstance(x, bytes) else x for x in f["obs"]["_index"][:]]
    split_vals = list(f["obs"]["split"]["codes"][:])
    split_cats = [x.decode() if isinstance(x, bytes) else x for x in f["obs"]["split"]["categories"][:]]
    splits_decoded = [split_cats[c] for c in split_vals]
    h5ad_ages = list(f["obs"]["age"][:])

# Build lookup: row_index → (gsm_id, age, split)
h5ad_by_idx = {i: (gsm, age, sp) for i, (gsm, age, sp) in enumerate(zip(all_gsm, h5ad_ages, splits_decoded))}

print("\n" + "=" * 60)
print("SPLIT COMPARISON — exact sample ID mapping")
print("=" * 60)
all_match = True
for split in ("train", "valid", "test"):
    tbl = pq.ParquetFile(f"{PARQUET_DIR}/{split}.parquet").read(columns=["age"])
    df  = tbl.to_pandas()
    matched = 0
    wrong_split = 0
    age_mismatch = 0
    not_found = 0
    gsm_ids_gpt = []
    for row_idx_str, row_age in zip(df.index.astype(str), df["age"]):
        row_idx = int(row_idx_str)
        if row_idx not in h5ad_by_idx:
            not_found += 1
            continue
        gsm, h5_age, h5_split = h5ad_by_idx[row_idx]
        gsm_ids_gpt.append(gsm)
        if abs(float(row_age) - float(h5_age)) > 0.01:
            age_mismatch += 1
        elif h5_split != split:
            wrong_split += 1
        else:
            matched += 1
    total = len(df)
    ok = matched == total
    all_match = all_match and ok
    print(f"\n{split.upper()} ({total} samples):")
    print(f"  Fully matched (ID + age + split): {matched}/{total}  {'✓' if ok else '✗'}")
    if age_mismatch: print(f"  Age mismatch: {age_mismatch}")
    if wrong_split:  print(f"  Wrong split : {wrong_split}")
    if not_found:    print(f"  Index not in h5ad: {not_found}")
    print(f"  Example GSM IDs: {gsm_ids_gpt[:3]}")

print("\n" + "=" * 60)
print(f"SPLITS 100% IDENTICAL (ID + age + split): {'YES ✓' if all_match else 'NO ✗'}")
print("=" * 60)

# ── Compare via AGE VALUES (MethylGPT uses integer row index, not GSM IDs) ───
print("\n" + "=" * 60)
print("SPLIT COMPARISON — via age values")
print("=" * 60)
import numpy as np

# Load MethylGPT ages per split
gpt_ages = {}
for split in ("train", "valid", "test"):
    tbl = pq.ParquetFile(f"{PARQUET_DIR}/{split}.parquet").read(columns=["age"])
    gpt_ages[split] = sorted(tbl["age"].to_pylist())

# Load MethylLlama ages per split via h5py
with h5py.File(H5AD_PATH, "r") as f:
    obs_grp = f["obs"]
    all_ids = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["_index"][:]]
    split_vals = list(obs_grp["split"]["codes"][:])
    split_cats = [x.decode() if isinstance(x, bytes) else x for x in obs_grp["split"]["categories"][:]]
    splits_decoded = [split_cats[c] for c in split_vals]
    all_ages = list(obs_grp["age"][:])

llama_ages = {}
for split in ("train", "valid", "test"):
    llama_ages[split] = sorted(a for a, s in zip(all_ages, splits_decoded) if s == split)

all_match = True
for split in ("train", "valid", "test"):
    g = gpt_ages[split]
    l = llama_ages[split]
    sizes_match = len(g) == len(l)
    ages_match  = np.allclose(g, l, atol=0.01)
    all_match   = all_match and sizes_match and ages_match
    print(f"\n{split.upper()}:")
    print(f"  MethylGPT  : {len(g):5d} samples  age range [{min(g):.1f}, {max(g):.1f}]  mean={np.mean(g):.2f}")
    print(f"  MethylLlama: {len(l):5d} samples  age range [{min(l):.1f}, {max(l):.1f}]  mean={np.mean(l):.2f}")
    print(f"  Sizes match : {'YES ✓' if sizes_match else 'NO ✗'}")
    print(f"  Ages match  : {'YES ✓' if ages_match else 'NO ✗'}")

print("\n" + "=" * 60)
print(f"SAME DATASET & SPLIT: {'YES ✓' if all_match else 'NO ✗ — comparison may be unfair!'}")
print("=" * 60)
