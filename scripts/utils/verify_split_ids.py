"""
Verify MethylGPT parquet integer index maps to h5ad row position → GSM ID.
Lightweight: reads only age column from parquet and obs from h5ad.
"""

import h5py
import numpy as np
import pyarrow.parquet as pq

PARQUET_DIR = "/sci/labs/benjamin.yakir/netanel.azran/MethylGPT/data/19k_data/finetuning_data"
H5AD_PATH   = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad/finetuning_19608_clean_stratified_no_outliers.h5ad"

# ── Load h5ad obs only (no matrix) ────────────────────────────────────────────
print("Reading h5ad obs...")
with h5py.File(H5AD_PATH, "r") as f:
    gsm_ids    = [x.decode() if isinstance(x, bytes) else str(x) for x in f["obs"]["_index"][:]]
    h5ad_ages  = np.array(f["obs"]["age"][:], dtype=np.float32)
    split_codes = list(f["obs"]["split"]["codes"][:])
    split_cats  = [x.decode() if isinstance(x, bytes) else x for x in f["obs"]["split"]["categories"][:]]
    h5ad_splits = [split_cats[c] for c in split_codes]

print(f"  h5ad rows: {len(gsm_ids)}  (e.g. {gsm_ids[:3]})")

# ── For each split: read parquet index + age, map to h5ad row ─────────────────
print()
all_ok = True
for split in ("train", "valid", "test"):
    tbl = pq.ParquetFile(f"{PARQUET_DIR}/{split}.parquet").read(columns=["age"])
    df  = tbl.to_pandas()
    parquet_indices = df.index.astype(int).tolist()
    parquet_ages    = df["age"].astype(float).tolist()

    matched  = 0
    mismatch = 0
    oob      = 0
    gsm_sample = []

    for pidx, page in zip(parquet_indices, parquet_ages):
        if pidx >= len(gsm_ids):
            oob += 1
            continue
        h_age   = float(h5ad_ages[pidx])
        h_split = h5ad_splits[pidx]
        h_gsm   = gsm_ids[pidx]
        if abs(page - h_age) < 0.01 and h_split == split:
            matched += 1
            if len(gsm_sample) < 5:
                gsm_sample.append(h_gsm)
        else:
            mismatch += 1

    total = len(parquet_indices)
    ok = matched == total
    all_ok = all_ok and ok
    print(f"{split.upper()} ({total} samples):")
    print(f"  Matched (index + age + split): {matched}/{total}  {'✓' if ok else '✗'}")
    if mismatch: print(f"  Mismatches: {mismatch}")
    if oob:      print(f"  Out-of-range indices: {oob}")
    print(f"  Sample GSM IDs from mapping: {gsm_sample}")
    print()

print("=" * 55)
print(f"SPLITS 100% IDENTICAL BY ID: {'YES ✓' if all_ok else 'NO ✗'}")
print("=" * 55)
