"""
Convert 49k fine-tuning parquet splits → single h5ad file.

Input layout (DATA_DIR):
  train.parquet  / valid.parquet  / test.parquet
    columns: id (str), data (list<double> len=49156), age (double)
  cpg_mapping/probe_ids_type3.csv
    column: probe_id  (or first column)

Output: finetuning_49k.h5ad
  X       : float32 (samples × 49156)  — NaN kept as-is
  obs     : id (str), age (float32), split (category: train/valid/test)
  var     : index = probe_id
  uns     : {"n_cpgs": 49156, "source": "finetuning_data_49k parquet"}

Usage:
  python scripts/utils/parquet_to_h5ad.py
  # or override paths:
  DATA_DIR=/path/to/parquet OUT=/path/to/out.h5ad python scripts/utils/parquet_to_h5ad.py
"""

import os
import sys
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import anndata as ad
from pathlib import Path

# ─── Paths ────────────────────────────────────────────────────────────────────
DATA_DIR = Path(
    os.environ.get(
        "DATA_DIR",
        "/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_49k",
    )
)
OUT = Path(
    os.environ.get(
        "OUT",
        "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad",
    )
)
# Authoritative pretrain probe IDs (same 49156 CpGs, known illumina_probe_id column)
PRETRAIN_PROBE_CSV = Path(
    os.environ.get(
        "PRETRAIN_PROBE_CSV",
        "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv",
    )
)
PROBE_CSV = DATA_DIR / "cpg_mapping" / "probe_ids_type3.csv"

print(f"DATA_DIR          : {DATA_DIR}")
print(f"OUT               : {OUT}")
print(f"PROBE_CSV         : {PROBE_CSV}")
print(f"PRETRAIN_PROBE_CSV: {PRETRAIN_PROBE_CSV}")


def _load_probe_ids(csv_path: Path) -> list:
    """Read probe IDs from a CSV, trying common column names. Returns list of strings."""
    df = pd.read_csv(csv_path)
    for col in ("illumina_probe_id", "probe_id", "cpg_id"):
        if col in df.columns:
            ids = df[col].astype(str).tolist()
            if ids[0].startswith("cg"):
                return ids
    # Try every column for cg... values
    for col in df.columns:
        ids = df[col].astype(str).tolist()
        if ids[0].startswith("cg"):
            return ids
    return None


# ─── 1. Load probe IDs ────────────────────────────────────────────────────────
print("\n[1] Loading probe IDs …")
probe_ids = None

# Try the fine-tuning mapping CSV first
if PROBE_CSV.exists():
    probe_ids = _load_probe_ids(PROBE_CSV)
    if probe_ids:
        print(f"    Loaded from {PROBE_CSV.name}: {probe_ids[0]} … {probe_ids[-1]}")

# Fall back to authoritative pretrain probe CSV
if probe_ids is None:
    if not PRETRAIN_PROBE_CSV.exists():
        raise FileNotFoundError(
            f"Could not find cg-named probe IDs in {PROBE_CSV}\n"
            f"and pretrain fallback not found: {PRETRAIN_PROBE_CSV}"
        )
    probe_ids = _load_probe_ids(PRETRAIN_PROBE_CSV)
    if probe_ids is None:
        raise ValueError(f"No cg-prefixed probe IDs found in {PRETRAIN_PROBE_CSV}")
    print(f"    Loaded from pretrain fallback: {probe_ids[0]} … {probe_ids[-1]}")

n_cpgs = len(probe_ids)
print(f"    {n_cpgs} probe IDs  (first: {probe_ids[0]}, last: {probe_ids[-1]})")

# ─── 2. Read each split ───────────────────────────────────────────────────────
splits = ["train", "valid", "test"]
all_X    = []
all_ids  = []
all_ages = []
all_splits = []

for split in splits:
    pq_file = DATA_DIR / f"{split}.parquet"
    print(f"\n[2] Reading {split}.parquet …")

    # Read columns one at a time to stay memory-safe
    tbl_meta = pq.read_table(pq_file, columns=["id", "age"])
    ids  = tbl_meta["id"].to_pylist()
    ages = tbl_meta["age"].to_pylist()
    n    = len(ids)
    print(f"    {n} rows")

    # Read data column (list<double> × n)
    print(f"    Reading data column …")
    tbl_data = pq.read_table(pq_file, columns=["data"])

    # Convert to float32 matrix row-by-row to avoid peak RAM spike
    X = np.empty((n, n_cpgs), dtype=np.float32)
    for i, row in enumerate(tbl_data["data"]):
        X[i] = np.array(row.as_py(), dtype=np.float32)
        if i % 1000 == 0 and i > 0:
            print(f"      … {i}/{n}")

    print(f"    Matrix shape: {X.shape}  dtype={X.dtype}")
    nan_pct = 100 * np.isnan(X).mean()
    print(f"    NaN: {nan_pct:.1f}%")

    all_X.append(X)
    all_ids.extend(ids)
    all_ages.extend(ages)
    all_splits.extend([split] * n)

    del tbl_data  # free memory

# ─── 3. Concatenate ──────────────────────────────────────────────────────────
print("\n[3] Concatenating splits …")
X_all    = np.concatenate(all_X, axis=0)
obs_df   = pd.DataFrame({
    "id":    all_ids,
    "age":   np.array(all_ages, dtype=np.float32),
    "split": pd.Categorical(all_splits, categories=splits),
}, index=all_ids)

print(f"    Total shape: {X_all.shape}")
print(f"    Split counts:\n{obs_df['split'].value_counts().to_string()}")
print(f"    Age range: {obs_df['age'].min():.1f} – {obs_df['age'].max():.1f}")
print(f"    Age NaN: {obs_df['age'].isna().sum()}")

# ─── 4. Build AnnData ────────────────────────────────────────────────────────
print("\n[4] Building AnnData …")
var_df = pd.DataFrame(index=probe_ids)
var_df.index.name = "probe_id"

adata = ad.AnnData(
    X=X_all,
    obs=obs_df,
    var=var_df,
    uns={
        "n_cpgs": n_cpgs,
        "source": "finetuning_data_49k parquet",
    },
)
print(f"    {adata}")

# ─── 5. Write h5ad ───────────────────────────────────────────────────────────
print(f"\n[5] Writing {OUT} …")
OUT.parent.mkdir(parents=True, exist_ok=True)
adata.write_h5ad(OUT, compression="gzip")
size_gb = OUT.stat().st_size / 1e9
print(f"    Done. File size: {size_gb:.2f} GB")

# ─── 6. Quick verification ───────────────────────────────────────────────────
print("\n[6] Verification read-back …")
adata2 = ad.read_h5ad(OUT)
print(f"    Shape: {adata2.shape}")
print(f"    obs columns: {list(adata2.obs.columns)}")
print(f"    var index[:5]: {list(adata2.var_names[:5])}")
print(f"    split counts:\n{adata2.obs['split'].value_counts().to_string()}")
print("\nDONE")
