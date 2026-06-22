"""
Inspect 49k fine-tuning parquet — memory-safe version.
Reads only id/age columns and schema metadata. Never loads data arrays.
"""
import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import pyarrow as pa
from pathlib import Path

DATA_DIR     = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_49k")
PRETRAIN_CSV = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/probe_ids_type3_pretrain.csv"

# ── 1. Row counts ──────────────────────────────────────────────────────────
print("=== Split sizes ===")
for split in ["train", "valid", "test"]:
    pf = pq.ParquetFile(DATA_DIR / f"{split}.parquet")
    print(f"  {split}: {pf.metadata.num_rows} rows")

# ── 2. Schema — infer data array size without reading ─────────────────────
print("\n=== Schema ===")
pf     = pq.ParquetFile(DATA_DIR / "train.parquet")
schema = pf.schema_arrow
for i, name in enumerate(schema.names):
    field = schema.field(name)
    print(f"  {name}: {field.type}")

# ── 3. Read ONLY id + age (very small) ────────────────────────────────────
print("\n=== Age column ===")
tbl = pq.read_table(DATA_DIR / "train.parquet", columns=["age"])
ages = tbl["age"].to_pylist()
ages = [a for a in ages if a is not None]
print(f"  count:  {len(ages)}")
print(f"  min:    {min(ages):.1f}")
print(f"  max:    {max(ages):.1f}")
print(f"  mean:   {sum(ages)/len(ages):.1f}")
print(f"  nulls:  {tbl['age'].null_count}")
print(f"  sample: {ages[:8]}")

# ── 4. cpg_mapping directory ───────────────────────────────────────────────
print("\n=== cpg_mapping/ ===")
mapping_dir = DATA_DIR / "cpg_mapping"
if mapping_dir.exists():
    for f in sorted(mapping_dir.iterdir()):
        sz = f.stat().st_size
        print(f"  {f.name}  ({sz/1e3:.1f} KB)")
        try:
            if f.suffix == ".csv":
                m = pd.read_csv(f)
                print(f"    shape={m.shape}  cols={list(m.columns)}")
                print(f"    {m.head(3).to_string()}")
            elif f.suffix == ".parquet":
                m = pq.read_table(f).to_pandas()
                print(f"    shape={m.shape}  cols={list(m.columns)}")
                print(f"    {m.head(3).to_string()}")
            elif f.suffix in (".json", ".txt", ".tsv"):
                print("    " + open(f).read()[:400].replace("\n", "\n    "))
        except Exception as e:
            print(f"    error: {e}")
else:
    print("  not found")

# ── 5. Compare data array size to pretrain vocab ───────────────────────────
print("\n=== Pretrain vocab comparison ===")
data_field = schema.field("data")
print(f"  data field type: {data_field.type}")
# Extract list size from type if fixed-size list
if hasattr(data_field.type, "list_size"):
    arr_len = data_field.type.list_size
    print(f"  data array length (fixed): {arr_len}")
elif str(data_field.type).startswith("list"):
    print("  data is variable-length list — size not in schema")
    arr_len = None
else:
    arr_len = None

try:
    pretrain = pd.read_csv(PRETRAIN_CSV)
    print(f"  pretrain vocab: {len(pretrain)} CpGs")
    if arr_len:
        match = "✓ MATCH" if arr_len == len(pretrain) else "✗ MISMATCH"
        print(f"  {match}: data[{arr_len}] vs pretrain[{len(pretrain)}]")
except Exception as e:
    print(f"  could not read pretrain csv: {e}")

# ── 6. Inspect one data array — NaN/zero pattern ──────────────────────────
print("\n=== One sample data array (first row only) ===")
import pyarrow.compute as pc
tbl_one = pq.read_table(DATA_DIR / "train.parquet", columns=["data"]).slice(0, 1)
arr = np.array(tbl_one["data"][0].as_py(), dtype=np.float64)
print(f"  length:     {len(arr)}")
print(f"  NaN count:  {np.isnan(arr).sum()} ({100*np.isnan(arr).mean():.1f}%)")
print(f"  Zero count: {(arr==0).sum()} ({100*(arr==0).mean():.1f}%)")
print(f"  Non-NaN non-zero: {((~np.isnan(arr)) & (arr!=0)).sum()}")
print(f"  value range (non-NaN): [{np.nanmin(arr):.4f}, {np.nanmax(arr):.4f}]")
print(f"  mean (non-NaN): {np.nanmean(arr):.4f}")

# Where are the NaNs — beginning, end, or scattered?
nan_positions = np.where(np.isnan(arr))[0]
val_positions = np.where(~np.isnan(arr))[0]
if len(nan_positions) > 0:
    print(f"  NaN positions: first={nan_positions[0]}, last={nan_positions[-1]}, scattered={len(np.unique(np.diff(nan_positions)))>1}")
if len(val_positions) > 0:
    print(f"  Valid positions: first={val_positions[0]}, last={val_positions[-1]}")

# Check if NaN CpGs match a specific subset (e.g. positions > 21k)
print(f"\n  Values at positions 0-5:      {arr[:5].tolist()}")
print(f"  Values at positions 21000-21005: {arr[21000:21005].tolist()}")
print(f"  Values at positions 40000-40005: {arr[40000:40005].tolist()}")
print(f"  Values at positions -5 to end:  {arr[-5:].tolist()}")

print("\nDONE")
