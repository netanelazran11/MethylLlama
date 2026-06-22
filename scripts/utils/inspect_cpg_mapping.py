#!/usr/bin/env python3
import pandas as pd, json
from pathlib import Path

BASE = "/sci/labs/benjamin.yakir/netanel.azran/repos/MethylGPT-Thesis/data/finetuning_data_21k"

# CpG mapping
for f in sorted(Path(BASE, "cpg_mapping").iterdir()):
    print(f"--- {f.name} ---")
    df = pd.read_parquet(f) if f.suffix == ".parquet" else pd.read_csv(f)
    print("shape:", df.shape, "  cols:", df.columns.tolist())
    print(df.head(10).to_string())
    print()

# dataset_summary.json
d = json.load(open(Path(BASE, "dataset_summary.json")))
print("=== dataset_summary.json ===")
for k, v in d.items():
    if isinstance(v, list):
        print(f"  {k}: {len(v)} items  first 5: {v[:5]}")
    else:
        print(f"  {k}: {v}")
