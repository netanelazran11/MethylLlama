"""
Compare train/valid/test splits between MethylGPT 21k (parquet) and
MethylLlama's 21k h5ad (altumage_21k_3way.h5ad).

Same logic as compare_splits.py (19k), adapted for the 21k dataset:
  - auto-detects MethylGPT 21k's data dir from its YAML config
  - row-level match: MethylGPT integer index → GSM ID via h5ad row order
  - fallback: distributional match by sorted age values per split

Verifies both models used identical valid & test sets → fair comparison.

Usage on cluster:
  cd <repository root>
  source bmfm_methyl_env/bin/activate
  python scripts/data/compare_splits_21k.py
"""

from collections import Counter
from pathlib import Path

import h5py
import numpy as np
import pandas as pd
import pyarrow.parquet as pq

import os
DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh
REPO = os.environ.get("METHYLLAMA_REPO", ".")             # scripts run from the repository root
METHYLGPT = os.environ.get("METHYLGPT_DIR", "MethylGPT")  # MethylGPT pipeline (comparison only)

# ── Paths ─────────────────────────────────────────────────────────────────────
GPT_YAML = Path(
    METHYLGPT + "/"
    "scripts/finetuning_age_prediction_medium/train_methylgpt_21k_altumage.yml"
)
H5AD_PATH = (
    DATA + "/data_methyl_21k_h5ad/"
    "altumage_21k_3way.h5ad"
)
# Fallbacks tried if the YAML doesn't reveal a parquet dir:
PARQUET_DIR_CANDIDATES = [
    METHYLGPT + "/data/21k_data/finetuning_data",
    METHYLGPT + "/data/21k_altumage/finetuning_data",
]


# ── Locate MethylGPT 21k parquet dir ──────────────────────────────────────────
def find_parquet_dir() -> str | None:
    # 1. Try to read a data dir out of the YAML
    if GPT_YAML.exists():
        print(f"Reading MethylGPT YAML: {GPT_YAML}")
        for line in GPT_YAML.read_text().splitlines():
            low = line.lower()
            if any(k in low for k in ("path", "dir", "data", "parquet", "root")) and "/" in line:
                print(f"  cfg> {line.strip()}")
                # extract a filesystem path token
                for tok in line.replace(":", " ").replace('"', " ").replace("'", " ").split():
                    if tok.startswith("/") and "MethylGPT" in tok:
                        p = Path(tok)
                        d = p if p.is_dir() else p.parent
                        for probe in (d, d / "finetuning_data"):
                            if (probe / "test.parquet").exists():
                                print(f"  → using parquet dir from YAML: {probe}")
                                return str(probe)
    else:
        print(f"YAML not found: {GPT_YAML} (using fallback candidates)")

    # 2. Fallback candidates
    for cand in PARQUET_DIR_CANDIDATES:
        if (Path(cand) / "test.parquet").exists():
            print(f"  → using fallback parquet dir: {cand}")
            return cand
    return None


PARQUET_DIR = find_parquet_dir()
if PARQUET_DIR is None:
    raise SystemExit(
        "Could not locate MethylGPT 21k parquet dir.\n"
        "  Inspect the YAML data path and set PARQUET_DIR manually, or run:\n"
        "  find ${METHYLGPT_DIR} -name 'test.parquet'"
    )

# ── MethylGPT (parquet) — sizes ───────────────────────────────────────────────
print("\nLoading MethylGPT 21k parquet files (schema inspect)...")
gpt_sizes = {}
for split in ("train", "valid", "test"):
    pf = pq.ParquetFile(f"{PARQUET_DIR}/{split}.parquet")
    print(f"  {split} columns: {pf.schema_arrow.names[:8]}")
    df = pf.read(columns=["age"]).to_pandas()
    gpt_sizes[split] = len(df)
    print(f"  MethylGPT  {split:5s}: {len(df):5d} samples")

# ── MethylLlama 21k h5ad — obs only ───────────────────────────────────────────
print("\nLoading MethylLlama 21k h5ad (obs only)...")
with h5py.File(H5AD_PATH, "r") as f:
    obs_grp = f["obs"]
    idx_key = obs_grp.attrs.get("_index", "_index")
    if idx_key not in obs_grp:
        idx_key = next(
            k for k in obs_grp.keys()
            if isinstance(obs_grp[k], h5py.Dataset) and obs_grp[k].dtype.kind in ("S", "O", "U")
        )
    all_gsm = [x.decode() if isinstance(x, bytes) else str(x) for x in obs_grp[idx_key][:]]
    sc = obs_grp["split"]
    split_vals = list(sc["codes"][:])
    split_cats = [x.decode() if isinstance(x, bytes) else str(x) for x in sc["categories"][:]]
    splits_decoded = [split_cats[c] for c in split_vals]
    h5ad_ages = list(obs_grp["age"][:])

print(f"  Index key    : '{idx_key}'")
print(f"  Total samples: {len(all_gsm)}")
print(f"  Split counts : {dict(Counter(splits_decoded))}")

h5ad_by_idx = {
    i: (gsm, age, sp)
    for i, (gsm, age, sp) in enumerate(zip(all_gsm, h5ad_ages, splits_decoded))
}

# ── Row-level match (MethylGPT integer index → GSM via h5ad row order) ────────
print("\n" + "=" * 60)
print("SPLIT COMPARISON — exact sample ID mapping")
print("=" * 60)
all_match_ids = True
for split in ("train", "valid", "test"):
    df = pq.ParquetFile(f"{PARQUET_DIR}/{split}.parquet").read(columns=["age"]).to_pandas()
    matched = wrong_split = age_mismatch = not_found = 0
    gsm_examples = []
    for row_idx_str, row_age in zip(df.index.astype(str), df["age"]):
        try:
            row_idx = int(row_idx_str)
        except ValueError:
            not_found += 1
            continue
        if row_idx not in h5ad_by_idx:
            not_found += 1
            continue
        gsm, h5_age, h5_split = h5ad_by_idx[row_idx]
        gsm_examples.append(gsm)
        if abs(float(row_age) - float(h5_age)) > 0.01:
            age_mismatch += 1
        elif h5_split != split:
            wrong_split += 1
        else:
            matched += 1
    total = len(df)
    ok = matched == total
    all_match_ids = all_match_ids and ok
    print(f"\n{split.upper()} ({total} samples):")
    print(f"  Fully matched (ID + age + split): {matched}/{total}  {'✓' if ok else '✗'}")
    if age_mismatch:
        print(f"  Age mismatch: {age_mismatch}")
    if wrong_split:
        print(f"  Wrong split : {wrong_split}")
    if not_found:
        print(f"  Index not in h5ad: {not_found}")
    print(f"  Example GSM IDs: {gsm_examples[:3]}")

print("\n" + "=" * 60)
print(f"SPLITS 100% IDENTICAL (ID + age + split): {'YES ✓' if all_match_ids else 'NO ✗'}")
print("  (row-level; may fail if MethylGPT reindexed — see age-value check below)")
print("=" * 60)

# ── Distributional match by sorted ages per split ─────────────────────────────
print("\n" + "=" * 60)
print("SPLIT COMPARISON — via age values (size + distribution)")
print("=" * 60)
gpt_ages = {
    s: sorted(pq.ParquetFile(f"{PARQUET_DIR}/{s}.parquet").read(columns=["age"])["age"].to_pylist())
    for s in ("train", "valid", "test")
}
llama_ages = {
    s: sorted(a for a, sp in zip(h5ad_ages, splits_decoded) if sp == s)
    for s in ("train", "valid", "test")
}

all_match_dist = True
for split in ("train", "valid", "test"):
    g, l = gpt_ages[split], llama_ages[split]
    sizes_match = len(g) == len(l)
    ages_match = sizes_match and np.allclose(g, l, atol=0.01)
    all_match_dist = all_match_dist and sizes_match and ages_match
    print(f"\n{split.upper()}:")
    print(f"  MethylGPT  : {len(g):5d} samples  age [{min(g):.1f}, {max(g):.1f}]  mean={np.mean(g):.2f}")
    print(f"  MethylLlama: {len(l):5d} samples  age [{min(l):.1f}, {max(l):.1f}]  mean={np.mean(l):.2f}")
    print(f"  Sizes match : {'YES ✓' if sizes_match else 'NO ✗'}")
    print(f"  Ages match  : {'YES ✓' if ages_match else 'NO ✗'}")

print("\n" + "=" * 60)
print(f"SAME DATASET & SPLIT (21k): {'YES ✓' if all_match_dist else 'NO ✗ — comparison may be unfair!'}")
print("=" * 60)
