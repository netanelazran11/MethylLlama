"""
Generate 5-fold stratified cross-validation splits for MethylLlama fine-tuning.

Strategy:
  - Fixed test set  : obs['split'] == 'test'  (1,927 samples, never touched)
  - Pool for CV     : obs['split'] in {'train','valid'}  (8,431 samples)
  - Stratification  : age_bin (10-yr) × tissue_type
  - 5-fold CV       : StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

Outputs (in OUT_DIR):
  test_ids.npy          — fixed test set GSM IDs (string array, 1,927)
  fold_0_train.npy      — train GSM IDs for fold 0  (~6,745)
  fold_0_val.npy        — val   GSM IDs for fold 0  (~1,686)
  ...
  fold_4_train.npy
  fold_4_val.npy

Usage on cluster:
  cd <repository root>
  source bmfm_methyl_env/bin/activate
  python scripts/data/create_kfold_splits.py
"""

import h5py
import numpy as np
import pandas as pd
from pathlib import Path
from sklearn.model_selection import StratifiedKFold

import os
DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh
REPO = os.environ.get("METHYLLAMA_REPO", ".")             # scripts run from the repository root
METHYLGPT = os.environ.get("METHYLGPT_DIR", "MethylGPT")  # MethylGPT pipeline (comparison only)

H5AD = Path(
    DATA + "/data_methyl_21k_h5ad/"
    "altumage_21k_3way.h5ad"
)
DUP_CSV = Path(
    REPO + "/"
    "dataset_fingerprint_outputs/duplicate_pairs.csv"
)
OUT_DIR = Path(
    REPO + "/"
    "outputs/kfold_splits"
)
N_FOLDS = 5
SEED    = 42

def _read_categorical(grp):
    """Read an h5ad categorical column (codes + categories) or plain string array."""
    if isinstance(grp, h5py.Group) and "codes" in grp:
        codes = grp["codes"][:]
        cats  = np.array(grp["categories"][:]).astype(str)
        return cats[codes]
    arr = grp[:]
    if arr.dtype.kind in ("S", "O"):
        arr = arr.astype(str)
    return arr

# ── 1. Load obs only (skip loading the full X matrix) ────────────────────────
print(f"Reading obs from {H5AD} ...")
with h5py.File(H5AD, "r") as f:
    obs = f["obs"]

    # GSM IDs — find the index key robustly (anndata stores it in attrs["_index"])
    idx_key = obs.attrs.get("_index", "_index")
    if idx_key not in obs:
        # Fall back: first string-like dataset in obs
        idx_key = next(
            k for k in obs.keys()
            if isinstance(obs[k], h5py.Dataset) and obs[k].dtype.kind in ("S", "O", "U")
        )
    gsm_ids = np.array(obs[idx_key][:]).astype(str)
    print(f"  Index key: '{idx_key}'  ({len(gsm_ids):,} samples)")

    # Age
    ages = obs["age"][:].astype(np.float32)

    # Split column (categorical)
    splits = _read_categorical(obs["split"])

    # Tissue type (categorical or plain string)
    if "tissue_type" in obs:
        tissue = _read_categorical(obs["tissue_type"])
    else:
        tissue = np.full(len(gsm_ids), "unknown", dtype=object)

n_total = len(gsm_ids)
print(f"  Total samples: {n_total:,}")
from collections import Counter
print(f"  Split counts: {dict(Counter(splits))}")

# ── 2. Apply same pre-split filters as the fine-tune pipeline ────────────────

# 2a. Age outlier filter (age<0 or age>120)
age_ok = (ages >= 0) & (ages <= 120)
n_age_removed = int((~age_ok).sum())
print(f"\nAge outlier filter: removing {n_age_removed} samples (age<0 or age>120)")

# 2b. Duplicate exclusion
dup_exclude = set()
if DUP_CSV.exists():
    dup_df = pd.read_csv(DUP_CSV)
    from collections import defaultdict, deque
    adj = defaultdict(set)
    all_nodes = set()
    for _, row in dup_df.iterrows():
        a, b = str(row["id_a"]), str(row["id_b"])
        adj[a].add(b); adj[b].add(a)
        all_nodes.update([a, b])
    visited = set()
    for start in sorted(all_nodes):
        if start in visited:
            continue
        comp = []
        q = deque([start])
        while q:
            node = q.popleft()
            if node in visited:
                continue
            visited.add(node)
            comp.append(node)
            q.extend(adj[node] - visited)
        comp.sort()
        dup_exclude.update(comp[1:])  # keep first alphabetically, exclude rest
    print(f"Duplicate exclusion: {len(dup_exclude)} samples excluded")
else:
    print(f"DUP_CSV not found ({DUP_CSV}) — skipping dedup")

dup_ok = np.array([g not in dup_exclude for g in gsm_ids])
keep   = age_ok & dup_ok
print(f"Samples after filters: {keep.sum():,} (removed {(~keep).sum():,})")

gsm_ids = gsm_ids[keep]
ages    = ages[keep]
splits  = splits[keep]
tissue  = tissue[keep]

# ── 3. Separate fixed test set ────────────────────────────────────────────────
test_mask = splits == "test"
pool_mask = np.isin(splits, ["train", "valid"])

test_ids = gsm_ids[test_mask]
pool_ids = gsm_ids[pool_mask]
pool_ages   = ages[pool_mask]
pool_tissue = tissue[pool_mask]

print(f"\nFixed test set : {len(test_ids):,} samples")
print(f"CV pool        : {len(pool_ids):,} samples")

# ── 4. Build stratification key: age_bin × tissue_type ───────────────────────
age_bin   = np.clip(np.floor(pool_ages / 10).astype(int), 0, 8)
strat_key = np.array([f"{b}__{t}" for b, t in zip(age_bin, pool_tissue)])

# Some strata may have <5 samples; StratifiedKFold will complain if any class
# has fewer than n_splits samples. Merge rare strata into "rare".
from collections import Counter as _Counter
counts = _Counter(strat_key)
rare   = {k for k, v in counts.items() if v < N_FOLDS}
if rare:
    print(f"  Merging {len(rare)} rare strata (count < {N_FOLDS}) → 'rare'")
    strat_key = np.where(np.isin(strat_key, list(rare)), "rare", strat_key)

print(f"  Unique strata: {len(np.unique(strat_key))}")

# ── 5. 5-fold stratified split ────────────────────────────────────────────────
skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)

OUT_DIR.mkdir(parents=True, exist_ok=True)

# Save fixed test set
np.save(OUT_DIR / "test_ids.npy", test_ids)
print(f"\nSaved test_ids.npy  ({len(test_ids):,} samples)")

print(f"\n{'Fold':>4}  {'Train':>6}  {'Val':>6}  {'Val age mean':>12}  {'Train age mean':>14}")
print("-" * 55)

for fold, (train_idx, val_idx) in enumerate(skf.split(pool_ids, strat_key)):
    t_ids = pool_ids[train_idx]
    v_ids = pool_ids[val_idx]

    np.save(OUT_DIR / f"fold_{fold}_train.npy", t_ids)
    np.save(OUT_DIR / f"fold_{fold}_val.npy",   v_ids)

    t_age_mean = pool_ages[train_idx].mean()
    v_age_mean = pool_ages[val_idx].mean()
    print(f"  {fold}    {len(t_ids):>6,}  {len(v_ids):>6,}  {v_age_mean:>12.2f}  {t_age_mean:>14.2f}")

print(f"\nAll splits saved to: {OUT_DIR}")
print("\nAge distribution in fixed test set:")
print(f"  mean={pool_ages[pool_mask[:len(pool_ages)]].mean():.1f}   range=[{ages[test_mask].min():.0f}, {ages[test_mask].max():.0f}]")

print("\nDone. Next step: submit 5 fine-tune jobs with FOLD=0..4")
print(f"  sbatch --export=FOLD=0,CHECKPOINT=<ckpt> scripts/finetune/finetune_kfold.sh")
