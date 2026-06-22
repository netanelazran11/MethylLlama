"""
Check duplicate sample IDs in the fine-tune dataset.
Answers:
  1. How many unique IDs are duplicated?
  2. Are duplicates within the same split or across splits?
  3. Is there data leakage (same ID in train AND test)?
"""

import pandas as pd
import scanpy as sc

FINETUNE_PATH = (
    "/sci/labs/benjamin.yakir/netanel.azran/data"
    "/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
)

print("Loading fine-tune dataset...")
adata = sc.read_h5ad(FINETUNE_PATH)
print(f"Loaded: {adata.n_obs:,} samples")

obs = adata.obs.copy()
obs["row_idx"] = range(len(obs))

print(f"\nColumns: {list(obs.columns)}")
print(f"\nSplit counts:\n{obs['split'].value_counts()}")

# ── 1. Basic duplicate stats ──────────────────────────────────────────────────
id_counts = obs["id"].value_counts()
dup_ids = id_counts[id_counts > 1]
print(f"\n{'='*60}")
print(f"  DUPLICATE SAMPLE IDs")
print(f"{'='*60}")
print(f"Total samples         : {len(obs):,}")
print(f"Unique IDs            : {obs['id'].nunique():,}")
print(f"IDs that appear >1x   : {len(dup_ids):,}")
print(f"Samples involved      : {dup_ids.sum():,}")
print(f"\nDuplicate count distribution:")
print(id_counts.value_counts().rename_axis("appears N times").rename("# of IDs").to_string())

# ── 2. Are duplicates within same split or across splits? ────────────────────
print(f"\n{'='*60}")
print(f"  DUPLICATE IDs — SPLIT BREAKDOWN")
print(f"{'='*60}")

within_split   = 0  # both copies in same split
across_splits  = 0  # copies in different splits — LEAKAGE RISK
leakage_train_test = 0
leakage_train_valid = 0
leakage_test_valid  = 0

cross_split_examples = []

for sample_id, group in obs[obs["id"].isin(dup_ids.index)].groupby("id"):
    splits_seen = set(group["split"].tolist())
    if len(splits_seen) == 1:
        within_split += 1
    else:
        across_splits += 1
        splits_list = sorted(splits_seen)
        if "train" in splits_seen and "test" in splits_seen:
            leakage_train_test += 1
        if "train" in splits_seen and "valid" in splits_seen:
            leakage_train_valid += 1
        if "test" in splits_seen and "valid" in splits_seen:
            leakage_test_valid += 1
        if len(cross_split_examples) < 10:
            cross_split_examples.append((sample_id, splits_list))

print(f"IDs appearing in same split only  : {within_split:,}")
print(f"IDs appearing across splits       : {across_splits:,}  ← LEAKAGE RISK")
print(f"  of which train ↔ test           : {leakage_train_test:,}")
print(f"  of which train ↔ valid          : {leakage_train_valid:,}")
print(f"  of which test  ↔ valid          : {leakage_test_valid:,}")

if cross_split_examples:
    print(f"\nExample cross-split duplicate IDs (first 10):")
    for sid, splits in cross_split_examples:
        print(f"  ID={sid}  →  splits: {splits}")

# ── 3. Age consistency for duplicates ────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  AGE CONSISTENCY FOR DUPLICATED IDs")
print(f"{'='*60}")
dup_obs = obs[obs["id"].isin(dup_ids.index)]
age_consistency = dup_obs.groupby("id")["age"].nunique()
same_age    = (age_consistency == 1).sum()
diff_age    = (age_consistency > 1).sum()
print(f"Duplicated IDs with same age  : {same_age:,}")
print(f"Duplicated IDs with diff age  : {diff_age:,}  ← longitudinal or mislabelled")

if diff_age > 0:
    diff_examples = age_consistency[age_consistency > 1].index[:5]
    print(f"\nExamples with different ages:")
    for sid in diff_examples:
        rows = obs[obs["id"] == sid][["id", "age", "split"]]
        print(rows.to_string(index=False))

# ── 4. Summary verdict ────────────────────────────────────────────────────────
print(f"\n{'='*60}")
print(f"  VERDICT")
print(f"{'='*60}")
if leakage_train_test > 0:
    print(f"  ⚠  DATA LEAKAGE DETECTED: {leakage_train_test} IDs appear in both TRAIN and TEST")
else:
    print(f"  ✓  No train↔test leakage")
if leakage_train_valid > 0:
    print(f"  ⚠  DATA LEAKAGE DETECTED: {leakage_train_valid} IDs appear in both TRAIN and VALID")
else:
    print(f"  ✓  No train↔valid leakage")
if diff_age > 0:
    print(f"  ℹ  {diff_age} IDs have different ages across copies → likely longitudinal samples")
else:
    print(f"  ✓  All duplicates have the same age → likely technical replicates")

print("\nDONE")
