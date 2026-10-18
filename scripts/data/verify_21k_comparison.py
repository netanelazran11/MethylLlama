"""
Verify that MethylGPT 21k and MethylLlama are trained/evaluated on the
SAME 21k dataset, split, and test set — the pre-flight fairness check before
launching MethylLlama k-fold CV.

Runs three checks:
  1. Read the MethylGPT 21k YAML config → print its data_path (confirm it's
     altumage_21k_3way.h5ad).
  2. Load altumage_21k_3way.h5ad obs → confirm split counts (expect
     train=7416 / valid=1308 / test=2264).
  3. Apply MethylLlama's pre-split filters (age 0-120 + duplicate exclusion) and report
     how many TEST samples survive. This tells us whether MethylGPT's raw 2264
     test set == MethylLlama's filtered test set, i.e. whether the metric comparison is
     on the exact same rows.

Usage on cluster:
  cd <repository root>
  source bmfm_methyl_env/bin/activate
  python scripts/data/verify_21k_comparison.py
"""

from collections import Counter, defaultdict, deque
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

import os
DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh
REPO = os.environ.get("METHYLLAMA_REPO", ".")             # scripts run from the repository root
METHYLGPT = os.environ.get("METHYLGPT_DIR", "MethylGPT")  # MethylGPT pipeline (comparison only)

# ── Paths ─────────────────────────────────────────────────────────────────────
GPT_YAML = Path(
    METHYLGPT + "/"
    "scripts/finetuning_age_prediction_medium/train_methylgpt_21k_altumage.yml"
)
H5AD = Path(
    DATA + "/data_methyl_21k_h5ad/"
    "altumage_21k_3way.h5ad"
)
DUP_CSV = Path(
    REPO + "/"
    "dataset_fingerprint_outputs/duplicate_pairs.csv"
)


def _read_categorical(grp):
    if isinstance(grp, h5py.Group) and "codes" in grp:
        codes = grp["codes"][:]
        cats = np.array(grp["categories"][:]).astype(str)
        return cats[codes]
    arr = grp[:]
    if arr.dtype.kind in ("S", "O"):
        arr = arr.astype(str)
    return arr


# ── CHECK 1: MethylGPT 21k YAML data_path ─────────────────────────────────────
print("=" * 70)
print("CHECK 1 — MethylGPT 21k config data path")
print("=" * 70)
if GPT_YAML.exists():
    text = GPT_YAML.read_text()
    print(f"YAML: {GPT_YAML}\n")
    for line in text.splitlines():
        low = line.lower()
        if any(k in low for k in ("path", "data", "h5ad", "parquet", "dir", "dataset", "file")):
            print(f"  {line.rstrip()}")
    gpt_uses_21k = "altumage_21k_3way" in text
    print(f"\n  → references 'altumage_21k_3way': {'YES ✓' if gpt_uses_21k else 'NO ✗ (inspect above)'}")
else:
    print(f"  YAML not found at {GPT_YAML}")
    print("  (set METHYLGPT_DIR to the MethylGPT fine-tuning checkout)")

# ── CHECK 2: 21k h5ad split counts ────────────────────────────────────────────
print("\n" + "=" * 70)
print("CHECK 2 — altumage_21k_3way.h5ad split counts")
print("=" * 70)
with h5py.File(H5AD, "r") as f:
    obs = f["obs"]
    idx_key = obs.attrs.get("_index", "_index")
    if idx_key not in obs:
        idx_key = next(
            k for k in obs.keys()
            if isinstance(obs[k], h5py.Dataset) and obs[k].dtype.kind in ("S", "O", "U")
        )
    gsm_ids = np.array(obs[idx_key][:]).astype(str)
    ages = obs["age"][:].astype(np.float32)
    splits = _read_categorical(obs["split"])

n_total = len(gsm_ids)
raw_counts = dict(Counter(splits))
print(f"  Index key    : '{idx_key}'")
print(f"  Total samples: {n_total:,}")
print(f"  Split counts : {raw_counts}")
expect = {"train": 7416, "valid": 1308, "test": 2264}
match = all(raw_counts.get(k) == v for k, v in expect.items())
print(f"  Expected     : {expect}")
print(f"  → matches expected: {'YES ✓' if match else 'NO ✗'}")

# ── CHECK 3: how many TEST samples survive MethylLlama's filters ──────────────────────
print("\n" + "=" * 70)
print("CHECK 3 — TEST set after MethylLlama pre-split filters (age + dedup)")
print("=" * 70)

age_ok = (ages >= 0) & (ages <= 120)
n_age_removed = int((~age_ok).sum())

dup_exclude = set()
if DUP_CSV.exists():
    dup_df = pd.read_csv(DUP_CSV)
    adj = defaultdict(set)
    all_nodes = set()
    for _, row in dup_df.iterrows():
        a, b = str(row["id_a"]), str(row["id_b"])
        adj[a].add(b)
        adj[b].add(a)
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
        dup_exclude.update(comp[1:])
    print(f"  Duplicate exclusion set: {len(dup_exclude)} samples")
else:
    print(f"  DUP_CSV not found ({DUP_CSV}) — skipping dedup")

dup_ok = np.array([g not in dup_exclude for g in gsm_ids])
keep = age_ok & dup_ok

test_mask_raw = splits == "test"
test_mask_filtered = test_mask_raw & keep

n_test_raw = int(test_mask_raw.sum())
n_test_filtered = int(test_mask_filtered.sum())
n_test_dropped = n_test_raw - n_test_filtered

# breakdown of what dropped in test
test_age_dropped = int((test_mask_raw & ~age_ok).sum())
test_dup_dropped = int((test_mask_raw & age_ok & ~dup_ok).sum())

print(f"\n  MethylGPT 21k test (raw split=='test') : {n_test_raw:,}")
print(f"  MethylLlama test (after age+dedup filters)     : {n_test_filtered:,}")
print(f"  Dropped by MethylLlama filters                 : {n_test_dropped:,}")
print(f"      - age outliers : {test_age_dropped}")
print(f"      - duplicates   : {test_dup_dropped}")

print("\n" + "=" * 70)
print("VERDICT")
print("=" * 70)
if n_test_dropped == 0:
    print("  ✓ Test sets are IDENTICAL rows — MethylGPT 21k and MethylLlama evaluate on")
    print("    exactly the same samples. Metric comparison is fully fair.")
else:
    print(f"  ⚠ MethylLlama test set is {n_test_filtered:,} vs MethylGPT's {n_test_raw:,}")
    print(f"    ({n_test_dropped} samples differ). Two options for a clean comparison:")
    print("      (a) Re-evaluate MethylGPT 21k on the MethylLlama-filtered test set, OR")
    print("      (b) Report MethylLlama test WITHOUT age/dedup filters on the test split")
    print("          (keep filters only on train/val).")
    print(f"    Note: {n_test_dropped}/{n_test_raw} = "
          f"{100*n_test_dropped/n_test_raw:.1f}% of test — likely small, but state it.")
