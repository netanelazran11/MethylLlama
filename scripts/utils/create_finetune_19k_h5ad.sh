#!/bin/bash -l
#SBATCH --job-name=create-19k-h5ad
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=00:30:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"

H5AD_21K="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
VALID_CPGS="${REPO}/outputs/cpg_coverage_analysis/cpgs_always_measured_finetune.csv"
OUT_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_19k_h5ad"
OUT_H5AD="${OUT_DIR}/finetuning_19608_clean.h5ad"

AGE_MIN=0
AGE_MAX=120

mkdir -p "${OUT_DIR}"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo "CREATE CLEAN 19,608-CpG FINETUNE H5AD"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Input:      ${H5AD_21K}"
echo "Valid CpGs: ${VALID_CPGS}"
echo "Output:     ${OUT_H5AD}"
echo "Age filter: [${AGE_MIN}, ${AGE_MAX}]"
echo "============================================================"

python3 - <<PY
import h5py
import numpy as np
import pandas as pd
import anndata as ad
import scipy.sparse

H5AD_21K   = "${H5AD_21K}"
VALID_CPGS = "${VALID_CPGS}"
OUT_H5AD   = "${OUT_H5AD}"
AGE_MIN    = float("${AGE_MIN}")
AGE_MAX    = float("${AGE_MAX}")

# ── 1. Load valid CpG list (19,608 in pretrain vocab) ───────────────────────
valid_cpgs = pd.read_csv(VALID_CPGS)["cpg_site"].tolist()
valid_cpg_set = set(valid_cpgs)
print(f"Valid CpGs (in pretrain vocab): {len(valid_cpgs):,}")

# ── 2. Read h5ad via h5py ────────────────────────────────────────────────────
print(f"\nReading {H5AD_21K} ...")
with h5py.File(H5AD_21K, "r") as f:
    # var (CpG names)
    var_grp  = f["var"]
    idx_key  = var_grp.attrs.get("_index", "_index")
    if idx_key not in var_grp: idx_key = list(var_grp.keys())[0]
    all_cpgs = np.array(var_grp[idx_key]).astype(str)

    # obs (sample metadata)
    obs_grp     = f["obs"]
    obs_idx_key = obs_grp.attrs.get("_index", "_index")
    if obs_idx_key not in obs_grp: obs_idx_key = list(obs_grp.keys())[0]
    obs_index   = np.array(obs_grp[obs_idx_key]).astype(str)

    obs_cols = {}
    for k in obs_grp.keys():
        if k == obs_idx_key: continue
        try:
            v = obs_grp[k]
            if isinstance(v, h5py.Dataset):
                arr = v[()]
                if len(arr) > 0 and hasattr(arr.flat[0], 'decode'):
                    arr = np.array([x.decode() for x in arr])
                obs_cols[k] = arr
            elif isinstance(v, h5py.Group) and 'categories' in v:
                cats  = np.array(v['categories'][()]).astype(str)
                codes = v['codes'][()]
                obs_cols[k] = np.array([cats[c] if c >= 0 else None for c in codes])
        except Exception:
            pass

    # X matrix
    X_grp = f["X"]
    if isinstance(X_grp, h5py.Dataset):
        X = X_grp[()].astype(np.float32)
    else:
        data    = X_grp["data"][()]
        indices = X_grp["indices"][()]
        indptr  = X_grp["indptr"][()]
        shape   = tuple(X_grp.attrs["shape"])
        X = scipy.sparse.csr_matrix(
            (data, indices, indptr), shape=shape
        ).toarray().astype(np.float32)

obs = pd.DataFrame(obs_cols, index=obs_index)
print(f"Loaded: {X.shape[0]:,} cells × {X.shape[1]:,} CpGs")
print(f"obs columns: {list(obs.columns)}")

# ── 3. Filter CpGs to 19,608 ─────────────────────────────────────────────────
cpg_mask    = np.array([c in valid_cpg_set for c in all_cpgs])
kept_cpgs   = all_cpgs[cpg_mask]
X           = X[:, cpg_mask]
dropped_cpg = int((~cpg_mask).sum())
print(f"\nCpG filtering:")
print(f"  Kept:    {cpg_mask.sum():,}  (in pretrain vocab)")
print(f"  Dropped: {dropped_cpg:,}  (not in pretrain vocab)")

# ── 4. Filter age ────────────────────────────────────────────────────────────
ages     = obs["age"].astype(float)
age_mask = (ages >= AGE_MIN) & (ages <= AGE_MAX)
n_before = len(obs)
print(f"\nAge filtering ({AGE_MIN}–{AGE_MAX}):")
print(f"  Before: {n_before:,}  (age min={ages.min():.0f} max={ages.max():.0f})")
print(f"  Removed: {(~age_mask).sum():,} samples outside range")

# ── 5. Remove leakage sample ─────────────────────────────────────────────────
# Find samples whose 8-value fingerprint appears in more than one split
splits      = obs["split"].values
fingerprints = {}
for i, (idx, row) in enumerate(obs.iterrows()):
    fp = tuple(X[i, :8].tolist())
    if fp not in fingerprints:
        fingerprints[fp] = []
    fingerprints[fp].append((i, splits[i]))

leakage_indices = set()
for fp, entries in fingerprints.items():
    split_set = set(s for _, s in entries)
    if len(split_set) > 1:
        for i, _ in entries:
            leakage_indices.add(i)

print(f"\nLeakage samples found: {len(leakage_indices)}")
for i in leakage_indices:
    print(f"  idx={i}  id={obs.index[i]}  split={splits[i]}  age={ages.iloc[i]:.0f}")

# Keep: age filter AND not leakage
keep_mask = age_mask.values.copy()
for i in leakage_indices:
    keep_mask[i] = False

X   = X[keep_mask]
obs = obs[keep_mask]
print(f"\nAfter all filters: {X.shape[0]:,} cells (removed {n_before - X.shape[0]:,})")

# ── 6. Final NaN check ───────────────────────────────────────────────────────
nan_count = int(np.isnan(X).sum())
print(f"\nNaN in final matrix: {nan_count} {'✓' if nan_count == 0 else '⚠ PROBLEM'}")

# ── 7. Final split distribution ──────────────────────────────────────────────
print(f"\nFinal split distribution:")
print(obs["split"].value_counts().to_string())

# ── 8. Leakage check on final data ───────────────────────────────────────────
print(f"\nFinal leakage check:")
final_splits = obs["split"].values
final_fps    = [tuple(X[i, :8].tolist()) for i in range(len(obs))]
for s1, s2 in [("train", "valid"), ("train", "test"), ("valid", "test")]:
    fp1 = set(final_fps[i] for i, s in enumerate(final_splits) if s == s1)
    fp2 = set(final_fps[i] for i, s in enumerate(final_splits) if s == s2)
    n   = len(fp1 & fp2)
    print(f"  {s1} ∩ {s2}: {n} {'✓' if n == 0 else '⚠ LEAKAGE'}")

# ── 9. Build var ─────────────────────────────────────────────────────────────
var_df = pd.DataFrame(index=kept_cpgs)
var_df.index.name = None

# ── 10. Save ─────────────────────────────────────────────────────────────────
print(f"\nSaving to {OUT_H5AD} ...")
adata = ad.AnnData(X=X, obs=obs, var=var_df)
adata.write_h5ad(OUT_H5AD, compression="gzip")

import os
size_mb = os.path.getsize(OUT_H5AD) / 1e6

print(f"""
============================================================
DONE
============================================================
Output:       {OUT_H5AD}
Shape:        {adata.n_obs:,} cells × {adata.n_vars:,} CpGs
NaN:          0
obs columns:  {list(adata.obs.columns)}
Splits:       {dict(adata.obs['split'].value_counts())}
File size:    {size_mb:.1f} MB

vs old finetune h5ad:
  Old: 11,453 cells × 49,156 CpGs (29,548 always-NaN columns)
  New: {adata.n_obs:,} cells × {adata.n_vars:,} CpGs (zero NaN, clean ages, no leakage)
============================================================
""")
PY

echo "============================================================"
echo "Finished: $(date)"
echo "Output: ${OUT_H5AD}"
echo "============================================================"
