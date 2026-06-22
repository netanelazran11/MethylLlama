#!/bin/bash -l
#SBATCH --job-name=analyze-21k-h5ad
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

H5AD_DIR="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_21k_h5ad"
H5AD_3WAY="${H5AD_DIR}/altumage_21k_3way.h5ad"
H5AD_COMB="${H5AD_DIR}/altumage_21k_combined.h5ad"
H5AD_49K="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
CPG_MAP="${H5AD_DIR}/probe_ids_type3_21k.csv"

cd "${REPO}"
source bmfm_methyl_env/bin/activate

echo "============================================================"
echo "ANALYZING EXISTING 21k H5AD FILES"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "============================================================"

python3 - <<PY
import h5py
import numpy as np
import pandas as pd
import scipy.sparse

H5AD_3WAY = "${H5AD_3WAY}"
H5AD_COMB = "${H5AD_COMB}"
H5AD_49K  = "${H5AD_49K}"
CPG_MAP   = "${CPG_MAP}"

def read_h5ad_raw(path):
    """Read X, obs, var from h5ad via h5py — avoids anndata obs validation bugs."""
    with h5py.File(path, "r") as f:
        # var (CpG names)
        var      = f["var"]
        idx_key  = var.attrs.get("_index", "_index")
        if idx_key not in var: idx_key = list(var.keys())[0]
        cpg_names = np.array(var[idx_key]).astype(str)

        # obs (sample metadata)
        obs_grp  = f["obs"]
        obs_idx_key = obs_grp.attrs.get("_index", "_index")
        if obs_idx_key not in obs_grp: obs_idx_key = list(obs_grp.keys())[0]
        obs_index = np.array(obs_grp[obs_idx_key]).astype(str)

        obs_cols = {}
        for k in obs_grp.keys():
            if k == obs_idx_key: continue
            try:
                v = obs_grp[k]
                if isinstance(v, h5py.Dataset):
                    arr = v[()]
                    if hasattr(arr[0], 'decode'):
                        arr = np.array([x.decode() for x in arr])
                    obs_cols[k] = arr
                elif isinstance(v, h5py.Group) and 'categories' in v:
                    # categorical
                    cats = np.array(v['categories'][()]).astype(str)
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
    return X, obs, cpg_names

def analyze(path, label):
    print(f"\n{'='*60}")
    print(f"File: {label}")
    print(f"Path: {path}")
    print('='*60)

    X, obs, cpg_names = read_h5ad_raw(path)
    n_cells, n_cpgs = X.shape

    print(f"Shape: {n_cells:,} cells × {n_cpgs:,} CpGs")

    # NaN
    nan_total = int(np.isnan(X).sum())
    print(f"NaN values: {nan_total} {'✓ CLEAN' if nan_total == 0 else '⚠ HAS NaN'}")

    # Beta range
    valid = X[~np.isnan(X)]
    print(f"Beta range: [{valid.min():.4f}, {valid.max():.4f}]  mean={valid.mean():.4f}")

    # CpG names
    print(f"\nCpG sites: {n_cpgs:,}  unique: {len(set(cpg_names)):,}")
    print(f"Sample CpGs: {cpg_names[:3].tolist()} ... {cpg_names[-3:].tolist()}")

    # obs columns
    print(f"\nobs columns: {list(obs.columns)}")
    print(f"obs index sample: {list(obs.index[:3])}")

    if 'age' in obs.columns:
        ages = obs['age'].astype(float)
        print(f"\nAge: min={ages.min():.0f}  max={ages.max():.0f}  mean={ages.mean():.1f}  NaN={ages.isna().sum()}")

    if 'split' in obs.columns:
        print(f"\nSplit distribution:\n{obs['split'].value_counts().to_string()}")
    else:
        print("\n⚠ No 'split' column found in obs")

    # Sample ID uniqueness
    n_ids    = len(obs.index)
    n_unique = len(set(obs.index))
    print(f"\nSample IDs: {n_ids:,} total, {n_unique:,} unique {'✓' if n_ids == n_unique else '⚠ DUPLICATES'}")

    return X, obs, set(cpg_names)

# ── Analyze both 21k files ──────────────────────────────────────────────────
X_3way, obs_3way, cpgs_3way = analyze(H5AD_3WAY, "altumage_21k_3way.h5ad")
X_comb, obs_comb, cpgs_comb = analyze(H5AD_COMB,  "altumage_21k_combined.h5ad")

# ── Compare CpG lists ───────────────────────────────────────────────────────
print(f"\n{'='*60}")
print("CpG LIST COMPARISON")
print('='*60)

# vs parquet CpG map
try:
    cpg_df      = pd.read_csv(CPG_MAP)
    parquet_cpgs = set(cpg_df['illumina_probe_id'].tolist())
    print(f"Parquet CpG map:     {len(parquet_cpgs):,}")
    print(f"3way h5ad CpGs:      {len(cpgs_3way):,}")
    print(f"Combined h5ad CpGs:  {len(cpgs_comb):,}")
    print(f"3way == Combined:    {cpgs_3way == cpgs_comb}")
    o = parquet_cpgs & cpgs_3way
    print(f"Overlap parquet ∩ 3way: {len(o):,}")
    print(f"Only in parquet:        {len(parquet_cpgs - cpgs_3way):,}")
    print(f"Only in 3way h5ad:      {len(cpgs_3way - parquet_cpgs):,}")
except FileNotFoundError:
    print(f"CpG map not found at {CPG_MAP}")

# vs old 49k h5ad
print(f"\nvs old 49k h5ad:")
with h5py.File(H5AD_49K, "r") as f:
    var     = f["var"]
    idx_key = var.attrs.get("_index", "_index")
    if idx_key not in var: idx_key = list(var.keys())[0]
    cpgs_49k = set(np.array(var[idx_key]).astype(str))

print(f"49k h5ad CpGs:         {len(cpgs_49k):,}")
print(f"21k ∩ 49k:             {len(cpgs_3way & cpgs_49k):,}")
print(f"In 21k but NOT in 49k: {len(cpgs_3way - cpgs_49k):,}")
print(f"In 49k but NOT in 21k: {len(cpgs_49k - cpgs_3way):,}  ← these are the NaN-padded columns")

# ── Cross-split leakage check ───────────────────────────────────────────────
if 'split' in obs_3way.columns:
    print(f"\n{'='*60}")
    print("CROSS-SPLIT LEAKAGE CHECK (3way h5ad)")
    print('='*60)
    splits = obs_3way['split'].unique()
    for s1 in splits:
        for s2 in splits:
            if s1 >= s2: continue
            X1 = X_3way[obs_3way['split'] == s1]
            X2 = X_3way[obs_3way['split'] == s2]
            fp1 = set(map(tuple, X1[:, :8].tolist()))
            fp2 = set(map(tuple, X2[:, :8].tolist()))
            n_ov = len(fp1 & fp2)
            print(f"  {s1} ∩ {s2}: {n_ov} identical samples {'⚠ LEAKAGE' if n_ov > 0 else '✓'}")

print(f"\n{'='*60}")
print("VERDICT")
print('='*60)
issues = []
if np.isnan(X_3way).sum() > 0: issues.append("3way h5ad has NaN values")
if np.isnan(X_comb).sum() > 0:  issues.append("combined h5ad has NaN values")
if len(set(obs_3way.index)) != len(obs_3way): issues.append("duplicate sample IDs in 3way")
if 'split' not in obs_3way.columns: issues.append("no split column in 3way h5ad")
if 'age' not in obs_3way.columns:   issues.append("no age column in 3way h5ad")

if issues:
    print("Issues found:")
    for iss in issues: print(f"  ⚠ {iss}")
else:
    print("✓ Both h5ad files look correct — ready to use for fine-tuning")
    print(f"✓ Recommended file: altumage_21k_3way.h5ad (has split column)")
PY

echo "============================================================"
echo "Analysis finished: $(date)"
echo "============================================================"
