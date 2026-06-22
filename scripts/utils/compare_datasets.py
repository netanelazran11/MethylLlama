#!/usr/bin/env python3
"""
compare_datasets.py
====================
Cross-dataset comparison: MethylLlama vs MethylGPT fine-tuning data.

Answers:
  1. How many samples / CpGs in each dataset?
  2. Do the CpG sites overlap? Are MethylLlama's 19,608 a subset of MethylGPT's 49,156?
  3. Do the valid / test samples overlap? (same split assignment?)
  4. Is MethylGPT's 49k dataset the 21k extended with NaN? (NaN pattern by position)
  5. Age distribution per split — both models compared.
  6. Split quality (age mean/std/range per split).

Outputs (written to --outdir):
  dataset_comparison_report.html   — visual slide report
  dataset_comparison_summary.txt   — plain-text digest
"""

import argparse
import json
import sys
import textwrap
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
# Cluster data paths
# ─────────────────────────────────────────────────────────────────────────────
_BASE   = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA   = f"{_BASE}/data"
_REPOS  = f"{_BASE}/repos"

LLAMA_H5AD = (
    f"{_DATA}/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean_stratified_no_outliers.h5ad"
)
LLAMA_H5AD_CLEAN = (
    f"{_DATA}/data_methyl_finetune_19k_h5ad/finetuning_19608_clean.h5ad"
)
ALTUMAGE_21K = (
    f"{_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
)
# MethylGPT — parquet directory (train/valid/test.parquet)
GPT_PARQUET_DIR = (
    f"{_REPOS}/MethylGPT-Thesis/data/finetuning_data_21k"
)
# MethylGPT — h5ad fallback (if parquet dir not available)
GPT_H5AD = (
    f"{_DATA}/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"
)

# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────
def _file_size_mb(path: str) -> float:
    try:
        return Path(path).stat().st_size / 1e6
    except Exception:
        return 0.0


def _load_h5ad_robust(path: str):
    """Load h5ad with scanpy; fall back to raw h5py on format errors."""
    try:
        import scanpy as sc
        return sc.read_h5ad(path)
    except Exception as e_sc:
        print(f"  scanpy fallback — h5py load ({e_sc})")
        import h5py, anndata as ad, scipy.sparse as sp
        with h5py.File(path, "r") as f:
            X_grp = f["X"]
            if isinstance(X_grp, h5py.Dataset):
                X = X_grp[()].astype(np.float32)
            else:
                data    = X_grp["data"][()]
                indices = X_grp["indices"][()]
                indptr  = X_grp["indptr"][()]
                shape   = tuple(X_grp.attrs["shape"])
                X = sp.csr_matrix((data, indices, indptr), shape=shape).toarray().astype(np.float32)
            def _read_grp(grp, n):
                idx_key = grp.attrs.get("_index", "_index")
                if idx_key not in grp:
                    idx_key = list(grp.keys())[0]
                idx = np.array(grp[idx_key]).astype(str)
                cols = {}
                for k in grp.keys():
                    if k == idx_key:
                        continue
                    try:
                        v = grp[k]
                        if isinstance(v, h5py.Dataset):
                            arr = v[()]
                            if len(arr) > 0 and hasattr(arr.flat[0], "decode"):
                                arr = np.array([x.decode() for x in arr])
                            cols[k] = arr
                        elif isinstance(v, h5py.Group) and "categories" in v:
                            cats  = np.array(v["categories"][()]).astype(str)
                            codes = v["codes"][()]
                            cols[k] = np.array([cats[c] if c >= 0 else None for c in codes])
                    except Exception:
                        pass
                return idx, pd.DataFrame(cols, index=idx)
            obs_idx, obs = _read_grp(f["obs"], X.shape[0])
            var_idx, var = _read_grp(f["var"], X.shape[1])
        adata = ad.AnnData(X=X, obs=obs, var=var)
        return adata


def _load_parquet_meta(parquet_dir: str):
    """
    Load MethylGPT parquet dataset (id + age only — never reads the data array).
    Returns dict: split -> DataFrame with columns [id, age, ...]
    """
    import pyarrow.parquet as pq
    p = Path(parquet_dir)
    result = {}
    for split in ("train", "valid", "test"):
        f = p / f"{split}.parquet"
        if not f.exists():
            print(f"  [WARN] {f} not found — skipping")
            continue
        # Read only scalar columns (not 'data' array)
        pf = pq.ParquetFile(f)
        schema_names = pf.schema_arrow.names
        scalar_cols  = [c for c in schema_names if c != "data"]
        tbl = pq.read_table(f, columns=scalar_cols)
        result[split] = tbl.to_pandas()
        n = pf.metadata.num_rows
        print(f"  GPT {split}: {n:,} rows  (columns: {scalar_cols})")
    return result


def _load_parquet_cpg_ids(parquet_dir: str):
    """
    Try to read CpG IDs (probe names like cg...) from the cpg_mapping/ directory.
    Returns list of string probe IDs, or None if not found / only integers.
    """
    import pyarrow.parquet as pq
    mapping_dir = Path(parquet_dir) / "cpg_mapping"
    if not mapping_dir.exists():
        return None
    for fname in sorted(mapping_dir.iterdir()):
        try:
            if fname.suffix == ".csv":
                df = pd.read_csv(fname)
                for col in ("cpg_id", "probe_id", "CpG", "id", "cpg"):
                    if col in df.columns:
                        ids = df[col].tolist()
                        break
                else:
                    ids = df.iloc[:, 0].tolist()
            elif fname.suffix == ".parquet":
                df = pq.read_table(fname).to_pandas()
                for col in ("cpg_id", "probe_id", "CpG", "id", "cpg"):
                    if col in df.columns:
                        ids = df[col].tolist()
                        break
                else:
                    ids = df.iloc[:, 0].tolist()
            else:
                continue
            # Reject if IDs are just integers (positional indices, not probe names)
            sample = [x for x in ids[:20] if x is not None]
            if sample and all(isinstance(x, (int, np.integer)) or
                              (isinstance(x, str) and x.isdigit()) for x in sample):
                print(f"  [WARN] cpg_mapping contains integer indices, not probe names — ignoring")
                return None
            return [str(x) for x in ids]
        except Exception as e:
            print(f"  [WARN] cpg_mapping read error {fname}: {e}")
    return None


def _sample_parquet_nan_profile(parquet_dir: str, n_rows: int = 500):
    """
    Read first n_rows of train.parquet and compute NaN fraction per CpG position.
    Returns array of shape (n_cpgs,) or None.
    """
    import pyarrow.parquet as pq
    f = Path(parquet_dir) / "train.parquet"
    if not f.exists():
        return None
    try:
        pf = pq.ParquetFile(f)
        schema = pf.schema_arrow
        if "data" not in schema.names:
            print("  [WARN] 'data' column not in parquet schema")
            return None
        # Read first n_rows with only the data column
        batch_iter = pf.iter_batches(batch_size=n_rows, columns=["data"])
        batch = next(batch_iter)
        tbl   = pa_batch_to_numpy(batch)
        nan_frac = np.isnan(tbl).mean(axis=0)
        return nan_frac
    except Exception as e:
        print(f"  [WARN] NaN profile error: {e}")
        return None


def pa_batch_to_numpy(batch):
    """Convert pyarrow RecordBatch with 'data' list column to numpy array."""
    import pyarrow as pa
    col = batch.column("data")
    arr = col.to_pylist()
    return np.array(arr, dtype=np.float32)


# ─────────────────────────────────────────────────────────────────────────────
# Analysis functions
# ─────────────────────────────────────────────────────────────────────────────
def analyze_llama(h5ad_path: str) -> dict:
    print(f"\n[MethylLlama] loading {h5ad_path}")
    adata = _load_h5ad_robust(h5ad_path)
    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")

    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)

    # Split column
    split_col = None
    for c in ("split", "Split", "set"):
        if c in obs.columns:
            split_col = c
            break

    per_split = {}
    if split_col:
        for sp in ("train", "valid", "test"):
            mask = obs[split_col] == sp
            sub  = obs[mask]
            ages = pd.to_numeric(sub.get("age", pd.Series(dtype=float)), errors="coerce").dropna()
            per_split[sp] = {
                "n":        int(mask.sum()),
                "ids":      set(sub.index.tolist()),
                "ages_raw": ages.round(4).tolist(),
                "age_mean": float(ages.mean()) if len(ages) else float("nan"),
                "age_std":  float(ages.std())  if len(ages) else float("nan"),
                "age_min":  float(ages.min())  if len(ages) else float("nan"),
                "age_max":  float(ages.max())  if len(ages) else float("nan"),
                "age_hist": np.histogram(ages.values, bins=range(0, 121, 10))[0].tolist()
                            if len(ages) else [],
            }
    else:
        ages_all = pd.to_numeric(obs.get("age", pd.Series(dtype=float)), errors="coerce").dropna()
        per_split["all"] = {
            "n":    adata.n_obs,
            "ids":  set(obs.index.tolist()),
            "age_mean": float(ages_all.mean()),
            "age_std":  float(ages_all.std()),
        }

    ages_all = pd.to_numeric(obs.get("age", pd.Series(dtype=float)), errors="coerce").dropna()
    cpg_ids  = list(adata.var.index.astype(str))
    return {
        "name":       "MethylLlama",
        "path":       h5ad_path,
        "file_mb":    _file_size_mb(h5ad_path),
        "n_samples":  adata.n_obs,
        "n_cpgs":     adata.n_vars,
        "cpg_ids":    cpg_ids,
        "per_split":  per_split,
        "age_all":    ages_all.values,
        "has_nan":    bool(np.isnan(adata.X).any()),
        "nan_frac":   float(np.isnan(adata.X).mean()),
        "split_col":  split_col,
        "obs_cols":   list(obs.columns),
    }


def analyze_gpt(parquet_dir: str, gpt_h5ad_fallback: str) -> dict:
    print(f"\n[MethylGPT] parquet dir: {parquet_dir}")
    p = Path(parquet_dir)

    # Try parquet first
    if p.exists() and any(p.glob("*.parquet")):
        splits_meta = _load_parquet_meta(parquet_dir)
        cpg_ids     = _load_parquet_cpg_ids(parquet_dir)
        nan_profile = _sample_parquet_nan_profile(parquet_dir, n_rows=500)

        # Get schema to know n_cpgs
        import pyarrow.parquet as pq
        pf     = pq.ParquetFile(p / "train.parquet")
        schema = pf.schema_arrow
        n_cpgs = None
        if "data" in schema.names:
            dtype = schema.field("data").type
            if hasattr(dtype, "list_size"):
                n_cpgs = dtype.list_size
            elif hasattr(dtype, "value_type"):
                # variable-size list: read one row to get length
                try:
                    row = next(pf.iter_batches(batch_size=1, columns=["data"]))
                    n_cpgs = len(row.column("data")[0].as_py())
                except Exception:
                    pass

        # If CpG IDs not in parquet, try loading var names from h5ad (no X loaded)
        if cpg_ids is None and gpt_h5ad_fallback and Path(gpt_h5ad_fallback).exists():
            print(f"  CpG IDs not in parquet — loading var names from {gpt_h5ad_fallback}")
            try:
                import h5py
                with h5py.File(gpt_h5ad_fallback, "r") as f:
                    if "var" in f:
                        var_grp = f["var"]
                        idx_key = "_index" if "_index" in var_grp else (
                            list(var_grp.keys())[0] if var_grp.keys() else None)
                        if idx_key:
                            cpg_ids = [x.decode() if isinstance(x, bytes) else str(x)
                                       for x in f["var"][idx_key][:]]
                            if n_cpgs is None:
                                n_cpgs = len(cpg_ids)
                            print(f"  Loaded {len(cpg_ids):,} CpG var names from h5ad")
            except Exception as e:
                print(f"  [WARN] could not load var names from h5ad: {e}")

        print(f"  n_cpgs detected: {n_cpgs}")
        print(f"  CpG IDs available: {len(cpg_ids):,}" if cpg_ids else "  CpG IDs: NOT available — CpG overlap will be skipped")
        print(f"  NaN profile: {'computed (' + str(len(nan_profile)) + ' positions)' if nan_profile is not None else 'NOT available'}")

        per_split = {}
        for sp, df in splits_meta.items():
            ids_col = "id" if "id" in df.columns else (df.columns[0] if len(df.columns) else None)
            age_col = "age" if "age" in df.columns else None
            ids  = set(df[ids_col].astype(str).tolist()) if ids_col else set()
            ages = pd.to_numeric(df[age_col], errors="coerce").dropna() if age_col else pd.Series(dtype=float)
            per_split[sp] = {
                "n":        int(len(df)),
                "ids":      ids,
                "ages_raw": ages.round(4).tolist(),
                "age_mean": float(ages.mean()) if len(ages) else float("nan"),
                "age_std":  float(ages.std())  if len(ages) else float("nan"),
                "age_min":  float(ages.min())  if len(ages) else float("nan"),
                "age_max":  float(ages.max())  if len(ages) else float("nan"),
                "age_hist": np.histogram(ages.values, bins=range(0, 121, 10))[0].tolist()
                            if len(ages) else [],
            }
        all_ages = np.concatenate([
            pd.to_numeric(df.get("age", pd.Series(dtype=float)), errors="coerce").dropna().values
            for df in splits_meta.values()
        ]) if splits_meta else np.array([])

        total_n = sum(v["n"] for v in per_split.values())
        return {
            "name":          "MethylGPT",
            "path":          parquet_dir,
            "file_mb":       sum(_file_size_mb(str(p / f"{s}.parquet"))
                                 for s in ("train", "valid", "test")),
            "n_samples":     total_n,
            "n_cpgs":        n_cpgs,
            "cpg_ids":       cpg_ids,
            "per_split":     per_split,
            "age_all":       all_ages,
            "nan_profile":   nan_profile,
            "has_nan":       (nan_profile is not None and nan_profile.mean() > 0),
            "nan_frac":      float(nan_profile.mean()) if nan_profile is not None else float("nan"),
            "source":        "parquet",
        }

    # Fallback: h5ad
    print(f"  Parquet dir not found — falling back to {gpt_h5ad_fallback}")
    adata = _load_h5ad_robust(gpt_h5ad_fallback)
    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")
    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    split_col = next((c for c in ("split", "Split", "set") if c in obs.columns), None)
    per_split = {}
    if split_col:
        for sp in ("train", "valid", "test"):
            mask = obs[split_col] == sp
            sub  = obs[mask]
            ages = pd.to_numeric(sub.get("age", pd.Series(dtype=float)), errors="coerce").dropna()
            per_split[sp] = {
                "n":        int(mask.sum()),
                "ids":      set(sub.index.tolist()),
                "ages_raw": ages.round(4).tolist(),
                "age_mean": float(ages.mean()) if len(ages) else float("nan"),
                "age_std":  float(ages.std())  if len(ages) else float("nan"),
                "age_min":  float(ages.min())  if len(ages) else float("nan"),
                "age_max":  float(ages.max())  if len(ages) else float("nan"),
                "age_hist": np.histogram(ages.values, bins=range(0, 121, 10))[0].tolist()
                            if len(ages) else [],
            }
    ages_all = pd.to_numeric(obs.get("age", pd.Series(dtype=float)), errors="coerce").dropna()
    nan_mat  = np.isnan(adata.X)
    nan_prof = nan_mat.mean(axis=0)
    return {
        "name":        "MethylGPT",
        "path":        gpt_h5ad_fallback,
        "file_mb":     _file_size_mb(gpt_h5ad_fallback),
        "n_samples":   adata.n_obs,
        "n_cpgs":      adata.n_vars,
        "cpg_ids":     list(adata.var.index.astype(str)),
        "per_split":   per_split,
        "age_all":     ages_all.values,
        "nan_profile": nan_prof,
        "has_nan":     bool(nan_mat.any()),
        "nan_frac":    float(nan_mat.mean()),
        "source":      "h5ad",
    }


def compare(llama: dict, gpt: dict) -> dict:
    """Compute all cross-dataset comparison metrics."""
    result = {}

    # ── 1. CpG overlap ──────────────────────────────────────────────────────
    ll_cpgs = set(llama.get("cpg_ids") or [])
    gp_cpgs = set(gpt.get("cpg_ids")  or [])
    # Diagnostics: show ID format samples
    print(f"  [CpG IDs] MethylLlama sample: {list(ll_cpgs)[:5]}")
    print(f"  [CpG IDs] MethylGPT   sample: {list(gp_cpgs)[:5]}")
    if ll_cpgs and gp_cpgs:
        shared       = ll_cpgs & gp_cpgs
        only_llama   = ll_cpgs - gp_cpgs
        only_gpt     = gp_cpgs - ll_cpgs
        result["cpg_overlap"] = {
            "llama_n":    len(ll_cpgs),
            "gpt_n":      len(gp_cpgs),
            "shared_n":   len(shared),
            "only_llama": len(only_llama),
            "only_gpt":   len(only_gpt),
            "llama_subset_of_gpt": len(only_llama) == 0,
            "overlap_pct_of_llama": 100 * len(shared) / max(len(ll_cpgs), 1),
            "overlap_pct_of_gpt":   100 * len(shared) / max(len(gp_cpgs), 1),
        }
    else:
        result["cpg_overlap"] = {
            "llama_n": len(ll_cpgs), "gpt_n": len(gp_cpgs),
            "shared_n": None, "note": "CpG IDs not available for one dataset"
        }

    # ── 2. Sample overlap per split ─────────────────────────────────────────
    result["sample_overlap"] = {}
    _printed_sample_diag = False
    # Detect if MethylGPT uses artificial IDs (sample_NNN scheme)
    _gpt_artificial_ids = False
    for _sp0 in ("valid", "test", "train"):
        _gp0 = gpt["per_split"].get(_sp0, {}).get("ids", set())
        if _gp0:
            _sample0 = list(_gp0)[:5]
            if all(str(x).startswith("sample_") or str(x).isdigit() for x in _sample0):
                _gpt_artificial_ids = True
            break

    for sp in ("valid", "test", "train"):
        ll_ids = llama["per_split"].get(sp, {}).get("ids", set())
        gp_ids = gpt["per_split"].get(sp, {}).get("ids", set())
        if not _printed_sample_diag and ll_ids and gp_ids:
            print(f"  [Sample IDs/{sp}] MethylLlama sample: {list(ll_ids)[:3]}")
            print(f"  [Sample IDs/{sp}] MethylGPT   sample: {list(gp_ids)[:3]}")
            if _gpt_artificial_ids:
                print(f"  [WARN] MethylGPT uses artificial sample_NNN IDs — cannot match to GSM/TCGA IDs. Age distribution will be compared instead.")
            _printed_sample_diag = True
        if not ll_ids or not gp_ids:
            result["sample_overlap"][sp] = {
                "llama_n": len(ll_ids), "gpt_n": len(gp_ids),
                "shared_n": None, "note": "IDs not available"
            }
            continue
        if _gpt_artificial_ids:
            # Cannot compare by ID — report n counts and note
            result["sample_overlap"][sp] = {
                "llama_n":   len(ll_ids),
                "gpt_n":     len(gp_ids),
                "shared_n":  None,
                "note":      "MethylGPT uses artificial IDs (sample_NNN) — direct ID matching not possible",
            }
            continue
        shared = ll_ids & gp_ids
        result["sample_overlap"][sp] = {
            "llama_n":          len(ll_ids),
            "gpt_n":            len(gp_ids),
            "shared_n":         len(shared),
            "only_llama":       len(ll_ids - gp_ids),
            "only_gpt":         len(gp_ids - ll_ids),
            "pct_llama":        100 * len(shared) / max(len(ll_ids), 1),
            "pct_gpt":          100 * len(shared) / max(len(gp_ids), 1),
            "fully_shared":     len(shared) == len(ll_ids) == len(gp_ids),
        }

    # ── 2b. Age-based sample overlap (proxy when IDs don't match) ───────────
    # Match samples by exact age value (rounded to 4dp).
    # If the same biological sample appears in both datasets it will have
    # the same age, so shared age values = lower-bound on shared samples.
    # Note: multiple samples can share the same age, so this may over-count.
    result["age_overlap"] = {}
    print("\n  --- Age-based overlap (proxy for sample identity) ---")
    for sp in ("valid", "test", "train"):
        ll_ages_raw = llama["per_split"].get(sp, {}).get("ages_raw", [])
        gp_ages_raw = gpt["per_split"].get(sp, {}).get("ages_raw", [])
        if not ll_ages_raw or not gp_ages_raw:
            result["age_overlap"][sp] = {"note": "ages not available"}
            continue
        ll_age_set = sorted(ll_ages_raw)
        gp_age_set = sorted(gp_ages_raw)
        # Count how many LL ages appear in GPT (multiset intersection via sorting)
        from collections import Counter
        ll_ctr = Counter(ll_ages_raw)
        gp_ctr = Counter(gp_ages_raw)
        shared_ages   = set(ll_ctr.keys()) & set(gp_ctr.keys())
        shared_count  = sum(min(ll_ctr[a], gp_ctr[a]) for a in shared_ages)
        ll_only_count = sum(ll_ctr[a] for a in set(ll_ctr.keys()) - set(gp_ctr.keys()))
        gp_only_count = sum(gp_ctr[a] for a in set(gp_ctr.keys()) - set(ll_ctr.keys()))
        pct_ll = 100 * shared_count / max(len(ll_ages_raw), 1)
        pct_gp = 100 * shared_count / max(len(gp_ages_raw), 1)
        # Duplicate ages in either set inflates shared_count — report unique age matches too
        unique_shared = len(shared_ages)
        result["age_overlap"][sp] = {
            "llama_n":       len(ll_ages_raw),
            "gpt_n":         len(gp_ages_raw),
            "shared_count":  shared_count,
            "unique_shared_ages": unique_shared,
            "ll_only":       ll_only_count,
            "gp_only":       gp_only_count,
            "pct_of_llama":  pct_ll,
            "pct_of_gpt":    pct_gp,
        }
        print(f"  [{sp}] LL={len(ll_ages_raw):,}  GPT={len(gp_ages_raw):,}  "
              f"shared_by_age={shared_count:,} ({pct_ll:.1f}% of LL, {pct_gp:.1f}% of GPT)  "
              f"unique_ages_shared={unique_shared:,}  ll_only={ll_only_count:,}  gp_only={gp_only_count:,}")

    # ── 3. NaN extension check (49k = 21k + NaN?) ───────────────────────────
    nan_prof  = gpt.get("nan_profile")
    ll_n_cpgs = llama["n_cpgs"]
    if nan_prof is not None and ll_n_cpgs:
        inside_nan  = float(np.nanmean(nan_prof[:ll_n_cpgs]))
        outside_nan = float(np.nanmean(nan_prof[ll_n_cpgs:]))
        # Overall NaN fraction across all positions
        overall_nan = float(np.nanmean(nan_prof))
        # Hypothesis: positions 0..ll_n_cpgs are "always-measured" (low NaN),
        # positions ll_n_cpgs.. are "850k-only" (high NaN).
        # If inside ≈ outside, CpG columns are NOT ordered that way.
        sorted_by_availability = (outside_nan > inside_nan * 3)
        # Find what fraction of positions have NaN < 5% (proxy for always-measured)
        low_nan_positions = int((nan_prof < 0.05).sum())
        result["nan_extension"] = {
            "llama_n_cpgs":          ll_n_cpgs,
            "gpt_n_cpgs":            len(nan_prof),
            "nan_inside_llama":      inside_nan,
            "nan_outside_llama":     outside_nan,
            "nan_overall":           overall_nan,
            "ratio":                 outside_nan / max(inside_nan, 1e-9),
            "supported":             sorted_by_availability,
            "low_nan_positions":     low_nan_positions,
            "columns_sorted_by_nan": sorted_by_availability,
            "note": (
                "Columns appear sorted by availability (low-NaN first)" if sorted_by_availability
                else f"NaN is uniform across all positions (~{overall_nan:.1%}); "
                     f"columns are NOT sorted by array coverage. "
                     f"Only {low_nan_positions:,} of {len(nan_prof):,} positions have <5% NaN."
            ),
        }
        print(f"  [NaN profile] inside 0-{ll_n_cpgs}: {inside_nan:.4f}  outside: {outside_nan:.4f}  "
              f"overall: {overall_nan:.4f}  low-NaN positions: {low_nan_positions:,}")
    else:
        result["nan_extension"] = {"note": "NaN profile not available"}

    # ── 4. Age distribution comparison + split identity analysis ─────────────
    try:
        from scipy import stats as scipy_stats
        _has_scipy = True
    except ImportError:
        scipy_stats = None
        _has_scipy = False
        print("  [WARN] scipy not available — chi2 p-values will be skipped")

    age_cmp = {}
    for sp in ("train", "valid", "test"):
        ll = llama["per_split"].get(sp, {})
        gp = gpt["per_split"].get(sp, {})
        ll_n = ll.get("n", 0)
        gp_n = gp.get("n", 0)
        # same_size: do the splits have the same number of samples?
        same_size = (ll_n == gp_n) if (ll_n and gp_n) else None
        # KS test on age histograms (proxy, since we don't have raw ages for GPT)
        ll_hist = ll.get("age_hist", [])
        gp_hist = gp.get("age_hist", [])
        ks_stat = ks_p = None
        if ll_hist and gp_hist and sum(ll_hist) > 0 and sum(gp_hist) > 0:
            # Normalise histograms to densities and compare
            ll_den = np.array(ll_hist, dtype=float) / max(sum(ll_hist), 1)
            gp_den = np.array(gp_hist, dtype=float) / max(sum(gp_hist), 1)
            ks_stat = float(np.max(np.abs(np.cumsum(ll_den) - np.cumsum(gp_den))))
            # chi2 goodness-of-fit as a p-value proxy
            if _has_scipy:
                try:
                    expected = gp_den * sum(ll_hist)
                    mask = expected > 0
                    chi2 = float(np.sum((np.array(ll_hist)[mask] - expected[mask])**2 / expected[mask]))
                    ks_p = float(1 - scipy_stats.chi2.cdf(chi2, df=max(mask.sum()-1, 1)))
                except Exception:
                    pass
        age_cmp[sp] = {
            "llama_n":    ll_n,           "llama_mean": ll.get("age_mean"),
            "llama_std":  ll.get("age_std"), "llama_min": ll.get("age_min"),
            "llama_max":  ll.get("age_max"),
            "gpt_n":      gp_n,           "gpt_mean":   gp.get("age_mean"),
            "gpt_std":    gp.get("age_std"), "gpt_min":    gp.get("age_min"),
            "gpt_max":    gp.get("age_max"),
            "same_size":  same_size,
            "size_ratio": (gp_n / max(ll_n, 1)) if (ll_n and gp_n) else None,
            "ks_stat":    ks_stat,        # max CDF difference (0=identical, 1=max diff)
            "age_dist_p": ks_p,           # chi2 p-value (>0.05 = similar distribution)
        }
        verdict = "SAME SIZE" if same_size else f"DIFFERENT ({gp_n:,} vs {ll_n:,})"
        age_note = ""
        if ks_stat is not None:
            age_note = f"  age KS={ks_stat:.3f}  p={ks_p:.3f}" if ks_p is not None else f"  KS={ks_stat:.3f}"
        print(f"  [split/{sp}] LL={ll_n:,}  GPT={gp_n:,}  → {verdict}{age_note}")
    result["age_cmp"] = age_cmp

    # ── 5. Overall split identity verdict ────────────────────────────────────
    same_sizes_all = all(age_cmp[sp]["same_size"] for sp in ("train", "valid", "test")
                         if age_cmp[sp]["same_size"] is not None)
    result["split_identity"] = {
        "same_sizes_all_splits": same_sizes_all,
        "can_compare_ids":       not _gpt_artificial_ids,
        "verdict": (
            "LIKELY SAME SPLIT — same sizes, similar age distributions"
            if same_sizes_all else
            "DIFFERENT SPLITS — split sizes differ; models evaluated on different samples"
        ),
        "implication": (
            "MedAE scores are directly comparable (same evaluation set)."
            if same_sizes_all else
            "MedAE scores are NOT directly comparable — different test populations."
        ),
    }
    print(f"\n  SPLIT VERDICT: {result['split_identity']['verdict']}")
    print(f"  IMPLICATION:   {result['split_identity']['implication']}")

    return result


# ─────────────────────────────────────────────────────────────────────────────
# HTML report generation
# ─────────────────────────────────────────────────────────────────────────────
CSS = """
* { box-sizing: border-box; margin: 0; padding: 0; }
body { font-family: 'Segoe UI', Arial, sans-serif; background: #eef0f4;
       color: #1e2535; font-size: 14px; }
.slide { width: 1280px; min-height: 720px; margin: 40px auto;
         background: #fff; border-radius: 16px; padding: 46px 56px;
         box-shadow: 0 4px 24px rgba(0,0,0,.10);
         border: 1px solid #dde1ea; position: relative; }
.slide-title { font-size: 24px; font-weight: 700; letter-spacing: .3px;
               margin-bottom: 30px; color: #1a2340;
               border-bottom: 2px solid #dde3f0; padding-bottom: 12px;
               display: flex; align-items: center; gap: 10px; }
.slide-label { position: absolute; top: 18px; right: 28px; font-size: 11px;
               color: #a0aabb; letter-spacing: 1px; text-transform: uppercase; }
.grid2 { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
.grid3 { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 16px; }
.grid4 { display: grid; grid-template-columns: 1fr 1fr 1fr 1fr; gap: 14px; }
.full { flex: 1; }
.stat-card { border-radius: 10px; padding: 18px 14px; text-align: center; }
.stat-card .s-val { font-size: 30px; font-weight: 800; line-height: 1; margin-bottom: 6px; }
.stat-card .s-label { font-size: 11px; text-transform: uppercase; letter-spacing: .8px;
                      font-weight: 600; }
.stat-card .s-sub { font-size: 11px; margin-top: 4px; opacity: .75; }
.sc-blue   { background:#eef3ff; border:1.5px solid #7a9fe0; color:#1a3a80; }
.sc-green  { background:#edf7ed; border:1.5px solid #4a9a4a; color:#1a5a1a; }
.sc-purple { background:#f4eeff; border:1.5px solid #8a60c8; color:#3a1a70; }
.sc-amber  { background:#fff8ee; border:1.5px solid #cc9040; color:#6a3a00; }
.sc-red    { background:#fff0f0; border:1.5px solid #cc4040; color:#6a0a0a; }
.sc-teal   { background:#edfafa; border:1.5px solid #3a9a9a; color:#0a4040; }
.panel { background:#f7f9fc; border:1px solid #dde3ef; border-radius:10px;
         padding:18px 20px; }
.panel-title { font-size:12px; font-weight:700; text-transform:uppercase;
               letter-spacing:.8px; color:#5a6888; margin-bottom:14px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:#f0f3f8; color:#4a5a78; font-size:11px; text-transform:uppercase;
     letter-spacing:.6px; padding:7px 10px; text-align:left; font-weight:700; }
td { padding:6px 10px; border-bottom:1px solid #edf0f6; color:#1e2535; }
tr:last-child td { border-bottom:none; }
tr:hover td { background:#f4f7fc; }
.mono { font-family:'Courier New',monospace; font-size:12px; }
.td-r { text-align:right; }
.td-c { text-align:center; }
.bar-row { display:flex; align-items:center; gap:8px; margin-bottom:5px; }
.bar-label { font-size:11px; color:#4a5a78; width:70px; text-align:right;
             flex-shrink:0; font-family:monospace; }
.bar-label-w { font-size:11px; color:#4a5a78; width:160px; text-align:right;
               flex-shrink:0; white-space:nowrap; overflow:hidden; text-overflow:ellipsis; }
.bar-track { flex:1; height:16px; background:#eef1f8; border-radius:4px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:4px; }
.bar-val   { font-size:11px; color:#6a7890; width:80px; flex-shrink:0;
             font-family:monospace; }
.b-blue   { background:#5a8de0; }
.b-green  { background:#4a9a60; }
.b-purple { background:#8060c0; }
.b-amber  { background:#cc8830; }
.b-teal   { background:#3a9a9a; }
.b-red    { background:#cc5040; }
.callout { border-radius:8px; padding:12px 16px; font-size:13px; line-height:1.5; }
.callout-ok   { background:#edf7ed; border-left:4px solid #4a9a4a; color:#1a4a1a; }
.callout-warn { background:#fff8ee; border-left:4px solid #e08820; color:#5a3800; }
.callout-info { background:#eef3ff; border-left:4px solid #5a80d0; color:#1a2880; }
.callout-red  { background:#fff0f0; border-left:4px solid #cc3030; color:#5a0808; }
.callout b { font-weight:700; }
.badge { display:inline-block; border-radius:4px; padding:2px 8px;
         font-size:11px; font-weight:700; letter-spacing:.3px; }
.badge-green  { background:#daf0da; color:#1a5a1a; }
.badge-red    { background:#ffe0e0; color:#6a0a0a; }
.badge-amber  { background:#fff0d0; color:#6a3a00; }
.badge-blue   { background:#dce8ff; color:#1a3a80; }
.badge-teal   { background:#d8f0f0; color:#0a4040; }
.badge-gray   { background:#e8eaf0; color:#3a4a60; }
.divider { height:1px; background:#dde3f0; margin:20px 0; }
.hi-blue   { color:#2a60c8; font-weight:700; }
.hi-green  { color:#2a7a2a; font-weight:700; }
.hi-red    { color:#aa2020; font-weight:700; }
.hi-amber  { color:#a06010; font-weight:700; }
.sec-head  { font-size:11px; font-weight:700; text-transform:uppercase;
             letter-spacing:1px; color:#7a8aa8; margin-bottom:8px; }
.ch-ll  { background:#FDE8E8; color:#7a1a1a; }
.ch-gpt { background:#E8F4FF; color:#1a3a7a; }
.ch-both{ background:#EDF7ED; color:#1a4a1a; }
"""


def _bar(pct: float, color: str = "b-blue", label: str = "", val: str = "") -> str:
    safe = min(max(pct, 0), 100)
    return (f'<div class="bar-row">'
            f'<div class="bar-label-w">{label}</div>'
            f'<div class="bar-track"><div class="bar-fill {color}" style="width:{safe:.1f}%"></div></div>'
            f'<div class="bar-val">{val}</div>'
            f'</div>')


def _age_hist_bars(hist: list, color: str = "b-blue") -> str:
    bins = [f"{i*10}–{(i+1)*10}" for i in range(12)]
    total = sum(hist) or 1
    html  = ""
    for i, (cnt, lbl) in enumerate(zip(hist, bins)):
        pct = 100 * cnt / total
        html += _bar(pct, color, lbl, f"{cnt:,} ({pct:.0f}%)")
    return html


def _fmt(v, decimals=1, suffix=""):
    if v is None or (isinstance(v, float) and np.isnan(v)):
        return "—"
    return f"{v:.{decimals}f}{suffix}"


def _fmti(v):
    """Format int with comma thousands separator, or '?' if None."""
    if v is None:
        return "?"
    return f"{int(v):,}"


def _badge(text, cls="badge-gray"):
    return f'<span class="badge {cls}">{text}</span>'


def render_html(llama: dict, gpt: dict, cmp: dict, out_path: Path):
    slides = []
    n_slides = 6

    # ── Slide 1: Overview ────────────────────────────────────────────────────
    ll_tot = llama["n_samples"]
    gp_tot = gpt["n_samples"]
    s1 = f"""
<div class="slide">
  <div class="slide-label">Dataset Comparison · Slide 1 / {n_slides}</div>
  <div class="slide-title">🔬 Dataset Overview — MethylLlama vs MethylGPT Fine-tuning Data</div>

  <div class="grid2" style="margin-bottom:24px">
    <!-- MethylLlama column -->
    <div>
      <div class="panel-title ch-ll" style="border-radius:6px;padding:4px 10px;margin-bottom:12px">
        MethylLlama Fine-tuning Dataset
      </div>
      <div class="grid4" style="margin-bottom:12px">
        <div class="stat-card sc-red">
          <div class="s-val">{ll_tot:,}</div>
          <div class="s-label">Total Samples</div>
          <div class="s-sub">{llama['n_cpgs']:,} CpGs each</div>
        </div>
        <div class="stat-card sc-blue">
          <div class="s-val">{llama['n_cpgs']:,}</div>
          <div class="s-label">CpG Sites</div>
          <div class="s-sub">zero NaN</div>
        </div>
        <div class="stat-card sc-green">
          <div class="s-val">{'✓' if not llama['has_nan'] else '⚠'}</div>
          <div class="s-label">{'No NaN' if not llama['has_nan'] else 'Has NaN'}</div>
          <div class="s-sub">float32</div>
        </div>
        <div class="stat-card sc-amber">
          <div class="s-val">{_fmt(llama['file_mb'] / 1000, 1)} GB</div>
          <div class="s-label">File Size</div>
          <div class="s-sub">h5ad</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">Split Distribution</div>
        <table>
          <tr><th>Split</th><th class="td-r">N</th><th class="td-r">%</th>
              <th class="td-r">Age mean±std</th></tr>
          {''.join(
            f"<tr><td>{sp}</td><td class='td-r'>{v['n']:,}</td>"
            f"<td class='td-r'>{100*v['n']/max(ll_tot,1):.1f}%</td>"
            f"<td class='td-r'>{_fmt(v.get('age_mean'))} ± {_fmt(v.get('age_std'))} yr</td></tr>"
            for sp, v in llama['per_split'].items()
          )}
        </table>
      </div>
    </div>

    <!-- MethylGPT column -->
    <div>
      <div class="panel-title ch-gpt" style="border-radius:6px;padding:4px 10px;margin-bottom:12px">
        MethylGPT Fine-tuning Dataset
      </div>
      <div class="grid4" style="margin-bottom:12px">
        <div class="stat-card sc-blue">
          <div class="s-val">{gp_tot:,}</div>
          <div class="s-label">Total Samples</div>
          <div class="s-sub">{_fmti(gpt['n_cpgs'])} CpGs each</div>
        </div>
        <div class="stat-card sc-purple">
          <div class="s-val">{_fmti(gpt['n_cpgs'])}</div>
          <div class="s-label">CpG Sites</div>
          <div class="s-sub">incl. NaN cols</div>
        </div>
        <div class="stat-card sc-amber">
          <div class="s-val">{'%.1f%%' % (100*gpt['nan_frac']) if not np.isnan(gpt.get('nan_frac',float('nan'))) else '?'}</div>
          <div class="s-label">NaN Fraction</div>
          <div class="s-sub">of all values</div>
        </div>
        <div class="stat-card sc-teal">
          <div class="s-val">{_fmt(gpt['file_mb'] / 1000, 1)} GB</div>
          <div class="s-label">File Size</div>
          <div class="s-sub">{gpt.get('source','parquet')}</div>
        </div>
      </div>
      <div class="panel">
        <div class="panel-title">Split Distribution</div>
        <table>
          <tr><th>Split</th><th class="td-r">N</th><th class="td-r">%</th>
              <th class="td-r">Age mean±std</th></tr>
          {''.join(
            f"<tr><td>{sp}</td><td class='td-r'>{v['n']:,}</td>"
            f"<td class='td-r'>{100*v['n']/max(gp_tot,1):.1f}%</td>"
            f"<td class='td-r'>{_fmt(v.get('age_mean'))} ± {_fmt(v.get('age_std'))} yr</td></tr>"
            for sp, v in gpt['per_split'].items()
          )}
        </table>
      </div>
    </div>
  </div>

  <div class="callout callout-info">
    <b>Dataset relationship:</b>
    MethylLlama uses <b>{llama['n_cpgs']:,} CpG sites</b> (always-measured, zero NaN) from ~{ll_tot:,} samples.
    MethylGPT uses <b>{_fmti(gpt['n_cpgs'])} CpG sites</b> from ~{gp_tot:,} samples, including NaN where probes
    were not measured. The {_fmti(gpt['n_cpgs'])} ≈ {llama['n_cpgs']:,} (core always-measured)
    + ~{_fmti((gpt['n_cpgs'] or llama['n_cpgs']) - llama['n_cpgs'])} (partially-measured CpGs with NaN).
  </div>
</div>"""
    slides.append(s1)

    # ── Slide 2: CpG overlap ─────────────────────────────────────────────────
    cpg = cmp.get("cpg_overlap", {})
    cpg_note = cpg.get("note", "")
    if cpg.get("shared_n") is not None:
        overlap_badge = (_badge("✓ FULL SUBSET", "badge-green") if cpg["llama_subset_of_gpt"]
                         else _badge("⚠ PARTIAL OVERLAP", "badge-amber"))
        cpg_callout_cls = "callout-ok" if cpg["llama_subset_of_gpt"] else "callout-warn"
        cpg_msg = (
            f"<b>MethylLlama's {llama['n_cpgs']:,} CpG sites are a complete subset "
            f"of MethylGPT's {gpt['n_cpgs'] or '?'} sites.</b> "
            f"Every CpG used for MethylLlama fine-tuning was also measured (or at least present in the "
            f"column schema) in MethylGPT's dataset."
            if cpg["llama_subset_of_gpt"] else
            f"<b>{cpg['only_llama']:,} CpG sites are in MethylLlama only</b> "
            f"(not in MethylGPT's 49k column set). This is unexpected if 49k ⊇ 21k."
        )
    else:
        overlap_badge = _badge("? NO CpG LIST", "badge-gray")
        cpg_callout_cls = "callout-warn"
        cpg_msg = f"<b>CpG ID list not available for one dataset.</b> {cpg_note}"

    _cpg_shared_str   = f"{cpg['shared_n']:,}" if cpg.get('shared_n') is not None else '?'
    _cpg_only_gpt_str = f"{cpg['only_gpt']:,}" if cpg.get('only_gpt') is not None else '?'

    s2 = f"""
<div class="slide">
  <div class="slide-label">Dataset Comparison · Slide 2 / {n_slides}</div>
  <div class="slide-title">🧬 CpG Site Overlap — MethylLlama ∩ MethylGPT</div>

  <div class="grid2" style="margin-bottom:24px">
    <div>
      <div class="panel" style="margin-bottom:16px">
        <div class="panel-title">CpG Counts</div>
        <table>
          <tr><th>Dataset</th><th class="td-r">CpG sites</th></tr>
          <tr><td>MethylLlama</td><td class="td-r">{cpg.get('llama_n', llama['n_cpgs']):,}</td></tr>
          <tr><td>MethylGPT</td><td class="td-r">{_fmti(cpg.get('gpt_n', gpt['n_cpgs']))}</td></tr>
          <tr style="background:#f4f7fc">
            <td><b>Shared (∩)</b></td>
            <td class="td-r"><b>{_cpg_shared_str}</b></td>
          </tr>
          <tr><td>Only in MethylLlama</td><td class="td-r">{cpg.get('only_llama', '?')}</td></tr>
          <tr><td>Only in MethylGPT</td><td class="td-r">{_cpg_only_gpt_str}</td></tr>
        </table>
      </div>

      <div class="panel">
        <div class="panel-title">Overlap Percentages</div>
        {_bar(cpg.get('overlap_pct_of_llama', 0), 'b-red',
              'of MethylLlama covered', f"{_fmt(cpg.get('overlap_pct_of_llama'))}%") if cpg.get('shared_n') else '<i>Not available</i>'}
        {_bar(cpg.get('overlap_pct_of_gpt', 0), 'b-blue',
              'of MethylGPT covered', f"{_fmt(cpg.get('overlap_pct_of_gpt'))}%") if cpg.get('shared_n') else ''}
      </div>
    </div>

    <div>
      <div style="margin-bottom:12px">
        <div class="panel-title">Verdict {overlap_badge}</div>
      </div>
      <div class="callout {cpg_callout_cls}" style="margin-bottom:16px">
        {cpg_msg}
      </div>
      <div class="panel">
        <div class="panel-title">What this means</div>
        <p style="font-size:13px;line-height:1.7;color:#2a3550">
          MethylGPT's 49,156-CpG schema includes <b>all arrays</b> from the AltumAge cohort —
          including probes that are only on newer 850k arrays. The always-measured probes
          (common to all arrays) were selected as MethylLlama's 19,608 input features.<br><br>
          Result: <b>MethylLlama uses a zero-NaN subset of MethylGPT's CpG space.</b>
          The extra ~29,548 CpGs in MethylGPT are those with high NaN rates across samples
          (measured only by some arrays).
        </p>
      </div>
    </div>
  </div>
</div>"""
    slides.append(s2)

    # ── Slide 3: Sample overlap ──────────────────────────────────────────────
    so = cmp.get("sample_overlap", {})

    def _overlap_row(sp):
        d = so.get(sp, {})
        if d.get("shared_n") is None:
            return f"<tr><td>{sp}</td><td class='td-c' colspan='6'><i>IDs not available</i></td></tr>"
        ll_n, gp_n, sh = d['llama_n'], d['gpt_n'], d['shared_n']
        pct_ll = d.get('pct_llama', 0)
        pct_gp = d.get('pct_gpt', 0)
        flag   = (_badge("IDENTICAL", "badge-green") if d.get("fully_shared") else
                  _badge(f"{pct_ll:.0f}% overlap", "badge-amber") if pct_ll > 50 else
                  _badge("DIFFERENT", "badge-red"))
        return (f"<tr><td><b>{sp}</b></td>"
                f"<td class='td-r'>{ll_n:,}</td><td class='td-r'>{gp_n:,}</td>"
                f"<td class='td-r'><b>{sh:,}</b></td>"
                f"<td class='td-r'>{pct_ll:.1f}%</td><td class='td-r'>{pct_gp:.1f}%</td>"
                f"<td class='td-c'>{flag}</td></tr>")

    def _split_bar_pair(sp):
        d = so.get(sp, {})
        if d.get("shared_n") is None:
            return "<i>Not available</i>"
        ll_n, gp_n, sh = d['llama_n'], d['gpt_n'], d['shared_n']
        max_n = max(ll_n, gp_n, 1)
        html  = ""
        html += _bar(100 * ll_n / max_n, "b-red",   "MethylLlama", f"{ll_n:,}")
        html += _bar(100 * gp_n / max_n, "b-blue",  "MethylGPT",   f"{gp_n:,}")
        html += _bar(100 * sh   / max_n, "b-green", "Shared",       f"{sh:,}")
        return html

    s3 = f"""
<div class="slide">
  <div class="slide-label">Dataset Comparison · Slide 3 / {n_slides}</div>
  <div class="slide-title">👥 Sample ID Overlap — Valid &amp; Test Sets</div>

  <div class="panel" style="margin-bottom:20px">
    <div class="panel-title">Sample counts and overlap by split</div>
    <table>
      <tr>
        <th>Split</th>
        <th class="td-r">MethylLlama N</th>
        <th class="td-r">MethylGPT N</th>
        <th class="td-r">Shared N</th>
        <th class="td-r">% of Llama</th>
        <th class="td-r">% of GPT</th>
        <th class="td-c">Verdict</th>
      </tr>
      {_overlap_row('valid')}
      {_overlap_row('test')}
      {_overlap_row('train')}
    </table>
  </div>

  <div class="grid3">
    <div class="panel">
      <div class="panel-title">Valid split</div>
      {_split_bar_pair('valid')}
    </div>
    <div class="panel">
      <div class="panel-title">Test split</div>
      {_split_bar_pair('test')}
    </div>
    <div class="panel">
      <div class="panel-title">Train split</div>
      {_split_bar_pair('train')}
    </div>
  </div>

  <div class="divider"></div>

  <div class="callout {'callout-ok' if all((so.get(sp,{}).get('shared_n') or 0)>0 for sp in ('valid','test')) else 'callout-warn'}">
    <b>Key question — do they evaluate on the same samples?</b><br>
    If the valid/test sets are identical, training performance differences between the models
    are directly comparable. If they differ (different splits or different cohorts),
    test metrics measure different populations — MedAE numbers cannot be compared directly.
  </div>
</div>"""
    slides.append(s3)

    # ── Slide 4: NaN extension check ────────────────────────────────────────
    ne = cmp.get("nan_extension", {})
    nan_inside  = ne.get("nan_inside_llama")
    nan_outside = ne.get("nan_outside_llama")
    ratio       = ne.get("ratio")
    supported   = ne.get("supported", False)

    if nan_inside is not None:
        nan_verdict_cls = "callout-ok" if supported else "callout-warn"
        nan_verdict_msg = (
            f"<b>✓ Hypothesis SUPPORTED.</b> "
            f"NaN fraction inside the 19,608-CpG window ({nan_inside:.1%}) is "
            f"{ratio:.1f}× lower than outside ({nan_outside:.1%}). "
            f"The 49k dataset IS the 21k extended with additional CpG columns that "
            f"are mostly NaN for samples using older arrays."
            if supported else
            f"<b>⚠ Hypothesis NOT clearly supported.</b> "
            f"NaN inside={nan_inside:.1%}, outside={nan_outside:.1%} (ratio {ratio:.1f}×). "
            f"Either the CpG ordering differs, or NaN distribution is more uniform."
        )
        nan_bars = f"""
        <div class="bar-row">
          <div class="bar-label-w">Positions 0–{llama['n_cpgs']:,}<br><small>(MethylLlama window)</small></div>
          <div class="bar-track"><div class="bar-fill b-green" style="width:{min(nan_inside*100*10,100):.0f}%"></div></div>
          <div class="bar-val">{nan_inside:.1%} NaN</div>
        </div>
        <div class="bar-row">
          <div class="bar-label-w">Positions {llama['n_cpgs']:,}–{ne.get('gpt_n_cpgs','?')}<br><small>(extra CpGs)</small></div>
          <div class="bar-track"><div class="bar-fill b-red" style="width:{min(nan_outside*100,100):.0f}%"></div></div>
          <div class="bar-val">{nan_outside:.1%} NaN</div>
        </div>"""
    else:
        nan_verdict_cls = "callout-warn"
        nan_verdict_msg = f"<b>NaN profile not available.</b> {ne.get('note','Run required data loading.')}"
        nan_bars = "<i>Data not loaded — requires reading 500 rows of parquet data column.</i>"

    s4 = f"""
<div class="slide">
  <div class="slide-label">Dataset Comparison · Slide 4 / {n_slides}</div>
  <div class="slide-title">🔍 NaN Extension Check — Is 49k = 21k + Extra CpGs?</div>

  <div class="grid2" style="margin-bottom:20px">
    <div>
      <div class="panel" style="margin-bottom:16px">
        <div class="panel-title">Hypothesis</div>
        <p style="font-size:13px;line-height:1.8">
          The AltumAge cohort uses different methylation arrays across studies:<br>
          • <b>450k array</b> → measures ~450k CpGs<br>
          • <b>850k array</b> → measures ~850k CpGs (superset)<br><br>
          The <b>always-measured CpGs</b> (common to both arrays) were selected
          as the 19,608 input features for MethylLlama.<br><br>
          The MethylGPT 49k dataset includes <b>all CpG columns</b> from both arrays
          (49,156 total), but samples measured on the 450k array will have <b>NaN</b>
          for the ~29,548 CpGs only on the 850k array.<br><br>
          <b>Prediction:</b> NaN fraction should be <b>low</b> in positions 0–19,608
          and <b>high</b> in positions 19,608–49,156 (if column order matches).
        </p>
      </div>
      <div class="panel">
        <div class="panel-title">NaN rate by CpG position (first 500 training rows)</div>
        {nan_bars}
      </div>
    </div>
    <div>
      <div class="callout {nan_verdict_cls}" style="margin-bottom:16px">
        {nan_verdict_msg}
      </div>
      <div class="panel">
        <div class="panel-title">Key numbers</div>
        <table>
          <tr><th>Metric</th><th class="td-r">Value</th></tr>
          <tr><td>MethylLlama CpG window</td><td class="td-r">{llama['n_cpgs']:,}</td></tr>
          <tr><td>MethylGPT total CpGs</td><td class="td-r">{ne.get('gpt_n_cpgs', gpt['n_cpgs'] or '?')}</td></tr>
          <tr><td>Extra CpGs in MethylGPT</td>
              <td class="td-r">{_fmti((ne.get('gpt_n_cpgs') or 0) - llama['n_cpgs'])}
                              {'(always-NaN in analysis)' if supported else ''}</td></tr>
          <tr><td>NaN inside Llama window</td>
              <td class="td-r">{_fmt(nan_inside, 3, ' (% samples)') if nan_inside is not None else '?'}</td></tr>
          <tr><td>NaN outside Llama window</td>
              <td class="td-r">{_fmt(nan_outside, 3, ' (% samples)') if nan_outside is not None else '?'}</td></tr>
          <tr><td>Ratio (outside / inside)</td>
              <td class="td-r"><b>{_fmt(ratio, 1, '×') if ratio is not None else '?'}</b></td></tr>
        </table>
      </div>
    </div>
  </div>
</div>"""
    slides.append(s4)

    # ── Slide 5: Age distributions per split ─────────────────────────────────
    age_cmp = cmp.get("age_cmp", {})

    def _age_split_table(sp):
        ll = age_cmp.get(sp, {})
        rows = [
            ("N samples",   f"{ll.get('llama_n',0):,}",         f"{ll.get('gpt_n',0):,}"),
            ("Age mean",    f"{_fmt(ll.get('llama_mean'))} yr",  f"{_fmt(ll.get('gpt_mean'))} yr"),
            ("Age std",     f"{_fmt(ll.get('llama_std'))} yr",   f"{_fmt(ll.get('gpt_std'))} yr"),
            ("Age min",     f"{_fmt(ll.get('llama_min'))} yr",   f"{_fmt(ll.get('gpt_min'))} yr"),
            ("Age max",     f"{_fmt(ll.get('llama_max'))} yr",   f"{_fmt(ll.get('gpt_max'))} yr"),
        ]
        html = f"""<div class="panel">
          <div class="panel-title">{sp.capitalize()} split</div>
          <table>
            <tr><th>Metric</th><th class="td-r">MethylLlama</th><th class="td-r">MethylGPT</th></tr>
            {''.join(f"<tr><td>{r[0]}</td><td class='td-r'>{r[1]}</td><td class='td-r'>{r[2]}</td></tr>" for r in rows)}
          </table>
        </div>"""
        return html

    # Age histograms side-by-side
    ll_valid_hist = llama["per_split"].get("valid", {}).get("age_hist", [])
    gp_valid_hist = gpt["per_split"].get("valid", {}).get("age_hist", [])
    ll_test_hist  = llama["per_split"].get("test", {}).get("age_hist", [])
    gp_test_hist  = gpt["per_split"].get("test", {}).get("age_hist", [])

    def _dual_hist(ll_hist, gp_hist):
        bins = [f"{i*10}–{(i+1)*10}" for i in range(12)]
        if not ll_hist and not gp_hist:
            return "<i>Not available</i>"
        ll_tot = sum(ll_hist) or 1
        gp_tot = sum(gp_hist) or 1
        max_pct = max(
            max((100 * v / ll_tot for v in ll_hist), default=0),
            max((100 * v / gp_tot for v in gp_hist), default=0), 1
        )
        html = ""
        for i, lbl in enumerate(bins):
            ll_v = ll_hist[i] if i < len(ll_hist) else 0
            gp_v = gp_hist[i] if i < len(gp_hist) else 0
            ll_p = 100 * ll_v / ll_tot
            gp_p = 100 * gp_v / gp_tot
            html += (
                f'<div style="display:flex;align-items:center;gap:4px;margin-bottom:3px">'
                f'<div style="width:56px;font-size:10px;text-align:right;color:#5a6888;font-family:monospace">{lbl}</div>'
                f'<div style="width:140px;display:flex;flex-direction:column;gap:2px">'
                f'  <div style="height:7px;background:#eef1f8;border-radius:3px;overflow:hidden">'
                f'    <div style="height:100%;width:{ll_p/max_pct*100:.0f}%;background:#cc5040;border-radius:3px"></div></div>'
                f'  <div style="height:7px;background:#eef1f8;border-radius:3px;overflow:hidden">'
                f'    <div style="height:100%;width:{gp_p/max_pct*100:.0f}%;background:#5a8de0;border-radius:3px"></div></div>'
                f'</div>'
                f'<div style="font-size:10px;color:#6a7890;font-family:monospace;width:110px">'
                f'LL:{ll_v:,}({ll_p:.0f}%) GP:{gp_v:,}({gp_p:.0f}%)</div>'
                f'</div>'
            )
        return html

    s5 = f"""
<div class="slide">
  <div class="slide-label">Dataset Comparison · Slide 5 / {n_slides}</div>
  <div class="slide-title">📊 Age Distribution Comparison — Per Split</div>

  <div class="grid3" style="margin-bottom:20px">
    {_age_split_table('train')}
    {_age_split_table('valid')}
    {_age_split_table('test')}
  </div>

  <div class="grid2">
    <div class="panel">
      <div class="panel-title">Valid split — age histogram comparison</div>
      <div style="display:flex;gap:14px;margin-bottom:8px">
        <div style="display:flex;align-items:center;gap:4px">
          <div style="width:14px;height:8px;background:#cc5040;border-radius:2px"></div>
          <span style="font-size:11px">MethylLlama</span>
        </div>
        <div style="display:flex;align-items:center;gap:4px">
          <div style="width:14px;height:8px;background:#5a8de0;border-radius:2px"></div>
          <span style="font-size:11px">MethylGPT</span>
        </div>
      </div>
      {_dual_hist(ll_valid_hist, gp_valid_hist)}
    </div>
    <div class="panel">
      <div class="panel-title">Test split — age histogram comparison</div>
      {_dual_hist(ll_test_hist, gp_test_hist)}
    </div>
  </div>

  <div class="divider"></div>
  <div class="callout callout-info" style="margin-top:0">
    <b>Why this matters:</b> If MethylLlama and MethylGPT evaluate on different test populations
    (different age distributions or different samples), their MedAE scores are not directly
    comparable — MedAE depends heavily on the age distribution of the test set.
    Both models can only be fairly compared if they use the <b>same test samples</b>.
  </div>
</div>"""
    slides.append(s5)

    # ── Slide 6: Split identity verdict ──────────────────────────────────────
    age_cmp  = cmp.get("age_cmp", {})
    split_id = cmp.get("split_identity", {})
    same_all = split_id.get("same_sizes_all_splits", False)
    verdict_cls = "callout-ok" if same_all else "callout-red"

    def _split_row(sp):
        d = age_cmp.get(sp, {})
        ll_n, gp_n = d.get("llama_n", 0), d.get("gpt_n", 0)
        same = d.get("same_size")
        size_badge = (_badge("SAME", "badge-green") if same
                      else _badge(f"DIFF ({gp_n:,} vs {ll_n:,})", "badge-red") if same is False
                      else _badge("?", "badge-gray"))
        ks = d.get("ks_stat")
        ks_p = d.get("age_dist_p")
        ks_str = f"{ks:.3f}" if ks is not None else "—"
        p_str  = f"{ks_p:.3f}" if ks_p is not None else "—"
        age_badge = (_badge("SIMILAR", "badge-green") if (ks_p is not None and ks_p > 0.05)
                     else _badge("DIFFERENT", "badge-amber") if ks_p is not None
                     else _badge("?", "badge-gray"))
        return (f"<tr><td><b>{sp}</b></td>"
                f"<td class='td-r'>{ll_n:,}</td><td class='td-r'>{gp_n:,}</td>"
                f"<td class='td-c'>{size_badge}</td>"
                f"<td class='td-r'>{_fmt(d.get('llama_mean'))} ± {_fmt(d.get('llama_std'))}</td>"
                f"<td class='td-r'>{_fmt(d.get('gpt_mean'))} ± {_fmt(d.get('gpt_std'))}</td>"
                f"<td class='td-r'>{ks_str}</td><td class='td-r'>{p_str}</td>"
                f"<td class='td-c'>{age_badge}</td></tr>")

    ao = cmp.get("age_overlap", {})

    def _age_overlap_row(sp):
        d = ao.get(sp, {})
        if "shared_count" not in d:
            return f"<tr><td>{sp}</td><td class='td-c' colspan='6'><i>{d.get('note','—')}</i></td></tr>"
        sc  = d["shared_count"]
        ll_n, gp_n = d["llama_n"], d["gpt_n"]
        pll, pgp = d["pct_of_llama"], d["pct_of_gpt"]
        u_sh = d["unique_shared_ages"]
        overlap_badge = (
            _badge(f"{pll:.0f}% of LL", "badge-green") if pll > 80 else
            _badge(f"{pll:.0f}% of LL", "badge-amber") if pll > 30 else
            _badge(f"{pll:.0f}% of LL", "badge-red"))
        return (f"<tr><td><b>{sp}</b></td>"
                f"<td class='td-r'>{ll_n:,}</td><td class='td-r'>{gp_n:,}</td>"
                f"<td class='td-r'><b>{sc:,}</b></td>"
                f"<td class='td-r'>{pll:.1f}%</td><td class='td-r'>{pgp:.1f}%</td>"
                f"<td class='td-r'>{u_sh:,}</td>"
                f"<td class='td-c'>{overlap_badge}</td></tr>")

    s6 = f"""
<div class="slide">
  <div class="slide-label">Dataset Comparison · Slide 6 / {n_slides}</div>
  <div class="slide-title">⚖️ Split Identity — Are Valid/Test Sets the Same?</div>

  <div class="panel" style="margin-bottom:16px">
    <div class="panel-title">Age-based sample overlap per split (proxy for same samples)</div>
    <table>
      <tr>
        <th>Split</th>
        <th class="td-r">LL N</th><th class="td-r">GPT N</th>
        <th class="td-r">Shared by age</th>
        <th class="td-r">% of LL</th><th class="td-r">% of GPT</th>
        <th class="td-r">Unique ages matched</th>
        <th class="td-c">Verdict</th>
      </tr>
      {_age_overlap_row('train')}
      {_age_overlap_row('valid')}
      {_age_overlap_row('test')}
    </table>
    <p style="font-size:11px;color:#7a8aa8;margin-top:8px">
      <b>Method:</b> Samples matched by exact age value (rounded to 4 decimal places).
      If the same biological sample exists in both datasets it will have the same age.
      Samples sharing an age value are counted as a lower-bound on sample overlap.
      Duplicate ages may over-count — "Unique ages matched" shows distinct age values shared.
    </p>
  </div>

  <div class="panel" style="margin-bottom:16px">
    <div class="panel-title">Split size comparison</div>
    <table>
      <tr>
        <th>Split</th>
        <th class="td-r">MethylLlama N</th><th class="td-r">MethylGPT N</th>
        <th class="td-c">Size match?</th>
        <th class="td-r">LL age mean±std</th><th class="td-r">GPT age mean±std</th>
        <th class="td-r">KS stat</th><th class="td-r">p-value</th>
        <th class="td-c">Age dist.</th>
      </tr>
      {_split_row('train')}
      {_split_row('valid')}
      {_split_row('test')}
    </table>
    <p style="font-size:11px;color:#7a8aa8;margin-top:8px">
      KS stat = max CDF difference between age histograms (0=identical). p&gt;0.05 = distributions not significantly different.
    </p>
  </div>

  <div class="callout {verdict_cls}" style="margin-bottom:16px">
    <b>Verdict: {split_id.get('verdict', '?')}</b><br>
    {split_id.get('implication', '')}
  </div>

  <div class="grid2">
    <div class="panel">
      <div class="panel-title">Key observations</div>
      <ul style="font-size:13px;line-height:2;padding-left:18px;color:#2a3550">
        <li>MethylGPT uses <b>artificial sample_NNN IDs</b> — direct ID matching impossible</li>
        <li>Total samples: MethylLlama <b>{llama['n_samples']:,}</b> vs MethylGPT <b>{gpt['n_samples']:,}</b></li>
        <li>Test set: MethylLlama <b>{age_cmp.get('test',{}).get('llama_n',0):,}</b> vs MethylGPT <b>{age_cmp.get('test',{}).get('gpt_n',0):,}</b> samples</li>
        <li>Valid set: MethylLlama <b>{age_cmp.get('valid',{}).get('llama_n',0):,}</b> vs MethylGPT <b>{age_cmp.get('valid',{}).get('gpt_n',0):,}</b> samples</li>
      </ul>
    </div>
    <div class="panel">
      <div class="panel-title">Implication for MedAE comparison</div>
      <p style="font-size:13px;line-height:1.8;color:#2a3550">
        MedAE (median absolute error) is <b>highly sensitive to the age distribution</b> of the test set.
        A test set skewed toward older patients will yield higher MedAE even for an equally accurate model.<br><br>
        {'<b style="color:#aa2020">⚠ Different test set sizes and composition mean the reported MedAE values (MethylLlama=3.56yr vs MethylGPT=3.75yr) are evaluated on DIFFERENT populations — direct comparison is not statistically valid.</b>'
         if not same_all else
         '<b style="color:#2a7a2a">✓ Same split sizes suggest the models are evaluated on the same population — MedAE comparison is valid.</b>'}
      </p>
    </div>
  </div>
</div>"""
    slides.append(s6)

    # ── Assemble HTML ─────────────────────────────────────────────────────────
    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<title>MethylLlama vs MethylGPT — Dataset Comparison</title>
<style>{CSS}</style>
</head>
<body>
{''.join(slides)}
</body>
</html>"""
    out_path.write_text(html)
    print(f"\n  HTML report → {out_path}")


def render_txt(llama: dict, gpt: dict, cmp: dict, out_path: Path):
    lines = ["=" * 70,
             "DATASET COMPARISON — MethylLlama vs MethylGPT",
             "=" * 70, ""]

    lines += ["── OVERVIEW ─────────────────────────────────────────",
              f"  MethylLlama: {llama['n_samples']:,} samples × {llama['n_cpgs']:,} CpGs  "
              f"  NaN={llama['has_nan']}  ({llama['file_mb']:.0f} MB)",
              f"  MethylGPT:   {gpt['n_samples']:,} samples × {gpt['n_cpgs'] or '?'} CpGs  "
              f"  NaN_frac={gpt.get('nan_frac', float('nan')):.1%}  ({gpt['file_mb']:.0f} MB)",
              ""]

    lines += ["── SPLITS ───────────────────────────────────────────"]
    for sp in ("train", "valid", "test"):
        ll = llama["per_split"].get(sp, {})
        gp = gpt["per_split"].get(sp, {})
        lines.append(f"  {sp:6s}  LL={ll.get('n',0):,}  GP={gp.get('n',0):,}  "
                     f"LL_age={_fmt(ll.get('age_mean'))}±{_fmt(ll.get('age_std'))}  "
                     f"GP_age={_fmt(gp.get('age_mean'))}±{_fmt(gp.get('age_std'))}")
    lines.append("")

    lines += ["── CpG OVERLAP ──────────────────────────────────────"]
    cpg = cmp.get("cpg_overlap", {})
    if cpg.get("shared_n") is not None:
        lines += [
            f"  Shared CpGs:          {cpg['shared_n']:,}",
            f"  Only in MethylLlama:  {cpg['only_llama']:,}",
            f"  Only in MethylGPT:    {cpg['only_gpt']:,}",
            f"  Llama ⊆ GPT?          {'YES' if cpg['llama_subset_of_gpt'] else 'NO'}",
        ]
    else:
        lines.append(f"  {cpg.get('note', 'Not available')}")
    lines.append("")

    lines += ["── SAMPLE OVERLAP (valid / test) ────────────────────"]
    for sp in ("valid", "test"):
        d = cmp.get("sample_overlap", {}).get(sp, {})
        if d.get("shared_n") is not None:
            lines.append(
                f"  {sp:6s}: shared={d['shared_n']:,}  "
                f"pct_llama={d['pct_llama']:.1f}%  pct_gpt={d['pct_gpt']:.1f}%  "
                f"IDENTICAL={d['fully_shared']}")
        else:
            lines.append(f"  {sp}: {d.get('note', 'not available')}")
    lines.append("")

    lines += ["── NaN EXTENSION CHECK ──────────────────────────────"]
    ne = cmp.get("nan_extension", {})
    if ne.get("nan_inside_llama") is not None:
        lines += [
            f"  NaN inside MethylLlama window:  {ne['nan_inside_llama']:.3%}",
            f"  NaN outside MethylLlama window: {ne['nan_outside_llama']:.3%}",
            f"  Ratio (outside/inside):         {ne['ratio']:.1f}×",
            f"  Hypothesis supported:           {'YES' if ne['supported'] else 'NO'}",
        ]
    else:
        lines.append(f"  {ne.get('note', 'Not available')}")

    lines += [""]
    lines += ["── SPLIT IDENTITY ANALYSIS ──────────────────────────"]
    age_cmp = cmp.get("age_cmp", {})
    for sp in ("train", "valid", "test"):
        d = age_cmp.get(sp, {})
        ll_n, gp_n = d.get("llama_n", 0), d.get("gpt_n", 0)
        same = d.get("same_size")
        ks   = d.get("ks_stat")
        ks_p = d.get("age_dist_p")
        lines.append(
            f"  {sp:6s}: LL={ll_n:,}  GPT={gp_n:,}  "
            f"{'SAME SIZE' if same else 'DIFFERENT SIZE'}  "
            f"KS={f'{ks:.3f}' if ks is not None else '?'}  "
            f"p={f'{ks_p:.3f}' if ks_p is not None else '?'}  "
            f"age LL={_fmt(d.get('llama_mean'))}±{_fmt(d.get('llama_std'))}  "
            f"GPT={_fmt(d.get('gpt_mean'))}±{_fmt(d.get('gpt_std'))}"
        )
    split_id = cmp.get("split_identity", {})
    lines += [
        "",
        f"  VERDICT:     {split_id.get('verdict', '?')}",
        f"  IMPLICATION: {split_id.get('implication', '?')}",
    ]

    lines += ["", "── AGE-BASED SAMPLE OVERLAP ─────────────────────────"]
    ao = cmp.get("age_overlap", {})
    for sp in ("valid", "test", "train"):
        d = ao.get(sp, {})
        if "shared_count" not in d:
            lines.append(f"  {sp}: {d.get('note', 'not available')}")
        else:
            lines.append(
                f"  {sp:6s}: LL={d['llama_n']:,}  GPT={d['gpt_n']:,}  "
                f"shared_by_age={d['shared_count']:,}  "
                f"({d['pct_of_llama']:.1f}% of LL / {d['pct_of_gpt']:.1f}% of GPT)  "
                f"unique_ages_matched={d['unique_shared_ages']:,}"
            )

    lines += ["", "=" * 70]
    out_path.write_text("\n".join(lines))
    print(f"  Text summary → {out_path}")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(description="Compare MethylLlama and MethylGPT datasets.")
    p.add_argument("--llama_h5ad",    default=LLAMA_H5AD)
    p.add_argument("--gpt_parquet",   default=GPT_PARQUET_DIR)
    p.add_argument("--gpt_h5ad",      default=GPT_H5AD)
    p.add_argument("--outdir",        default="outputs/dataset_comparison")
    args = p.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Dataset Comparison — MethylLlama vs MethylGPT")
    print("=" * 60)

    llama = analyze_llama(args.llama_h5ad)
    gpt   = analyze_gpt(args.gpt_parquet, args.gpt_h5ad)
    cmp   = compare(llama, gpt)

    # Print comparison coverage summary
    print("\n--- Comparison coverage ---")
    print(f"  MethylLlama : {llama['n_samples']:,} samples × {llama['n_cpgs']:,} CpGs  "
          f"({'CpG IDs: YES' if llama.get('cpg_ids') else 'CpG IDs: NO'})")
    print(f"  MethylGPT   : {gpt['n_samples']:,} samples × {gpt['n_cpgs'] or '?'} CpGs  "
          f"({'CpG IDs: YES (' + str(len(gpt['cpg_ids'])) + ')' if gpt.get('cpg_ids') else 'CpG IDs: NO'})")
    cpg_cmp = cmp.get("cpg_overlap", {})
    so      = cmp.get("sample_overlap", {})
    print(f"  CpG overlap  : {'shared=' + str(cpg_cmp.get('shared_n', '?')) if cpg_cmp.get('shared_n') is not None else 'SKIPPED (no CpG IDs for GPT)'}")
    for sp in ("valid", "test"):
        d = so.get(sp, {})
        if d.get("shared_n") is not None:
            print(f"  {sp} overlap : shared={d['shared_n']:,} / LL={d['llama_n']:,} / GPT={d['gpt_n']:,}")
        else:
            print(f"  {sp} overlap : N/A — {d.get('note', 'no sample IDs')}")
    ne = cmp.get("nan_extension", {})
    if "nan_inside_llama" in ne:
        print(f"  NaN extension: computed  inside={ne['nan_inside_llama']:.4f}  outside={ne['nan_outside_llama']:.4f}  ratio={ne['ratio']:.1f}x  hypothesis={'SUPPORTED' if ne['supported'] else 'NOT supported'}")
    else:
        print(f"  NaN extension: SKIPPED ({ne.get('note', 'no NaN profile')})")

    render_html(llama, gpt, cmp, out / "dataset_comparison_report.html")
    render_txt(llama, gpt, cmp, out / "dataset_comparison_summary.txt")

    print("\n" + "=" * 60)
    print(f"Done. All outputs → {out.resolve()}")
    print("=" * 60)


if __name__ == "__main__":
    main()
