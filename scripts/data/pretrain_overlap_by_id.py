#!/usr/bin/env python3
"""
pretrain_overlap_by_id.py
==========================
Exact overlap, by sample identifier, between the pretraining corpus and the
downstream (AltuMAge) cohort -- and which pretraining split each shared sample
fell in.  READ-ONLY.  Reads only the obs index of each h5ad (no methylation
values), so it runs in seconds and needs no GPU and no job.

An identifier identifies one sample; nothing here is inferred from similarity.

The pretraining split is recomputed exactly as the training code defines it
(bmfm_methylation/shared/data_module.py, auto-split): rng(42).permutation(n),
first 80% train, next 10% valid, last 10% held-out.

Outputs (--outdir):
  pretrain_ids.csv.gz        row, sample_id, pretrain_split   (the corpus index, for provenance)
  overlap_by_id.csv          one row per downstream sample: in corpus? which split?
  overlap_by_id_summary.json counts by downstream split, benchmark test set, study

Usage (a login node is fine):
  python scripts/data/pretrain_overlap_by_id.py
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", default=f"{DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad")
    p.add_argument("--downstream", default=f"{DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad")
    p.add_argument("--test_ids", default="outputs/kfold_splits/test_ids.npy")
    p.add_argument("--outdir", default="outputs/pretrain_overlap")
    return p.parse_args()


def decode(arr):
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in arr], dtype=object)


def read_obs(path):
    """obs index plus any per-sample columns, straight from HDF5 (works for old and new anndata files)."""
    with h5py.File(path, "r") as f:
        obs = f["obs"]
        n = f["X"].shape[0] if isinstance(f["X"], h5py.Dataset) else int(f["X"].attrs["shape"][0])
        # The index is the per-sample string column whose length equals the number of rows.
        # Do not trust the '_index' attribute alone: in the pretraining file it names 'sample_id',
        # which holds a single entry, while the real 169,120 ids sit in obs['_index'] -- the same
        # inconsistency that makes anndata report "obs has 1 rows" for that file.
        attr = obs.attrs.get("_index", "_index")
        attr = attr.decode() if isinstance(attr, bytes) else attr
        candidates = [k for k in dict.fromkeys([attr, "_index", "index", "obs_names", *obs.keys()])
                      if k in obs and isinstance(obs[k], h5py.Dataset)
                      and obs[k].shape == (n,) and obs[k].dtype.kind in ("S", "O", "U")]
        assert candidates, (f"{path}: no per-sample string column of length {n} in obs "
                            f"(keys and shapes: { {k: getattr(obs[k], 'shape', 'group') for k in obs.keys()} })")
        index_key = candidates[0]
        ids = decode(obs[index_key][:])
        print(f"  {Path(path).name}: sample ids read from obs['{index_key}'] ({n:,} rows; "
              f"'_index' attribute says '{attr}')")
        cols = {}
        for k in obs.keys():
            if k == index_key:
                continue
            item = obs[k]
            if isinstance(item, h5py.Dataset) and item.shape == (n,):
                cols[k] = decode(item[:]) if item.dtype.kind in ("S", "O") else item[:]
            elif isinstance(item, h5py.Group) and "codes" in item and "categories" in item:
                cats, codes = decode(item["categories"][:]), item["codes"][:]
                cols[k] = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats) - 1)], "")
    return ids, cols


def prefix_counts(ids):
    s = pd.Series(ids).astype(str).str.extract(r"^([A-Za-z_]+)")[0].fillna("(numeric/other)")
    return s.value_counts().to_dict()


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    pre_ids, _ = read_obs(a.pretrain)
    n_pre = len(pre_ids)
    perm = np.random.default_rng(42).permutation(n_pre)
    tr_end, va_end = int(0.8 * n_pre), int(0.9 * n_pre)
    split = np.empty(n_pre, dtype=object)
    split[perm[:tr_end]], split[perm[tr_end:va_end]], split[perm[va_end:]] = "train", "valid", "heldout"
    n_unique = len(set(pre_ids.tolist()))
    print(f"pretraining corpus: {n_pre:,} rows, {n_unique:,} unique ids "
          f"| train {tr_end:,} / valid {va_end - tr_end:,} / held-out {n_pre - va_end:,}")
    print("pretraining id prefixes:", prefix_counts(pre_ids))
    pd.DataFrame({"row": np.arange(n_pre), "sample_id": pre_ids, "pretrain_split": split}).to_csv(
        outdir / "pretrain_ids.csv.gz", index=False)

    dn_ids, dn_cols = read_obs(a.downstream)
    print(f"downstream cohort: {len(dn_ids):,} rows | id prefixes: {prefix_counts(dn_ids)}")
    print("downstream obs columns:", sorted(dn_cols))

    # a sample id may occur more than once in the corpus; report every split it appears in
    where = {}
    for sid, sp in zip(pre_ids.tolist(), split.tolist()):
        where.setdefault(sid, set()).add(sp)
    in_corpus = np.array([s in where for s in dn_ids.tolist()])
    splits_of = np.array(["+".join(sorted(where[s])) if s in where else "" for s in dn_ids.tolist()], dtype=object)

    test_ids = set(decode(np.load(a.test_ids, allow_pickle=True)).tolist()) if Path(a.test_ids).exists() else set()
    df = pd.DataFrame({"sample_id": dn_ids, "in_pretraining_corpus": in_corpus, "pretrain_split": splits_of,
                       "in_benchmark_test_set": [s in test_ids for s in dn_ids.tolist()]})
    for k in ("split", "dataset", "tissue_type"):
        if k in dn_cols:
            df[k] = dn_cols[k]
    df.to_csv(outdir / "overlap_by_id.csv", index=False)

    def tab(mask):
        sub = df[mask]; m = sub[sub.in_pretraining_corpus]
        return {"n": int(len(sub)), "in_pretraining_corpus": int(len(m)),
                "pct": round(100 * len(m) / max(len(sub), 1), 1),
                "of_which_in_pretrain_split": m["pretrain_split"].value_counts().to_dict(),
                "seen_during_optimisation (train)": int(m["pretrain_split"].str.contains("train").sum())}

    summary = {"method": "exact intersection of sample identifiers (obs index of each h5ad)",
               "pretraining_rows": int(n_pre), "pretraining_unique_ids": int(n_unique),
               "pretraining_id_prefixes": prefix_counts(pre_ids),
               "downstream_rows": int(len(df)),
               "all_downstream": tab(np.ones(len(df), bool)),
               "benchmark_test_set_2149": tab(df.in_benchmark_test_set.values)}
    if "split" in df:
        summary["by_downstream_split"] = {s: tab((df["split"] == s).values) for s in sorted(df["split"].unique())}
    if "dataset" in df:
        summary["by_study"] = {s: tab((df["dataset"] == s).values) for s in df["dataset"].value_counts().index}
    (outdir / "overlap_by_id_summary.json").write_text(json.dumps(summary, indent=2, default=str))
    print(json.dumps({k: v for k, v in summary.items() if k != "by_study"}, indent=2, default=str))
    print(f"Saved -> {outdir}/pretrain_ids.csv.gz, overlap_by_id.csv, overlap_by_id_summary.json")


if __name__ == "__main__":
    main()
