#!/usr/bin/env python3
"""
hard_negative_heldout.py
========================
Hard-negative cross-view retrieval on the held-out pretraining partition (all saved view draws;
Fig. 2a,b and the pattern-only result against same-study negatives in Results).

Pure post-processing of the view embeddings saved by view_consistency_heldout_full.py
(per_run/<cond>_embeddings.npz, about 31 MB per draw; not shipped, regenerate them with
scripts/evaluation/run_view_consistency_heldout_full.sh); no GPU, no re-encoding.

Per query i (view 1) the positive is its own view 2; K = 31 negatives (the number one per-GPU
InfoNCE batch sees) are view-2 embeddings of other profiles drawn from a restricted pool.
Score: hit = cos(pos) > max cos(neg); margin = cos(pos) - max cos(neg).

Fairness rules:
  * queries need non-missing study AND tissue; a pool keyed on a field uses only negatives with
    that field present (the old script pooled all missing-tissue profiles as one "tissue")
  * never a negative: exact/near duplicates (heldout vs heldout) or a known same individual
    (same study and same PATIENT_ID); "profile", not "person", since IDs exist for ~20 %
  * negatives are drawn once per (pool, draw) and reused for every view condition, so overlap
    vs disjoint is a paired comparison on identical negatives
  * every pool is also scored with random negatives on the SAME queries, so a drop is never a
    change of population
  * 5 independent negative draws; 95 % CI by bootstrap over queries
Pools: random, same tissue, same study, same study+tissue, +sex, +age decade,
K nearest neighbours of the target (by view-2 similarity), all candidates (K = every eligible).

Other inputs
  --metadata  annotations that accompany the pretraining corpus (study, tissue, sex, age, PATIENT_ID);
              ${METHYLLAMA_DATA}/pretrain_metadata.csv.gz, see configs/paths.example.sh
  --dups      duplicate and near-duplicate pairs of scripts/data/pretrain_heldout_duplicates.py
              (shipped: results/data_audit/heldout_duplicates_flagged_pairs.csv)

Usage (CPU, from the repository root):
    python scripts/evaluation/hard_negative_heldout.py
Writes results/pretrained_cls/hard_negative_heldout/{summary.json, per_query.npz}:
per (view condition, pool) the top-1 rate, its bootstrap CI and range over negative draws, the median
margin; per query the matched-view cosine and the hardest negative cosine of the first draw.
"""
import argparse
import json
import os
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", default="results/pretrained_cls/view_consistency_heldout_full/per_run")
ap.add_argument("--conds", nargs="+", default=["overlap_seed0", "overlap_seed1", "overlap_seed2",
                                                "disjoint_seed0", "disjoint_seed1", "disjoint_seed2",
                                                "pattern_only_seed0"])
ap.add_argument("--metadata", default=f'{os.environ.get("METHYLLAMA_DATA", "data")}/pretrain_metadata.csv.gz')
ap.add_argument("--dups", default="results/data_audit/heldout_duplicates_flagged_pairs.csv")
ap.add_argument("--k", type=int, default=31)
ap.add_argument("--draws", type=int, default=5)
ap.add_argument("--boot", type=int, default=2000)
ap.add_argument("--out-dir", default="results/pretrained_cls/hard_negative_heldout")
a = ap.parse_args()
K = a.k

# ---- ids, metadata, exclusions (shared by all conditions) -------------------------------
z0 = np.load(f"{a.emb_dir}/{a.conds[0]}_embeddings.npz")
ids = z0["sample_ids"].astype(str)
N = len(ids)
for c in a.conds[1:]:
    assert np.array_equal(np.load(f"{a.emb_dir}/{c}_embeddings.npz")["sample_ids"].astype(str), ids), c

md = pd.read_csv(a.metadata).drop_duplicates("GSM_ID").set_index("GSM_ID").reindex(ids)
def field(col):
    v = md[col].astype(object).where(md[col].notna(), None).values
    return np.array([None if x is None else str(x) for x in v], dtype=object)
ds, ti, pid = field("dataset"), field("tissue"), field("PATIENT_ID")
sx = np.array([s if s in ("M", "F") else None for s in field("sex")], dtype=object)
age = pd.to_numeric(md["age"], errors="coerce").values
dec = np.where(np.isnan(age), np.nan, np.floor(age / 10))
query_ok = np.array([d is not None and t is not None for d, t in zip(ds, ti)])

excl = [set() for _ in range(N)]
pos = {s: i for i, s in enumerate(ids)}
dup = pd.read_csv(a.dups, usecols=["comparison", "id_a", "id_b", "duplicate", "near_duplicate"])
dup = dup[(dup.comparison == "heldout vs heldout") & (dup.duplicate | dup.near_duplicate)]
n_dup = 0
for x, y in zip(dup.id_a.astype(str), dup.id_b.astype(str)):
    if x in pos and y in pos:
        excl[pos[x]].add(pos[y]); excl[pos[y]].add(pos[x]); n_dup += 1
key = pd.Series([f"{d}|{p}" if d is not None and p is not None else None for d, p in zip(ds, pid)])
n_same_ind = 0
for _, g in key.dropna().groupby(key.dropna()):
    g = g.index.values
    if len(g) > 1:
        for i in g:
            excl[i].update(int(j) for j in g if j != i); n_same_ind += len(g) - 1

def groups(keys):
    out = {}
    for i, k in enumerate(keys):
        if k is not None:
            out.setdefault(k, []).append(i)
    return {k: np.array(v) for k, v in out.items()}

def keyed(*arrs):
    return [None if any(x is None or (isinstance(x, float) and np.isnan(x)) for x in t) else t
            for t in zip(*arrs)]

POOLS = {  # name -> per-profile key (None = profile not eligible as query or negative)
    "random": [0 if q else None for q in query_ok],
    "same tissue": keyed(ti),
    "same study": keyed(ds),
    "same study+tissue": keyed(ds, ti),
    "same study+tissue+sex": keyed(ds, ti, sx),
    "same study+tissue+age decade": keyed(ds, ti, list(dec)),
}
eligible_all = np.where(query_ok)[0]          # every negative must itself be a valid profile
rng = np.random.default_rng(20261018)

def draw(pool_keys):
    """negatives[d][i] -> array of K indices or None; same draw reused for every condition."""
    g = groups([k if query_ok[i] else None for i, k in enumerate(pool_keys)])
    out = []
    for _ in range(a.draws):
        neg = [None] * N
        for i in eligible_all:
            k = pool_keys[i]
            if k is None:
                continue
            cand = g[k]
            cand = cand[cand != i]
            if excl[i]:
                cand = cand[~np.isin(cand, list(excl[i]))]
            if len(cand) >= K:
                neg[i] = rng.choice(cand, K, replace=False)
        out.append(neg)
    return out

NEG = {p: draw(k) for p, k in POOLS.items()}
print(f"N={N}, queries with study+tissue={query_ok.sum()}, dup pairs={n_dup}, same-individual links={n_same_ind}")

def boot_ci(x):
    x = np.asarray(x, float)
    b = np.random.default_rng(0).integers(0, len(x), (a.boot, len(x)))
    m = x[b].mean(1)
    return [float(np.percentile(m, 2.5)), float(np.percentile(m, 97.5))]

summary = {"n_profiles": N, "n_queries_with_study_and_tissue": int(query_ok.sum()), "k_negatives": K,
           "negative_draws": a.draws, "excluded_negatives": {"duplicate_pairs": n_dup, "same_individual_links": n_same_ind},
           "conditions": {}}
per_query = {}
mask_valid = np.zeros(N, bool); mask_valid[eligible_all] = True

for cond in a.conds:
    z = np.load(f"{a.emb_dir}/{cond}_embeddings.npz")
    e1 = z["e1"] / np.linalg.norm(z["e1"], axis=1, keepdims=True)
    e2 = z["e2"] / np.linalg.norm(z["e2"], axis=1, keepdims=True)
    pcos = np.einsum("ij,ij->i", e1, e2)
    res = {}
    hard = {p: np.full((a.draws, N), np.nan) for p in POOLS}
    knn_hard = np.full(N, np.nan); all_hard = np.full(N, np.nan)
    for s in range(0, N, 1024):
        blk = np.arange(s, min(s + 1024, N))
        S = e1[blk] @ e2.T                        # query view 1 vs every view 2
        T = e2[blk] @ e2.T                        # target view 2 vs every view 2 (for kNN)
        for r, i in enumerate(blk):
            if not query_ok[i]:
                continue
            for p in POOLS:
                for d in range(a.draws):
                    n = NEG[p][d][i]
                    if n is not None:
                        hard[p][d, i] = S[r, n].max()
            bad = np.fromiter(excl[i], int) if excl[i] else np.empty(0, int)
            row = T[r].copy(); row[i] = -9; row[bad] = -9; row[~mask_valid] = -9
            knn = np.argpartition(-row, K)[:K]
            knn_hard[i] = S[r, knn].max()
            srow = S[r].copy(); srow[i] = -9; srow[bad] = -9; srow[~mask_valid] = -9
            all_hard[i] = srow.max()
    per_query[f"{cond}__pos"] = pcos.astype(np.float32)
    for p in POOLS:
        ok = ~np.isnan(hard[p][0])
        hits = pcos[None, ok] > hard[p][:, ok]
        rnd = pcos[None, ok] > hard["random"][:, ok]   # random negatives on the same queries
        marg = pcos[ok] - hard[p][0, ok]
        res[p] = {"n_queries": int(ok.sum()),
                  "top1": float(hits.mean()), "top1_ci95": boot_ci(hits.mean(0)),  # per-query mean over draws
                  "top1_range_over_draws": [float(hits.mean(1).min()), float(hits.mean(1).max())],
                  "top1_random_same_queries": float(rnd.mean()),
                  "median_margin": float(np.median(marg)), "frac_margin_below_0.01": float((marg < 0.01).mean())}
        per_query[f"{cond}__{p}"] = hard[p][0].astype(np.float32)
    for p, h in (("K nearest neighbours of target", knn_hard), ("all candidates", all_hard)):
        ok = ~np.isnan(h); hits = pcos[ok] > h[ok]; marg = pcos[ok] - h[ok]
        res[p] = {"n_queries": int(ok.sum()), "top1": float(hits.mean()), "top1_ci95": boot_ci(hits),
                  "median_margin": float(np.median(marg)), "frac_margin_below_0.01": float((marg < 0.01).mean())}
        per_query[f"{cond}__{p}"] = h.astype(np.float32)
    summary["conditions"][cond] = res
    print(cond)
    for p, r in res.items():
        print(f"  {p:32s} n={r['n_queries']:6d} top1={100*r['top1']:6.2f}%  CI [{100*r['top1_ci95'][0]:.2f}, {100*r['top1_ci95'][1]:.2f}]"
              f"  median margin={r['median_margin']:.4f}" + (f"  random-on-same={100*r['top1_random_same_queries']:.2f}%" if 'top1_random_same_queries' in r else ""))

out = Path(a.out_dir); out.mkdir(parents=True, exist_ok=True)
summary["inputs"] = {"emb_dir": a.emb_dir, "metadata": Path(a.metadata).name, "dups": a.dups, "negative_seed": 20261018}
(out / "summary.json").write_text(json.dumps(summary, indent=1))
np.savez_compressed(out / "per_query.npz", sample_ids=ids, query_ok=query_ok, **per_query)
print("wrote", out)
