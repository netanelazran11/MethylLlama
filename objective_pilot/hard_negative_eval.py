#!/usr/bin/env python3
"""
Hard-negative cross-view retrieval on saved held-out embeddings (no re-encoding).

For each held-out profile with metadata: is view 1 closer to its own view 2 than to
K view-2 negatives drawn from a restricted pool (same tissue / study / sex / age bin /
coverage), or from the K nearest neighbours of the target? K = 31 matches the negatives
one per-GPU InfoNCE batch sees in pretraining. Exact and near duplicates
(heldout_duplicates_pairs.csv) are never used as negatives. Reports top-1 (1 vs K)
and the median margin (matched cosine - hardest negative cosine).

Inputs are the outputs of scripts/evaluation/view_consistency_heldout_full.py.
Run from the repository root:
    python -m objective_pilot.hard_negative_eval [--emb-dir DIR] [--conds overlap_seed0 disjoint_seed0]
Point --emb-dir at another checkpoint's per_run/ folder to compare arms with the same protocol.

First run (2026-10-05, published checkpoint, seed 0, n = 13,699 with metadata):
disjoint views: random 99.9 %, same study 93.8 %, same study+tissue 93.0 %, +sex 90.6 %,
31 nearest neighbours of the target 90.5 %; median margin 0.27 -> ~0.01.
"""

import argparse
import json
import os

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--emb-dir", default="outputs/objective_pilot/pilot_eval/lam0.05_seed42_ep19/view_consistency/per_run/")
ap.add_argument("--conds", nargs="+", default=["overlap_seed0", "disjoint_seed0"])
ap.add_argument("--metadata", default=f'{os.environ.get("METHYLLAMA_DATA", "data")}/pretrain_metadata.csv.gz')
ap.add_argument("--dups", default="results/data_audit/heldout_duplicates_flagged_pairs.csv")
ap.add_argument("--k", type=int, default=31)
ap.add_argument("--seed", type=int, default=0)
ap.add_argument("--out", default=None, help="optional JSON output path")
ap.add_argument("--per-query-out", default=None,
                help="optional .npz with per-query hits/margins per pool (for paired comparisons across checkpoints: "
                     "the random pools depend only on metadata, so with the same --seed every checkpoint gets the "
                     "same negatives for the same query)")
a = ap.parse_args()

m = pd.read_csv(a.metadata).drop_duplicates("GSM_ID").set_index("GSM_ID")
dup = pd.read_csv(a.dups)
dup = dup[(dup.comparison == "heldout vs heldout") & (dup.duplicate | dup.near_duplicate)]
rng = np.random.default_rng(a.seed)
K = a.k
results = {}
per_query = {}

for cond in a.conds:
    z = np.load(a.emb_dir + cond + "_embeddings.npz")
    ids = z["sample_ids"]
    N = len(ids)
    e1 = z["e1"] / np.linalg.norm(z["e1"], axis=1, keepdims=True)
    e2 = z["e2"] / np.linalg.norm(z["e2"], axis=1, keepdims=True)
    md = m.reindex(ids)
    mf = z["measured_fraction"]
    ds = md["dataset"].values.astype(str)
    ti = md["tissue"].values.astype(str)
    sx = md["sex"].values.astype(str)
    ab = np.floor(pd.to_numeric(md["age"], errors="coerce").values / 10)
    has = md["dataset"].notna().values

    pos = {s: i for i, s in enumerate(ids)}
    excl = [set() for _ in range(N)]
    for x, y in zip(dup.id_a, dup.id_b):
        if x in pos and y in pos:
            excl[pos[x]].add(pos[y])
            excl[pos[y]].add(pos[x])

    S = e1 @ e2.T
    diag = np.diag(S).copy()
    res = {"n_profiles": int(N), "n_with_metadata": int(has.sum()), "k_negatives": K, "pools": {}}

    def run(name, mask_fn):
        hits, margins, skipped, qs = [], [], 0, []
        for i in range(N):
            if not has[i]:
                continue
            cand = np.where(mask_fn(i))[0]
            cand = cand[cand != i]
            if excl[i]:
                cand = np.array([c for c in cand if c not in excl[i]])
            if len(cand) < K:
                skipped += 1
                continue
            neg = rng.choice(cand, K, replace=False)
            hard = S[i, neg].max()
            hits.append(diag[i] > hard)
            margins.append(diag[i] - hard)
            qs.append(i)
        r = {"n": len(hits), "skipped_pool_too_small": skipped,
             "top1": float(np.mean(hits)) if hits else None,
             "median_margin": float(np.median(margins)) if margins else None}
        res["pools"][name] = r
        per_query[f"{cond}|{name}|q"] = np.array(qs, dtype=np.int32)
        per_query[f"{cond}|{name}|hit"] = np.array(hits, dtype=bool)
        per_query[f"{cond}|{name}|margin"] = np.array(margins, dtype=np.float32)
        print(f"  {name:40s} n={r['n']:6d} skipped={skipped:5d}  top1={r['top1']:.4f}  median margin={r['median_margin']:.4f}")

    print(f"{cond}  (N={N}, with metadata {has.sum()})")
    run("random", lambda i: np.ones(N, bool))
    run("same tissue", lambda i: ti == ti[i])
    run("same study", lambda i: ds == ds[i])
    run("same study+tissue", lambda i: (ds == ds[i]) & (ti == ti[i]))
    run("same study+tissue+sex", lambda i: (ds == ds[i]) & (ti == ti[i]) & (sx == sx[i]) & (sx[i] != "nan"))
    run("same study+tissue+age decade", lambda i: (ds == ds[i]) & (ti == ti[i]) & (ab == ab[i]) & ~np.isnan(ab[i]))
    run("same study+tissue+|coverage diff|<0.005", lambda i: (ds == ds[i]) & (ti == ti[i]) & (np.abs(mf - mf[i]) < 0.005))

    # K nearest neighbours of the target (by view-2 similarity; the query's own view is not used)
    T = e2 @ e2.T
    np.fill_diagonal(T, -9)
    hits, qs = [], []
    for i in range(N):
        if not has[i]:
            continue
        o = np.argsort(-T[i])
        o = [c for c in o[: K + len(excl[i]) + 5] if c not in excl[i]][:K]
        hits.append(diag[i] > S[i, o].max())
        qs.append(i)
    per_query[f"{cond}|K nearest neighbours of target|q"] = np.array(qs, dtype=np.int32)
    per_query[f"{cond}|K nearest neighbours of target|hit"] = np.array(hits, dtype=bool)
    res["pools"]["K nearest neighbours of target"] = {"n": len(hits), "top1": float(np.mean(hits))}
    print(f"  {'K nearest neighbours of target':40s} n={len(hits):6d}  top1={np.mean(hits):.4f}")
    results[cond] = res

from pathlib import Path
for _p in (a.per_query_out, a.out):
    if _p:
        Path(_p).parent.mkdir(parents=True, exist_ok=True)

if a.per_query_out:
    np.savez_compressed(a.per_query_out, **per_query)
    print("wrote", a.per_query_out)

if a.out:
    with open(a.out, "w") as f:
        json.dump(results, f, indent=1)
    print("wrote", a.out)
