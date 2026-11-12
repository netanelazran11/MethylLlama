#!/usr/bin/env python3
"""
within_tissue_study_structure.py
================================
Study structure inside a tissue (Results 2.3, Methods), with age and sex held fixed (addresses "study separation may be biology").

Tissues: those with at least two studies of >= 100 profiles (rule fixed before looking at results);
E-GEOD-56105 (corrupted downstream copy) is excluded. For each tissue and each profile, the 10 nearest
neighbours by cosine similarity among profiles of the same tissue:
  same-study fraction           observed fraction of neighbours from the profile's own study
  chance                        expected fraction if neighbours were drawn at random within the tissue
  matched same-study fraction   the same, with candidate neighbours restricted to the same sex and age decade
  matched chance                its random expectation within the same (sex, age-decade) cells
If study agreement stays far above its matched chance, age and sex do not explain the study separation.
Disease, platform and processing are not annotated and cannot be separated here.
Works on any embedding file with rows aligned to metadata.csv (CLS, mean pooling, raw-beta PCA).
Usage (CPU, under a minute):
  python scripts/evaluation/within_tissue_study_structure.py --emb results/pretrained_cls/embeddings_cls.npy \
      --out results/pretrained_cls/within_tissue_study_cls.json
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--emb", required=True)
ap.add_argument("--meta", default="results/pretrained_cls/metadata.csv")
ap.add_argument("--k", type=int, default=10)
ap.add_argument("--out", default=None)
a = ap.parse_args()

m = pd.read_csv(a.meta)
X = np.load(a.emb).astype(np.float64)
assert len(X) == len(m)
X /= np.linalg.norm(X, axis=1, keepdims=True) + 1e-12
ok = (m.dataset != "E-GEOD-56105").values
res = {"embedding": a.emb, "k": a.k, "rule": "tissues with >= 2 studies of >= 100 profiles; E-GEOD-56105 excluded", "tissues": {}}

for t, g in m[ok].groupby("tissue_type"):
    vc = g.dataset.value_counts()
    if (vc >= 100).sum() < 2:
        continue
    idx = g.index.values
    ds = g.dataset.astype(str).values
    sex = g.gender.astype(str).values
    dec = np.floor(pd.to_numeric(g.age, errors="coerce").values / 10)
    S = X[idx] @ X[idx].T
    np.fill_diagonal(S, -np.inf)
    nn = np.argsort(-S, axis=1)[:, : a.k]
    obs = (ds[nn] == ds[:, None]).mean()
    p = vc / vc.sum()
    chance = float(((vc - 1) / (len(g) - 1) * p).sum())
    # matched: neighbours only among same sex and same age decade
    hits, exp, nq = [], [], 0
    cell = pd.Series([f"{s}|{d}" for s, d in zip(sex, dec)])
    valid = (sex != "nan") & ~np.isnan(dec)
    for i in range(len(g)):
        if not valid[i]:
            continue
        cand = np.where(valid & (cell.values == cell.values[i]))[0]
        cand = cand[cand != i]
        if len(cand) < a.k:
            continue
        top = cand[np.argsort(-S[i, cand])[: a.k]]
        hits.append((ds[top] == ds[i]).mean())
        exp.append((ds[cand] == ds[i]).mean())
        nq += 1
    res["tissues"][t] = {"n_profiles": int(len(g)), "n_studies": int(len(vc)),
                         "same_study_fraction": float(obs), "chance": chance,
                         "matched_same_study_fraction": float(np.mean(hits)) if hits else None,
                         "matched_chance": float(np.mean(exp)) if exp else None,
                         "matched_n_queries": nq}
print(json.dumps(res, indent=1))
if a.out:
    Path(a.out).write_text(json.dumps(res, indent=1))
