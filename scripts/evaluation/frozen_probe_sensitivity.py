"""
frozen_probe_sensitivity.py
===========================
Sensitivity of the frozen-representation results (Results section 2.3, Fig. 3c-e) to two data
problems found by the data-structure audit:

  1. E-GEOD-56105 (282 whole-blood profiles). Its downstream values do not correlate with the
     same accessions in the pretraining corpus (r ~ 0.002 for every sampled pair), and its CLS
     embeddings sit far from every other blood study. It is in the probe-fitting partition and is
     one of the 11 leave-one-study-out studies.
  2. Redundant profiles. The frozen probes use the original AltuMAge partition without the
     duplicate filter; 24 duplicate components (49 profiles) span probe fitting and evaluation.

Same probes as extract_pretrain_cls.py and leave_one_dataset_out_probe.py (Ridge alpha=1 on
standardised CLS; LogisticRegression C=1 on classes with >= 30 profiles), recomputed under:
  as_reported      all profiles (must reproduce the manuscript)
  no_56105         E-GEOD-56105 removed everywhere
  no_56105_dedup   additionally one profile per duplicate component (benchmark rule)

The LODO block uses the same rule as leave_one_dataset_out_probe.py (studies with >= 100
profiles, R² summarised over studies with age SD >= 10 years); its no_56105 block equals
results/pretrained_cls/lodo_*.

Pure post-processing of results/pretrained_cls/{embeddings_cls.npy, metadata.csv} and the
downstream duplicate pairs of scripts/data/analyze_full_dataset.py
(results/data_audit/downstream_duplicate_pairs.csv); no GPU.
Writes results/data_audit/frozen_probe_sensitivity.json and lodo_sensitivity.csv.

Usage:  python scripts/evaluation/frozen_probe_sensitivity.py
The manuscript values were produced with the pinned environment (requirements-lock.txt:
scikit-learn 1.8.0, numpy 1.26.4); other versions can move the balanced accuracies in the
third decimal.
"""
import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

REPO = Path(__file__).resolve().parents[2]
BAD = "E-GEOD-56105"


def redundant_ids():
    dup = pd.read_csv(REPO / "results/data_audit/downstream_duplicate_pairs.csv", dtype={"id_a": str, "id_b": str})
    adj = defaultdict(set)
    for a, b in zip(dup.id_a.astype(str), dup.id_b.astype(str)):
        adj[a].add(b); adj[b].add(a)
    seen, out = set(), set()
    for s in sorted(adj):
        if s in seen:
            continue
        q, comp = deque([s]), []
        while q:
            x = q.popleft()
            if x not in seen:
                seen.add(x); comp.append(x); q.extend(adj[x] - seen)
        out.update(sorted(comp)[1:])
    return out


def age_probe(X, age, split):
    tr, te = split != "test", split == "test"
    sc = StandardScaler().fit(X[tr])
    p = Ridge(alpha=1.0).fit(sc.transform(X[tr]), age[tr]).predict(sc.transform(X[te]))
    return {"r2": round(float(r2_score(age[te], p)), 3), "medae": round(float(np.median(np.abs(p - age[te]))), 2),
            "n_fit": int(tr.sum()), "n_eval": int(te.sum())}


def clf_probe(X, labels, split):
    labels = np.asarray(pd.Series(labels).astype(str), dtype=object)
    vc = pd.Series(labels).value_counts()
    keep = np.isin(labels, vc[vc >= 30].index) & (labels != "nan")
    tr, te = keep & (split != "test"), keep & (split == "test")
    le = LabelEncoder().fit(labels[keep])
    sc = StandardScaler().fit(X[tr])
    m = LogisticRegression(max_iter=2000).fit(sc.transform(X[tr]), le.transform(labels[tr]))
    k = int(pd.Series(labels[keep]).nunique())
    return {"balanced_acc": round(float(balanced_accuracy_score(le.transform(labels[te]), m.predict(sc.transform(X[te])))), 3),
            "n_classes": k, "n_eval": int(te.sum())}


def neighbours(X, tissue, dataset, k=10):
    Z = X / np.linalg.norm(X, axis=1, keepdims=True)
    both = t_only = d_only = 0
    for s in range(0, len(Z), 2000):
        sim = Z[s:s + 2000] @ Z.T
        sim[np.arange(sim.shape[0]), s + np.arange(sim.shape[0])] = -np.inf
        nn = np.argpartition(-sim, k, axis=1)[:, :k]
        st = tissue[nn] == tissue[s:s + 2000, None]
        sd = dataset[nn] == dataset[s:s + 2000, None]
        both += (st & sd).sum(); t_only += (st & ~sd).sum(); d_only += (~st & sd).sum()
    n = len(Z) * k
    return {"both": round(100 * both / n, 1), "tissue_only": round(100 * t_only / n, 1), "dataset_only": round(100 * d_only / n, 1)}


def lodo(X, age, ds, min_n=100, min_sd=10.0):
    rng = np.random.default_rng(0)
    idx = rng.permutation(len(X)); cut = int(0.8 * len(X))
    sc = StandardScaler().fit(X[idx[:cut]])
    p = Ridge(alpha=1.0).fit(sc.transform(X[idx[:cut]]), age[idx[:cut]]).predict(sc.transform(X[idx[cut:]]))
    ref = float(r2_score(age[idx[cut:]], p))
    vc = pd.Series(ds).value_counts()
    rows = []
    for name in vc[vc >= min_n].index:
        te = ds == name
        sc = StandardScaler().fit(X[~te])
        pr = Ridge(alpha=1.0).fit(sc.transform(X[~te]), age[~te]).predict(sc.transform(X[te]))
        sd = float(age[te].std())
        rows.append({"dataset": name, "n": int(te.sum()), "age_std": round(sd, 1),
                     "R2_LODO": round(float(r2_score(age[te], pr)), 3) if age[te].var() > 1 else np.nan,
                     "MedAE": round(float(np.median(np.abs(pr - age[te]))), 2)})
    r = pd.DataFrame(rows)
    w = r[(r.age_std >= min_sd) & r.R2_LODO.notna()]
    return r, {"in_distribution_r2": round(ref, 3), "n_studies": int(len(w)), "median_r2": round(float(w.R2_LODO.median()), 3),
               "median_medae": round(float(w.MedAE.median()), 2), "negative_r2": int((w.R2_LODO < 0).sum()),
               "studies": dict(zip(w.dataset, w.R2_LODO))}


def main():
    X = np.load(REPO / "results/pretrained_cls/embeddings_cls.npy").astype(np.float64)
    m = pd.read_csv(REPO / "results/pretrained_cls/metadata.csv")
    assert len(X) == len(m)
    red = redundant_ids()
    masks = {"as_reported": np.ones(len(m), bool),
             "no_56105": (m.dataset != BAD).values,
             "no_56105_dedup": ((m.dataset != BAD) & ~m.sample_id.isin(red)).values}
    out, lodo_tables = {}, []
    for name, k in masks.items():
        Xk, mk = X[k], m[k].reset_index(drop=True)
        split, age = mk.split.values, mk.age.values.astype(float)
        r, s = lodo(Xk, age, mk.dataset.astype(str).values)
        r["condition"] = name; lodo_tables.append(r)
        out[name] = {"n_profiles": int(k.sum()), "age_probe": age_probe(Xk, age, split),
                     "tissue_probe": clf_probe(Xk, mk.tissue_type.values, split),
                     "dataset_probe": clf_probe(Xk, mk.dataset.values, split),
                     "neighbours_10": neighbours(Xk, mk.tissue_type.astype(str).values, mk.dataset.astype(str).values),
                     "lodo": s}
        print(name, json.dumps(out[name]))
    outdir = REPO / "results/data_audit"
    outdir.mkdir(parents=True, exist_ok=True)
    (outdir / "frozen_probe_sensitivity.json").write_text(json.dumps(out, indent=2))
    pd.concat(lodo_tables).to_csv(outdir / "lodo_sensitivity.csv", index=False)


if __name__ == "__main__":
    main()
