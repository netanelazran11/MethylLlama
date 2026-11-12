#!/usr/bin/env python3
"""
probe_cis.py
============
Frozen-probe point estimates and 95 % bootstrap CIs for Fig. 3c,d.

Probes are exactly those of scripts/evaluation/extract_pretrain_cls.py:
  age      Ridge(alpha=1) on standardised features, fit on train+valid (8,724), evaluated on test (2,264)
  tissue / dataset  LogisticRegression(max_iter=2000), classes with >= 30 profiles, balanced accuracy on test
CI: percentile bootstrap over evaluation profiles (10,000 resamples, seed 0); the fitted probe is fixed,
so the interval reflects evaluation-set uncertainty.

Must run with the pinned environment (requirements-lock.txt: scikit-learn 1.8.0, numpy 1.26.4, Linux);
other versions can move the balanced accuracies in the third decimal. From the repository root:
  python scripts/evaluation/probe_cis.py
It asserts that the point estimates equal the manuscript values before writing anything.
Writes results/pretrained_cls/probe_cis.json.
"""
import json
from pathlib import Path

import numpy as np
import pandas as pd
import sklearn
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.metrics import balanced_accuracy_score, r2_score
from sklearn.preprocessing import LabelEncoder, StandardScaler

R = Path("results/pretrained_cls")
MANUSCRIPT = {"age_cls": 0.866, "age_mean": 0.795, "tissue_type": 0.936, "dataset": 0.956}
NB, SEED = 10000, 0


def boot(stat, n, rng):
    return np.array([stat(rng.integers(0, n, n)) for _ in range(NB)])


def main():
    X = np.load(R / "embeddings_cls.npy").astype(np.float64)
    Xm = np.load(R / "embeddings_mean.npy").astype(np.float64)
    m = pd.read_csv(R / "metadata.csv")
    split = m.split.astype(str).values
    age = m.age.values.astype(float)
    rng = np.random.default_rng(SEED)
    out = {"env": {"sklearn": sklearn.__version__, "numpy": np.__version__}, "n_boot": NB, "seed": SEED}

    tr = (split != "test") & ~np.isnan(age)
    te = (split == "test") & ~np.isnan(age)
    for key, F in (("age_cls", X), ("age_mean", Xm)):
        sc = StandardScaler().fit(F[tr])
        p = Ridge(alpha=1.0).fit(sc.transform(F[tr]), age[tr]).predict(sc.transform(F[te]))
        y = age[te]
        b = boot(lambda i: r2_score(y[i], p[i]), len(y), rng)
        out[key] = {"r2": float(r2_score(y, p)), "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                    "n_fit": int(tr.sum()), "n_eval": int(te.sum())}

    for c in ("tissue_type", "dataset"):
        labels = np.asarray(pd.Series(m[c]).astype(str), dtype=object)
        vc = pd.Series(labels).value_counts()
        keep = np.isin(labels, vc[vc >= 30].index) & (labels != "nan")
        trc, tec = keep & (split != "test"), keep & (split == "test")
        le = LabelEncoder().fit(labels[keep])
        sc = StandardScaler().fit(X[trc])
        mdl = LogisticRegression(max_iter=2000).fit(sc.transform(X[trc]), le.transform(labels[trc]))
        y, p = le.transform(labels[tec]), mdl.predict(sc.transform(X[tec]))
        b = boot(lambda i: balanced_accuracy_score(y[i], p[i]), len(y), rng)
        k = int(pd.Series(labels[keep]).nunique())
        out[c] = {"balanced_acc": float(balanced_accuracy_score(y, p)),
                  "ci95": [float(np.percentile(b, 2.5)), float(np.percentile(b, 97.5))],
                  "n_classes": k, "chance": 1.0 / k, "n_eval": int(tec.sum()),
                  "n_labels_total": int(m[c].nunique()), "min_class_size": 30}

    for k, v in MANUSCRIPT.items():
        got = out[k]["r2"] if k.startswith("age") else out[k]["balanced_acc"]
        assert round(got, 3) == v, f"{k}: {got:.4f} does not reproduce the manuscript value {v}"
    (R / "probe_cis.json").write_text(json.dumps(out, indent=1))
    print(json.dumps(out, indent=1))


if __name__ == "__main__":
    main()
