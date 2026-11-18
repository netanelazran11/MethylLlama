"""
probe_external_sparse_B.py  (CNS task, stage 6 - external cohort, head trained on sparse profiles)
=================================================================================================
Scenario B on GSE215240: at each level and mask seed, a head is trained on the 2,798 Capper reference profiles seen at
that level (the existing Capper masks and sparse embeddings) and tested on the external profiles at the same level
(external masks and embeddings). Nothing is re-embedded. Raw baselines: probe_cns.features on the masked betas, so CpG
selection, imputation means and PCA come from the sparse training part.

C: chosen per cell by inner 3-fold CV on the sparse Capper training profiles (same folds and grid as probe_cns.fit_predict;
the external profiles take no part). Every C of the grid is also fitted and scored on the external cohort, for the
sensitivity table only (c_sensitivity.csv): a head trained at a sparse level is sensitive to C on external data.
Also writes the 14-category accuracy of Scenario A (full-profile heads, as probe_external_sparse.py) per level.

Outputs  outputs/cns/external/sparse/probe_B/{per_seed.csv, c_sensitivity.csv, accuracy.csv, differences.csv, report.txt},
         outputs/cns/external/sparse/probe/category_accuracy.csv
  OMP_NUM_THREADS=1 python scripts/cns/probe_external_sparse_B.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_labels import EVAL_GROUP  # noqa: E402
from fig_cns_landscape import CATEGORY, HGNET  # noqa: E402
from make_sparse_masks import n_keep, tag  # noqa: E402
from probe_cns import C_GRID, MAX_ITER, features  # noqa: E402
from probe_external import EMB, EXT, MODELS, RAW, REPO, aligned, group_ba  # noqa: E402
from probe_external_sparse import masked  # noqa: E402

CS = REPO / "outputs/cns/sparse"
LEVELS = (20.0, 10.0, 5.0, 2.0, 1.0)
SEEDS = (0, 1, 2)
PANEL = 49156
to_group = np.vectorize(lambda c: EVAL_GROUP.get(c, c))


def fit_all(Ftr, y, Fte):
    """Inner-CV score and external prediction for every C of the grid (head as probe_cns.fit_predict, seed 7)."""
    sc = StandardScaler().fit(Ftr)
    Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    skf = StratifiedKFold(int(min(3, pd.Series(y).value_counts().min())), shuffle=True, random_state=7)
    cv, pred = {}, {}
    for c in C_GRID:
        cv[c] = np.mean([balanced_accuracy_score(y[ite], LogisticRegression(C=c, max_iter=MAX_ITER).fit(
            Ftr[itr], y[itr]).predict(Ftr[ite])) for itr, ite in skf.split(Ftr, y)])
        pred[c] = LogisticRegression(C=c, max_iter=MAX_ITER).fit(Ftr, y).predict(Fte)
    return cv, pred


def cell(rep, lv, s, cid, eid, y, Xc, Xe, ref):
    if rep in EMB:
        short = rep.split("_")[0]
        Ftr = aligned(CS / f"emb_{short}" / tag(lv, s), cid)
        Fte = aligned(EXT / f"sparse/emb_{short}" / tag(lv, s), eid)
    else:
        k = n_keep(lv, PANEL)
        Mc = masked(Xc, np.load(CS / f"masks/order_s{s}.npy", mmap_mode="r")[ref], k)
        Me = masked(Xe, np.load(EXT / f"sparse/masks/order_s{s}.npy", mmap_mode="r"), k)
        D = np.vstack([Mc, Me])
        Ftr, Fte = features(rep, D, np.arange(len(Mc)), np.arange(len(Mc), len(D)))
    return (rep, lv, s) + fit_all(Ftr, y, Fte)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--n_jobs", type=int, default=4)
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args()
    import anndata as ad
    out = EXT / "sparse/probe_B"
    out.mkdir(parents=True, exist_ok=True)
    C = ad.read_h5ad(REPO / "outputs/cns/data/cns_capper.h5ad", backed="r")
    ref = np.where(C.obs.set == "reference")[0]
    cid = C.obs.index.astype(str).to_numpy()[ref]
    y = C.obs.methylation_class.astype(str).to_numpy()[ref]
    Xc = np.asarray(C.X[:][ref], dtype=np.float32)
    pre = dict(zip(C.obs.methylation_class.astype(str), C.obs.prefix_group.astype(str)))
    cat = {c: HGNET.get(c) or CATEGORY[f] for c, f in pre.items()}
    for k, v in EVAL_GROUP.items():
        cat[v] = cat[k]
    E = ad.read_h5ad(EXT / "GSE215240.h5ad")
    eid = E.obs.index.astype(str).to_numpy()
    Xe = np.asarray(E.X, dtype=np.float32)
    truth = E.obs.eval_group.astype(str).to_numpy()
    sc = truth != ""
    groups = np.array(sorted(set(truth[sc])))
    gi = np.searchsorted(groups, truth[sc])
    reps = MODELS + RAW
    jobs = [(r, lv, s) for r in reps for lv in LEVELS for s in SEEDS]
    res = Parallel(n_jobs=a.n_jobs, verbose=5)(delayed(cell)(r, lv, s, cid, eid, y, Xc, Xe, ref) for r, lv, s in jobs)
    hits, rows, sens = {}, [], []
    for r, lv, s, cv, pred in res:
        for c in C_GRID:
            ok = (to_group(pred[c])[sc] == truth[sc]).astype(float)
            sens.append({"rep": r, "level": lv, "seed": s, "C": c, "inner_cv_ba": cv[c],
                         "external_ba": group_ba(ok, gi, len(groups))})
        best = max(C_GRID, key=lambda c: cv[c])
        pg = to_group(pred[best])[sc]
        ok = (pg == truth[sc]).astype(float)
        hits.setdefault((r, lv), []).append(ok)
        rows.append({"rep": r, "level": lv, "seed": s, "C": best, "ba": group_ba(ok, gi, len(groups)),
                     "acc": ok.mean(), "cat_acc": np.mean([cat[q] == cat[t] for q, t in zip(pg, truth[sc])])})
    pd.DataFrame(sens).to_csv(out / "c_sensitivity.csv", index=False)
    S = pd.DataFrame(rows)
    S.to_csv(out / "per_seed.csv", index=False)
    correct = {k: np.mean(v, 0) for k, v in hits.items()}
    rng = np.random.default_rng(215242)
    n = int(sc.sum())
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.n_boot)])
    boot = {k: np.array([group_ba(v, gi, len(groups), w) for w in W]) for k, v in correct.items()}
    A, D = [], []
    for lv in LEVELS:
        pt = {r: group_ba(correct[(r, lv)], gi, len(groups)) for r in reps}
        for r in reps:
            lo, hi = np.percentile(boot[(r, lv)], [2.5, 97.5])
            A.append({"rep": r, "level": lv, "ba": pt[r], "ci_lo": lo, "ci_hi": hi, "acc": correct[(r, lv)].mean(),
                      "cat_acc": S[(S.rep == r) & (S.level == lv)].cat_acc.mean()})
        for z in ("gpt_cls", max(RAW, key=pt.get)):
            d = boot[("llama_cls", lv)] - boot[(z, lv)]
            lo, hi = np.percentile(d, [2.5, 97.5])
            D.append({"level": lv, "comparison": f"llama_cls - {z}", "diff": pt["llama_cls"] - pt[z], "ci_lo": lo,
                      "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    A, D = pd.DataFrame(A), pd.DataFrame(D)
    A.to_csv(out / "accuracy.csv", index=False)
    D.to_csv(out / "differences.csv", index=False)
    lines = ["External test, head trained at the same sparse level (Scenario B), BA", "",
             A.pivot(index="level", columns="rep", values="ba")[reps].sort_index(ascending=False).round(3).to_string(),
             "", D.round(3).to_string(index=False)]
    open(out / "report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)

    # 14-category accuracy, Scenario A
    from probe_external_sparse import raw_transform
    Cs = pd.read_csv(EXT / "probe/metrics.csv").set_index("rep").C
    R = []
    for r in reps:
        if r in EMB:
            Ftr, tf = aligned(EMB[r][0], cid), None
        else:
            tf = raw_transform(r, Xc)
            Ftr = tf(Xc)
        s0 = StandardScaler().fit(Ftr)
        m = LogisticRegression(C=float(Cs[r]), max_iter=MAX_ITER).fit(s0.transform(Ftr), y)
        for lv in (100.0,) + LEVELS + (0.5,):
            for s in ([0] if lv == 100 else SEEDS):
                if r in EMB:
                    Fte = aligned(EXT / f"sparse/emb_{r.split('_')[0]}" / tag(lv, s), eid)
                elif lv == 100:
                    Fte = tf(Xe)
                else:
                    Fte = tf(masked(Xe, np.load(EXT / f"sparse/masks/order_s{s}.npy", mmap_mode="r"),
                                    n_keep(lv, PANEL)))
                pg = to_group(m.predict(s0.transform(Fte)))[sc]
                R.append({"rep": r, "level": lv, "seed": s, "cat_acc": np.mean([cat[q] == cat[t] for q, t in
                                                                                 zip(pg, truth[sc])])})
    Q = pd.DataFrame(R).groupby(["rep", "level"]).cat_acc.mean().unstack(0)[reps].sort_index(ascending=False)
    Q.to_csv(EXT / "sparse/probe/category_accuracy.csv")
    print(Q.round(3).to_string(), flush=True)


if __name__ == "__main__":
    main()
