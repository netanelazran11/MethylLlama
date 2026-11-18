"""
probe_sparse.py  (CNS task, stages 3-4 - linear probe on sparse profiles)
=========================================================================
The probe of probe_cns.py (same folds, few-shot draws, head, C grid, 1-NN; features() and fit_predict() imported
unchanged) applied to profiles that show only the CpGs kept by the sparse masks.

Scenarios
  A (primary)    head trained on FULL-profile representations of the training part, tested on SPARSE profiles of
                 the test part: a full reference atlas, a sparse new sample.
  B (secondary)  trained and tested at the same sparse level.

Representations (--reps, comma separated)
  <name>=<emb root>[#file]   sparse embeddings <root>/<L.._s..>/<file> (default embeddings_cls.npy); the full-profile
                             ones are <root>/L100 (written by the same extraction script)
  raw_all, raw_top5k, raw_top32k, raw_pca256
                             betas; unselected CpGs are missing and imputed, like every missing value, with the
                             training-part CpG mean (probe_cns.features)

Checks: no test profile in its training set (asserted per task); --shuffle_labels gives the label-shuffle control
(labels permuted within the reference and within the validation set; expected about 1/91).

Outputs  outputs/cns/sparse/probe/<A|B>/<rep>/<L100 | L5_s0 | ...>[_shuffled]/
  pred_full.csv, pred_fewshot.csv, pred_validation.csv, metrics.json   (columns as probe_cns.py)

CPU, from the repository root:
  python scripts/cns/probe_sparse.py --scenario A --levels 100,5 --seeds 0 \\
      --reps llama_cls=outputs/cns/sparse/emb_llama,gpt_cls=outputs/cns/sparse/emb_gpt,raw_pca256,raw_top5k
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.metrics import balanced_accuracy_score

sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sparse_masks import keep_columns, load_masks, n_keep, tag  # noqa: E402
from probe_cns import C_GRID, MAX_ITER, features, fit_predict, load_rep  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "outputs/cns/data"


def run(kind, Dtr_src, Dte_src, tr, te, y, classes, seed):
    """Features fitted on the training part of Dtr_src, applied to the test part of Dte_src."""
    assert np.intersect1d(tr, te).size == 0, "test profile in the training set"
    Z = np.vstack([Dtr_src[tr], Dte_src[te]])
    Ftr, Fte = features(kind, Z, np.arange(len(tr)), np.arange(len(tr), len(tr) + len(te)))
    return fit_predict(Ftr, y[tr], Fte, classes, seed)


def sparse_raw(X, order, k):
    Xs = np.full_like(X, np.nan)
    for i in range(len(X)):
        keep = keep_columns(order[i], k)
        Xs[i, keep] = X[i, keep]
    return Xs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--scenario", choices=["A", "B"], default="A")
    p.add_argument("--reps", required=True)
    p.add_argument("--levels", default="100,5")
    p.add_argument("--seeds", default="0")
    p.add_argument("--settings", default="full,fewshot,validation")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--shuffle_labels", action="store_true")
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--masks", default=str(REPO / "outputs/cns/sparse/masks"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/sparse/probe"))
    p.add_argument("--n_jobs", type=int, default=-1)
    a = p.parse_args()
    import anndata as ad
    A = ad.read_h5ad(Path(a.data) / "cns_capper.h5ad", backed="r")
    obs = A.obs.copy()
    gsm = obs.index.astype(str).to_numpy()
    y = obs.methylation_class.astype(str).to_numpy()
    ref = np.where(obs.set == "reference")[0]
    val = np.where((obs.set == "validation") & ~obs.val_leak_dup.astype(bool))[0]
    classes = np.array(sorted(set(y[ref])))
    if a.shuffle_labels:
        rng = np.random.default_rng(12345)
        y = y.copy()
        y[ref], y[val] = rng.permutation(y[ref]), rng.permutation(y[val])
    pos = pd.Series(np.arange(len(gsm)), index=gsm)
    folds = pd.read_csv(Path(a.data) / "folds.csv")
    shots = pd.read_csv(Path(a.data) / "fewshot.csv")
    info, _ = load_masks(a.masks)
    if pd.read_csv(Path(a.masks) / "metadata.csv").sample_id.astype(str).tolist() != list(gsm):
        raise SystemExit("h5ad row order differs from the mask file")
    specs = a.reps.split(",")
    X = np.asarray(A.X[:], dtype=np.float32) if any(s.startswith("raw_") for s in specs) else None
    settings = a.settings.split(",")

    for spec in specs:
        if spec.startswith("raw_"):
            name, kind, root, fname = spec, spec, None, None
            D_full = X
        else:
            name, path = spec.split("=", 1)
            root, fname = (path.split("#", 1) + ["embeddings_cls.npy"])[:2]
            kind = "emb"
            D_full = load_rep(f"emb:{name}={root}/L100#{fname}", gsm, None)[2]
        for level in [float(x) for x in a.levels.split(",")]:
            for seed in ([0] if level == 100 else [int(s) for s in a.seeds.split(",")]):
                t0 = time.time()
                lt = tag(level, seed)
                if kind == "emb":
                    D_sp = D_full if level == 100 else load_rep(f"emb:{name}={root}/{lt}#{fname}", gsm, None)[2]
                else:
                    D_sp = X if level == 100 else sparse_raw(
                        X, np.load(Path(a.masks) / f"order_s{seed}.npy", mmap_mode="r"), n_keep(level, info["panel"]))
                D_tr = D_full if a.scenario == "A" else D_sp
                out = Path(a.outdir) / a.scenario / name / (lt + ("_shuffled" if a.shuffle_labels else ""))
                out.mkdir(parents=True, exist_ok=True)
                M = {"rep": spec, "scenario": a.scenario, "level": level, "seed": seed, "C_grid": list(C_GRID),
                     "max_iter": MAX_ITER, "shuffled_labels": a.shuffle_labels}

                if "full" in settings:
                    tasks = []
                    for r in range(a.repeats):
                        f = folds[folds.repeat == r]
                        for fo in range(int(f.fold.max()) + 1):
                            te = pos.loc[f[f.fold == fo].gsm].to_numpy()
                            tasks.append((r, fo, np.setdiff1d(ref, te), te))
                    res = Parallel(n_jobs=a.n_jobs)(delayed(run)(kind, D_tr, D_sp, tr, te, y, classes, 100 * r + fo)
                                                    for r, fo, tr, te in tasks)
                    pf = pd.concat([pd.DataFrame({"repeat": r, "fold": fo, "gsm": gsm[te], "y_true": y[te],
                                                  "y_pred_lr": classes[P.argmax(1)],
                                                  "p_true_lr": P[np.arange(len(te)), np.searchsorted(classes, y[te])],
                                                  "y_pred_1nn": nn, "C": c})
                                    for (r, fo, tr, te), (P, nn, c) in zip(tasks, res)])
                    pf.to_csv(out / "pred_full.csv", index=False)
                    ba = [balanced_accuracy_score(q.y_true, q.y_pred_lr) for _, q in pf.groupby("repeat")]
                    nn1 = [balanced_accuracy_score(q.y_true, q.y_pred_1nn) for _, q in pf.groupby("repeat")]
                    M["full"] = {"balanced_acc": float(np.mean(ba)), "sd": float(np.std(ba, ddof=1)),
                                 "balanced_acc_1nn": float(np.mean(nn1)),
                                 "C_counts": pf.groupby(["repeat", "fold"]).C.first().value_counts().to_dict()}

                if "fewshot" in settings:
                    tasks = []
                    for (r, fo, k, sd), g in shots[shots.repeat < a.repeats].groupby(["repeat", "fold", "k", "seed"]):
                        te = pos.loc[folds[(folds.repeat == r) & (folds.fold == fo)].gsm].to_numpy()
                        tasks.append((r, fo, k, sd, pos.loc[g.gsm].to_numpy(), te))
                    res = Parallel(n_jobs=a.n_jobs)(
                        delayed(run)(kind, D_tr, D_sp, tr, te, y, classes, 1000 * r + 100 * fo + 10 * k + sd)
                        for r, fo, k, sd, tr, te in tasks)
                    pfs = pd.concat([pd.DataFrame({"repeat": r, "fold": fo, "k": k, "seed": sd, "gsm": gsm[te],
                                                   "y_true": y[te], "y_pred_lr": classes[P.argmax(1)],
                                                   "p_true_lr": P[np.arange(len(te)), np.searchsorted(classes, y[te])],
                                                   "y_pred_1nn": nn, "C": c})
                                     for (r, fo, k, sd, tr, te), (P, nn, c) in zip(tasks, res)])
                    pfs.to_csv(out / "pred_fewshot.csv", index=False)
                    M["fewshot"] = {int(k): float(np.mean([balanced_accuracy_score(q.y_true, q.y_pred_lr)
                                                          for _, q in g.groupby(["repeat", "seed"])]))
                                    for k, g in pfs.groupby("k")}

                if "validation" in settings:
                    P, nn, c = run(kind, D_tr, D_sp, ref, val, y, classes, 7)
                    pd.DataFrame({"gsm": gsm[val], "y_true": y[val], "y_pred_lr": classes[P.argmax(1)],
                                  "p_true_lr": P[np.arange(len(val)), np.searchsorted(classes, y[val])],
                                  "y_pred_1nn": nn}).to_csv(out / "pred_validation.csv", index=False)
                    M["validation"] = {"n": int(len(val)), "C": c,
                                       "balanced_acc": float(balanced_accuracy_score(y[val], classes[P.argmax(1)]))}
                M["seconds"] = round(time.time() - t0, 1)
                json.dump(M, open(out / "metrics.json", "w"), indent=1, default=str)
                print(a.scenario, name, lt, json.dumps({k: M[k] for k in ("full", "fewshot", "validation", "seconds")
                                                        if k in M}, default=str), flush=True)


if __name__ == "__main__":
    main()
