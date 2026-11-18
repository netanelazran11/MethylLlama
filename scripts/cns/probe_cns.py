"""
probe_cns.py  (CNS task, stages 2 and 4)
=========================================
One protocol for every representation of the Capper profiles, so that the comparison between MethylLlama,
MethylGPT and the raw data is decided by the representation alone.

Representations (--reps, comma separated)
  raw_all                  betas of all 49,156 CpGs (the models' full input)
  raw_top5k / raw_top10k   betas of the 5k / 10k CpGs with the highest variance in the TRAINING part of each fold
  raw_pca256               PCA (256 components) fitted on the training part, all 49,156 CpGs
  emb:<name>=<dir>         embeddings from an extraction directory: <dir>/embeddings_cls.npy (or the file named
                           after '#', e.g. emb:llama_mean=<dir>#embeddings_mean.npy) + <dir>/metadata.csv (sample_id)
  Raw NaNs are imputed with the training-part CpG mean. Nothing is fitted on test profiles.

Head (identical for every representation)
  StandardScaler + multinomial LogisticRegression (lbfgs); C from {1e-3, 1e-2, 1e-1, 1, 10, 100} (C_GRID) by stratified 3-fold inner
  CV on the training part (balanced accuracy). Plus a parameter-free 1-NN (cosine after the same scaling).

Settings
  full        the shared 5 x 5 stratified folds (folds.csv); every reference profile predicted once per repeat
  fewshot     k = 3 and 5 profiles per class (fewshot.csv, 5 seeds) from the training part; tested on the fold
  validation  trained on all reference profiles, tested on the validation set minus the 6 profiles that have a
              near-duplicate in the reference (labels there are the Capper classifier's predictions: secondary)

Outputs  outputs/cns/probe_all/<rep>/
  pred_full.csv        repeat, fold, gsm, y_true, y_pred_lr, p_true_lr, y_pred_1nn
  pred_fewshot.csv     repeat, fold, k, seed, gsm, y_true, y_pred_lr, p_true_lr, y_pred_1nn
  pred_validation.csv  gsm, y_true, y_pred_lr, p_true_lr, y_pred_1nn
  metrics.json         balanced accuracy, accuracy, macro-F1, log-loss per setting (mean and sd over repeats/draws),
                       chosen C values
Comparisons between representations (paired bootstrap, breakdowns) are made by compare_cns.py from these files.

CPU only (scikit-learn, anndata; 16 cores, 48 GB for the raw baselines), from the repository root:
  python scripts/cns/probe_cns.py --reps raw_top5k,raw_top10k,raw_pca256
"""
import argparse
import json
import time
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, balanced_accuracy_score, f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "outputs/cns/data"
C_GRID = (1e-3, 1e-2, 1e-1, 1.0, 10.0, 100.0)   # wide enough that no representation's choice sits at the edge
MAX_ITER = 500                                     # separable few-shot fits stop here instead of running to 3,000


# ── representations ───────────────────────────────────────────────────────────
def features(kind, D, tr, te):
    """Fold-local features. kind 'emb': D is the embedding matrix. kind 'raw_*': D is the raw beta matrix; NaN ->
    training-part CpG mean; then top-variance CpGs or PCA fitted on the training part only."""
    Ftr, Fte = D[tr], D[te]
    if kind == "emb":
        return Ftr, Fte
    mu = np.nanmean(Ftr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.5)
    Ftr = np.where(np.isnan(Ftr), mu, Ftr)
    Fte = np.where(np.isnan(Fte), mu, Fte)
    if kind == "raw_all":                       # all 49,156 CpGs - the same input the models receive
        return Ftr, Fte
    if kind.startswith("raw_top"):
        keep = np.argsort(Ftr.var(0))[::-1][:int(kind[len("raw_top"):-1]) * 1000]
        return Ftr[:, keep], Fte[:, keep]
    if kind == "raw_pca256":
        pca = PCA(n_components=min(256, len(tr) - 1, Ftr.shape[1]), svd_solver="randomized", random_state=0).fit(Ftr)
        return pca.transform(Ftr), pca.transform(Fte)
    raise ValueError(kind)


def load_rep(spec, gsm, X):
    """-> (name, kind, matrix aligned to gsm)."""
    if spec.startswith("raw_"):
        return spec, spec, X
    name, path = spec[len("emb:"):].split("=", 1)
    d, fname = (path.split("#", 1) + ["embeddings_cls.npy"])[:2]
    E = np.load(Path(d) / fname).astype(np.float32)
    meta = pd.read_csv(Path(d) / "metadata.csv")
    idx = pd.Series(np.arange(len(meta)), index=meta.sample_id.astype(str))
    if not set(gsm) <= set(idx.index):
        raise SystemExit(f"{name}: embeddings miss {len(set(gsm) - set(idx.index))} profiles")
    E = E[idx.loc[gsm].to_numpy()]
    if not np.isfinite(E).all():
        raise SystemExit(f"{name}: non-finite embedding values")
    return name, "emb", E


# ── head ──────────────────────────────────────────────────────────────────────
def fit_predict(Ftr, ytr, Fte, classes, seed):
    sc = StandardScaler().fit(Ftr)
    Ftr, Fte = sc.transform(Ftr), sc.transform(Fte)
    n_inner = int(min(3, pd.Series(ytr).value_counts().min()))
    best_c, best = C_GRID[0], -1.0
    if n_inner >= 2:
        skf = StratifiedKFold(n_inner, shuffle=True, random_state=seed)
        for c in C_GRID:
            sc_ = []
            for itr, ite in skf.split(Ftr, ytr):
                m = LogisticRegression(C=c, max_iter=MAX_ITER).fit(Ftr[itr], ytr[itr])
                sc_.append(balanced_accuracy_score(ytr[ite], m.predict(Ftr[ite])))
            if np.mean(sc_) > best:
                best, best_c = float(np.mean(sc_)), c
    m = LogisticRegression(C=best_c, max_iter=MAX_ITER).fit(Ftr, ytr)
    P = np.zeros((len(Fte), len(classes)), dtype=np.float32)
    P[:, np.searchsorted(classes, m.classes_)] = m.predict_proba(Fte)
    # 1-NN, cosine on the scaled features
    A = Ftr / (np.linalg.norm(Ftr, axis=1, keepdims=True) + 1e-9)
    B = Fte / (np.linalg.norm(Fte, axis=1, keepdims=True) + 1e-9)
    nn = ytr[np.argmax(B @ A.T, axis=1)]
    return P, nn, best_c


def run_task(kind, D, tr, te, y, classes, seed):
    Ftr, Fte = features(kind, D, tr, te)
    P, nn, c = fit_predict(Ftr, y[tr], Fte, classes, seed)
    return P, nn, c


def summarise(rows):
    df = pd.DataFrame(rows)
    return {k: {"mean": float(df[k].mean()), "sd": float(df[k].std(ddof=1)) if len(df) > 1 else 0.0}
            for k in df.columns if k not in ("repeat", "fold", "k", "seed")}


# ── main ──────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", required=True)
    p.add_argument("--data", default=str(DATA))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/probe_all"))
    p.add_argument("--settings", default="full,fewshot,validation")
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--n_jobs", type=int, default=-1)
    a = p.parse_args()
    import anndata as ad
    A = ad.read_h5ad(Path(a.data) / "cns_capper.h5ad", backed="r")   # obs only unless raw features are needed
    obs = A.obs.copy()
    obs["gsm"] = obs.index.astype(str)
    need_raw = any(s.startswith("raw_") for s in a.reps.split(","))
    X = np.asarray(A.X[:], dtype=np.float32) if need_raw else None
    gsm = obs.gsm.to_numpy()
    y = obs.methylation_class.astype(str).to_numpy()
    ref = np.where(obs.set == "reference")[0]
    val = np.where((obs.set == "validation") & ~obs.val_leak_dup.astype(bool))[0]
    classes = np.array(sorted(set(y[ref])))
    pos = pd.Series(np.arange(len(gsm)), index=gsm)
    folds = pd.read_csv(Path(a.data) / "folds.csv")
    shots = pd.read_csv(Path(a.data) / "fewshot.csv")
    settings = a.settings.split(",")

    for spec in a.reps.split(","):
        t0 = time.time()
        name, kind, D = load_rep(spec, gsm, X)
        out = Path(a.outdir) / name
        out.mkdir(parents=True, exist_ok=True)
        M = {"rep": spec, "n_reference": int(len(ref)), "n_validation_used": int(len(val)),
             "n_classes": int(len(classes)), "C_grid": list(C_GRID), "max_iter": MAX_ITER}

        if "full" in settings:
            tasks = []
            for r in range(a.repeats):
                f = folds[folds.repeat == r]
                for fo in range(int(f.fold.max()) + 1):
                    te = pos.loc[f[f.fold == fo].gsm].to_numpy()
                    tr = np.setdiff1d(ref, te)
                    tasks.append((r, fo, tr, te))
            res = Parallel(n_jobs=a.n_jobs)(delayed(run_task)(kind, D, tr, te, y, classes, 100 * r + fo)
                                            for r, fo, tr, te in tasks)
            rows, preds = [], []
            for (r, fo, tr, te), (P, nn, c) in zip(tasks, res):
                ci = np.searchsorted(classes, y[te])
                preds.append(pd.DataFrame({"repeat": r, "fold": fo, "gsm": gsm[te], "y_true": y[te],
                                           "y_pred_lr": classes[P.argmax(1)], "p_true_lr": P[np.arange(len(te)), ci],
                                           "y_pred_1nn": nn, "C": c}))
            pf = pd.concat(preds)
            pf.to_csv(out / "pred_full.csv", index=False)
            for r in range(a.repeats):          # metrics on the out-of-fold predictions of each repeat
                q = pf[pf.repeat == r]
                rows.append({"repeat": r,
                             "balanced_acc": balanced_accuracy_score(q.y_true, q.y_pred_lr),
                             "acc": accuracy_score(q.y_true, q.y_pred_lr),
                             "macro_f1": f1_score(q.y_true, q.y_pred_lr, average="macro", zero_division=0),
                             "neg_log_p_true": float(-np.log(np.clip(q.p_true_lr, 1e-7, 1)).mean()),
                             "balanced_acc_1nn": balanced_accuracy_score(q.y_true, q.y_pred_1nn)})
            M["full"] = summarise(rows)
            M["full_C_counts"] = pf.groupby(["repeat", "fold"]).C.first().value_counts().to_dict()

        if "fewshot" in settings:
            tasks = []
            for (r, fo, k, sd), g in shots[shots.repeat < a.repeats].groupby(["repeat", "fold", "k", "seed"]):
                te = pos.loc[folds[(folds.repeat == r) & (folds.fold == fo)].gsm].to_numpy()
                tasks.append((r, fo, k, sd, pos.loc[g.gsm].to_numpy(), te))
            res = Parallel(n_jobs=a.n_jobs)(delayed(run_task)(kind, D, tr, te, y, classes, 1000 * r + 100 * fo + 10 * k + sd)
                                            for r, fo, k, sd, tr, te in tasks)
            preds, rows = [], []
            for (r, fo, k, sd, tr, te), (P, nn, c) in zip(tasks, res):
                ci = np.searchsorted(classes, y[te])
                preds.append(pd.DataFrame({"repeat": r, "fold": fo, "k": k, "seed": sd, "gsm": gsm[te],
                                           "y_true": y[te], "y_pred_lr": classes[P.argmax(1)],
                                           "p_true_lr": P[np.arange(len(te)), ci], "y_pred_1nn": nn, "C": c}))
            pfs = pd.concat(preds)
            pfs.to_csv(out / "pred_fewshot.csv", index=False)
            for (r, k, sd), q in pfs.groupby(["repeat", "k", "seed"]):   # one full out-of-fold pass per draw
                rows.append({"repeat": r, "k": k, "seed": sd,
                             "balanced_acc": balanced_accuracy_score(q.y_true, q.y_pred_lr),
                             "balanced_acc_1nn": balanced_accuracy_score(q.y_true, q.y_pred_1nn)})
            fr = pd.DataFrame(rows)
            M["fewshot"] = {int(k): summarise(g.drop(columns=["k"]).to_dict("records")) for k, g in fr.groupby("k")}

        if "validation" in settings:
            P, nn, c = run_task(kind, D, ref, val, y, classes, 7)
            ci = np.searchsorted(classes, y[val])
            pd.DataFrame({"gsm": gsm[val], "y_true": y[val], "y_pred_lr": classes[P.argmax(1)],
                          "p_true_lr": P[np.arange(len(val)), ci], "y_pred_1nn": nn}).to_csv(
                out / "pred_validation.csv", index=False)
            yv = y[val]
            M["validation"] = {"n": int(len(val)), "C": c,
                               "balanced_acc": float(balanced_accuracy_score(yv, classes[P.argmax(1)])),
                               "acc": float(accuracy_score(yv, classes[P.argmax(1)])),
                               "balanced_acc_1nn": float(balanced_accuracy_score(yv, nn))}
        M["seconds"] = round(time.time() - t0, 1)
        json.dump(M, open(out / "metrics.json", "w"), indent=1, default=str)
        print(name, json.dumps({k: M[k] for k in M if k in ("full", "fewshot", "validation", "seconds")},
                               default=str)[:1500], flush=True)


if __name__ == "__main__":
    main()
