"""
probe_external_sparse.py  (CNS task, stage 6 - sparse input on the external cohort)
===================================================================================
Scenario A on GSE215240: the heads of probe_external.py (trained once on the 2,798 Capper reference profiles, full
profiles) applied to the same external profiles seen at fewer CpGs. Masks: make_sparse_masks.py run on the external
h5ad; embeddings: extract_sparse_llama.py / extract_sparse_gpt.py on those masks (scripts/cns/README.md, step 7). Raw
baselines: CpGs outside the mask become NaN and take the training-part mean, as in probe_cns.features.

C is read from external/probe/metrics.csv (the head is the same fit, so no inner CV is repeated). Gate (the script
stops on a failure): at level 100 every representation must reproduce external/probe/pred_<rep>.csv exactly.

Metric: balanced accuracy over the scoring groups. A level's value is the mean over mask seeds; its 95 % CI comes from
a bootstrap over profiles (B = 2,000, same resamples for every representation and level) of each profile's share of
correct calls over the seeds. Paired differences: MethylLlama - MethylGPT and MethylLlama - best raw baseline at that
level (picked on the point estimate, which favours the raw side).

Outputs  outputs/cns/external/sparse/probe/{accuracy.csv, per_seed.csv, differences.csv, report.txt}
  python scripts/cns/probe_external_sparse.py [--levels 100,20,10,5,2,1,0.5] [--seeds 0,1,2]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import StandardScaler

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_labels import EVAL_GROUP  # noqa: E402
from make_sparse_masks import keep_columns, load_masks, n_keep, tag  # noqa: E402
from probe_cns import MAX_ITER  # noqa: E402
from probe_external import EMB, EXT, MODELS, RAW, REPO, aligned, group_ba  # noqa: E402


def raw_transform(kind, Xtr):
    """Fitted on the training part only, as probe_cns.features; returns a function of a beta matrix."""
    mu = np.nanmean(Xtr, 0)
    mu = np.where(np.isfinite(mu), mu, 0.5)
    F = np.where(np.isnan(Xtr), mu, Xtr)
    if kind == "raw_top5k":
        keep = np.argsort(F.var(0))[::-1][:5000]
        return lambda X: np.where(np.isnan(X), mu, X)[:, keep]
    pca = PCA(n_components=min(256, len(F) - 1, F.shape[1]), svd_solver="randomized", random_state=0).fit(F)
    return lambda X: pca.transform(np.where(np.isnan(X), mu, X))


def masked(Xe, order, k):
    out = np.full_like(Xe, np.nan)
    for i in range(len(Xe)):
        keep = keep_columns(order[i], k)
        out[i, keep] = Xe[i, keep]
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capper", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--external", default=str(EXT / "GSE215240.h5ad"))
    p.add_argument("--sparse", default=str(EXT / "sparse"))
    p.add_argument("--full_probe", default=str(EXT / "probe"))
    p.add_argument("--levels", default="100,20,10,5,2,1,0.5")
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--reps", default=",".join(MODELS + RAW))
    p.add_argument("--n_boot", type=int, default=2000)
    a = p.parse_args()
    import anndata as ad
    sp = Path(a.sparse)
    out = sp / "probe"
    out.mkdir(parents=True, exist_ok=True)
    levels = [float(x) for x in a.levels.split(",")]
    seeds = [int(x) for x in a.seeds.split(",")]
    reps = a.reps.split(",")
    Cs = pd.read_csv(Path(a.full_probe) / "metrics.csv").set_index("rep").C

    C = ad.read_h5ad(a.capper, backed="r")
    ref = np.where(C.obs.set == "reference")[0]
    cid = C.obs.index.astype(str).to_numpy()[ref]
    y = C.obs.methylation_class.astype(str).to_numpy()[ref]
    Xc = np.asarray(C.X[:][ref], dtype=np.float32)
    E = ad.read_h5ad(a.external)
    eid = E.obs.index.astype(str).to_numpy()
    Xe = np.asarray(E.X, dtype=np.float32)
    truth = E.obs.eval_group.astype(str).to_numpy()
    scored = truth != ""
    groups = np.array(sorted(set(truth[scored])))
    gi = np.searchsorted(groups, truth[scored])
    to_group = np.vectorize(lambda c: EVAL_GROUP.get(c, c))
    info, _ = load_masks(sp / "masks")
    meta = pd.read_csv(sp / "masks/metadata.csv").sample_id.astype(str).to_numpy()
    if not np.array_equal(meta, eid):
        raise SystemExit("mask row order differs from the external h5ad")

    correct, rows = {}, []
    for rep in reps:
        if rep in EMB:
            Ftr, tf = aligned(EMB[rep][0], cid), None
        else:
            tf = raw_transform(rep, Xc)
            Ftr = tf(Xc)
        sc = StandardScaler().fit(Ftr)
        m = LogisticRegression(C=float(Cs[rep]), max_iter=MAX_ITER).fit(sc.transform(Ftr), y)
        for lv in levels:
            hits = []
            for s in ([0] if lv == 100 else seeds):
                if rep in EMB:
                    Fte = aligned(sp / f"emb_{rep.split('_')[0]}" / tag(lv, s), eid)
                elif lv == 100:
                    Fte = tf(Xe)
                else:
                    order = np.load(sp / "masks" / f"order_s{s}.npy", mmap_mode="r")
                    Fte = tf(masked(Xe, order, n_keep(lv, info["panel"])))
                pred = m.predict(sc.transform(Fte))
                if lv == 100:
                    saved = pd.read_csv(Path(a.full_probe) / f"pred_{rep}.csv").y_pred.astype(str).to_numpy()
                    if not np.array_equal(pred, saved):
                        raise SystemExit(f"gate failed: {rep} differs from pred_{rep}.csv at "
                                         f"{(pred != saved).sum()} profiles")
                ok = (to_group(pred)[scored] == truth[scored]).astype(float)
                hits.append(ok)
                rows.append({"rep": rep, "level": lv, "seed": s, "ba": group_ba(ok, gi, len(groups)), "acc": ok.mean()})
            correct[(rep, lv)] = np.mean(hits, 0)
        print(rep, "done", flush=True)
    S = pd.DataFrame(rows)
    S.to_csv(out / "per_seed.csv", index=False)

    rng = np.random.default_rng(215241)
    n = int(scored.sum())
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.n_boot)])
    boot = {k: np.array([group_ba(v, gi, len(groups), w) for w in W]) for k, v in correct.items()}
    A, D = [], []
    for lv in levels:
        pt = {r: group_ba(correct[(r, lv)], gi, len(groups)) for r in reps}
        for r in reps:
            lo, hi = np.percentile(boot[(r, lv)], [2.5, 97.5])
            A.append({"rep": r, "level": lv, "ba": pt[r], "ci_lo": lo, "ci_hi": hi,
                      "acc": correct[(r, lv)].mean()})
        raw = [r for r in reps if r in RAW]
        best = max(raw, key=pt.get) if raw else None
        for z in ("gpt_cls", best):
            if z in pt and "llama_cls" in pt:
                d = boot[("llama_cls", lv)] - boot[(z, lv)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                D.append({"level": lv, "comparison": f"llama_cls - {z}", "diff": pt["llama_cls"] - pt[z], "ci_lo": lo,
                          "ci_hi": hi, "excludes_zero": bool(lo > 0 or hi < 0)})
    A, D = pd.DataFrame(A), pd.DataFrame(D)
    A.to_csv(out / "accuracy.csv", index=False)
    D.to_csv(out / "differences.csv", index=False)
    lines = [f"External sparse test (Scenario A): {n} profiles, {len(groups)} groups, seeds {seeds}", "",
             A.pivot(index="level", columns="rep", values="ba")[reps].sort_index(ascending=False).round(3).to_string(),
             "", D.round(3).to_string(index=False)]
    open(out / "report.txt", "w").write("\n".join(lines) + "\n")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
