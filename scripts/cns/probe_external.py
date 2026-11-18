"""
probe_external.py  (CNS task, stage 6 - external test on GSE215240)
===================================================================
The probe of probe_cns.py (features() and fit_predict() imported unchanged: same head, C grid, 1-NN) trained once
on all 2,798 Capper reference profiles and applied to the external cohort (build_external_cns.py): another
laboratory, another preprocessing, paediatric tumours, profiles that neither model saw in pretraining.

Label: the v11b4 class of the reference classifier with calibrated score >= 0.9 (external_labels.py). A prediction
among the 91 Capper classes is scored after mapping to the scoring group (MB G3/G4 and GBM RTK I/II/III merged);
profiles below the score threshold or without a mapped class are not scored.

Metrics, per representation: balanced accuracy over the scoring groups (main), accuracy, 1-NN balanced accuracy,
and the same split by array (EPIC / 450K). 95 % CI by bootstrap over profiles (B = 2,000, the same resamples for
every representation); paired differences MethylLlama - MethylGPT and MethylLlama - best raw baseline (picked on
the point estimate, which favours the raw side).

Outputs  outputs/cns/external/probe/{pred_<rep>.csv, metrics.csv, differences.csv, report.txt}
  python scripts/cns/probe_external.py [--reps llama_cls,gpt_cls,raw_top5k,raw_pca256] [--n_boot 2000]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from external_labels import EVAL_GROUP  # noqa: E402
from probe_cns import features, fit_predict  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
EXT = REPO / "outputs/cns/external"
EMB = {"llama_cls": (REPO / "outputs/cns/cls_methylllama", EXT / "emb_llama"),
       "gpt_cls": (REPO / "outputs/cns/cls_methylgpt", EXT / "emb_gpt")}
MODELS, RAW = ["llama_cls", "gpt_cls"], ["raw_top5k", "raw_pca256"]


def aligned(d, ids):
    meta = pd.read_csv(Path(d) / "metadata.csv").sample_id.astype(str)
    idx = pd.Series(np.arange(len(meta)), index=meta)
    E = np.load(Path(d) / "embeddings_cls.npy").astype(np.float32)[idx.loc[ids].to_numpy()]
    if not np.isfinite(E).all():
        raise SystemExit(f"{d}: non-finite embeddings")
    return E


def group_ba(correct, g, n_g, w=None):
    """Balanced accuracy over groups; w = bootstrap counts per profile (None = all once)."""
    w = np.ones(len(g)) if w is None else w
    hit = np.bincount(g, weights=w * correct, minlength=n_g)
    tot = np.bincount(g, weights=w, minlength=n_g)
    ok = tot > 0
    return float((hit[ok] / tot[ok]).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--capper", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--external", default=str(EXT / "GSE215240.h5ad"))
    p.add_argument("--reps", default=",".join(MODELS + RAW))
    p.add_argument("--n_boot", type=int, default=2000)
    p.add_argument("--outdir", default=str(EXT / "probe"))
    a = p.parse_args()
    import anndata as ad
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    reps = a.reps.split(",")

    C = ad.read_h5ad(a.capper, backed="r")
    ref = np.where(C.obs.set == "reference")[0]
    cid = C.obs.index.astype(str).to_numpy()[ref]
    y = C.obs.methylation_class.astype(str).to_numpy()[ref]
    classes = np.array(sorted(set(y)))
    E = ad.read_h5ad(a.external)
    eid = E.obs.index.astype(str).to_numpy()
    grp_true = E.obs.eval_group.astype(str).to_numpy()
    scored = grp_true != ""
    platform = E.obs.platform.astype(str).to_numpy()
    groups = np.array(sorted(set(grp_true[scored])))
    gi = np.searchsorted(groups, grp_true[scored])
    to_group = np.vectorize(lambda c: EVAL_GROUP.get(c, c))
    Xc = Xe = None
    if any(r.startswith("raw_") for r in reps):
        Xc = np.asarray(C.X[ref], dtype=np.float32) if not C.isbacked else np.asarray(C.X[:][ref], dtype=np.float32)
        Xe = np.asarray(E.X, dtype=np.float32)

    correct, correct_nn, rows = {}, {}, []
    for rep in reps:
        if rep in EMB:
            Dtr, Dte, kind = aligned(EMB[rep][0], cid), aligned(EMB[rep][1], eid), "emb"
        else:
            Dtr, Dte, kind = Xc, Xe, rep
        D = np.vstack([Dtr, Dte])
        Ftr, Fte = features(kind, D, np.arange(len(Dtr)), np.arange(len(Dtr), len(D)))
        P, nn, c = fit_predict(Ftr, y, Fte, classes, 7)
        pred = classes[P.argmax(1)]
        pg, nng = to_group(pred), to_group(nn)
        pd.DataFrame({"gsm": eid, "platform": platform, "eval_group": grp_true, "y_pred": pred, "pred_group": pg,
                      "p_max": P.max(1), "y_pred_1nn": nn}).to_csv(out / f"pred_{rep}.csv", index=False)
        correct[rep] = (pg[scored] == grp_true[scored]).astype(float)
        correct_nn[rep] = (nng[scored] == grp_true[scored]).astype(float)
        r = {"rep": rep, "C": c, "n_scored": int(scored.sum()), "n_groups": len(groups),
             "ba": group_ba(correct[rep], gi, len(groups)), "acc": float(correct[rep].mean()),
             "ba_1nn": group_ba(correct_nn[rep], gi, len(groups))}
        for plat, tag in (("GPL21145", "epic"), ("GPL13534", "450k")):
            m = platform[scored] == plat
            r[f"ba_{tag}"] = group_ba(correct[rep][m], gi[m], len(groups))
            r[f"n_{tag}"] = int(m.sum())
        rows.append(r)
        print(json.dumps(r), flush=True)

    rng = np.random.default_rng(215240)
    n = int(scored.sum())
    W = np.stack([np.bincount(rng.integers(0, n, n), minlength=n) for _ in range(a.n_boot)])
    boot = {rep: np.array([group_ba(correct[rep], gi, len(groups), w) for w in W]) for rep in reps}
    M = pd.DataFrame(rows)
    M["ci_lo"] = [np.percentile(boot[r], 2.5) for r in M.rep]
    M["ci_hi"] = [np.percentile(boot[r], 97.5) for r in M.rep]
    M.to_csv(out / "metrics.csv", index=False)

    D = []
    raw = [r for r in reps if r in RAW]
    best_raw = max(raw, key=lambda r: float(M.set_index("rep").ba[r])) if raw else None
    for x, z in (("llama_cls", "gpt_cls"), ("llama_cls", best_raw)):
        if x in boot and z in boot:
            d = boot[x] - boot[z]
            est = float(M.set_index("rep").ba[x] - M.set_index("rep").ba[z])
            lo, hi = np.percentile(d, [2.5, 97.5])
            D.append({"comparison": f"{x} - {z}", "diff": est, "ci_lo": lo, "ci_hi": hi,
                      "excludes_zero": bool(lo > 0 or hi < 0)})
    pd.DataFrame(D).to_csv(out / "differences.csv", index=False)

    lines = [f"External test GSE215240: {n} profiles scored (v11b4 score >= 0.9) in {len(groups)} groups; "
             f"head trained on {len(ref)} Capper reference profiles", ""]
    for _, r in M.iterrows():
        lines.append(f"  {r.rep:11s} BA {r.ba:.3f} [{r.ci_lo:.3f},{r.ci_hi:.3f}]  acc {r.acc:.3f}  1-NN BA {r.ba_1nn:.3f}"
                     f"  EPIC {r.ba_epic:.3f} (n={r.n_epic})  450K {r.ba_450k:.3f} (n={r.n_450k})  C={r.C:g}")
    lines.append("")
    for d in D:
        lines.append(f"  {d['comparison']:28s} {d['diff']:+.3f} [{d['ci_lo']:+.3f},{d['ci_hi']:+.3f}]"
                     f"{' *' if d['excludes_zero'] else ''}")
    (out / "report.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
