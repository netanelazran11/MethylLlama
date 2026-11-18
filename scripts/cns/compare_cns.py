"""
compare_cns.py  (CNS task, stage 5)
====================================
Turns the per-profile predictions of probe_cns.py into the comparison the task was designed for, and checks the
success criteria fixed before running (CNS plan, 2026-10-25):
  1. MethylLlama balanced accuracy > MethylGPT, 95 % CI of the difference excluding zero, full data AND 5-shot.
  2. In 5-shot, MethylLlama > the best raw-data baseline (same CI rule).
  3. Otherwise: report no advantage.

Statistics
  balanced accuracy (BA) = mean per-class recall over the 91 reference classes. Each repeat (and each few-shot draw)
  predicts every reference profile once; a representation's BA is the mean over repeats (and draws).
  Paired bootstrap over PROFILES (2,000 resamples, the same resample for both representations): in each resample the
  BA of each repeat/draw is recomputed with the profiles' resample weights and averaged -> CI of the difference.

Breakdowns (same statistic, restricted to a subset of profiles; BA over the classes present in the subset)
  small_classes      reference class size < 20
  mixed_material     only the 56 classes that have FFPE and frozen profiles, split into FFPE / frozen
  pretrain_seen      profiles in the pretraining train split vs not (valid / held-out)
  high_nan           NaN fraction > 5 % vs <= 5 %
  tumour_only        the 82 tumour classes (control-tissue classes removed)
  family             accuracy at prefix_group level (prediction counted correct if its family matches)
  validation         train on reference, test on validation minus the 6 duplicated profiles (secondary)

Representation geometry (for embedding directories): effective rank, mean pairwise cosine, within- vs
between-class mean cosine on raw and on standardised embeddings.

Outputs  outputs/cns/compare/{comparison.json, breakdowns.csv, report.txt}
CPU only:  python scripts/cns/compare_cns.py --reps llama_cls,gpt_cls,llama_mean,gpt_mean,raw_top5k,raw_top10k,raw_pca256
"""
import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
B = 2000


def ba_weighted(correct, cls_idx, w, n_cls):
    """Balanced accuracy with profile weights w (bootstrap counts). correct: [R, n] 0/1; cls_idx: [n]."""
    tot = np.bincount(cls_idx, weights=w, minlength=n_cls)
    present = tot > 0
    out = []
    for r in range(correct.shape[0]):
        hit = np.bincount(cls_idx, weights=w * correct[r], minlength=n_cls)
        out.append((hit[present] / tot[present]).mean())
    return float(np.mean(out))


def correctness(pred, gsm_order, col="y_pred_lr", by=("repeat",)):
    """-> [R, n] matrix of 0/1, rows = repeats (or repeat x seed), columns = profiles in gsm_order."""
    rows = []
    for _, g in pred.groupby(list(by)):
        s = g.set_index("gsm")
        rows.append((s.loc[gsm_order, col] == s.loc[gsm_order, "y_true"]).to_numpy(float))
    return np.vstack(rows)


def paired(cA, cB, cls_idx, n_cls, mask, rng):
    idx = np.where(mask)[0]
    ci = cls_idx[idx]
    a, b = cA[:, idx], cB[:, idx]
    w1 = np.ones(len(idx))
    est_a, est_b = ba_weighted(a, ci, w1, n_cls), ba_weighted(b, ci, w1, n_cls)
    diffs = np.empty(B)
    for i in range(B):
        w = np.bincount(rng.integers(0, len(idx), len(idx)), minlength=len(idx)).astype(float)
        diffs[i] = ba_weighted(a, ci, w, n_cls) - ba_weighted(b, ci, w, n_cls)
    lo, hi = np.percentile(diffs, [2.5, 97.5])
    return {"A": est_a, "B": est_b, "diff": est_a - est_b, "ci95": [float(lo), float(hi)],
            "excludes_zero": bool(lo > 0 or hi < 0), "n_profiles": int(len(idx))}


def geometry(E, y):
    Ec = E - E.mean(0)
    s = np.linalg.svd(Ec, compute_uv=False)
    p = s ** 2 / (s ** 2).sum()
    out = {"effective_rank": float(np.exp(-(p * np.log(p + 1e-12)).sum()))}
    for tag, M in (("raw", E), ("standardised", (E - E.mean(0)) / (E.std(0) + 1e-9))):
        Mn = M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-9)
        G = Mn @ Mn.T
        same = y[:, None] == y[None, :]
        iu = np.triu(np.ones_like(G, dtype=bool), 1)
        out[f"{tag}_cos_mean"] = float(G[iu].mean())
        out[f"{tag}_cos_within_class"] = float(G[iu & same].mean())
        out[f"{tag}_cos_between_class"] = float(G[iu & ~same].mean())
    return out


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--reps", required=True)
    p.add_argument("--probe", default=str(REPO / "outputs/cns/probe_all"))
    p.add_argument("--data", default=str(REPO / "outputs/cns/data"))
    p.add_argument("--emb_dirs", default=f"llama={REPO}/outputs/cns/cls_methylllama,"
                                         f"gpt={REPO}/outputs/cns/cls_methylgpt")
    p.add_argument("--primary", default="llama_cls,gpt_cls")
    p.add_argument("--raw", default="raw_top5k,raw_top10k,raw_pca256")
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/compare"))
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(0)
    import anndata as ad
    obs = ad.read_h5ad(Path(a.data) / "cns_capper.h5ad", backed="r").obs.copy()
    obs["gsm"] = obs.index.astype(str)
    ref = obs[obs.set == "reference"].copy()
    gsm = ref.gsm.to_numpy()
    classes = np.array(sorted(ref.methylation_class.astype(str).unique()))
    cls_idx = np.searchsorted(classes, ref.methylation_class.astype(str).to_numpy())
    size = ref.methylation_class.astype(str).map(ref.methylation_class.astype(str).value_counts())
    mixed = ref.groupby("methylation_class", observed=True).material.nunique()
    mixed_cls = set(mixed[mixed > 1].index.astype(str))
    in_mixed = ref.methylation_class.astype(str).isin(mixed_cls).to_numpy()
    ctrl = ref.methylation_class.astype(str).str.startswith("CONTR").to_numpy()
    masks = {
        "all": np.ones(len(ref), bool),
        "small_classes_lt20": (size < 20).to_numpy(),
        "mixed_material_FFPE": in_mixed & (ref.material == "FFPE").to_numpy(),
        "mixed_material_frozen": in_mixed & (ref.material == "Frozen").to_numpy(),
        "pretrain_train": (ref.pretrain_split == "train").to_numpy(),
        "pretrain_not_train": (ref.pretrain_split != "train").to_numpy(),
        "nan_le_5pct": (ref.nan_frac <= 0.05).to_numpy(),
        "nan_gt_5pct": (ref.nan_frac > 0.05).to_numpy(),
        "tumour_only": ~ctrl,
    }
    fam = dict(zip(ref.methylation_class.astype(str), ref.prefix_group.astype(str)))

    reps = a.reps.split(",")
    C_full, C_shot, R = {}, {}, {}
    for r in reps:
        d = Path(a.probe) / r
        pf = pd.read_csv(d / "pred_full.csv")
        C_full[r] = correctness(pf, gsm)
        pf["fam_ok"] = pf.y_pred_lr.map(fam) == pf.y_true.map(fam)
        ps = pd.read_csv(d / "pred_fewshot.csv") if (d / "pred_fewshot.csv").exists() else None
        if ps is not None:
            for k in sorted(ps.k.unique()):
                C_shot[(r, int(k))] = correctness(ps[ps.k == k], gsm, by=("repeat", "seed"))
        w1 = np.ones(len(gsm))
        R[r] = {"full_BA": ba_weighted(C_full[r], cls_idx, w1, len(classes)),
                "full_BA_1nn": ba_weighted(correctness(pf, gsm, "y_pred_1nn"), cls_idx, w1, len(classes)),
                "family_acc": float(pf.fam_ok.mean()),
                "breakdown_BA": {m: ba_weighted(C_full[r][:, mk], cls_idx[mk], w1[mk], len(classes))
                                 for m, mk in masks.items()}}
        for k in (3, 5):
            if (r, k) in C_shot:
                R[r][f"fewshot{k}_BA"] = ba_weighted(C_shot[(r, k)], cls_idx, w1, len(classes))
        pv = d / "pred_validation.csv"
        if pv.exists():
            v = pd.read_csv(pv)
            vc = np.searchsorted(classes, v.y_true.astype(str))
            R[r]["validation_BA"] = ba_weighted((v.y_pred_lr == v.y_true).to_numpy(float)[None], vc,
                                                np.ones(len(v)), len(classes))

    # paired comparisons
    A_, B_ = a.primary.split(",")
    raw = [x for x in a.raw.split(",") if x in reps]
    P = {}
    if A_ in reps and B_ in reps:
        P["primary_full"] = paired(C_full[A_], C_full[B_], cls_idx, len(classes), masks["all"], rng)
        for m, mk in masks.items():
            if m != "all":
                P[f"primary_full_{m}"] = paired(C_full[A_], C_full[B_], cls_idx, len(classes), mk, rng)
        for k in (3, 5):
            if (A_, k) in C_shot and (B_, k) in C_shot:
                P[f"primary_fewshot{k}"] = paired(C_shot[(A_, k)], C_shot[(B_, k)], cls_idx, len(classes),
                                                  masks["all"], rng)
    if raw and A_ in reps:
        best5 = max(raw, key=lambda x: R[x].get("fewshot5_BA", -1))
        if (A_, 5) in C_shot and (best5, 5) in C_shot:
            P[f"{A_}_vs_{best5}_fewshot5"] = paired(C_shot[(A_, 5)], C_shot[(best5, 5)], cls_idx, len(classes),
                                                    masks["all"], rng)
        bestf = max(raw, key=lambda x: R[x]["full_BA"])
        P[f"{A_}_vs_{bestf}_full"] = paired(C_full[A_], C_full[bestf], cls_idx, len(classes), masks["all"], rng)

    # success criteria
    crit = {}
    if "primary_full" in P and "primary_fewshot5" in P:
        crit["1_llama_beats_gpt_full_and_5shot"] = all(
            P[k]["diff"] > 0 and P[k]["excludes_zero"] for k in ("primary_full", "primary_fewshot5"))
    k5 = [k for k in P if k.endswith("_fewshot5") and k.startswith(f"{A_}_vs_raw")]
    if k5:
        crit["2_llama_beats_best_raw_5shot"] = P[k5[0]]["diff"] > 0 and P[k5[0]]["excludes_zero"]
    crit["3_report_no_advantage"] = not any(v for v in crit.values() if isinstance(v, bool))

    # geometry
    G = {}
    for spec in a.emb_dirs.split(","):
        name, d = spec.split("=", 1)
        d = Path(d)
        if not (d / "metadata.csv").exists():
            continue
        meta = pd.read_csv(d / "metadata.csv")
        ix = pd.Series(np.arange(len(meta)), index=meta.sample_id.astype(str)).loc[gsm].to_numpy()
        for f in ("embeddings_cls.npy", "embeddings_mean.npy"):
            if (d / f).exists():
                G[f"{name}_{f.split('_')[1].split('.')[0]}"] = geometry(np.load(d / f)[ix].astype(np.float64),
                                                                       cls_idx)

    res = {"per_representation": R, "paired": P, "success_criteria": crit, "geometry": G,
           "n_reference": int(len(gsm)), "n_classes": int(len(classes)), "bootstrap": B}
    json.dump(res, open(out / "comparison.json", "w"), indent=1, default=float)
    rows = [{"rep": r, "subset": m, "BA": v} for r in R for m, v in R[r]["breakdown_BA"].items()]
    pd.DataFrame(rows).pivot(index="subset", columns="rep", values="BA").to_csv(out / "breakdowns.csv")
    lines = ["CNS (Capper) - frozen representation + identical logistic head", ""]
    for r in reps:
        q = R[r]
        lines.append(f"{r:12s} full BA {q['full_BA']:.3f}  1-NN {q['full_BA_1nn']:.3f}  "
                     f"3-shot {q.get('fewshot3_BA', float('nan')):.3f}  5-shot {q.get('fewshot5_BA', float('nan')):.3f}  "
                     f"family acc {q['family_acc']:.3f}  validation {q.get('validation_BA', float('nan')):.3f}")
    lines += ["", "Paired differences (A - B, 95 % bootstrap CI):"]
    for k, v in P.items():
        lines.append(f"  {k:45s} {v['diff']:+.4f}  [{v['ci95'][0]:+.4f}, {v['ci95'][1]:+.4f}]"
                     f"{'  *' if v['excludes_zero'] else ''}")
    lines += ["", f"Success criteria: {crit}"]
    (out / "report.txt").write_text("\n".join(lines))
    print("\n".join(lines))


if __name__ == "__main__":
    main()
