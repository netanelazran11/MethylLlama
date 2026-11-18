"""
compare_sparse.py  (CNS task, stage 5 - sparse-input comparison)
================================================================
Turns the per-profile predictions of probe_sparse.py into the comparison of the sparse-input experiment and checks
the criteria fixed before running (CNS plan, "Sparse-input experiment"):
  1. MethylLlama above MethylGPT at every level <= 20 %, 95 % CI of the difference excluding zero.
  2. MethylLlama above the best raw baseline at the same level, at 5 % and 1 %, 95 % CI excluding zero.
  3. Otherwise: report no advantage.
The criteria are judged in Scenario A (primary) on the full-data setting; Scenario B and the few-shot / validation
settings are reported beside them.

Statistics (as compare_cns.py)
  balanced accuracy (BA) = mean per-class recall. At a level, every repeat (and few-shot draw) of every mask seed
  predicts each profile once; a representation's BA is the mean over all of them. Because the class totals do not
  depend on the repeat, that mean equals the BA of each profile's mean correctness, which is what is resampled.
  Paired bootstrap over PROFILES (2,000 resamples, the same resample for both representations) -> 95 % CI.
  Seed spread = min / max of the per-seed BA (the masks are the only thing that changes between seeds).
  "Best raw" is picked per scenario, level and setting by point estimate among the raw reps present, which favours
  the raw side.

Embedding stability (panel f; from the sparse embeddings, not from the probe)
  cos_self      cosine between a profile's sparse and full CLS
  cos_other     mean cosine between a profile's sparse CLS and the full CLS of all OTHER profiles
  top1          fraction of profiles whose nearest full CLS (cosine, among all 3,902) is their own
  median_rank   median rank of the own full CLS (0 = nearest)
  Raw cosines are not comparable across models (MethylGPT's CLS vectors share one direction, mean pairwise cosine
  ~0.94 vs ~0.52 for MethylLlama); top1 and the gap cos_self - cos_other are.

Outputs  outputs/cns/sparse/compare_wideC_all/
  accuracy.csv     scenario, rep, level, setting, n_seeds, ba, ci_lo, ci_hi, seed_min, seed_max, retention
  differences.csv  scenario, level, setting, comparison, a, b, diff, ci_lo, ci_hi, excludes_zero
  stability.csv    model, level, seed, cos_self, cos_other, top1, median_rank
  criteria.json, report.txt
CPU only, from the repository root:  python scripts/cns/compare_sparse.py
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_cns import B, correctness  # noqa: E402
from make_sparse_masks import LEVELS, tag  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
SPARSE = REPO / "outputs/cns/sparse"
MODELS = ("llama_cls", "gpt_cls")
RAW = ("raw_pca256", "raw_top5k", "raw_top32k", "raw_all")
SETTINGS = ("full", "5-shot", "3-shot", "validation")


def ba_of(cbar, cls_idx, w, n_cls):
    """BA from per-profile mean correctness cbar with profile weights w."""
    tot = np.bincount(cls_idx, weights=w, minlength=n_cls)
    hit = np.bincount(cls_idx, weights=w * cbar, minlength=n_cls)
    present = tot > 0
    return float((hit[present] / tot[present]).mean())


def boot(cbars, cls_idx, n_cls, rng):
    """Point estimates and B bootstrap replicates for several reps on the SAME profile resamples."""
    n = len(cls_idx)
    est = [ba_of(c, cls_idx, np.ones(n), n_cls) for c in cbars]
    reps = np.empty((B, len(cbars)))
    for i in range(B):
        w = np.bincount(rng.integers(0, n, n), minlength=n).astype(float)
        reps[i] = [ba_of(c, cls_idx, w, n_cls) for c in cbars]
    return np.array(est), reps


def load_setting(d, setting, order):
    """Correctness rows of one (rep, level, seed) directory for one setting, or None."""
    if setting == "full":
        f = d / "pred_full.csv"
        return correctness(pd.read_csv(f), order) if f.exists() else None
    if setting.endswith("-shot"):
        f = d / "pred_fewshot.csv"
        if not f.exists():
            return None
        p = pd.read_csv(f)
        p = p[p.k == int(setting.split("-")[0])]
        return correctness(p, order, by=("repeat", "seed")) if len(p) else None
    f = d / "pred_validation.csv"
    return correctness(pd.read_csv(f).assign(repeat=0), order) if f.exists() else None


def stability(sparse_root, name, levels, seeds):
    full = Path(sparse_root) / f"emb_{name}" / "L100" / "embeddings_cls.npy"
    if not full.exists():
        return []
    F = np.load(full)
    F = F / np.linalg.norm(F, axis=1, keepdims=True)
    n = len(F)
    off = (F @ F.T).sum(1) - 1
    rows = [{"model": name, "level": 100.0, "seed": 0, "cos_self": 1.0, "cos_other": float((off / (n - 1)).mean()),
             "top1": 1.0, "median_rank": 0.0}]
    for lv in levels:
        for s in seeds:
            f = Path(sparse_root) / f"emb_{name}" / tag(lv, s) / "embeddings_cls.npy"
            if not f.exists():
                continue
            S = np.load(f)
            S = S / np.linalg.norm(S, axis=1, keepdims=True)
            M = S @ F.T
            own = np.diag(M)
            rank = (M > own[:, None]).sum(1)
            rows.append({"model": name, "level": float(lv), "seed": s, "cos_self": float(own.mean()),
                         "cos_other": float(((M.sum(1) - own) / (n - 1)).mean()),
                         "top1": float((rank == 0).mean()), "median_rank": float(np.median(rank))})
    return rows


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default=str(SPARSE / "probe_wideC"))
    p.add_argument("--sparse", default=str(SPARSE))
    p.add_argument("--data", default=str(REPO / "outputs/cns/data"))
    p.add_argument("--seeds", default="0,1,2")
    p.add_argument("--outdir", default=str(SPARSE / "compare_wideC_all"))
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    seeds = [int(s) for s in a.seeds.split(",")]
    import anndata as ad
    obs = ad.read_h5ad(Path(a.data) / "cns_capper.h5ad", backed="r").obs.copy()
    obs["gsm"] = obs.index.astype(str)
    ycls = obs.methylation_class.astype(str)
    classes = np.array(sorted(ycls[obs.set == "reference"].unique()))
    order = {"ref": obs.gsm[obs.set == "reference"].to_numpy(),
             "val": obs.gsm[(obs.set == "validation") & ~obs.val_leak_dup.astype(bool)].to_numpy()}
    cidx = {k: np.searchsorted(classes, ycls.loc[v].to_numpy()) for k, v in order.items()}

    # per-profile mean correctness, pooled over repeats / draws / mask seeds
    cbar, seed_ba, missing = {}, {}, []
    for sc in ("A", "B"):
        for rep in MODELS + RAW:
            for lv in LEVELS:
                for st in SETTINGS:
                    grp = "val" if st == "validation" else "ref"
                    rows, per_seed = [], []
                    for s in ([0] if lv == 100 else seeds):
                        d = Path(a.probe) / sc / rep / tag(lv, s)
                        # metrics.json is written last: a directory without it is a cell cut off mid-run
                        c = load_setting(d, st, order[grp]) if (d / "metrics.json").exists() else None
                        if c is None:
                            continue
                        rows.append(c)
                        per_seed.append(ba_of(c.mean(0), cidx[grp], np.ones(len(cidx[grp])), len(classes)))
                    if not rows:
                        continue
                    if lv != 100 and len(rows) < len(seeds):
                        missing.append(f"{sc}/{rep}/L{lv:g}/{st}: {len(rows)} of {len(seeds)} seeds")
                    cbar[(sc, rep, lv, st)] = np.vstack(rows).mean(0)
                    seed_ba[(sc, rep, lv, st)] = per_seed
    # Scenario B at 100 % is Scenario A at 100 % (no mask): reuse it as the reference point for retention
    for (sc, rep, lv, st), v in list(cbar.items()):
        if sc == "A" and lv == 100 and ("B", rep, 100, st) not in cbar:
            cbar[("B", rep, 100, st)], seed_ba[("B", rep, 100, st)] = v, seed_ba[(sc, rep, lv, st)]

    acc, dif = [], []
    for sc in ("A", "B"):
        for lv in LEVELS:
            for st in SETTINGS:
                grp = "val" if st == "validation" else "ref"
                reps = [r for r in MODELS + RAW if (sc, r, lv, st) in cbar]
                if not reps:
                    continue
                rng = np.random.default_rng(LEVELS.index(lv) * 100 + SETTINGS.index(st) * 10 + (sc == "B"))
                est, bs = boot([cbar[(sc, r, lv, st)] for r in reps], cidx[grp], len(classes), rng)
                for j, r in enumerate(reps):
                    lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
                    full = cbar.get((sc, r, 100, st))
                    base = ba_of(full, cidx[grp], np.ones(len(full)), len(classes)) if full is not None else np.nan
                    acc.append({"scenario": sc, "rep": r, "level": lv, "setting": st,
                                "n_seeds": len(seed_ba[(sc, r, lv, st)]), "ba": est[j], "ci_lo": lo, "ci_hi": hi,
                                "seed_min": min(seed_ba[(sc, r, lv, st)]), "seed_max": max(seed_ba[(sc, r, lv, st)]),
                                "retention": est[j] / base})
                raws = [r for r in reps if r in RAW]
                pairs = []
                if all(m in reps for m in MODELS):
                    pairs.append(("llama_cls - gpt_cls", "llama_cls", "gpt_cls"))
                if "llama_cls" in reps and raws:
                    best = max(raws, key=lambda r: est[reps.index(r)])
                    pairs.append((f"llama_cls - best raw ({best})", "llama_cls", best))
                for name, x, y in pairs:
                    i, j = reps.index(x), reps.index(y)
                    d = bs[:, i] - bs[:, j]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    dif.append({"scenario": sc, "level": lv, "setting": st, "comparison": name, "a": est[i],
                                "b": est[j], "diff": est[i] - est[j], "ci_lo": lo, "ci_hi": hi,
                                "excludes_zero": bool(lo > 0 or hi < 0)})
    acc, dif = pd.DataFrame(acc), pd.DataFrame(dif)
    acc.to_csv(out / "accuracy.csv", index=False)
    dif.to_csv(out / "differences.csv", index=False)

    stab = pd.DataFrame([r for m in ("llama", "gpt") for r in stability(a.sparse, m, [l for l in LEVELS if l < 100],
                                                                        seeds)])
    stab.to_csv(out / "stability.csv", index=False)

    # criteria: Scenario A, full-data setting
    def judged(levels, prefix):
        q = dif[(dif.scenario == "A") & (dif.setting == "full") & dif.comparison.str.startswith(prefix)]
        rows = {f"{lv:g}": (q[q.level == lv].iloc[0] if (q.level == lv).any() else None) for lv in levels}
        res = {k: (None if r is None else {"diff": round(float(r["diff"]), 4),
                                           "ci": [round(float(r.ci_lo), 4), round(float(r.ci_hi), 4)],
                                           "met": bool(r["diff"] > 0 and r.excludes_zero),
                                           "vs": r.comparison}) for k, r in rows.items()}
        done = all(v is not None for v in res.values())
        return {"levels": res, "complete": done, "met": done and all(v["met"] for v in res.values())}

    crit = {"1_llama_above_gpt_all_levels_le20": judged([l for l in LEVELS if l <= 20], "llama_cls - gpt_cls"),
            "2_llama_above_best_raw_5_and_1": judged([5, 1], "llama_cls - best raw")}
    crit["3_report_no_advantage"] = (not crit["1_llama_above_gpt_all_levels_le20"]["met"]
                                     and not crit["2_llama_above_best_raw_5_and_1"]["met"])
    shuffled = {}
    for rep in MODELS + RAW:
        f = Path(a.probe) / "A" / rep / f"{tag(5, 0)}_shuffled" / "metrics.json"
        if f.exists():
            shuffled[rep] = json.load(open(f))["full"]["balanced_acc"]
    crit["label_shuffle_control_A_5pct"] = {"ba": shuffled, "chance": round(1 / len(classes), 4)}
    crit["missing"] = missing
    json.dump(crit, open(out / "criteria.json", "w"), indent=1)

    with open(out / "report.txt", "w") as f:
        for sc in ("A", "B"):
            f.write(f"== Scenario {sc}: balanced accuracy, full-data setting (95% CI; seed min-max)\n")
            t = acc[(acc.scenario == sc) & (acc.setting == "full")]
            for lv in LEVELS:
                r = t[t.level == lv]
                if len(r):
                    f.write(f"  {lv:>5g}%  " + "  ".join(f"{x.rep} {x.ba:.3f} [{x.ci_lo:.3f},{x.ci_hi:.3f}]"
                                                         for x in r.itertuples()) + "\n")
            f.write(f"-- Scenario {sc}: differences (all settings)\n")
            for x in dif[dif.scenario == sc].itertuples():
                f.write(f"  {x.level:>5g}% {x.setting:<10} {x.comparison:<34} {x.diff:+.3f} "
                        f"[{x.ci_lo:+.3f},{x.ci_hi:+.3f}]{' *' if x.excludes_zero else ''}\n")
        f.write("== Embedding stability (mean over seeds)\n")
        if len(stab):
            f.write(stab.groupby(["model", "level"])[["cos_self", "cos_other", "top1", "median_rank"]].mean()
                    .round(3).to_string() + "\n")
        f.write("== Criteria\n" + json.dumps(crit, indent=1) + "\n")
    print(open(out / "report.txt").read())


if __name__ == "__main__":
    main()
