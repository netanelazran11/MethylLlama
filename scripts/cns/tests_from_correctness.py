"""
tests_from_correctness.py  (CNS task - Fig. 6 and Supplementary Table S3 from the shipped results)
=================================================================================================
Recomputes the balanced accuracies of Fig. 6a and the paired tests of Supplementary Table S3 from
results/cns/sparse_correctness.npz, without the methylation data: the npz holds, for every cell of the
Scenario A and B grids, each reference profile's mean correctness over the cross-validation repeats.

The computation is that of fig_cns_sparse_B.tests(): a cell's correctness is averaged over the three mask
seeds; balanced accuracy and the paired bootstrap over profiles (2,000 resamples) use compare_sparse.boot with
the same random seed per level, so the confidence intervals match; the best raw-beta baseline is chosen per level
on its point estimate; p-values are Holm-adjusted across levels within each scenario and comparison.

  python scripts/cns/tests_from_correctness.py [--npz results/cns/sparse_correctness.npz] [--out DIR]
Writes fig6_accuracy.csv and fig6_tests.csv (same columns as results/cns/figure_data/) and compares them with
the shipped files.
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[1]
sys.path.insert(0, str(HERE))
from compare_sparse import RAW, SETTINGS, boot  # noqa: E402
from make_sparse_masks import LEVELS, tag  # noqa: E402

REPS = ("llama_cls", "gpt_cls", "raw_top5k", "raw_pca256")
HIGH = (60.0, 70.0, 80.0, 90.0)
ALL = tuple(sorted(set(LEVELS) | set(HIGH)))


def holm(p):
    p = np.asarray(p, float)
    o = np.argsort(p)
    adj = np.empty_like(p)
    run = 0.0
    for k, i in enumerate(o):
        run = max(run, min(1.0, (len(p) - k) * p[i]))
        adj[i] = run
    return adj


def tests(z, seeds=(0, 1, 2)):
    y = z["methylation_class"]
    classes = np.array(sorted(set(y)))
    ci = np.searchsorted(classes, y)
    A, T = [], []
    for sc in ("A", "B"):
        for lv in ALL:
            if sc == "A" and lv not in LEVELS:
                continue
            cbar = {r: np.vstack([z[f"{'A' if lv == 100 else sc}|{r}|{tag(lv, s)}"]
                                  for s in ([0] if lv == 100 else seeds)]).mean(0) for r in REPS}
            seed0 = LEVELS.index(lv) * 100 if lv in LEVELS else 900 + int(lv)
            rng = np.random.default_rng(seed0 + SETTINGS.index("full") * 10 + (sc == "B"))
            est, bs = boot([cbar[r] for r in REPS], ci, len(classes), rng)
            for j, r in enumerate(REPS):
                lo, hi = np.percentile(bs[:, j], [2.5, 97.5])
                A.append({"scenario": sc, "level": lv, "rep": r, "ba": est[j], "ci_lo": lo, "ci_hi": hi})
            best = max(RAW[:2], key=lambda r: est[REPS.index(r)])
            for name, other in (("best raw", best), ("MethylGPT", "gpt_cls")):
                d = bs[:, 0] - bs[:, REPS.index(other)]
                lo, hi = np.percentile(d, [2.5, 97.5])
                p = min(1.0, max(2 * min((d <= 0).mean(), (d >= 0).mean()), 1 / len(d)))
                T.append({"scenario": sc, "level": lv, "vs": name, "against": other,
                          "diff": est[0] - est[REPS.index(other)], "ci_lo": lo, "ci_hi": hi, "p": p})
    A, T = pd.DataFrame(A), pd.DataFrame(T)
    T["p_holm"] = np.nan
    for _, g in T.groupby(["scenario", "vs"]):
        T.loc[g.index, "p_holm"] = holm(g.p.to_numpy())
    T["significant"] = T.p_holm < 0.05
    return A, T


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--npz", default=str(REPO / "results/cns/sparse_correctness.npz"))
    p.add_argument("--shipped", default=str(REPO / "results/cns/figure_data"))
    p.add_argument("--out", default=str(REPO / "outputs/cns/from_correctness"))
    a = p.parse_args()
    A, T = tests(np.load(a.npz, allow_pickle=False))
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    A.to_csv(out / "fig6_accuracy.csv", index=False)
    T.to_csv(out / "fig6_tests.csv", index=False)
    for name, new in (("fig6_accuracy.csv", A), ("fig6_tests.csv", T)):
        old = pd.read_csv(Path(a.shipped) / name)
        num = [c for c in new.columns if c in old.columns and new[c].dtype.kind in "fi"]
        dev = max(float(np.nanmax(np.abs(new[c].to_numpy(float) - old[c].to_numpy(float)))) for c in num)
        same = all((new[c].astype(str) == old[c].astype(str)).all() for c in new.columns if c not in num)
        print(f"{name}: {len(new)} rows, max |diff| vs shipped {dev:.2e}, labels identical {same}")


if __name__ == "__main__":
    main()
