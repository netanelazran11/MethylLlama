"""
fig_cns_data.py  (CNS task, stage 2 - figure data)
==================================================
Builds one tidy CSV per figure panel from the frozen-representation results, so that every plotted point can be
checked against a file. Drawing is done separately by fig_cns.py.

Inputs   outputs/cns/{data/cns_capper.h5ad (obs only), probe_all/<rep>/pred_*.csv}
Outputs  outputs/cns/figures/data/
  panel_b_families.csv     family, n_classes, n_profiles, n_classes_lt20
  panel_c_accuracy.csv     rep, setting (full / 5-shot / 3-shot), ba, ci_lo, ci_hi       (bootstrap over profiles)
  panel_c_difference.csv   setting, diff (MethylLlama - MethylGPT), ci_lo, ci_hi
  panel_d_family.csv       family, n_classes, n_profiles, ba_llama, ba_gpt, diff         (family = mean per-class recall)
  s1_per_class.csv         class, family, n_profiles, recall_llama, recall_gpt

Families are the prefix of the class abbreviation (e.g. 'GBM, MES' -> GBM), a provisional grouping until the
families of Capper et al.'s supplement are used. Balanced accuracy and its bootstrap follow compare_cns.py.
  python scripts/cns/fig_cns_data.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from compare_cns import ba_weighted, correctness, paired  # noqa: E402

B = 2000
REPS = {"llama_cls": "MethylLlama", "gpt_cls": "MethylGPT", "raw_pca256": "Raw PCA-256", "raw_top5k": "Raw 5k CpGs",
        "raw_all": "Raw all CpGs"}


def boot_ba(C, ci, n_cls, rng):
    n = C.shape[1]
    est = ba_weighted(C, ci, np.ones(n), n_cls)
    bs = [ba_weighted(C, ci, np.bincount(rng.integers(0, n, n), minlength=n).astype(float), n_cls) for _ in range(B)]
    lo, hi = np.percentile(bs, [2.5, 97.5])
    return est, lo, hi


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--root", default=str(REPO / "outputs/cns"))
    a = p.parse_args()
    root = Path(a.root)
    out = root / "figures/data"
    out.mkdir(parents=True, exist_ok=True)
    import anndata as ad
    obs = ad.read_h5ad(root / "data/cns_capper.h5ad", backed="r").obs.copy()
    obs["gsm"] = obs.index.astype(str)
    ref = obs[obs.set == "reference"].copy()
    ref["methylation_class"] = ref.methylation_class.astype(str)
    ref["family"] = ref.prefix_group.astype(str)
    gsm = ref.gsm.to_numpy()
    classes = np.array(sorted(ref.methylation_class.unique()))
    ci = np.searchsorted(classes, ref.methylation_class.to_numpy())
    size = ref.methylation_class.value_counts()
    rng = np.random.default_rng(0)

    # panel b
    cf = ref.groupby("methylation_class").agg(family=("family", "first"), n=("gsm", "size"))
    fam = cf.groupby("family").agg(n_classes=("n", "size"), n_profiles=("n", "sum"),
                                   n_classes_lt20=("n", lambda x: int((x < 20).sum())))
    fam.sort_values("n_profiles", ascending=False).reset_index().to_csv(out / "panel_b_families.csv", index=False)

    # panel c
    rows, C = [], {}
    for rep in REPS:
        d = root / "probe_all" / rep
        C[(rep, "full")] = correctness(pd.read_csv(d / "pred_full.csv"), gsm)
        fs = pd.read_csv(d / "pred_fewshot.csv")
        for k in (5, 3):
            C[(rep, f"{k}-shot")] = correctness(fs[fs.k == k], gsm, by=("repeat", "seed"))
        for s in ("full", "5-shot", "3-shot"):
            est, lo, hi = boot_ba(C[(rep, s)], ci, len(classes), rng)
            rows.append({"rep": rep, "label": REPS[rep], "setting": s, "ba": est, "ci_lo": lo, "ci_hi": hi})
    pd.DataFrame(rows).to_csv(out / "panel_c_accuracy.csv", index=False)
    diff = []
    for s in ("full", "5-shot", "3-shot"):
        r = paired(C[("llama_cls", s)], C[("gpt_cls", s)], ci, len(classes), np.ones(len(gsm), bool), rng)
        diff.append({"setting": s, "diff": r["diff"], "ci_lo": r["ci95"][0], "ci_hi": r["ci95"][1]})
    pd.DataFrame(diff).to_csv(out / "panel_c_difference.csv", index=False)

    # per-class recall (full setting, mean over repeats) -> S1 and panel d
    rec = {}
    for rep in ("llama_cls", "gpt_cls"):
        pf = pd.read_csv(root / "probe_all" / rep / "pred_full.csv")
        rec[rep] = pf.assign(ok=pf.y_pred_lr == pf.y_true).groupby("y_true").ok.mean()
    s1 = pd.DataFrame({"class": classes, "family": cf.loc[classes, "family"].values,
                       "n_profiles": size.loc[classes].values,
                       "recall_llama": rec["llama_cls"].loc[classes].values,
                       "recall_gpt": rec["gpt_cls"].loc[classes].values})
    s1.to_csv(out / "s1_per_class.csv", index=False)
    pdf = s1.groupby("family").agg(n_classes=("class", "size"), n_profiles=("n_profiles", "sum"),
                                   ba_llama=("recall_llama", "mean"), ba_gpt=("recall_gpt", "mean")).reset_index()
    pdf["diff"] = pdf.ba_llama - pdf.ba_gpt
    pdf.sort_values("n_profiles", ascending=False).to_csv(out / "panel_d_family.csv", index=False)
    print("wrote", sorted(x.name for x in out.iterdir()))


if __name__ == "__main__":
    main()
