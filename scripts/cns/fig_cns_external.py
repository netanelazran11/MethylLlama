"""
fig_cns_external.py  (CNS task, stage 6 - external cohort figure)
=================================================================
Draws the external test (GSE215240) from the files of probe_external.py (nothing is recomputed but the curves):

  a  balanced accuracy over the scoring groups per representation, 95 % bootstrap CI; markers: the same measure per
     array (EPIC / 450K) over the groups present on both arrays, so the two markers compare like with like
  b  accuracy against coverage when the classifier may abstain: profiles are ranked by the probe's top
     probability; for each threshold t, coverage = share of scored profiles with top probability >= t, accuracy =
     share of those classified into the right group. The reference classifier is used the same way in practice
     (a class is reported only above a calibrated score of 0.9), so this is how the representations would serve.
     Markers at t = 0.9.
  c  sparse input, heads trained on full Capper profiles (Scenario A, probe_external_sparse.py), 95 % bootstrap CI
  d  sparse input, heads trained on Capper profiles at the same sparse level (Scenario B, probe_external_sparse_B.py)

Outputs  outputs/cns/figures/{pdf,png,svg}/fig_cns_external.*, figures/data/fig_cns_external_{bars,coverage}.csv
  python scripts/cns/fig_cns_external.py [--probe outputs/cns/external/probe]
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/figures"))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

REPS = {"llama_cls": ("MethylLlama", COL_LLAMA, "-"), "gpt_cls": ("MethylGPT", COL_GPT, "-"),
        "raw_top5k": ("Raw, 5k CpGs", "#8a8a8a", "-"), "raw_pca256": ("Raw, PCA-256", "#8a8a8a", "--")}


def save(fig, out, stem):
    for ext in ("pdf", "png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"{stem}.{ext}", bbox_inches="tight", dpi=300)
    print("saved", stem)


def coverage_curve(pr):
    s = pr[pr.eval_group.fillna("") != ""].sort_values("p_max", ascending=False)
    ok = (s.pred_group == s.eval_group).to_numpy(float)
    n = np.arange(1, len(s) + 1)
    return pd.DataFrame({"threshold": s.p_max.to_numpy(), "coverage": n / len(s), "accuracy": np.cumsum(ok) / n})


def shared_ba(pr):
    """Balanced accuracy per array over the groups scored on both arrays; returns EPIC, 450K, number of groups."""
    s = pr[pr.eval_group.fillna("") != ""]
    ep, k4 = s[s.platform == "GPL21145"], s[s.platform == "GPL13534"]
    both = set(ep.eval_group) & set(k4.eval_group)
    ba = [t[t.eval_group.isin(both)].assign(ok=lambda d: d.eval_group == d.pred_group).groupby("eval_group").ok.mean().mean()
          for t in (ep, k4)]
    return ba[0], ba[1], len(both)


def curves(ax, t, xlabel, ylabel=None):
    for rep, (lab, col, ls) in REPS.items():
        r = t[t.rep == rep].sort_values("level")
        if not len(r):
            continue
        model = not rep.startswith("raw")
        ax.fill_between(r.level, r.ci_lo, r.ci_hi, color=col, alpha=0.18 if model else 0.10, lw=0)
        ax.plot(r.level, r.ba, color=col, ls=ls, marker="o" if model else ("s" if rep == "raw_top5k" else "D"),
                ms=3 if model else 2.6, lw=1.3 if model else 0.9, mfc=col if model else "white", label=lab)
    lv = sorted(t.level.unique())
    ax.set_xscale("log")
    ax.set_xticks(lv, [f"{x:g}" for x in lv])
    ax.minorticks_off()
    ax.set_xlim(max(lv) * 1.3, min(lv) / 1.3)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel(xlabel)
    if ylabel:
        ax.set_ylabel(ylabel)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--probe", default=str(REPO / "outputs/cns/external/probe"))
    p.add_argument("--root", default=str(REPO / "outputs/cns/figures"))
    p.add_argument("--stem", default="fig_cns_external")
    a = p.parse_args()
    M = pd.read_csv(Path(a.probe) / "metrics.csv")
    reps = [r for r in REPS if r in set(M.rep)]
    out = Path(a.root)
    (out / "data").mkdir(parents=True, exist_ok=True)
    M.to_csv(out / "data" / f"{a.stem}_bars.csv", index=False)

    apply_style()
    fig = plt.figure(figsize=(7.2, 6.0))
    gs = fig.add_gridspec(2, 2, width_ratios=[1, 1.25], wspace=0.35, hspace=0.42)
    axa, axb = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    for i, rep in enumerate(reps):
        r = M.set_index("rep").loc[rep]
        ba_e, ba_k, n_both = shared_ba(pd.read_csv(Path(a.probe) / f"pred_{rep}.csv"))
        lab, col, _ = REPS[rep]
        raw = rep.startswith("raw")
        axa.bar(i, r.ba, 0.7, color="white" if raw else col, edgecolor=col, lw=0.9,
                hatch="////" if rep == "raw_pca256" else "")
        axa.errorbar(i, r.ba, yerr=[[r.ba - r.ci_lo], [r.ci_hi - r.ba]], color="black", capsize=2, elinewidth=0.7)
        axa.scatter(i - 0.17, ba_e, marker="^", s=12, color="black", zorder=3, label=f"EPIC ({n_both} shared classes)" if i == 0 else None)
        axa.scatter(i + 0.17, ba_k, marker="v", s=12, color="black", facecolor="white", zorder=3,
                    label=f"450K ({n_both} shared classes)" if i == 0 else None)
        axa.text(i, 0.02, f"{r.ba:.2f}", ha="center", va="bottom", fontsize=6.5, color="white" if not raw else "black",
                  bbox=dict(fc="white", ec="none", pad=0.6) if raw else None)
    axa.set_xticks(range(len(reps)), [REPS[r][0].replace(", ", ",\n") for r in reps], fontsize=6.5)
    axa.set_ylim(0, 1)
    n = int(M.n_scored.iloc[0])
    axa.set_ylabel(f"Balanced accuracy\n({n} tumours, {int(M.n_groups.iloc[0])} classes)")
    axa.legend(frameon=False, fontsize=6.3, loc="lower center", bbox_to_anchor=(0.5, 1.0), ncol=2, handletextpad=0.2)
    panel_label(axa, "a", dx=-0.3)

    cov = []
    for rep in reps:
        c = coverage_curve(pd.read_csv(Path(a.probe) / f"pred_{rep}.csv")).assign(rep=rep)
        cov.append(c)
        lab, col, ls = REPS[rep]
        axb.plot(c.coverage, c.accuracy, color=col, ls=ls, lw=1.2, label=lab)
        k = c[c.threshold >= 0.9]
        if len(k):
            axb.scatter(k.coverage.iloc[-1], k.accuracy.iloc[-1], color=col, s=16, zorder=3,
                        facecolor="white" if rep.startswith("raw") else col)
    pd.concat(cov).to_csv(out / "data" / f"{a.stem}_coverage.csv", index=False)
    axb.set_xlim(0, 1.0)
    axb.set_ylim(min(0.5, axb.get_ylim()[0]), 1.005)
    axb.set_xlabel("Share of tumours classified (most confident first)")
    axb.set_ylabel("Accuracy of those classified")
    axb.legend(frameon=False, fontsize=6.3, loc="lower left")
    axb.text(0.99, 0.02, "dots: top probability >= 0.9", transform=axb.transAxes, ha="right", va="bottom", fontsize=6.2,
             color="#666666")
    panel_label(axb, "b", dx=-0.18)
    sp = Path(a.probe).parent / "sparse"
    axc, axd = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])
    At, Bt = pd.read_csv(sp / "probe/accuracy.csv"), pd.read_csv(sp / "probe_B/accuracy.csv")
    curves(axc, At, "CpGs measured, test only (%)", "Balanced accuracy\n(trained on full profiles)")
    curves(axd, Bt, "CpGs measured, train and test (%)", "Balanced accuracy\n(trained at the same level)")
    axd.legend(frameon=False, fontsize=6.3, loc="upper right", handlelength=1.8)
    panel_label(axc, "c", dx=-0.3)
    panel_label(axd, "d", dx=-0.18)
    pd.concat([At.assign(scenario="A"), Bt.assign(scenario="B")]).to_csv(out / "data" / f"{a.stem}_sparse.csv", index=False)
    save(fig, out, a.stem)
    plt.close(fig)


if __name__ == "__main__":
    main()
