"""
fig4_age_benchmark.py -- Figure 4 and Supplementary Fig. S4 of the manuscript.

Figure 4 (figures/paper/fig4_age_benchmark.*):
  a  cumulative distribution of the absolute age error, MethylLlama and MethylGPT ensembles
  b  primary endpoints MedAE and MAE: five-fold ensemble (bar) + the five folds (circles) per model;
     ElasticNetCV single model (reported point estimate)
  c  paired MethylGPT - MethylLlama differences in MedAE and MAE: full test set + four sensitivity analyses
  d  per-study MAE of the two ensembles, 77 test studies
Supplementary Fig. S4 (figures/paper/figS5_age_benchmark.*, the file name of the manuscript sources):
  a  predicted vs chronological age, both ensembles, hexbin density
  b  fold-0 validation MedAE, pretrained and randomly initialised runs, every recorded epoch, best epoch marked
No number is re-fitted: inputs are the fold predictions, paired predictions, stored bootstrap JSONs and
fold-0 histories in results/age_benchmark/ and the sensitivity outputs in results/data_audit/
(scripts/evaluation/paired_bootstrap_grouped_sensitivity.py). Font sizes >= common_style.MIN_FONT_PT.
Usage:  python scripts/figures/fig4_age_benchmark.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_GPT, COL_LLAMA, MIN_FONT_PT, apply_style, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
BP, AUD, H = REPO / "results/age_benchmark", REPO / "results/data_audit", REPO / "results/age_benchmark"
OUT = REPO / "figures/paper"
ENET = {"medae": 3.437, "mae": 4.731, "r2": 0.9322}      # Table 1, as reported
FS = MIN_FONT_PT                                          # 6.8 pt floor
COL_RANDOM = "#6f6f6f"


def metrics(y, p):
    e = np.abs(p - y)
    return {"medae": float(np.median(e)), "mae": float(e.mean()), "r2": float(r2_score(y, p))}


def load():
    paired = pd.read_csv(BP / "paired_per_subject_predictions.csv", dtype={"sample_id": str})
    assert len(paired) == 2149
    d = {"y": paired.true_age.to_numpy(float), "pl": paired.predicted_age_llama.to_numpy(float),
         "pg": paired.predicted_age_gpt.to_numpy(float)}
    d["ens"] = {"MethylLlama": metrics(d["y"], d["pl"]), "MethylGPT": metrics(d["y"], d["pg"])}
    assert round(d["ens"]["MethylLlama"]["medae"], 3) == 3.112 and round(d["ens"]["MethylGPT"]["medae"], 3) == 3.701
    d["folds"] = {}
    for m, sub in (("MethylLlama", "methyllama"), ("MethylGPT", "methylgpt")):
        rows = []
        for f in sorted((BP / sub).glob("fold_*_predictions.csv")):
            t = pd.read_csv(f); assert len(t) == 2149
            rows.append(metrics(t.true_age.to_numpy(float), t.predicted_age.to_numpy(float)))
        assert len(rows) == 5
        d["folds"][m] = rows
    g = json.loads((AUD / "paired_bootstrap_grouped_sensitivity.json").read_text())
    a = json.loads((BP / "age_restricted_paired_bootstrap_summary.json").read_text())["subsets"]
    pick = lambda src, k: (src[k]["estimate"], src[k]["ci_95"])
    d["forest"] = [
        ("All test profiles", "n = 2,149", {k: pick(g["table_1_profile_bootstrap"], k) for k in ("medae", "mae")}),
        ("Age ≥ 18 years", "n = 1,776", {k: (a["adults_age_ge_18"][f"delta_{k}"], a["adults_age_ge_18"][f"delta_{k}_ci_95"]) for k in ("medae", "mae")}),
        ("Age ≥ 40 years", "n = 1,449", {k: (a["older_age_ge_40"][f"delta_{k}"], a["older_age_ge_40"][f"delta_{k}_ci_95"]) for k in ("medae", "mae")}),
        ("Shared individuals removed", "n = 2,116", {k: pick(g["A_donor_disjoint_test_set"], k) for k in ("medae", "mae")}),
        ("Studies resampled", "77 studies", {k: pick(g["B_study_cluster_bootstrap"], k) for k in ("medae", "mae")})]
    assert a["adults_age_ge_18"]["n"] == 1776 and a["older_age_ge_40"]["n"] == 1449 and g["A_donor_disjoint_test_set"]["n"] == 2116
    assert round(d["forest"][0][2]["medae"][0], 3) == 0.589
    d["pre"] = pd.read_csv(H / "fold0_train_history.csv")
    d["ran"] = pd.read_csv(H / "random_init_fold0_history.csv")
    return d


# ── panels ───────────────────────────────────────────────────────────────────
def panel_scatter(axs, d):
    lim = (-3, 108)
    for ax, key, m, cmap in ((axs[0], "pl", "MethylLlama", "Blues"), (axs[1], "pg", "MethylGPT", "Oranges")):
        hb = ax.hexbin(d["y"], d[key], gridsize=36, extent=(*lim, *lim), mincnt=1, cmap=cmap, bins="log",
                       linewidths=0.1, vmin=1, vmax=60)
        ax.plot(lim, lim, color="0.25", lw=0.7, ls=(0, (3, 2)))
        ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
        e = d["ens"][m]
        ax.text(0.04, 0.96, f"{m}\nMedAE {e['medae']:.2f} y\n$R^2$ {e['r2']:.3f}", transform=ax.transAxes,
                va="top", fontsize=FS, color=COL_LLAMA if m == "MethylLlama" else COL_GPT, linespacing=1.15)
        ax.set_xlabel("Chronological age (years)", fontsize=FS + 0.4)
        ax.tick_params(labelsize=FS)
    axs[0].set_ylabel("Predicted age (years)", fontsize=FS + 0.4)
    axs[1].tick_params(labelleft=False)
    return hb


def panel_ecdf(ax, d):
    """Cumulative distribution of absolute error; the horizontal gap at 0.5 is the MedAE difference."""
    for key, m, col in (("pl", "MethylLlama", COL_LLAMA), ("pg", "MethylGPT", COL_GPT)):
        e = np.sort(np.abs(d[key] - d["y"]))
        ax.step(e, np.arange(1, len(e) + 1) / len(e), where="post", color=col, lw=1.4, label=m)
        md = d["ens"][m]["medae"]
        ax.plot([md, md], [0, 0.5], color=col, lw=0.7, ls=(0, (2, 1.5)))
        ax.text(md, 0.02, f"{md:.2f}", color=col, fontsize=FS, ha="right" if m == "MethylLlama" else "left",
                va="bottom")
    ax.axhline(0.5, color="0.6", lw=0.6, ls=(0, (2, 1.5)))
    ax.set_xlim(0, 15); ax.set_ylim(0, 1.0)
    ax.set_xlabel("Absolute error (years)", fontsize=FS + 0.4)
    ax.set_ylabel("Fraction of test profiles\nwith error ≤ x", fontsize=FS + 0.4)
    ax.tick_params(labelsize=FS); ax.legend(fontsize=FS, frameon=False, loc="lower right")


def panel_endpoints(axs, d):
    """Horizontal bars from zero = five-fold ensemble (ElasticNetCV: single model); points = the five folds."""
    models = [("MethylLlama", COL_LLAMA), ("MethylGPT", COL_GPT), ("ElasticNetCV", COL_ENET)]
    ys = [2, 1, 0]
    for j, (ax, (k, lab, xmax)) in enumerate(zip(axs, (("medae", "MedAE (years)", 5.2), ("mae", "MAE (years)", 7.0)))):
        for yi, (m, col) in zip(ys, models):
            v = ENET[k] if m == "ElasticNetCV" else d["ens"][m][k]
            ax.barh(yi, v, height=0.62, color=col, alpha=0.9 if m != "ElasticNetCV" else 0.75, linewidth=0)
            ax.text(v - xmax * 0.03, yi, f"{v:.2f}", va="center", ha="right", fontsize=FS, color="white",
                    fontweight="bold")
            if m != "ElasticNetCV":
                fv = [f[k] for f in d["folds"][m]]
                ax.scatter(fv, np.full(5, yi), s=9, facecolors="white", edgecolors="0.15", linewidths=0.5, zorder=3)
        ax.set_xlim(0, xmax); ax.set_ylim(-0.55, 2.55); ax.tick_params(labelsize=FS)
        ax.set_xlabel(lab, fontsize=FS + 0.4); ax.set_yticks(ys)
        ax.set_yticklabels([m for m, _ in models] if j == 0 else [], fontsize=FS)
        ax.tick_params(axis="y", length=0)
        if j == 0:
            for lab_, (_, col) in zip(ax.get_yticklabels(), models):
                lab_.set_color(col)
        for sname in ("top", "right"):
            ax.spines[sname].set_visible(False)


def panel_per_study(ax, d):
    """Per-study test-set MAE of the two ensembles; points below the diagonal favour MethylLlama."""
    s = pd.read_csv(AUD / "per_study_mae.csv")
    assert len(s) == 77 and int(s.n.sum()) == 2149
    better = s.mae_llama < s.mae_gpt
    big = s.n >= 10
    lim = (0, max(s.mae_gpt.max(), s.mae_llama.max()) * 1.05)
    ax.plot(lim, lim, color="0.35", lw=0.7, ls=(0, (3, 2)), zorder=1)
    size = lambda n: np.asarray(n, float) * 0.4             # marker area proportional to test profiles
    ax.scatter(s.mae_gpt, s.mae_llama, s=size(s.n), c=np.where(better, COL_LLAMA, COL_GPT), alpha=0.6,
               linewidths=0.3, edgecolors="white", zorder=2)
    h_col = [plt.Line2D([], [], marker="o", ls="", color=COL_LLAMA, alpha=0.7, ms=5, label="MethylLlama lower"),
             plt.Line2D([], [], marker="o", ls="", color=COL_GPT, alpha=0.7, ms=5, label="MethylGPT lower")]
    h_size = [ax.scatter([], [], s=size(n_ref), color="0.6", alpha=0.6, linewidths=0, label=f"{n_ref}")
              for n_ref in (10, 100, 300)]
    leg1 = ax.legend(handles=h_col, fontsize=FS, frameon=False, loc="upper left", handletextpad=0.3, borderpad=0.2)
    ax.add_artist(leg1)
    ax.legend(handles=h_size, title="Test profiles", fontsize=FS, title_fontsize=FS, frameon=False,
              loc="upper left", bbox_to_anchor=(0.0, 0.8), labelspacing=1.1, borderpad=0.2, handletextpad=0.6)
    ax.set_xlim(lim); ax.set_ylim(lim); ax.set_aspect("equal")
    ax.set_xlabel("MethylGPT MAE per study (years)", fontsize=FS + 0.4)
    ax.set_ylabel("MethylLlama MAE per study (years)", fontsize=FS + 0.4)
    ax.tick_params(labelsize=FS)
    ax.text(0.98, 0.02, f"MethylLlama lower in\n{int(better.sum())} of {len(s)} studies\n"
            f"{int((better & big).sum())} of {int(big.sum())} with n ≥ 10", transform=ax.transAxes, ha="right",
            va="bottom", fontsize=FS, color="0.2", bbox=dict(facecolor="white", edgecolor="none", alpha=0.85, pad=1.5))
    return int(better.sum()), int((better & big).sum()), int(big.sum())


def panel_forest(ax_lab, axs, d):
    rows = d["forest"]
    yy = np.arange(len(rows))[::-1].astype(float); yy[1:] -= 0.4
    for yi, (name, n, _) in zip(yy, rows):
        ax_lab.text(1.0, yi + 0.13, name, ha="right", va="center", fontsize=FS,
                    fontweight="bold" if name.startswith("All") else None)
        ax_lab.text(1.0, yi - 0.25, n, ha="right", va="center", fontsize=FS, color="0.4")
    ax_lab.set_xlim(0, 1); ax_lab.set_ylim(yy.min() - 0.7, yy.max() + 1.4); ax_lab.axis("off")
    for ax, (k, lab) in zip(axs, (("medae", "ΔMedAE (years)"), ("mae", "ΔMAE (years)"))):
        ax.axvline(0, color="0.35", lw=0.7, ls=(0, (3, 2)))
        for yi, (_, _, v) in zip(yy, rows):
            est, (lo, hi) = v[k]
            ax.plot([lo, hi], [yi, yi], color=COL_LLAMA, lw=1.3, solid_capstyle="butt")
            ax.scatter([est], [yi], s=18, color=COL_LLAMA, zorder=3, linewidths=0)
        ax.set_xlim(-0.1, 1.25); ax.set_xticks([0, 0.5, 1.0]); ax.tick_params(labelsize=FS)
        ax.set_yticks([]); ax.spines["left"].set_visible(False)
        ax.set_xlabel(lab, fontsize=FS + 0.4)
        ax.set_ylim(ax_lab.get_ylim())
        top = ax_lab.get_ylim()[1]
        ax.annotate("", xy=(1.2, top - 0.3), xytext=(0.05, top - 0.3),
                    arrowprops=dict(arrowstyle="->", lw=0.7, color=COL_LLAMA))


def panel_curves(ax, d):
    for t, col, name in ((d["pre"], COL_LLAMA, "WCED-pretrained"), (d["ran"], COL_RANDOM, "Random initialisation")):
        e, v = t["epoch"].to_numpy(), t["val/medae"].to_numpy(float)
        ax.plot(e, v, color=col, lw=0.5, alpha=0.35)
        i = int(np.argmin(v))
        ax.plot(e, pd.Series(v).rolling(7, center=True, min_periods=4).median(), color=col, lw=1.4,
                label=f"{name}\nlowest {v[i]:.3f} y (epoch {e[i]})")
        ax.scatter([e[i]], [v[i]], s=18, color=col, zorder=4, edgecolors="white", linewidths=0.5)
    ax.set_yscale("log"); ax.set_yticks([3, 5, 10, 20]); ax.set_yticklabels(["3", "5", "10", "20"]); ax.minorticks_off()
    ax.set_ylim(2.0, 30); ax.set_xlim(0, 192)
    ax.set_xlabel("Fine-tuning epoch", fontsize=FS + 0.4); ax.set_ylabel("Validation MedAE (years)", fontsize=FS + 0.4)
    ax.tick_params(labelsize=FS); ax.legend(fontsize=FS, frameon=False, loc="upper right", labelspacing=0.8)
    assert d["pre"]["val/medae"].min() == 2.625 and int(d["pre"].loc[d["pre"]["val/medae"].idxmin(), "epoch"]) == 92


def title(ax, s):
    ax.set_title(s, fontsize=FS + 1.0, pad=14, loc="left")


# ── compose ──────────────────────────────────────────────────────────────────
def build(d, version):
    w, h = (7.2, 5.6) if version == "A" else (7.2, 5.0)
    fig = plt.figure(figsize=(w, h))
    outer = GridSpec(2, 1, figure=fig, hspace=0.42, height_ratios=[1.05, 1.0])
    top = outer[0].subgridspec(1, 2, wspace=0.55, width_ratios=[1.0, 1.0])
    axa = fig.add_subplot(top[0])
    panel_ecdf(axa, d)
    title(axa, "Absolute age error, common test set")
    panel_label(axa, "a", dx=-0.16, dy=1.12)
    sb = top[1].subgridspec(1, 2, wspace=0.18)
    axb = [fig.add_subplot(sb[0, 0]), fig.add_subplot(sb[0, 1])]
    panel_endpoints(axb, d)
    title(axb[0], "Primary endpoints, test set")
    panel_label(axb[0], "b", dx=-0.55, dy=1.12)
    if version == "A":
        bot = outer[1].subgridspec(1, 2, wspace=0.28, width_ratios=[1.25, 1.0])
        sc = bot[0].subgridspec(1, 3, wspace=0.12, width_ratios=[0.8, 1, 1])
    else:
        bot = outer[1].subgridspec(1, 1)
        sc = bot[0].subgridspec(1, 3, wspace=0.12, width_ratios=[0.55, 1, 1])
    axl = fig.add_subplot(sc[0, 0]); axc = [fig.add_subplot(sc[0, 1]), fig.add_subplot(sc[0, 2])]
    panel_forest(axl, axc, d)
    axl.set_title("Error difference, MethylGPT − MethylLlama (95% CI)", fontsize=FS + 1.0, pad=14, loc="left")
    axl.text(1.0, axl.get_ylim()[1] - 0.3, "MethylLlama lower error", ha="right", va="center", fontsize=FS,
             color=COL_LLAMA)
    panel_label(axl, "c", dx=-0.1, dy=1.12)
    if version == "A":
        axd = fig.add_subplot(bot[1])
        panel_per_study(axd, d)
        title(axd, "Per-study error, 77 test studies")
        panel_label(axd, "d", dx=-0.2, dy=1.12)
    return fig


def main():
    apply_style(); OUT.mkdir(parents=True, exist_ok=True)
    d = load()
    # Supplementary figure: a predicted vs chronological (both ensembles), b fold-0 validation curves
    f = plt.figure(figsize=(7.2, 2.6)); gs = GridSpec(1, 3, figure=f, width_ratios=[1, 1, 1.35], wspace=0.32)
    a1, a2 = f.add_subplot(gs[0]), f.add_subplot(gs[1]); panel_scatter([a1, a2], d)
    a1.set_title("Predicted and chronological age", fontsize=FS + 1.0, loc="left", pad=6); panel_label(a1, "a", dx=-0.3, dy=1.04)
    a3 = f.add_subplot(gs[2]); panel_curves(a3, d)
    a3.set_title("Fold-0 validation, exploratory", fontsize=FS + 1.0, loc="left", pad=6); panel_label(a3, "b", dx=-0.2, dy=1.04)
    save(f, str(OUT / "figS5_age_benchmark")); plt.close(f)
    fig = build(d, "A"); save(fig, str(OUT / "fig4_age_benchmark")); plt.close(fig)
    for m in ("MethylLlama", "MethylGPT"):
        print(m, {k: round(v, 4) for k, v in d["ens"][m].items()})


if __name__ == "__main__":
    main()
