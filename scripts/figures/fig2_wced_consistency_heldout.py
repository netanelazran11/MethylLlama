"""
Figure 2 -- WCED representation consistency and held-out reconstruction,
with panels A/B computed on the HELD-OUT partition of the pretraining corpus.

This is a new script. fig2_wced_consistency.py (panels A/B from 2,000 downstream
profiles) is left untouched and writes to a different file.

Panels:
  A  Matched-versus-unmatched cross-view similarity, every held-out profile.
     The unmatched density uses ALL N*(N-1) pairs (no subsampling).
  B  Top-1 cross-view retrieval among all N held-out candidates: overlapping views,
     disjoint views, and the pattern-only negative control; one point per view draw.
  C  Observed vs predicted beta at withheld CpGs (hexbin).      } same artifact and same
  D  Reconstruction vs controls under identical masks.          } drawing as the original

Inputs
  results/pretrained_cls/view_consistency_heldout_full/summary.json and per_run/
      written by scripts/evaluation/view_consistency_heldout_full.py
  results/pretrained_cls/reconstruction_withheld/  (already held-out: split=test)

Every number drawn in A/B is RECOMPUTED from the per-run arrays and asserted against
the per-run JSON and summary.json before anything is plotted.

Usage:  python scripts/figures/fig2_wced_consistency_heldout.py
Output: figures/paper/fig2_wced_consistency_reconstruction_heldout.{pdf,png,svg}
        figures/paper/fig2_heldout_provenance.json
"""

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import (COL_ENET, COL_GPT, COL_LLAMA, apply_style,
                          panel_label, save)
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
VIEWDIR = REPO / "results/pretrained_cls/view_consistency_heldout_full"
RECON_DIR = REPO / "results/pretrained_cls/reconstruction_withheld"
OUTDIR = REPO / "figures/paper"

POOL = "all"
CONDS = [("overlap", "Overlapping\nviews", COL_LLAMA),
         ("disjoint", "Disjoint\nviews", COL_GPT),
         ("pattern_only", "Pattern-only\ncontrol", COL_ENET)]


def load_runs(viewdir):
    """Per-run statistics, each recomputed from the stored arrays and checked."""
    runs = {}
    for f in sorted((viewdir / "per_run").glob("*_seed*.json")):
        j = json.loads(f.read_text())
        z = np.load(f.with_suffix(".npz"))
        s = j["pools"][POOL]
        n = s["n_candidates"]
        pos, rank, hist = z[f"{POOL}_matched_cos"], z[f"{POOL}_rank"], z[f"{POOL}_unmatched_hist"]
        assert len(pos) == len(rank) == n == j["n_profiles"], f.name
        assert hist.sum() == n * (n - 1), f"{f.name}: histogram holds {hist.sum()} pairs, expected {n * (n - 1)}"
        assert abs(pos.mean() - s["matched_cos"]) < 1e-5, f.name
        assert abs((rank < 1).mean() - s["retrieval_at1"]) < 1e-9, f.name
        assert j["split"] != "train", f.name
        runs[(j["condition"], j["seed"])] = {"json": j, "pos": pos, "rank": rank, "hist": hist,
                                             "bins": z["hist_bins"], "n": n}
    assert runs, f"no runs in {viewdir / 'per_run'}"
    assert len({r["n"] for r in runs.values()}) == 1, "runs were evaluated on different numbers of profiles"
    return runs


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--viewdir", default=str(VIEWDIR))
    p.add_argument("--outdir", default=str(OUTDIR))
    a = p.parse_args()
    viewdir, outdir = Path(a.viewdir), Path(a.outdir)
    apply_style()
    outdir.mkdir(parents=True, exist_ok=True)

    runs = load_runs(viewdir)
    summary = json.loads((viewdir / "summary.json").read_text())
    n = next(iter(runs.values()))["n"]
    assert summary["n_profiles"] == n
    top1 = {c: np.array([100 * (r["rank"] < 1).mean() for (cc, _), r in sorted(runs.items()) if cc == c])
            for c, _, _ in CONDS}
    for c, v in top1.items():
        if len(v):
            ref = summary["conditions"][c]["pools"][POOL]["retrieval_at1"]["mean"]
            assert abs(v.mean() / 100 - ref) < 1e-9, f"{c}: figure {v.mean() / 100} vs summary.json {ref}"
    assert len(top1["overlap"]) and len(top1["disjoint"]), "overlap and disjoint runs are both required"

    have_recon = (RECON_DIR / "reconstruction_withheld_summary.json").exists()
    if have_recon:
        rsum = json.loads((RECON_DIR / "reconstruction_withheld_summary.json").read_text())
        assert rsum["split"] == "test", "reconstruction artifact is not from the held-out partition"
        scat = np.load(RECON_DIR / "scatter_sample.npz")
        fig = plt.figure(figsize=(7.2, 5.6))
        gs = GridSpec(2, 2, figure=fig, hspace=0.55, wspace=0.42, width_ratios=[1.25, 1.0])
    else:
        print("NOTE: reconstruction artifact missing -- rendering A/B only.")
        fig = plt.figure(figsize=(7.2, 2.8))
        gs = GridSpec(1, 2, figure=fig, wspace=0.42, width_ratios=[1.25, 1.0])

    # ── Panel A: matched vs unmatched, overlapping views, first draw ─────────
    r0 = runs[("overlap", 0)]
    bins, width = r0["bins"], np.diff(r0["bins"])
    off_density = r0["hist"] / (r0["hist"].sum() * width)
    pos_counts = np.histogram(r0["pos"], bins=bins)[0]
    assert pos_counts.sum() == n, "matched similarities fall outside the histogram range"
    pos_density = pos_counts / (n * width)
    centers = 0.5 * (bins[:-1] + bins[1:])
    off_mean = r0["json"]["pools"][POOL]["unmatched_cos"]
    pos_mean = float(r0["pos"].mean())

    axA = fig.add_subplot(gs[0, 0])
    axA.bar(centers, off_density, width=width, alpha=0.62, color="0.62",
            label=f"Unmatched profiles (mean {off_mean:.3f})")
    axA.bar(centers, pos_density, width=width, alpha=0.78, color=COL_LLAMA,
            label=f"Matched views (mean {pos_mean:.3f})")
    axA.axvline(off_mean, color="0.35", lw=0.8, ls="--")
    axA.axvline(pos_mean, color=COL_LLAMA, lw=0.8, ls="--")
    axA.set_xlabel("Cosine similarity of profile representations")
    axA.set_ylabel("Density (log scale)")
    y_floor = 0.02
    lo = bins[np.nonzero(off_density >= y_floor)[0][0]]          # first bin visible on the log axis
    axA.set_xlim(min(-0.2, np.floor(lo * 10) / 10), 1.01)
    axA.set_yscale("log")
    axA.set_ylim(y_floor, max(200, 2 * pos_density.max()))
    leg = axA.legend(frameon=True, fontsize=6.8, loc="upper left", handlelength=1.1, borderpad=0.3)
    leg.get_frame().set_facecolor("white"); leg.get_frame().set_edgecolor("none"); leg.get_frame().set_alpha(1.0)
    panel_label(axA, "a")

    # ── Panel B: top-1 retrieval among all N candidates ──────────────────────
    axB = fig.add_subplot(gs[0, 1])
    shown = [(c, lab, col) for c, lab, col in CONDS if len(top1[c])]
    for x, (c, lab, col) in enumerate(shown):
        v = top1[c]
        axB.bar(x, v.mean(), width=0.62, color=col, zorder=2)
        if len(v) > 1:
            axB.vlines(x, v.min(), v.max(), color="0.15", lw=0.9, zorder=4)
        axB.scatter(np.full(len(v), x), v, s=4, color="white", edgecolors="0.15",
                    linewidths=0.4, zorder=5)
        axB.text(x, v.max() + 2.5, f"{v.mean():.1f}" if v.mean() >= 1 else f"{v.mean():.2f}",
                 ha="center", va="bottom", fontsize=6.8, color=col)
    chance = 100.0 / n
    axB.axhline(chance, color="0.35", ls="--", lw=0.8, zorder=3, label=f"chance ({chance:.3f}%)")
    axB.set_xticks(range(len(shown)))
    axB.set_xticklabels([lab for _, lab, _ in shown], fontsize=6.8)
    axB.set_ylabel("Top-1 cross-view retrieval (%)")
    axB.set_ylim(0, 112)
    if len(top1["pattern_only"]):
        # The chance line is invisible on a 0--100% axis. A log-scale inset
        # keeps both the pattern-only control and exact 1/N chance legible
        # without compressing the two primary retrieval bars.
        axBi = axB.inset_axes([0.68, 0.70, 0.30, 0.23])
        low_values = [float(top1["pattern_only"].mean()), chance]
        low_colors = [COL_ENET, "0.35"]
        for xx, (yy, col) in enumerate(zip(low_values, low_colors)):
            axBi.vlines(xx, chance / 2, yy, color=col, lw=1.0)
            axBi.scatter(xx, yy, s=12, color=col, edgecolors="white",
                         linewidths=0.4, zorder=3)
        axBi.set_yscale("log")
        axBi.set_ylim(chance / 2, max(low_values) * 2.4)
        axBi.set_xticks([0, 1])
        axBi.set_xticklabels(["Pattern\nonly", "Chance"], fontsize=5.0)
        axBi.tick_params(axis="y", labelsize=4.8, length=2)
        axBi.set_title("Low-range inset", fontsize=5.3, pad=1.5)
        axBi.spines["top"].set_visible(True)
        axBi.spines["right"].set_visible(True)
    axB.text(0.5, -0.30, f"{n:,} held-out candidates; points: view draws",
             transform=axB.transAxes, ha="center", va="top", fontsize=5.9,
             color="0.45", style="italic")
    panel_label(axB, "b", dx=-0.2)

    # ── Panels C/D: withheld reconstruction (same drawing as the original) ───
    ci_txt = []
    if have_recon:
        axC = fig.add_subplot(gs[1, 0])
        obs = scat["observed"].astype(np.float32)
        prd = scat["predicted"].astype(np.float32)
        hb = axC.hexbin(obs, prd, gridsize=60, cmap="Blues", mincnt=1,
                        bins="log", extent=(0, 1, 0, 1), linewidths=0.1)
        axC.plot([0, 1], [0, 1], color="0.35", lw=0.7, ls="--")
        axC.text(0.04, 0.95, f"$r$ = {rsum['pearson_withheld_obs_vs_pred']:.3f}",
                 transform=axC.transAxes, va="top", fontsize=7)
        axC.set_xlabel(r"Observed $\beta$ (withheld CpGs)")
        axC.set_ylabel(r"Predicted $\beta$")
        axC.set_xlim(0, 1); axC.set_ylim(0, 1)
        cb2 = fig.colorbar(hb, ax=axC, fraction=0.046, pad=0.03)
        cb2.set_label("positions (log)", fontsize=6.8)
        cb2.ax.tick_params(labelsize=6.8)
        panel_label(axC, "c", dx=-0.14)

        axD = fig.add_subplot(gs[1, 1])
        pm = pd.read_csv(RECON_DIR / "per_sample_metrics.csv")
        conds = [("Model\n(real CLS)", "model_mse", COL_LLAMA),
                 ("Per-CpG\nmean", "b_mean_mse", COL_ENET),
                 ("Shuffled\nCLS", "b_shuffle_mse", COL_ENET),
                 ("Random\nCLS", "b_random_mse", COL_ENET)]
        data = [pm[k].to_numpy() for _, k, _ in conds]
        for j, (arr, (_, _, colr)) in enumerate(zip(data, conds)):
            mean = arr.mean()
            p10, p90 = np.percentile(arr, [10, 90])
            axD.bar(j, mean, width=0.62, color=colr, zorder=2)
            axD.vlines(j, p10, p90, color="0.15", lw=0.9, zorder=3)
            axD.text(j, max(mean, p90) + 0.002, f"{mean:.4f}", ha="center",
                     va="bottom", fontsize=6.8)
        axD.set_xticks(range(len(conds)))
        axD.set_xticklabels([c[0] for c in conds], fontsize=6.8)
        axD.set_ylabel("MSE at withheld CpGs")
        axD.set_ylim(0, max(np.percentile(arr, 90) for arr in data) + 0.012)

        rngb = np.random.default_rng(0); NB = 10000
        m = data[0]
        for (lab, _, _), arr in list(zip(conds, data))[1:]:
            d = arr - m
            bs = np.array([d[rngb.integers(0, len(d), len(d))].mean() for _ in range(NB)])
            lo_, hi_ = np.percentile(bs, [2.5, 97.5])
            ci_txt.append((lab.replace("\n", " "), d.mean(), lo_, hi_))
        axD.text(0.5, -0.30, "bars: mean; whiskers: 10th-90th percentile",
                 transform=axD.transAxes, ha="center", va="top", fontsize=6.2,
                 color="0.45", style="italic")
        panel_label(axD, "d", dx=-0.2)

    save(fig, str(outdir / "fig2_wced_consistency_reconstruction_heldout"))

    def rel(path):
        try:
            return str(Path(path).resolve().relative_to(REPO))
        except ValueError:
            return str(path)

    prov = {
        "population_panels_A_B": summary["population"],
        "checkpoint": summary["checkpoint"],
        "n_profiles": int(n), "chance_top1_percent": chance,
        "panel_A": {"source": rel(viewdir / "per_run/overlap_seed0.npz"), "pool": POOL,
                    "matched_cos_mean": round(pos_mean, 4), "unmatched_cos_mean": round(off_mean, 4),
                    "unmatched_pairs_in_density": int(r0["hist"].sum())},
        "panel_B_top1_percent": {c: {"per_draw": [round(float(x), 4) for x in v],
                                     "mean": round(float(v.mean()), 4)}
                                 for c, v in top1.items() if len(v)},
        "all_pools_from_summary_json": summary["conditions"],
        "panel_C_D": rel(RECON_DIR) if have_recon else "PENDING",
        "panel_D_paired_ci": {lab: {"mean_diff": round(float(md), 5),
                                    "ci95": [round(float(lo_), 5), round(float(hi_), 5)]}
                              for lab, md, lo_, hi_ in ci_txt},
    }
    (outdir / "fig2_heldout_provenance.json").write_text(json.dumps(prov, indent=2))
    print(f"Panel A: matched {pos_mean:.4f}, unmatched {off_mean:.4f} over {int(r0['hist'].sum()):,} pairs")
    for c, v in top1.items():
        if len(v):
            print(f"Panel B: {c:13s} top-1 {v.mean():.2f}% (draws: {', '.join(f'{x:.2f}' for x in v)})  N={n:,}")


if __name__ == "__main__":
    main()
