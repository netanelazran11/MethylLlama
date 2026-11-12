"""
fig3_representation.py -- Figure 3 of the manuscript.

  a  UMAP of the frozen CLS representations of all 10,988 profiles, by tissue (bars: composition)
  b  the boxed region of a: whole-blood profiles coloured by study; E-GEOD-56105 (corrupted downstream
     copy, see Methods) is left out of this panel only
  c  age-probe R^2 with 95 % bootstrap CIs, CLS and mean pooling
  d  balanced accuracy of the tissue and study probes with 95 % bootstrap CIs, chance marked
  e  age probe with one study held out (10 studies; E-GEOD-56105 excluded by the study-level QC)
Coordinates are the stored UMAP of the 10,988 profiles (PCA-50 then UMAP), used as is.
Inputs (results/pretrained_cls/): metadata.csv, cls_umap50_coords.npy, probe_cis.json
(scripts/evaluation/probe_cis.py), lodo_age_probe.csv and lodo_reference.json
(scripts/evaluation/leave_one_dataset_out_probe.py --exclude_study E-GEOD-56105).
Usage:  python scripts/figures/fig3_representation.py
Output: figures/paper/fig3_representation.{pdf,png,svg} + _provenance.json
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_LLAMA, COL_NEG, apply_style, panel_label, save  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.gridspec import GridSpec  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
D = REPO / "results/pretrained_cls"
OUT = REPO / "figures/paper/fig3_representation"
LODO_DIR = D
EXCLUDED = "E-GEOD-56105"
N_LEG = 6
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#a3a9b0"  # remaining categories; non-blood background in panel b is lighter (#eceef1)


PAL_STUDY = ["#882255", "#117733", "#332288", "#DDCC77", "#AA4499", "#44AA99"]  # distinct from the tissue colours
TISSUE_NAME = {"blood whole": "Whole blood", "blood buffy coat": "Buffy coat", "blood cord": "Cord blood",
               "placenta": "Placenta", "kidney": "Kidney", "blood leukocyte": "Leukocytes"}


def study_name(s):
    """Display name: ArrayExpress mirrors E-GEOD-N are the GEO series GSE N; TCGA_X -> TCGA-X."""
    s = str(s)
    if s.startswith("E-GEOD-"):
        return "GSE" + s[len("E-GEOD-"):]
    return s.replace("TCGA_", "TCGA-")


def umap_panel(ax, xy, labels, title, pal, background=None):
    """Scatter coloured by the six largest categories; returns (names, counts, colours) for the bar legend."""
    if background is not None:
        ax.scatter(background[:, 0], background[:, 1], s=0.6, c="#eceef1", linewidths=0, rasterized=True, zorder=0)
    vc = labels.value_counts()
    top = vc.index[:N_LEG]
    other = ~labels.isin(top)
    ax.scatter(xy[other, 0], xy[other, 1], s=1.0, c=GREY, linewidths=0, rasterized=True, zorder=1)
    for i, label in enumerate(top):
        mask = (labels == label).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1.4, c=pal[i], linewidths=0, rasterized=True, zorder=2)
    ax.set_xticks([]); ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True); spine.set_linewidth(0.6); spine.set_color("0.7")
    ax.set_title(title, fontsize=7.6, pad=3)
    return list(top) + ["Other"], [int(vc[t]) for t in top] + [int(other.sum())], pal[:len(top)] + [GREY]


def bar_legend(ax, names, counts, colours, rename=lambda s: s):
    """Composition as a horizontal bar chart (doubles as the colour legend)."""
    y = np.arange(len(names))[::-1]
    ax.barh(y, counts, color=colours, height=0.72)
    for yi, c in zip(y, counts):
        ax.text(c + max(counts) * 0.03, yi, f"{c:,}", va="center", fontsize=6.2, color="0.3")
    ax.set_yticks(y); ax.set_yticklabels([rename(n) if n != "Other" else "Other" for n in names], fontsize=6.6)
    ax.set_xlim(0, max(counts) * 1.45); ax.set_xticks([])
    for side in ("top", "right", "bottom"):
        ax.spines[side].set_visible(False)
    ax.tick_params(axis="y", length=0, pad=2)
    ax.set_xlabel("Profiles", fontsize=6.4, labelpad=2)


def dot_ci(ax, x, v, lo, hi, colour, label_side=0.12):
    ax.vlines(x, lo, hi, color=colour, lw=1.6, zorder=2)
    ax.plot(x, v, "o", ms=5.5, color=colour, mec="white", mew=0.6, zorder=3)
    ax.text(x + label_side, v, f"{v:.3f}", ha="left", va="center", fontsize=6.8)


def main():
    apply_style()
    meta = pd.read_csv(D / "metadata.csv")
    xy = np.load(D / "cls_umap50_coords.npy")
    assert len(meta) == len(xy)
    ci = json.loads((D / "probe_cis.json").read_text())
    lodo = pd.read_csv(LODO_DIR / "lodo_age_probe.csv").dropna(subset=["R2_LODO"])
    ref = json.loads((LODO_DIR / "lodo_reference.json").read_text())["in_distribution_random_80_20"]
    assert EXCLUDED not in set(lodo.dataset)

    fig = plt.figure(figsize=(7.2, 5.1))
    outer = GridSpec(2, 1, figure=fig, hspace=0.34, height_ratios=[0.8, 1.0])
    # [a composition][a map][b map][b composition]: the two maps sit side by side so the zoom connectors stay short
    top = outer[0].subgridspec(1, 5, wspace=0.12, width_ratios=[0.62, 1.0, 1.0, 0.36, 0.62])

    ax_a = fig.add_subplot(top[0, 1])
    tissue = meta["tissue_type"].astype(str)
    na, ca, cola = umap_panel(ax_a, xy, tissue, "All profiles, by tissue", PAL)
    ax_a.set_aspect("equal", adjustable="datalim")
    ax_al = fig.add_subplot(top[0, 0])
    bar_legend(ax_al, na, ca, cola, rename=lambda s: TISSUE_NAME.get(s, s))
    panel_label(ax_al, "a", dx=-0.55, dy=1.02)

    blood = ((meta["tissue_type"] == "blood whole") & (meta["dataset"] != EXCLUDED)).to_numpy()
    ax_b = fig.add_subplot(top[0, 2])
    nb, cb, colb = umap_panel(ax_b, xy[blood], meta.loc[blood, "dataset"].astype(str).reset_index(drop=True),
                              "Whole blood, by study", PAL_STUDY, background=xy)
    lo, hi = np.percentile(xy[blood], 1, axis=0), np.percentile(xy[blood], 99, axis=0)
    pad = (hi - lo) * 0.08
    x0, x1, y0, y1 = lo[0] - pad[0], hi[0] + pad[0], lo[1] - pad[1], hi[1] + pad[1]
    ax_b.set_xlim(x0, x1); ax_b.set_ylim(y0, y1)
    ax_bl = fig.add_subplot(top[0, 4])
    bar_legend(ax_bl, nb, cb, colb, rename=study_name)
    panel_label(ax_b, "b", dx=-0.04, dy=1.02)
    # zoom box on a, joined to the corners of b
    from matplotlib.patches import ConnectionPatch, Rectangle
    ax_a.add_patch(Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec="0.25", lw=0.8, ls=(0, (2, 1.5)), zorder=5))
    for ya, yb in ((y1, 1), (y0, 0)):
        fig.add_artist(ConnectionPatch(xyA=(x1, ya), coordsA=ax_a.transData, xyB=(0, yb), coordsB=ax_b.transAxes,
                                       color="0.45", lw=0.6, ls=(0, (2, 1.5))))

    bottom = outer[1].subgridspec(1, 3, wspace=0.65, width_ratios=[0.76, 0.82, 2.0])

    ax_c = fig.add_subplot(bottom[0, 0])
    for j, (key, colour) in enumerate((("age_cls", COL_LLAMA), ("age_mean", COL_ENET))):
        v = ci[key]
        dot_ci(ax_c, j, v["r2"], *v["ci95"], colour)
    ax_c.set_xticks([0, 1]); ax_c.set_xticklabels(["CLS", "Mean\npooling"], fontsize=6.8)
    ax_c.set_xlim(-0.5, 1.6)
    ax_c.set_ylabel("Age probe $R^2$")
    ax_c.set_ylim(0.6, 1.0)
    ax_c.set_title("Linear age probe", fontsize=7.2, pad=4)
    panel_label(ax_c, "c", dx=-0.23)

    ax_d = fig.add_subplot(bottom[0, 1])
    XD = (0, 1.4)
    for x, (key, colour) in zip(XD, (("tissue_type", COL_LLAMA), ("dataset", COL_ENET))):
        v = ci[key]
        dot_ci(ax_d, x, v["balanced_acc"], *v["ci95"], colour)
        ax_d.hlines(v["chance"], x - 0.3, x + 0.3, color="0.25", lw=1.0, ls=(0, (3, 1.5)), zorder=1)
    ax_d.text(XD[1] + 0.35, ci["dataset"]["chance"] + 0.02, "chance", fontsize=6.4, color="0.35", ha="left", va="bottom")
    ax_d.set_xticks(XD)
    ax_d.set_xticklabels([f"Tissue\n({ci['tissue_type']['n_classes']})", f"Study\n({ci['dataset']['n_classes']})"],
                         fontsize=6.8)
    ax_d.set_xlim(-0.55, 2.3)
    ax_d.set_ylabel("Balanced accuracy")
    ax_d.set_ylim(0, 1.03)
    ax_d.set_title("Frozen linear probes", fontsize=7.2, pad=4)
    panel_label(ax_d, "d", dx=-0.20)

    e_grid = bottom[0, 2].subgridspec(1, 2, wspace=0.04, width_ratios=[1.15, 1.85])
    ax_e_labels, ax_e = fig.add_subplot(e_grid[0, 0]), fig.add_subplot(e_grid[0, 1])
    keep = lodo[(lodo["n"] >= 100) & (lodo["age_std"] >= 10.0)].sort_values("R2_LODO")
    y = np.arange(len(keep)); r2 = keep["R2_LODO"].to_numpy()
    lo_x = min(-1.0, float(r2.min()) - 0.15)
    ax_e.axvspan(lo_x, 0, color=COL_NEG, alpha=0.05, zorder=0)
    ax_e.axvline(0, color="0.3", lw=0.9, zorder=1)
    ax_e.axvline(ref["r2"], color=COL_LLAMA, ls="--", lw=1.0, zorder=1)
    ax_e.text(ref["r2"] + 0.025, (len(keep) - 1) / 2, f"mixed-study split {ref['r2']:.3f}", rotation=90,
              ha="left", va="center", fontsize=6.8, color=COL_LLAMA)
    ax_e.hlines(y, lo_x, r2, color="0.78", lw=0.7, zorder=2)
    ax_e.scatter(r2, y, s=17, c=[COL_NEG if v < 0 else COL_LLAMA for v in r2], zorder=3, edgecolors="white", linewidths=0.4)
    ax_e.set_yticks([]); ax_e.set_xlabel("Age probe $R^2$, study held out")
    ax_e.set_xlim(lo_x, 1.06); ax_e.set_ylim(-0.7, len(keep) - 0.3)
    ax_e.spines["left"].set_visible(False)
    ax_e_labels.set_xlim(0, 1); ax_e_labels.set_ylim(-0.7, len(keep) - 0.3); ax_e_labels.axis("off")
    for i, row in enumerate(keep.itertuples()):
        ax_e_labels.text(0.98, i, f"{study_name(row.dataset)} (n={int(row.n)})", ha="right", va="center", fontsize=6.8)
    ax_e_labels.text(0.80, 1.05, "Age transfer with study held out", transform=ax_e_labels.transAxes,
                     ha="left", va="bottom", fontsize=7.2)
    panel_label(ax_e_labels, "e", dx=-0.08)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    save(fig, str(OUT))
    bl = meta.loc[blood]
    prov = {"file": "fig3_representation.pdf",
            "panel_a": {"profiles": int(len(meta)), "coords": "results/pretrained_cls/cls_umap50_coords.npy (stored)"},
            "panel_b": {"profiles": int(blood.sum()), "studies": int(bl.dataset.nunique()), "tissue": "blood whole",
                        "excluded_study": EXCLUDED, "background": "all other profiles in light grey"},
            "panel_c_d": {"source": "results/pretrained_cls/probe_cis.json", **{k: ci[k] for k in ("age_cls", "age_mean", "tissue_type", "dataset")}},
            "panel_e": {"source": "results/pretrained_cls/lodo_age_probe.csv", "n_studies": int(len(keep)),
                        "median_r2": float(keep["R2_LODO"].median()), "reference_r2": ref["r2"]}}
    Path(f"{OUT}_provenance.json").write_text(json.dumps(prov, indent=1))


if __name__ == "__main__":
    main()
