"""Figure 3 - biological and study structure in frozen representations.

Panels a and b show the same frozen-CLS UMAP, coloured by tissue and dataset.
Panels c and d deliberately separate regression from classification metrics.
Panel e reports study-held-out transfer and names the sample size of each
study. Values are read from saved analysis artifacts.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_GPT, COL_LLAMA, COL_NEG, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
D = REPO / "results/pretrained_cls"
OUT = REPO / "figures/paper"

N_LEG = 6
UMAP_FILE = "cls_umap50_coords.npy"
YFLOOR = -1.3
PAL = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9"]
GREY = "#cfd3d8"


def umap_panel(ax, ax_legend, xy, labels, title):
    top = labels.value_counts().index[:N_LEG]
    other = ~labels.isin(top)
    ax.scatter(xy[other, 0], xy[other, 1], s=1.0, c=GREY, linewidths=0,
               rasterized=True, zorder=1, label=f"Other ({int(other.sum()):,})")
    for i, label in enumerate(top):
        mask = (labels == label).to_numpy()
        ax.scatter(xy[mask, 0], xy[mask, 1], s=1.4, c=PAL[i], linewidths=0,
                   rasterized=True, zorder=2,
                   label=f"{str(label)[:18]} ({int(mask.sum()):,})")
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_linewidth(0.6)
        spine.set_color("0.7")
    ax.set_title(title, fontsize=7.6, pad=3)
    ax.set_aspect("equal", adjustable="datalim")
    handles, names = ax.get_legend_handles_labels()
    ax_legend.axis("off")
    ax_legend.legend(handles, names, frameon=False, fontsize=6.8,
                     loc="center left", bbox_to_anchor=(-0.08, 0.5), ncol=1,
                     markerscale=4.0, handletextpad=0.15, labelspacing=0.22)


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(D / "metadata.csv")
    xy = np.load(D / UMAP_FILE)
    assert len(meta) == len(xy), "metadata / UMAP row mismatch"
    summary = json.loads((D / "analysis_summary.json").read_text())
    lodo = pd.read_csv(D / "lodo_age_probe.csv").dropna(subset=["R2_LODO"])
    ref = json.loads((D / "lodo_reference.json").read_text())["in_distribution_random_80_20"]

    embeddings = np.load(D / "embeddings_cls.npy").astype(np.float64)
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-9
    neighbours = NearestNeighbors(n_neighbors=11).fit(embeddings).kneighbors(embeddings)[1][:, 1:]
    tissue = np.asarray(meta["tissue_type"].astype(str))
    dataset = np.asarray(meta["dataset"].astype(str))
    same_tissue = tissue[neighbours] == tissue[:, None]
    same_dataset = dataset[neighbours] == dataset[:, None]
    knn = {
        "tissue": float(same_tissue.mean()),
        "dataset": float(same_dataset.mean()),
        "both": float((same_tissue & same_dataset).mean()),
        "tissue_only": float((same_tissue & ~same_dataset).mean()),
        "dataset_only": float((same_dataset & ~same_tissue).mean()),
    }
    assert abs(knn["both"] - 0.846) < 0.01, knn["both"]

    # UMAPs are descriptive, so they share the top row with the two probe
    # panels at roughly half their former area; the study-held-out panel
    # takes the full width of the bottom row.
    fig = plt.figure(figsize=(7.2, 4.9))
    outer = GridSpec(2, 1, figure=fig, hspace=0.32, height_ratios=[0.78, 1.0])
    # Each UMAP sits beside its own legend axis, so the map stays square at
    # about half its former area and the legend can never spill sideways.
    top = outer[0].subgridspec(1, 4, wspace=0.05,
                               width_ratios=[1.0, 0.85, 1.0, 0.85])

    ax_a = fig.add_subplot(top[0, 0])
    umap_panel(ax_a, fig.add_subplot(top[0, 1]), xy,
               meta["tissue_type"].astype(str), "Tissue")
    panel_label(ax_a, "a", dx=-0.06, dy=1.02)

    ax_b = fig.add_subplot(top[0, 2])
    umap_panel(ax_b, fig.add_subplot(top[0, 3]), xy,
               meta["dataset"].astype(str), "Dataset")
    panel_label(ax_b, "b", dx=-0.06, dy=1.02)

    bottom = outer[1].subgridspec(1, 3, wspace=0.65,
                                  width_ratios=[0.76, 0.82, 2.0])

    ax_c = fig.add_subplot(bottom[0, 0])
    age_items = [
        ("CLS", summary["age_probe_cls"]["linear_ridge_r2"], COL_LLAMA),
        ("Mean\npooling", summary["age_probe_mean"]["linear_ridge_r2"], COL_ENET),
    ]
    for j, (label, value, colour) in enumerate(age_items):
        ax_c.bar(j, value, width=0.62, color=colour, zorder=2)
        ax_c.text(j, value + 0.025, f"{value:.3f}", ha="center", va="bottom",
                  fontsize=6.8)
    ax_c.set_xticks(range(len(age_items)))
    ax_c.set_xticklabels([item[0] for item in age_items], fontsize=6.8)
    ax_c.set_ylabel("Age probe $R^2$")
    ax_c.set_ylim(0, 1.03)
    ax_c.set_title("Linearly accessible age", fontsize=7.2, pad=4)
    panel_label(ax_c, "c", dx=-0.23)

    ax_d = fig.add_subplot(bottom[0, 1])
    class_probes = summary["class_probes_cls"]
    class_items = [
        ("Tissue\n37 classes", class_probes["tissue_type"]["balanced_acc"],
         class_probes["tissue_type"]["chance"], COL_LLAMA),
        ("Dataset\n85 classes", class_probes["dataset"]["balanced_acc"],
         class_probes["dataset"]["chance"], COL_ENET),
    ]
    for j, (label, value, chance, colour) in enumerate(class_items):
        ax_d.bar(j, value, width=0.62, color=colour, zorder=2)
        ax_d.text(j, value + 0.025, f"{value:.3f}", ha="center", va="bottom",
                  fontsize=6.8)
        ax_d.hlines(chance, j - 0.33, j + 0.33, color="0.15", lw=1.2,
                    linestyle=(0, (3, 1.5)), zorder=3)
        ax_d.text(j, 0.055, f"chance\n{chance:.3f}", ha="center",
                  va="bottom", fontsize=5.8, color="white", fontweight="bold",
                  linespacing=1.05)
    ax_d.set_xticks(range(len(class_items)))
    ax_d.set_xticklabels([item[0] for item in class_items], fontsize=6.8)
    ax_d.set_ylabel("Balanced accuracy")
    ax_d.set_ylim(0, 1.03)
    ax_d.set_title("Frozen-label probes", fontsize=7.2, pad=4)
    panel_label(ax_d, "d", dx=-0.20)

    # Reserve a dedicated label column for panel e so long study identifiers
    # cannot intrude into panel d at manuscript scale.
    e_grid = bottom[0, 2].subgridspec(1, 2, wspace=0.04,
                                      width_ratios=[1.15, 1.85])
    ax_e_labels = fig.add_subplot(e_grid[0, 0])
    ax_e = fig.add_subplot(e_grid[0, 1])
    keep = lodo[(lodo["n"] >= 100) & (lodo["age_std"] >= 10.0)].sort_values("R2_LODO")
    y = np.arange(len(keep))
    r2 = keep["R2_LODO"].to_numpy()
    shown_r2 = np.maximum(r2, YFLOOR)
    ax_e.axvspan(YFLOOR - 0.28, 0, color=COL_NEG, alpha=0.05, zorder=0)
    ax_e.axvline(0, color="0.3", lw=0.9, zorder=1)
    ax_e.axvline(ref["r2"], color=COL_LLAMA, ls="--", lw=1.0, zorder=1)
    # Reference label runs along its own dashed line so it cannot collide
    # with the bottom row or the truncated-value annotation.
    ax_e.text(ref["r2"] + 0.025, (len(keep) - 1) / 2,
              f"mixed-study split {ref['r2']:.3f}", rotation=90,
              ha="left", va="center", fontsize=6.8, color=COL_LLAMA)
    ax_e.hlines(y, YFLOOR - 0.28, shown_r2, color="0.78", lw=0.7, zorder=2)
    ax_e.scatter(shown_r2, y, s=17,
                 c=[COL_NEG if value < 0 else COL_LLAMA for value in r2],
                 zorder=3, edgecolors="white", linewidths=0.4)
    for i, value in enumerate(r2):
        if value < YFLOOR:
            ax_e.annotate(f"{value:.1f}", (YFLOOR, i), xytext=(5, 0),
                          textcoords="offset points", ha="left", va="center",
                          fontsize=6.8, color=COL_NEG, fontweight="bold")
    ax_e.set_yticks([])
    ax_e.set_xlabel("Age probe $R^2$, study held out")
    ax_e.set_xlim(YFLOOR - 0.28, 1.06)
    ax_e.set_ylim(-1.1, len(keep) - 0.3)
    ax_e.spines["left"].set_visible(False)
    ax_e_labels.set_xlim(0, 1)
    ax_e_labels.set_ylim(-1.1, len(keep) - 0.3)
    ax_e_labels.axis("off")
    for i, row in enumerate(keep.itertuples()):
        ax_e_labels.text(0.98, i, f"{row.dataset} (n={int(row.n)})",
                         ha="right", va="center", fontsize=6.8)
    ax_e_labels.text(0.80, 1.05, "Age transfer with study held out",
                     transform=ax_e_labels.transAxes, ha="left", va="bottom",
                     fontsize=7.2)
    ax_e.text(0.5, -0.30,
              f"{len(keep)} studies with $n\\geq100$ and age SD $\\geq10$ y; "
              f"median $R^2$ = {keep['R2_LODO'].median():.3f}",
              transform=ax_e.transAxes, ha="center", va="top", fontsize=6.8,
              color="0.45", style="italic")
    panel_label(ax_e_labels, "e", dx=-0.08)

    save(fig, str(OUT / "fig3_representation_encoding"))

    caption = (
        "\\textbf{Frozen MethylLlama representations encode biological and study-associated "
        "variation.} \\textbf{a,b}, The same 256-dimensional frozen CLS representations "
        "from 10,988 AltuMAge profiles, projected with UMAP and coloured by tissue or dataset. "
        "The six largest categories are named and the remainder are grey. Neighbourhood "
        "statistics were computed in the original 256-dimensional space: "
        f"{knn['both'] * 100:.1f}\\% of ten-nearest-neighbour relations shared both tissue "
        f"and dataset labels, {knn['tissue_only'] * 100:.1f}\\% shared tissue only, and "
        f"{knn['dataset_only'] * 100:.1f}\\% shared dataset only. These values show that the "
        "two sources of structure are strongly entangled in this cohort. \\textbf{c}, "
        "Ridge age probes compare the global CLS representation with mean-pooled contextual "
        "CpG representations. \\textbf{d}, Balanced accuracy of frozen linear probes for "
        "tissue and dataset identity; dashed segments and numeric annotations mark the chance "
        "level for each task. These are separate classification tests with different label "
        "spaces and evaluation sizes: each result is interpreted relative to its own chance "
        "level, not by comparing the two bar heights. \\textbf{e}, Age-probe $R^2$ when one "
        "entire study was excluded from probe fitting and then used only for evaluation, "
        f"for {len(keep)} studies with $n\\geq100$ and age SD $\\geq10$ years. "
        "Study sample sizes are shown in parentheses. "
        "The dashed line is the in-distribution mixed-study random-split reference "
        f"($R^2={ref['r2']:.3f}$), and the shaded region denotes "
        f"negative $R^2$. The axis is truncated at ${YFLOOR}$; a more extreme estimate is "
        f"printed at the boundary. Median study-held-out $R^2$ was "
        f"{keep['R2_LODO'].median():.3f}."
    )
    (OUT / "fig3_representation_encoding_caption.txt").write_text(caption + "\n")
    (OUT / "fig3_representation_encoding_provenance.json").write_text(json.dumps({
        "panel_a_b": {
            "coords": f"results/pretrained_cls/{UMAP_FILE}",
            "metadata": "results/pretrained_cls/metadata.csv",
            "population": "10,988 AltuMAge downstream profiles, not the pretraining corpus",
            "embedding": "256-d frozen pretrained CLS; PCA-50 followed by UMAP for display",
        },
        "knn10_agreement_pct": {key: round(value * 100, 1) for key, value in knn.items()},
        "panel_c": {
            "age_cls_r2": summary["age_probe_cls"]["linear_ridge_r2"],
            "age_mean_pooled_r2": summary["age_probe_mean"]["linear_ridge_r2"],
        },
        "panel_d": {
            "tissue_balanced_accuracy": class_probes["tissue_type"]["balanced_acc"],
            "tissue_chance": class_probes["tissue_type"]["chance"],
            "dataset_balanced_accuracy": class_probes["dataset"]["balanced_acc"],
            "dataset_chance": class_probes["dataset"]["chance"],
            "note": "classification metrics are separated from age regression and are not "
                    "compared across tasks; sex was omitted from the main figure",
        },
        "panel_e": {
            "source": "results/pretrained_cls/lodo_age_probe.csv",
            "reference": ref,
            "n_studies": int(len(keep)),
            "median_r2": round(float(keep["R2_LODO"].median()), 4),
            "display_floor": YFLOOR,
            "sample_sizes_printed": True,
        },
    }, indent=2))


if __name__ == "__main__":
    main()
