"""Supplementary figure - representation changes after age fine-tuning.

The two UMAPs are fitted independently and are therefore not compared
point-for-point. Regression and classification probe metrics occupy separate
panels. The figure is a representation diagnostic, not an independent test of
cross-study generalization.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
PRE = REPO / "results/pretrained_cls"
FT = REPO / "results/finetuned_cls"
OUT = REPO / "figures/paper"
CLIP = (2, 98)


def age_locality(embedding, age, k=10):
    x = embedding / (np.linalg.norm(embedding, axis=1, keepdims=True) + 1e-9)
    neighbours = NearestNeighbors(n_neighbors=k + 1).fit(x).kneighbors(x)[1][:, 1:]
    return float(np.abs(age[neighbours] - age[:, None]).mean())


def paired_bars(ax, labels, before, after, ylabel, title):
    x = np.arange(len(labels))
    width = 0.34
    ax.bar(x - width / 2, before, width=width * 0.92, color=COL_ENET,
           zorder=2, label="pretrained")
    ax.bar(x + width / 2, after, width=width * 0.92, color=COL_LLAMA,
           zorder=2, label="fine-tuned")
    for position, left, right in zip(x, before, after):
        ax.text(position - width / 2, left + 0.018, f"{left:.3f}",
                ha="center", va="bottom", fontsize=6.8, color=COL_ENET)
        ax.text(position + width / 2, right + 0.018, f"{right:.3f}",
                ha="center", va="bottom", fontsize=6.8, color=COL_LLAMA)
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=6.8)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel(ylabel, fontsize=6.8)
    ax.set_title(title, fontsize=7.2, pad=4)


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    meta = pd.read_csv(PRE / "metadata.csv")
    age = meta["age"].to_numpy(float)
    rng = np.random.default_rng(0)
    shuffled_gap = float(np.abs(age[rng.permutation(len(age))] - age).mean())

    locality = {}
    for tag, directory in [("pretrained", PRE), ("fine_tuned", FT)]:
        locality[tag] = age_locality(
            np.load(directory / "embeddings_cls.npy").astype(np.float64), age
        )

    pre = json.loads((PRE / "analysis_summary.json").read_text())
    fine = json.loads((FT / "analysis_summary.json").read_text())
    vmin, vmax = np.percentile(age, CLIP)

    fig = plt.figure(figsize=(7.2, 3.0))
    grid = GridSpec(1, 4, figure=fig, wspace=0.52,
                    width_ratios=[1.04, 1.04, 0.72, 1.08])

    first = None
    for j, (tag, directory, title, letter) in enumerate([
        ("pretrained", PRE, "Pretrained encoder", "a"),
        ("fine_tuned", FT, "Age-fine-tuned encoder", "b"),
    ]):
        ax = fig.add_subplot(grid[0, j])
        coordinates = np.load(directory / "cls_umap_coords.npy")
        order = np.argsort(np.abs(age - np.median(age)))
        scatter = ax.scatter(coordinates[order, 0], coordinates[order, 1],
                             c=age[order], s=1.3, cmap="viridis", vmin=vmin,
                             vmax=vmax, linewidths=0, rasterized=True)
        ax.set_xticks([])
        ax.set_yticks([])
        for spine in ax.spines.values():
            spine.set_visible(True)
            spine.set_linewidth(0.6)
            spine.set_color("0.7")
        ax.set_title(title, fontsize=7.2, pad=4)
        ax.set_xlabel(f"mean 10-NN age gap\n{locality[tag]:.1f} y",
                      fontsize=6.8, labelpad=3, color="0.35")
        panel_label(ax, letter, dx=-0.04, dy=1.02)
        if j == 0:
            first = ax
        else:
            colourbar = fig.colorbar(scatter, ax=ax, orientation="horizontal",
                                     fraction=0.055, pad=0.17, aspect=28)
            colourbar.set_label("Age (years)", fontsize=6.8, labelpad=1)
            colourbar.ax.tick_params(labelsize=6.8)
    fig.text(0.055, 0.015,
             f"Independent UMAP fits; shared {CLIP[0]}nd-{CLIP[1]}th percentile colour "
             f"scale. Shuffled-pairing age gap: {shuffled_gap:.1f} y.",
             ha="left", va="bottom", fontsize=6.8, color="0.45", style="italic")

    ax_c = fig.add_subplot(grid[0, 2])
    age_before = [pre["age_probe_cls"]["linear_ridge_r2"]]
    age_after = [fine["age_probe_cls"]["linear_ridge_r2"]]
    paired_bars(ax_c, ["Age"], age_before, age_after,
                "Age probe $R^2$", "Age information")
    panel_label(ax_c, "c", dx=-0.28)

    ax_d = fig.add_subplot(grid[0, 3])
    labels = ["Tissue", "Dataset"]
    class_before = [
        pre["class_probes_cls"]["tissue_type"]["balanced_acc"],
        pre["class_probes_cls"]["dataset"]["balanced_acc"],
    ]
    class_after = [
        fine["class_probes_cls"]["tissue_type"]["balanced_acc"],
        fine["class_probes_cls"]["dataset"]["balanced_acc"],
    ]
    paired_bars(ax_d, labels, class_before, class_after,
                "Balanced accuracy", "Labels remain decodable")
    # Headroom above the bars for the legend, so it never sits on a bar.
    ax_d.set_ylim(0, 1.45)
    ax_d.set_yticks(np.arange(0, 1.01, 0.2))
    ax_d.legend(frameon=False, fontsize=6.8, loc="upper center",
                bbox_to_anchor=(0.5, 1.0), ncol=2, handlelength=1.1,
                columnspacing=0.8)
    panel_label(ax_d, "d", dx=-0.22)

    fig.subplots_adjust(bottom=0.17)

    save(fig, str(OUT / "figS_finetuning_effect"))

    caption = (
        "\\textbf{Representation changes associated with chronological-age fine-tuning.} "
        "\\textbf{a,b}, Age-coloured UMAPs of pretrained and age-fine-tuned CLS "
        "representations for the same 10,988 AltuMAge profiles. Each UMAP was fitted "
        "independently, so positions are not comparable point-for-point; both panels use the "
        f"same age colour scale clipped to the 2nd--98th percentiles. Labels "
        "report the mean absolute age difference between each profile and its ten nearest "
        f"neighbours in the original representation space; the shuffled-pairing reference was "
        f"{shuffled_gap:.1f} years. \\textbf{{c}}, Linear age-probe $R^2$ before and after "
        "fine-tuning. \\textbf{d}, Tissue and dataset balanced accuracy before and after "
        "fine-tuning. These classification tasks contain different numbers of classes and are "
        "not compared directly. The fine-tuned embeddings were extracted from one fold-specific "
        "checkpoint and include profiles used during supervised fine-tuning; this figure is "
        "therefore a representation diagnostic, not an independent generalization test."
    )
    (OUT / "figS_finetuning_effect_caption.txt").write_text(caption + "\n")
    (OUT / "figS_finetuning_effect_provenance.json").write_text(json.dumps({
        "inputs": [
            "results/pretrained_cls/{embeddings_cls.npy,cls_umap_coords.npy,analysis_summary.json}",
            "results/finetuned_cls/{embeddings_cls.npy,cls_umap_coords.npy,analysis_summary.json}",
        ],
        "population": "10,988 AltuMAge profiles; fine-tuned embedding set includes training profiles",
        "finetuned_checkpoint": fine.get("checkpoint", "see analysis_summary.json"),
        "age_neighbour_gap_years": {
            "pretrained": round(locality["pretrained"], 3),
            "fine_tuned": round(locality["fine_tuned"], 3),
            "shuffled_baseline": round(shuffled_gap, 3),
            "k": 10,
        },
        "panel_c_age_r2": {"pretrained": age_before[0], "fine_tuned": age_after[0]},
        "panel_d_balanced_accuracy": {
            label: {"pretrained": before, "fine_tuned": after}
            for label, before, after in zip(labels, class_before, class_after)
        },
        "colour_clip_percentiles": list(CLIP),
        "interpretation_limit": "independent UMAP fits; diagnostic only; no fine-tuned LODO analysis",
    }, indent=2))


if __name__ == "__main__":
    main()
