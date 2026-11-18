"""
fig_cns_taxonomy.py  (CNS task, stage 5 - the label space)
==========================================================
The 91 Capper methylation classes (left, one row each, ordered by category and then by number of profiles) joined to
the 14 tumour categories used in fig_cns_landscape.py (right; WHO CNS5 chapters, mapping CATEGORY / HGNET there).
Each class is one line that converges on its category's node; node height grows with the number of classes. No counts
are drawn; the class sizes are written to figures/data/fig_cns_taxonomy.csv. This panel only states the task.

Outputs  outputs/cns/figures/{pdf,png,svg}/fig_cns_taxonomy.*, figures/data/fig_cns_taxonomy.csv
  python scripts/cns/fig_cns_taxonomy.py
"""
import argparse
import sys
from pathlib import Path

import pandas as pd
from matplotlib.patches import PathPatch, Rectangle
from matplotlib.path import Path as MPath

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/figures"))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common_style import apply_style  # noqa: E402
from fig_cns_landscape import CATEGORY, COLORS, HGNET  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402

PITCH = 1.0          # one row per class
NODE = 0.42          # node height per class, in rows (lines converge onto a node shorter than their rows)
X0, X1 = 0.03, 1.0   # line start (after the colour dash) and node position


def curve(ax, x0, y0, y1, color):
    xm = (x0 + X1) / 2
    ax.add_patch(PathPatch(MPath([(x0, y0), (xm, y0), (xm, y1), (X1, y1)],
                                 [MPath.MOVETO, MPath.CURVE4, MPath.CURVE4, MPath.CURVE4]),
                           fc="none", ec=color, lw=0.8, alpha=0.5, capstyle="butt"))


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--root", default=str(REPO / "outputs/cns/figures"))
    a = p.parse_args()
    import anndata as ad
    obs = ad.read_h5ad(a.data, backed="r").obs
    cls = obs.methylation_class.astype(str)
    grp = [HGNET.get(c) or CATEGORY[f] for c, f in zip(cls, obs.prefix_group.astype(str))]
    T = pd.DataFrame({"cls": cls.to_numpy(), "group": grp}).value_counts().rename("n").reset_index()
    order = list(COLORS)
    T["k"] = T.group.map(order.index)
    T = T.sort_values(["k", "n"], ascending=[True, False]).reset_index(drop=True)
    if len(T) != 91:
        raise SystemExit(f"expected 91 classes, found {len(T)}")
    out = Path(a.root)
    (out / "data").mkdir(parents=True, exist_ok=True)
    T[["cls", "group", "n"]].to_csv(out / "data" / "fig_cns_taxonomy.csv", index=False)

    apply_style()
    fig, ax = plt.subplots(figsize=(4.6, 7.8))
    T["y"] = -T.index * PITCH
    ax.set_xlim(-0.01, X1 + 0.03)
    ax.set_ylim(T.y.min() - 1, 2.8)
    ends = {}
    for y0, name, g in zip(T.y, T.cls, T.group):
        ax.add_patch(Rectangle((0, y0 - 0.3), 0.022, 0.6, color=COLORS[g], lw=0))
        ends[name] = ax.text(X0 + 0.006, y0, name, ha="left", va="center", fontsize=5.3, color="#2b2b2b")
    fig.canvas.draw()                          # where each name ends, so its line starts right after it
    inv = ax.transData.inverted()
    xend = {k: inv.transform(t.get_window_extent().get_points())[1, 0] + 0.012 for k, t in ends.items()}
    for g in order:
        t = T[T.group == g]
        col = COLORS[g]
        h = NODE * len(t)
        mid = t.y.mean()
        tgt = mid + h / 2 - NODE * (pd.RangeIndex(len(t)) + 0.5)
        for y0, name, y1 in zip(t.y, t.cls, tgt):
            curve(ax, xend[name], y0, y1, col)
        ax.add_patch(Rectangle((X1, mid - h / 2), 0.022, h, color=col, lw=0, zorder=3))
        ax.text(X1 + 0.04, mid, g, ha="left", va="center", fontsize=6.6, color=col, fontweight="bold")
    ax.text(0, 1.6, "91 methylation classes", ha="left", va="bottom", fontsize=7.4)
    ax.text(X1, 1.6, "14 tumour categories", ha="left", va="bottom", fontsize=7.4)
    ax.axis("off")
    for ext in ("pdf", "png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"fig_cns_taxonomy.{ext}", bbox_inches="tight", dpi=400)
    print("saved fig_cns_taxonomy")


if __name__ == "__main__":
    main()
