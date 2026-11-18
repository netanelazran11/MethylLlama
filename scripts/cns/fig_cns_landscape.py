"""
fig_cns_landscape.py  (CNS task, stage 5 - the tumour landscape in the representation)
=====================================================================================
One figure from embeddings that already exist (nothing is retrained):

  a  UMAP of MethylLlama CLS for the 3,902 Capper profiles (reference + validation), coloured by tumour category.
     Categories group the 91 methylation classes by the chapters of the WHO CNS5 classification (2021); the
     mapping is CATEGORY below and is written to figures/data/fig_cns_landscape_categories.csv.
  b  the same representation for two families alone (UMAP refitted on their profiles only):
       ependymal tumours coloured by anatomical compartment (posterior fossa / supratentorial / spinal), the axis
       along which their methylation classes are defined, and medulloblastoma by molecular subgroup.
  c  the full-profile UMAP of each model kept fixed; CLS of the same profiles seen at 100, 20, 5 and 1 % of their
     CpGs (mask seed 0) placed onto it with umap.transform.
  d  in the representation itself (not on the map), as a function of CpGs kept, mean and range over 3 mask seeds:
       share of profiles whose 10 nearest full-profile neighbours (cosine, standardised; the profile's own full
       version excluded) are mostly of its own category, and share of those neighbours of its own class.
  e  the first measure per category and level for MethylLlama (MethylGPT is not shown per category: below 10 % all
     its sparse profiles fall next to one cluster, so a per-category share says nothing about the category).
UMAPs here use min_dist 0.5 (an earlier exploratory map: 0.3) so the small categories stay readable; the measures in d and e
do not depend on the map.

Outputs  outputs/cns/figures/{pdf,png,svg}/fig_cns_landscape.*, figures/data/fig_cns_landscape_*.csv
  python scripts/cns/fig_cns_landscape.py
"""
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/figures"))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import matplotlib.patheffects as pe  # noqa: E402

K = 10
LEVELS = [100, 50, 20, 10, 5, 2, 1, 0.5]
MAP_LEVELS = [100, 20, 5, 1]
SEEDS = [0, 1, 2]

# class prefix -> WHO CNS5 chapter (adult/paediatric diffuse gliomas split by IDH status for readability)
CATEGORY = {
    "A IDH": "IDH-mutant glioma", "O IDH": "IDH-mutant glioma",
    "GBM": "Glioblastoma & paediatric HGG", "DMG": "Glioblastoma & paediatric HGG",
    "IHG": "Glioblastoma & paediatric HGG",
    "LGG": "Low-grade glial & glioneuronal", "ANA PA": "Low-grade glial & glioneuronal",
    "PXA": "Low-grade glial & glioneuronal", "DLGNT": "Low-grade glial & glioneuronal",
    "CN": "Low-grade glial & glioneuronal", "LIPN": "Low-grade glial & glioneuronal",
    "CHGL": "Low-grade glial & glioneuronal", "HGNET": None,
    "EPN": "Ependymal", "SUBEPN": "Ependymal",
    "MB": "Embryonal", "ATRT": "Embryonal", "ETMR": "Embryonal", "CNS NB": "Embryonal", "EFT": "Embryonal",
    "RETB": "Embryonal",
    "PLEX": "Choroid plexus", "PIN T": "Pineal", "PTPR": "Pineal",
    "PITAD": "Sellar", "PITUI": "Sellar", "CPH": "Sellar",
    "MNG": "Meningioma & mesenchymal", "SFT HMPC": "Meningioma & mesenchymal",
    "EWS": "Meningioma & mesenchymal", "CHORDM": "Meningioma & mesenchymal",
    "HMB": "Meningioma & mesenchymal",
    "SCHW": "Nerve sheath & paraganglioma", "PGG": "Nerve sheath & paraganglioma",
    "MELAN": "Melanocytic", "MELCYT": "Melanocytic",
    "LYMPHO": "Haematolymphoid", "PLASMA": "Haematolymphoid",
    "ENB": "Olfactory neuroblastoma",  # sinonasal, outside the CNS5 chapters
    "CONTR": "Non-tumour control",
}
HGNET = {"HGNET, BCOR": "Embryonal", "HGNET, MN1": "Low-grade glial & glioneuronal"}  # MN1 = astroblastoma
COLORS = {
    "Embryonal": "#2E5EAA", "Ependymal": "#2A9D8F", "Glioblastoma & paediatric HGG": "#C0392B",
    "IDH-mutant glioma": "#E67E22", "Low-grade glial & glioneuronal": "#E2B93B", "Choroid plexus": "#8E44AD",
    "Pineal": "#D16BA5", "Sellar": "#5C9A3A", "Meningioma & mesenchymal": "#8D6E63",
    "Nerve sheath & paraganglioma": "#4FA3D9", "Melanocytic": "#2F3E4E", "Haematolymphoid": "#B07AA1",
    "Olfactory neuroblastoma": "#C9A0DC",
    "Non-tumour control": "#A0A0A0",
}
SHORT = {"Glioblastoma & paediatric HGG": "Glioblastoma &\npaediatric HGG",
         "Low-grade glial & glioneuronal": "Low-grade glial &\nglioneuronal",
         "Meningioma & mesenchymal": "Meningioma &\nmesenchymal",
         "Nerve sheath & paraganglioma": "Nerve sheath &\nparaganglioma", "Non-tumour control": "Control tissue",
         "Olfactory neuroblastoma": "Olfactory\nneuroblastoma"}
EPN_SITE = {"EPN, PF A": "PF", "EPN, PF B": "PF", "SUBEPN, PF": "PF", "EPN, RELA": "ST", "EPN, YAP": "ST",
            "SUBEPN, ST": "ST", "EPN, SPINE": "SP", "EPN, MPE": "SP", "SUBEPN, SPINE": "SP"}
EPN_COL = {"PF": "#1B7F79", "ST": "#7B4FA0", "SP": "#D08A2E"}
EPN_SITE_NAME = {"PF": "Posterior fossa", "ST": "Supratentorial", "SP": "Spinal"}
EPN_LAB = {"EPN, PF A": "PFA", "EPN, PF B": "PFB", "SUBEPN, PF": "SE-PF", "EPN, RELA": "ZFTA",
           "EPN, YAP": "YAP1", "SUBEPN, ST": "SE-ST", "EPN, SPINE": "SP-EPN", "EPN, MPE": "MPE",
           "SUBEPN, SPINE": "SE-SP"}
MB_COL = {"MB, WNT": "#3A86C8", "MB, SHH INF": "#C0392B", "MB, SHH CHL AD": "#E8836B", "MB, G3": "#F2B134",
          "MB, G4": "#4E9A53"}
MB_LAB = {"MB, WNT": "WNT", "MB, SHH INF": "SHH infant", "MB, SHH CHL AD": "SHH child/adult", "MB, G3": "Group 3",
          "MB, G4": "Group 4"}
# label positions on the panel-a map (UMAP seed 0), as fractions of the map span from its lower-left corner
LABEL_AT = {"Pineal": (0.36, 1.03), "Olfactory neuroblastoma": (0.82, 0.99), "Embryonal": (0.30, 0.86),
            "Choroid plexus": (0.90, 0.81), "Sellar": (1.02, 0.66), "Glioblastoma & paediatric HGG": (0.86, 0.56),
            "Haematolymphoid": (1.02, 0.45), "Low-grade glial & glioneuronal": (0.10, 0.64),
            "Ependymal": (-0.02, 0.41), "Non-tumour control": (0.12, 0.27),
            "Nerve sheath & paraganglioma": (0.34, 0.17), "Melanocytic": (0.90, 0.31),
            "IDH-mutant glioma": (0.92, 0.12), "Meningioma & mesenchymal": (0.20, 0.03)}
# panel b labels, same convention, per sub-map
EPN_AT = {"EPN, RELA": (0.93, 0.72), "EPN, YAP": (0.40, 0.68), "SE": (0.30, 0.50), "EPN, PF A": (0.08, 0.34),
          "EPN, PF B": (0.55, -0.06), "EPN, MPE": (1.08, 0.52), "EPN, SPINE": (1.08, 0.12)}
MB_AT = {"MB, WNT": (0.66, 0.98), "MB, SHH INF": (0.06, -0.07), "MB, SHH CHL AD": (0.08, 0.58),
         "MB, G3": (0.50, 0.48), "MB, G4": (0.92, 0.22)}
HALO = [pe.withStroke(linewidth=2.2, foreground="white")]


def save(fig, out, stem):
    for ext in ("pdf", "png", "svg"):
        (out / ext).mkdir(parents=True, exist_ok=True)
        fig.savefig(out / ext / f"{stem}.{ext}", bbox_inches="tight", dpi=400)
    print("saved", stem)


def aligned(d, ids):
    meta = pd.read_csv(Path(d) / "metadata.csv").sample_id.astype(str)
    idx = pd.Series(np.arange(len(meta)), index=meta)
    return np.load(Path(d) / "embeddings_cls.npy").astype(np.float32)[idx.loc[ids].to_numpy()]


def unit(Z):
    return Z / (np.linalg.norm(Z, axis=1, keepdims=True) + 1e-9)


def knn_excl_self(Q, R, k):
    """k nearest rows of R (cosine) for each row of Q, row i of R excluded for row i of Q (Q, R same profiles)."""
    S = unit(Q) @ unit(R).T
    np.fill_diagonal(S, -np.inf)
    return np.argsort(-S, axis=1)[:, :k]


def umap_fit(Z, n_neighbors=30, min_dist=0.5, seed=0):
    import umap
    return umap.UMAP(n_neighbors=n_neighbors, min_dist=min_dist, metric="cosine", random_state=seed).fit(Z)


def repel(anchors, targets, span, dmin=0.11, aspect=0.45, it=400):
    """Move label positions apart (x distances weighted by aspect, since labels are wide) while pulling each back
    towards its target; returns the new positions."""
    P = np.array(targets, float)
    w = np.array([aspect, 1.0])
    for _ in range(it):
        for i in range(len(P)):
            for j in range(i + 1, len(P)):
                d = (P[i] - P[j]) * w / span
                n = np.linalg.norm(d)
                if n < dmin:
                    step = (dmin - n) / 2 * span * (d / (n + 1e-9)) / w
                    P[i] += step * 0.5
                    P[j] -= step * 0.5
        P += 0.02 * (np.array(targets) - P)
    return P


def labels(ax, xy_by_key, text, color, span, push=0.10, dmin=0.11, fs=6.9, leader=True):
    keys = list(xy_by_key)
    A = np.array([xy_by_key[k] for k in keys])
    c = A.mean(0)
    T = A + push * span * (A - c) / (np.linalg.norm(A - c, axis=1, keepdims=True) + 1e-9)
    P = repel(A, T, span, dmin)
    for k, a0, p1 in zip(keys, A, P):
        ax.annotate(text[k], a0, p1, fontsize=fs, color=color[k], fontweight="bold", ha="center", va="center",
                    path_effects=HALO, arrowprops=dict(arrowstyle="-", lw=0.5, color=color[k], shrinkA=0,
                                                       shrinkB=1.0) if leader else None)


def clean(ax):
    ax.set_xticks([])
    ax.set_yticks([])
    for sp in ax.spines.values():
        sp.set_visible(False)


def anchor(xy, r):
    """Point of xy with the most other points of xy within radius r (the dense core, robust to split clusters)."""
    d = np.linalg.norm(xy[:, None] - xy[None], axis=2)
    return xy[np.argmax((d < r).sum(1))]


def callouts(ax, xy, masks, text, color, at, fs=6.4):
    """Label each group at its dense core with a leader line to a fixed position outside the points (at: fractions
    of the map span from its lower-left corner)."""
    span, lo = np.ptp(xy, 0).max(), xy.min(0)
    for k, m in masks.items():
        ax.annotate(text[k], anchor(xy[m], 0.05 * span), lo + span * np.array(at[k]), fontsize=fs, color=color[k],
                    fontweight="bold", ha="center", va="center", path_effects=HALO,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=color[k], shrinkA=2, shrinkB=1.5))
    return span, lo


def scatter_cat(ax, xy, cat, s=2.0, alpha=0.85):
    for c, col in COLORS.items():
        m = cat == c
        ax.scatter(xy[m, 0], xy[m, 1], s=s, c=col, lw=0, alpha=alpha, rasterized=True)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--root", default=str(REPO / "outputs/cns/figures"))
    a = p.parse_args()
    import anndata as ad
    obs = ad.read_h5ad(a.data, backed="r").obs
    ids = obs.index.astype(str).to_numpy()
    cls = obs.methylation_class.astype(str).to_numpy()
    pre = obs.prefix_group.astype(str).to_numpy()
    cat = np.array([HGNET.get(c) or CATEGORY[f] for c, f in zip(cls, pre)])
    out = Path(a.root)
    (out / "data").mkdir(parents=True, exist_ok=True)
    pd.DataFrame({"methylation_class": cls, "category": cat}).drop_duplicates().sort_values(
        ["category", "methylation_class"]).to_csv(out / "data" / "fig_cns_landscape_categories.csv", index=False)

    full = {"llama": aligned(REPO / "outputs/cns/cls_methylllama", ids),
            "gpt": aligned(REPO / "outputs/cns/cls_methylgpt", ids)}
    sc = {k: StandardScaler().fit(v) for k, v in full.items()}
    Z = {k: sc[k].transform(v) for k, v in full.items()}
    sparse = {}
    for k, d in (("llama", "emb_llama"), ("gpt", "emb_gpt")):
        for lv in LEVELS:
            for s in SEEDS:
                sparse[(k, lv, s)] = Z[k] if lv == 100 else sc[k].transform(
                    aligned(REPO / f"outputs/cns/sparse/{d}/L{lv:g}_s{s}", ids))

    rows, per_cat = [], []
    for (k, lv, s), S in sparse.items():
        if lv == 100 and s > 0:
            continue
        nn = knn_excl_self(S, Z[k], K)
        maj = np.array([pd.Series(r).mode().iloc[0] for r in cat[nn]])
        kept = maj == cat
        rows.append({"model": k, "level": lv, "seed": s, "category_majority": kept.mean(),
                     "class_knn": (cls[nn] == cls[:, None]).mean()})
        if k == "llama":
            per_cat += [{"level": lv, "seed": s, "category": c, "n": int((cat == c).sum()),
                         "category_majority": kept[cat == c].mean()} for c in COLORS]
    M = pd.DataFrame(rows)
    M.to_csv(out / "data" / "fig_cns_landscape_neighbours.csv", index=False)
    PC = pd.DataFrame(per_cat).groupby(["level", "category", "n"], as_index=False).category_majority.mean()
    PC.to_csv(out / "data" / "fig_cns_landscape_per_category_llama.csv", index=False)
    print(M.groupby(["model", "level"])[["category_majority", "class_knn"]].mean().round(3), flush=True)

    maps = {k: umap_fit(Z[k]) for k in Z}
    placed = {(k, lv): maps[k].embedding_ if lv == 100 else maps[k].transform(sparse[(k, lv, 0)])
              for k in Z for lv in MAP_LEVELS}
    epn = np.isin(cls, list(EPN_SITE))
    mb = np.isin(cls, list(MB_COL))
    sub = {"epn": umap_fit(Z["llama"][epn], 10, 0.7).embedding_, "mb": umap_fit(Z["llama"][mb], 10, 0.7).embedding_}
    pd.concat([pd.DataFrame({"panel": "a", "gsm": ids, "category": cat, "x": maps["llama"].embedding_[:, 0],
                             "y": maps["llama"].embedding_[:, 1]}),
               pd.DataFrame({"panel": "b_epn", "gsm": ids[epn], "category": cls[epn], "x": sub["epn"][:, 0],
                             "y": sub["epn"][:, 1]}),
               pd.DataFrame({"panel": "b_mb", "gsm": ids[mb], "category": cls[mb], "x": sub["mb"][:, 0],
                             "y": sub["mb"][:, 1]})]).to_csv(out / "data" / "fig_cns_landscape_coords.csv",
                                                             index=False)

    # ------------------------------------------------------------------------------------------------ draw
    apply_style()
    fig = plt.figure(figsize=(7.2, 9.9))
    G = fig.add_gridspec(5, 1, height_ratios=[3.4, 1.5, 1.5, 0.45, 2.05], hspace=0.26)
    top = G[0].subgridspec(2, 2, width_ratios=[1.85, 1], hspace=0.22, wspace=0.04)

    # a: the landscape
    ax = fig.add_subplot(top[:, 0])
    xy = maps["llama"].embedding_
    scatter_cat(ax, xy, cat, s=4.5)
    span, lo = np.ptp(xy, 0).max(), xy.min(0)
    for c in COLORS:
        a0 = anchor(xy[cat == c], 0.04 * span)
        ax.annotate(SHORT.get(c, c), a0, lo + span * np.array(LABEL_AT[c]), fontsize=6.9, color=COLORS[c],
                    fontweight="bold", ha="center", va="center", path_effects=HALO,
                    arrowprops=dict(arrowstyle="-", lw=0.5, color=COLORS[c], shrinkA=2, shrinkB=1.5))
    ax.set_xlim(lo[0] - 0.17 * span, lo[0] + 1.13 * span)
    ax.set_ylim(lo[1] - 0.05 * span, lo[1] + 1.08 * span)
    ax.set_aspect("equal", adjustable="box")
    clean(ax)
    ax.set_title("MethylLlama: 3,902 profiles, 91 methylation classes", fontsize=8.2, loc="left", pad=14)
    full_row = M[(M.model == "llama") & (M.level == 100)].iloc[0]
    ax.text(0.0, 1.01, f"10 nearest neighbours of the same class: {full_row.class_knn:.0%} "
                      f"(MethylGPT {M[(M.model == 'gpt') & (M.level == 100)].class_knn.iloc[0]:.0%})",
            transform=ax.transAxes, fontsize=6.8, color="#555555", va="bottom")
    panel_label(ax, "a", dx=-0.01)

    # b: two families alone
    axe = fig.add_subplot(top[0, 1])
    ce = cls[epn]
    sp_e = np.ptp(sub["epn"], 0).max()
    for c in EPN_SITE:
        m = ce == c
        axe.scatter(*sub["epn"][m].T, s=6, c=EPN_COL[EPN_SITE[c]], lw=0, alpha=0.85,
                    marker="o" if c.startswith("EPN") else "^", rasterized=True)
    se = np.char.startswith(ce.astype(str), "SUBEPN")
    masks = {c: ce == c for c in EPN_SITE if c.startswith("EPN")}
    masks["SE"] = se
    span, lo = callouts(axe, sub["epn"], masks, {**EPN_LAB, "SE": "subependymoma\n(all sites)"},
                        {**{c: EPN_COL[EPN_SITE[c]] for c in EPN_SITE}, "SE": "#555555"}, EPN_AT)
    axe.set_xlim(lo[0] - 0.12 * span, lo[0] + 1.2 * span)
    axe.set_ylim(lo[1] - 0.13 * span, lo[1] + 0.8 * span)
    axe.set_title("Ependymal tumours, by site", fontsize=7.6, loc="left")
    axe.legend(handles=[plt.Line2D([], [], marker="o", ls="", ms=3.5, color=v, label=EPN_SITE_NAME[k])
                        for k, v in EPN_COL.items()] +
               [plt.Line2D([], [], marker="^", ls="", ms=3.5, color="#777777", label="subependymoma")],
               frameon=False, fontsize=6.3, loc="upper left", bbox_to_anchor=(1.0, 1.0), handletextpad=0.1,
               borderaxespad=0)
    axm = fig.add_subplot(top[1, 1])
    cm = cls[mb]
    sp_m = np.ptp(sub["mb"], 0).max()
    for c, col in MB_COL.items():
        axm.scatter(*sub["mb"][cm == c].T, s=6, c=col, lw=0, alpha=0.85, rasterized=True)
    span, lo = callouts(axm, sub["mb"], {c: cm == c for c in MB_COL}, MB_LAB, MB_COL, MB_AT)
    axm.set_xlim(lo[0] - 0.2 * span, lo[0] + 1.08 * span)
    axm.set_ylim(lo[1] - 0.13 * span, lo[1] + 1.06 * span)
    axm.set_title("Medulloblastoma, by subgroup", fontsize=7.6, loc="left")
    for x in (axe, axm):
        x.set_aspect("equal", adjustable="box")
        clean(x)
        for sp in x.spines.values():
            sp.set_visible(True)
            sp.set_color("#d0d0d0")
            sp.set_linewidth(0.6)
    panel_label(axe, "b", dx=-0.04)

    # c: profiles with fewer CpGs placed on the fixed full-profile map
    for i, k in enumerate(("llama", "gpt")):
        row = G[1 + i].subgridspec(1, len(MAP_LEVELS), wspace=0.04)
        base = maps[k].embedding_
        lim = None
        for j, lv in enumerate(MAP_LEVELS):
            x = fig.add_subplot(row[j])
            if lv < 100:
                x.scatter(*base.T, s=1.5, c="#ebebeb", lw=0, rasterized=True)
            scatter_cat(x, placed[(k, lv)], cat, s=2.4)
            if lim is None:
                x.set_aspect("equal", adjustable="datalim")
                x.margins(0.04)
                fig.canvas.draw()
                lim = x.get_xlim(), x.get_ylim()
            else:
                x.set_xlim(lim[0])
                x.set_ylim(lim[1])
                x.set_aspect("equal", adjustable="box")
            clean(x)
            r = M[(M.model == k) & (M.level == lv) & (M.seed == 0)].iloc[0]
            x.text(0.5, -0.02, f"{r.category_majority:.0%} near own category", transform=x.transAxes, fontsize=7,
                   va="top", ha="center",
                   fontweight="bold", color={"llama": COL_LLAMA, "gpt": COL_GPT}[k])
            if i == 0:
                x.set_title("full profile" if lv == 100 else f"{lv:g} % of CpGs", fontsize=7.4)
            if j == 0:
                x.set_ylabel({"llama": "MethylLlama", "gpt": "MethylGPT"}[k], fontsize=7.8,
                             color={"llama": COL_LLAMA, "gpt": COL_GPT}[k])
                x.yaxis.label.set_visible(True)
                if i == 0:
                    panel_label(x, "c", dx=-0.1)
    fig.text(0.5, 0.0, "", fontsize=1)

    note = fig.add_subplot(G[3])
    note.axis("off")
    note.text(0.0, 0.75, "Every map: the same 3,902 profiles, one 256-d CLS vector each, for both models. MethylLlama "
                        "places profiles of one class almost on one spot\n(63 % of profiles lie within 0.2 % of the map "
                        "width of another, MethylGPT 19 %), so its clusters look smaller.",
              transform=note.transAxes, fontsize=6.6, color="#555555", va="center", ha="left")

    # d, e: in the representation
    bot = G[4].subgridspec(1, 2, width_ratios=[1, 1.15], wspace=0.95)
    axd = fig.add_subplot(bot[0])
    for k, col in (("llama", COL_LLAMA), ("gpt", COL_GPT)):
        for meas, ls, mk in (("category_majority", "-", "o"), ("class_knn", "--", "s")):
            t = M[M.model == k].groupby("level")[meas].agg(["mean", "min", "max"]).reset_index()
            axd.fill_between(t.level, t["min"], t["max"], color=col, alpha=0.18, lw=0)
            axd.plot(t.level, t["mean"], ls=ls, marker=mk, ms=2.8, lw=1.2, color=col,
                     mfc=col if meas == "category_majority" else "white")
    axd.set_xscale("log")
    axd.set_xticks(LEVELS, [f"{x:g}" for x in LEVELS])
    axd.minorticks_off()
    axd.invert_xaxis()
    axd.set_ylim(0, 1.02)
    axd.set_xlabel("CpGs measured per profile (%)")
    axd.set_ylabel("Share kept in the representation")
    axd.legend(handles=[plt.Line2D([], [], color=COL_LLAMA, lw=1.4, label="MethylLlama"),
                        plt.Line2D([], [], color=COL_GPT, lw=1.4, label="MethylGPT"),
                        plt.Line2D([], [], color="#555555", marker="o", ms=3, lw=1, label="own category (majority)"),
                        plt.Line2D([], [], color="#555555", ls="--", marker="s", ms=3, mfc="white", lw=1,
                                   label="own class (of 10)")],
               frameon=False, fontsize=6.3, loc="lower left", bbox_to_anchor=(0, 1.04), ncol=2, handlelength=1.8,
               columnspacing=1.0, borderaxespad=0)
    panel_label(axd, "d", dx=-0.2)

    axc = fig.add_subplot(bot[1])
    H = PC.pivot(index="category", columns="level", values="category_majority")[sorted(LEVELS, reverse=True)]
    H = H.sort_values(1, ascending=False)
    im = axc.imshow(H.to_numpy(), cmap="RdYlBu", vmin=0, vmax=1, aspect="auto")
    for i in range(H.shape[0]):
        for j in range(H.shape[1]):
            v = H.iat[i, j]
            axc.text(j, i, f"{v:.2f}"[1:] if v < 0.995 else "1", ha="center", va="center", fontsize=5.6,
                     color="white" if (v > 0.85 or v < 0.15) else "#222222")
    axc.set_xticks(range(H.shape[1]), [f"{x:g}" for x in H.columns])
    axc.set_yticks(range(H.shape[0]), H.index, fontsize=6.3)
    for t, c in zip(axc.get_yticklabels(), H.index):
        t.set_color(COLORS[c])
    axc.tick_params(length=0)
    for sp in axc.spines.values():
        sp.set_visible(False)
    axc.set_xlabel("CpGs measured per profile (%)")
    axc.set_title("MethylLlama, own category kept", fontsize=7.4, loc="left")
    cb = fig.colorbar(im, ax=axc, fraction=0.05, pad=0.03, ticks=[0, 0.5, 1])
    cb.ax.tick_params(labelsize=6.3, length=2)
    cb.outline.set_visible(False)
    cb.set_label("share near own category", fontsize=6.5)
    panel_label(axc, "e", dx=-0.02)
    save(fig, out, "fig_cns_landscape")
    plt.close(fig)


if __name__ == "__main__":
    main()
