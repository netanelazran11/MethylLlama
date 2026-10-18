"""
Supplementary figure -- robustness of pretrained representations to random
CpG subsampling.

Motivated by the gene-panel-independence experiment of scConcept (Bahrami
et al. 2025, Fig. 3), where a query set restricted to a targeted gene panel is
shown to co-embed with whole-transcriptome profiles. The motivation carries
over because array generations assay different CpG sets and missing values are
common. The experiment here is deliberately weaker than that claim, and the
wording is kept correspondingly narrow:

  * the subsets are RANDOM 50% draws, not real 450K / EPIC / EPICv2 probe
    panels, so this does not demonstrate cross-platform or technology
    independence;
  * a single view seed is used (the draw saved by two_view_consistency.py);
  * performance is not identical -- the age probe falls from R2 0.863 to
    0.835 -- so this is substantial robustness, not equivalence.

Panels:
  A  Co-embedding of full profiles and 50%-CpG subsets of the same profiles in
     a shared PCA space, with a segment joining each profile to its own subset.
     Short segments mean the subset lands where the full profile does.
  B  Per-profile cosine similarity between the full-profile embedding and the
     embedding of a 50% subset of the same profile, against a shuffled-pairing
     control.
  C  Practical consequence: a ridge age probe fitted on full-profile embeddings
     applied to full versus 50%-subset embeddings of held-out profiles.

Everything is computed from existing artifacts; no new inference is required.
Index alignment between the full-profile and view embeddings is reconstructed
from the generating script's fixed seed and asserted before use.

Usage:  python scripts/figures/figS_panel_independence.py
Output: figures/paper/figS_panel_independence.{pdf,png}
"""

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
from sklearn.linear_model import Ridge
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_ENET, COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
D = REPO / "results/pretrained_cls"
OUTDIR = REPO / "figures/paper"

VIEW_SEED = 0        # two_view_consistency.py: np.random.default_rng(0)
N_SEGMENTS = 250     # profiles drawn for the displacement panel


def unit(x):
    return x / (np.linalg.norm(x, axis=1, keepdims=True) + 1e-9)


def main():
    apply_style()
    OUTDIR.mkdir(parents=True, exist_ok=True)

    full = np.load(D / "embeddings_cls.npy").astype(np.float64)
    v1 = np.load(D / "two_view_v1n.npy").astype(np.float64)
    meta = pd.read_csv(D / "metadata.csv")
    assert len(meta) == full.shape[0], "metadata / embedding row mismatch"

    # Reconstruct which profiles the view embeddings correspond to.
    idx = np.random.default_rng(VIEW_SEED).choice(full.shape[0], v1.shape[0],
                                                  replace=False)
    full_n = unit(full)
    cos_pair = (full_n[idx] * v1).sum(1)
    shuf = np.random.default_rng(1).permutation(len(idx))
    cos_shuf = (full_n[idx][shuf] * v1).sum(1)
    # If the index reconstruction were wrong, paired and shuffled would both
    # sit near the population mean; fail loudly rather than plot nonsense.
    assert cos_pair.mean() > 0.9 and cos_shuf.mean() < 0.7, (
        f"index alignment looks wrong: paired={cos_pair.mean():.3f} "
        f"shuffled={cos_shuf.mean():.3f}")
    print(f"cos(full, own 50% subset): mean={cos_pair.mean():.4f} "
          f"median={np.median(cos_pair):.4f} min={cos_pair.min():.4f}")
    print(f"shuffled control:          mean={cos_shuf.mean():.4f}")

    fig = plt.figure(figsize=(7.2, 2.6))
    gs = GridSpec(1, 3, figure=fig, wspace=0.45, width_ratios=[1.1, 1.0, 0.95])

    # ── Panel A: shared PCA space, full profile -> its own subset ────────────
    axA = fig.add_subplot(gs[0, 0])
    pca = PCA(n_components=2, random_state=0).fit(full_n)   # fitted on full only
    P_full = pca.transform(full_n)
    P_view = pca.transform(v1)
    axA.scatter(P_full[:, 0], P_full[:, 1], s=1.5, color="0.82", linewidths=0,
                rasterized=True, label="full profile")
    rng = np.random.default_rng(0)
    pick = rng.choice(len(idx), size=min(N_SEGMENTS, len(idx)), replace=False)
    for p in pick:
        a, b = P_full[idx[p]], P_view[p]
        axA.plot([a[0], b[0]], [a[1], b[1]], color=COL_LLAMA, lw=0.3, alpha=0.5,
                 zorder=2)
    axA.scatter(P_view[pick, 0], P_view[pick, 1], s=2.5, color=COL_LLAMA,
                linewidths=0, zorder=3, label="50% CpG subset")
    axA.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]*100:.0f}%)")
    axA.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]*100:.0f}%)")
    axA.legend(frameon=False, fontsize=7.0, loc="upper right", markerscale=2.5,
               handletextpad=0.3)
    panel_label(axA, "a", dx=-0.26)

    # ── Panel B: paired vs shuffled cosine ──────────────────────────────────
    axB = fig.add_subplot(gs[0, 1])
    bins = np.linspace(0.3, 1.0, 90)
    axB.hist(cos_shuf, bins=bins, density=True, color="0.65", alpha=0.85,
             label=f"different profiles ({cos_shuf.mean():.3f})")
    axB.hist(cos_pair, bins=bins, density=True, color=COL_LLAMA, alpha=0.85,
             label=f"same profile ({cos_pair.mean():.3f})")
    axB.set_xlabel("Cosine similarity to full-profile embedding")
    axB.set_ylabel("Density")
    axB.legend(frameon=False, fontsize=6.8, loc="upper left")
    panel_label(axB, "b", dx=-0.28)

    # ── Panel C: age probe fitted on full, applied to subsets ───────────────
    axC = fig.add_subplot(gs[0, 2])
    split = meta["split"].to_numpy()
    age = meta["age"].to_numpy(float)
    fit_mask = (split == "train") & np.isfinite(age)
    probe = Ridge(alpha=1.0).fit(full_n[fit_mask], age[fit_mask])

    # evaluate on held-out profiles that also have a subset embedding
    pos_in_view = {int(g): p for p, g in enumerate(idx)}
    eval_g = [g for g in np.where((split == "test") & np.isfinite(age))[0]
              if int(g) in pos_in_view]
    eval_v = [pos_in_view[int(g)] for g in eval_g]
    y = age[eval_g]
    r2_full = r2_score(y, probe.predict(full_n[eval_g]))
    r2_view = r2_score(y, probe.predict(v1[eval_v]))
    med_full = float(np.median(np.abs(probe.predict(full_n[eval_g]) - y)))
    med_view = float(np.median(np.abs(probe.predict(v1[eval_v]) - y)))
    print(f"age probe on n={len(eval_g)} held-out profiles: "
          f"full R2={r2_full:.3f} (MedAE {med_full:.2f}) | "
          f"50% subset R2={r2_view:.3f} (MedAE {med_view:.2f})")

    # Paired bootstrap over the same held-out profiles: the two conditions share
    # subjects, so the difference is what carries uncertainty, not each bar.
    n_boot, rng_b = 5000, np.random.default_rng(0)
    pf = probe.predict(full_n[eval_g])
    pv = probe.predict(v1[eval_v])
    d_r2 = np.empty(n_boot)
    for b in range(n_boot):
        s_ = rng_b.integers(0, len(y), size=len(y))
        d_r2[b] = r2_score(y[s_], pf[s_]) - r2_score(y[s_], pv[s_])
    lo, hi = np.percentile(d_r2, [2.5, 97.5])
    print(f"paired bootstrap dR2 (full - subset): {r2_full - r2_view:+.4f} "
          f"95% CI [{lo:+.4f}, {hi:+.4f}]  (n={len(y)}, {n_boot} resamples)")

    for j, (lab, r2v, colr) in enumerate([("Full\nprofile", r2_full, COL_LLAMA),
                                          ("50% CpG\nsubset", r2_view, COL_ENET)]):
        axC.bar(j, r2v, width=0.55, color=colr)
        axC.text(j, r2v + 0.015, f"{r2v:.3f}", ha="center", va="bottom", fontsize=6.8)
    axC.set_xticks([0, 1])
    axC.set_xticklabels(["Full\nprofile", "50% CpG\nsubset"], fontsize=6.8)
    axC.set_ylabel("Age probe $R^2$ (held-out)")
    axC.set_ylim(0, 1.15)
    axC.set_xlim(-0.6, 1.6)
    axC.text(0.5, 1.10, f"$\\Delta R^2$ = {r2_full - r2_view:.3f}\n"
                        f"95% CI [{lo:.3f}, {hi:.3f}]",
             ha="center", va="top", fontsize=6.8, color="0.25")
    axC.text(0.5, -0.30, f"n = {len(y)} held-out profiles, single view seed",
             transform=axC.transAxes, ha="center", va="top", fontsize=5.8,
             color="0.45", style="italic")
    panel_label(axC, "c", dx=-0.3)

    save(fig, str(OUTDIR / "figS_panel_independence"))

    prov = {
        "inputs": [str((D / f).relative_to(REPO)) for f in
                    ["embeddings_cls.npy", "two_view_v1n.npy", "metadata.csv"]],
        "index_alignment": f"np.random.default_rng({VIEW_SEED}).choice(n_profiles, 2000, replace=False), asserted",
        "cos_full_vs_subset": {"mean": round(float(cos_pair.mean()), 4),
                                 "median": round(float(np.median(cos_pair)), 4),
                                 "min": round(float(cos_pair.min()), 4)},
        "cos_shuffled_control": round(float(cos_shuf.mean()), 4),
        "subsets_are": "random 50% CpG draws, NOT real 450K/EPIC/EPICv2 panels",
        "n_view_seeds": 1,
        "age_probe": {"n_eval": len(eval_g),
                       "r2_full": round(float(r2_full), 4),
                       "r2_subset": round(float(r2_view), 4),
                       "medae_full": round(med_full, 3),
                       "medae_subset": round(med_view, 3),
                       "probe": "Ridge(alpha=1.0) fitted on full-profile embeddings, train split; applied without adaptation to partial-view embeddings",
                       "delta_r2": round(float(r2_full - r2_view), 4),
                       "delta_r2_ci95": [round(float(lo), 4), round(float(hi), 4)],
                       "n_boot": n_boot},
    }
    (OUTDIR / "figS_panel_independence_provenance.json").write_text(json.dumps(prov, indent=2))


if __name__ == "__main__":
    main()
