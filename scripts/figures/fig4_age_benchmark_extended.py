"""Figure 4 - chronological-age benchmark on the common held-out cohort.

The main figure combines the prespecified held-out benchmark with two clearly
labelled secondary diagnostics: a single-fold validation comparison of the
pretrained and from-scratch training protocols, and nested post-hoc age
restrictions of the paired test-set comparison.
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import r2_score

sys.path.insert(0, str(Path(__file__).parent))
from common_style import COL_GPT, COL_LLAMA, apply_style, panel_label, save
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

REPO = Path(__file__).resolve().parents[2]
# Shipped copies of outputs/bootstrap_predictions/ and kfold_full_history_analysis/ (see results/README.md)
RES = REPO / "results/age_benchmark"
PAIRED = RES / "paired_per_subject_predictions.csv"
BOOT = RES / "paired_bootstrap_summary.json"
AGE_BOOT = RES / "age_restricted_paired_bootstrap_summary.json"
FOLDS = RES / "methyllama"
OUT = REPO / "figures/paper"
PRETRAINED_CURVE = RES / "fold0_train_history.csv"
RANDOM_CURVE = RES / "random_init_fold0_history.csv"
CURVE_LIMIT = 165
CURVE_START = 20
MATCHED_EPOCH = 148
COL_RANDOM = "#6f6f6f"


def medae(prediction, truth):
    return float(np.median(np.abs(prediction - truth)))


def main():
    apply_style()
    OUT.mkdir(parents=True, exist_ok=True)

    data = pd.read_csv(PAIRED, dtype={"sample_id": str})
    assert len(data) == 2149
    bootstrap = json.loads(BOOT.read_text())
    age_bootstrap = json.loads(AGE_BOOT.read_text())
    gap = bootstrap["paired_bootstrap_gap_gpt_minus_llama"]

    folds = []
    for path in sorted(FOLDS.glob("fold_*_predictions.csv")):
        frame = pd.read_csv(path)
        prediction = [column for column in frame.columns if "pred" in column.lower()][0]
        truth = [column for column in frame.columns
                 if "true" in column.lower() or column.lower() == "age"][0]
        folds.append({
            "fold": int(path.stem.split("_")[1]),
            "medae": medae(frame[prediction], frame[truth]),
            "mae": float(np.mean(np.abs(frame[prediction] - frame[truth]))),
            "r2": float(r2_score(frame[truth], frame[prediction])),
        })
    folds = pd.DataFrame(folds).sort_values("fold")
    assert len(folds) == 5

    pretrained_curve = pd.read_csv(PRETRAINED_CURVE).sort_values("epoch")
    random_curve = pd.read_csv(RANDOM_CURVE).sort_values("epoch")
    shared_end = int(min(CURVE_LIMIT, pretrained_curve["epoch"].max(),
                         random_curve["epoch"].max()))
    pretrained_display = pretrained_curve[
        pretrained_curve["epoch"].between(CURVE_START, shared_end)
    ].copy()
    random_display = random_curve[
        random_curve["epoch"].between(CURVE_START, shared_end)
    ].copy()
    pretrained_at_match = pretrained_curve.loc[
        pretrained_curve["epoch"] == MATCHED_EPOCH
    ].iloc[0]
    random_at_match = random_curve.loc[random_curve["epoch"] == MATCHED_EPOCH].iloc[0]
    pretrained_best = pretrained_curve.loc[pretrained_curve["val/medae"].idxmin()]
    random_best = random_curve.loc[random_curve["val/medae"].idxmin()]

    fig = plt.figure(figsize=(7.2, 6.55))
    outer = GridSpec(3, 12, figure=fig, hspace=0.82, wspace=0.95,
                     height_ratios=[0.92, 0.88, 0.82])

    grid_a = outer[0, :5].subgridspec(1, 3, wspace=1.10)
    first_a = None
    for i, (column, label) in enumerate([
        ("medae", "MedAE (y)"),
        ("mae", "MAE (y)"),
        ("r2", "$R^2$"),
    ]):
        ax = fig.add_subplot(grid_a[0, i])
        values = folds[column].to_numpy()
        ax.scatter(np.linspace(-0.11, 0.11, len(values)), values, s=16,
                   color=COL_LLAMA, zorder=3, edgecolors="white", linewidths=0.4)
        ax.hlines(values.mean(), -0.30, 0.30, color=COL_LLAMA, lw=1.3, zorder=2)
        ax.set_xlim(-0.5, 0.5)
        ax.set_xticks([])
        ax.set_ylabel(label, fontsize=6.8)
        spread = (values.max() - values.min()) * 1.55 + 1e-3
        ax.set_ylim(values.mean() - spread, values.mean() + spread)
        ax.tick_params(labelsize=6.8)
        if i == 0:
            first_a = ax
    first_a.text(0.0, 1.24, "five training folds; common held-out test cohort",
                 transform=first_a.transAxes, fontsize=6.8, color="0.45",
                 style="italic")
    panel_label(first_a, "a", dx=-0.62, dy=1.24)

    grid_b = outer[0, 5:].subgridspec(1, 2, wspace=0.12)
    first_b = None
    observed = data["true_age"].to_numpy()
    for i, (column, name, colour) in enumerate([
        ("predicted_age_llama", "MethylLlama", COL_LLAMA),
        ("predicted_age_gpt", "MethylGPT", COL_GPT),
    ]):
        ax = fig.add_subplot(grid_b[0, i])
        predicted = data[column].to_numpy()
        ax.scatter(observed, predicted, s=1.7, color=colour, alpha=0.30,
                   linewidths=0, rasterized=True)
        ax.plot([0, 103], [0, 103], ls="--", lw=0.8, color="0.35", zorder=3)
        ax.set_xlim(-4, 108)
        ax.set_ylim(-4, 108)
        ax.set_aspect("equal", adjustable="box")
        ax.set_xlabel("Chronological age (y)", fontsize=6.8)
        ax.tick_params(labelsize=6.8)
        if i == 0:
            ax.set_ylabel("Predicted age (y)", fontsize=6.8)
            first_b = ax
        else:
            ax.set_yticklabels([])
        ax.text(0.04, 0.96, f"{name}\nMedAE {medae(predicted, observed):.2f} y",
                transform=ax.transAxes, ha="left", va="top", fontsize=6.8,
                color=colour, linespacing=1.2)
    panel_label(first_b, "b", dx=-0.26, dy=1.02)

    # Panel c: one-fold validation trajectories for the pretrained and
    # from-scratch protocols. Raw values remain visible beneath a short rolling
    # median so smoothing never conceals the observations.
    ax_c = fig.add_subplot(outer[1, 1:11])
    for frame, colour, label in [
        (pretrained_display, COL_LLAMA, "WCED-pretrained protocol"),
        (random_display, COL_RANDOM, "From-scratch protocol"),
    ]:
        smooth = frame["val/medae"].rolling(7, center=True, min_periods=1).median()
        ax_c.plot(frame["epoch"], frame["val/medae"], color=colour, lw=0.45,
                  alpha=0.28, zorder=1)
        ax_c.plot(frame["epoch"], smooth, color=colour, lw=1.25,
                  label=label, zorder=3)
    ax_c.axvline(MATCHED_EPOCH, color="0.72", ls=":", lw=0.8, zorder=0)
    for row, colour, y_offset in [
        (pretrained_at_match, COL_LLAMA, -0.42),
        (random_at_match, COL_RANDOM, 0.48),
    ]:
        ax_c.scatter(row["epoch"], row["val/medae"], s=24, color=colour,
                     edgecolors="white", linewidths=0.5, zorder=5)
        # Offset well clear of the curve, with a white backing so the value
        # stays legible where the raw trace is noisy.
        ax_c.text(row["epoch"] + 2.0, row["val/medae"] + y_offset,
                  f"{row['val/medae']:.3f}", ha="left", va="center",
                  fontsize=6.8, color=colour, zorder=6,
                  bbox=dict(facecolor="white", edgecolor="none",
                            alpha=0.85, pad=0.8))
    ax_c.set_xlim(CURVE_START, CURVE_LIMIT)
    ax_c.set_ylim(2.3, 8.0)
    ax_c.set_xlabel("Fine-tuning epoch")
    ax_c.set_ylabel("Validation MedAE (years)")
    ax_c.set_title("Pretrained and from-scratch training protocols (fold 0)",
                   fontsize=7.2, pad=4)
    ax_c.legend(frameon=False, fontsize=6.8, loc="upper right", ncol=1,
                handlelength=1.4)
    panel_label(ax_c, "c", dx=-0.08)

    # Panel d: paired test-set improvements for the two error metrics. R^2 is
    # retained in Table 1 but omitted here to keep a single interpretable unit.
    ax_d = fig.add_subplot(outer[2, :6])
    forest = []
    error_specs = [("medae", "MedAE"), ("mae", "MAE")]
    y_effect = np.arange(len(error_specs))[::-1]
    for yy, (key, label) in zip(y_effect, error_specs):
        estimate = gap[key]["point_estimate"]
        low, high = sorted([
            gap[key]["ci_95_low"],
            gap[key]["ci_95_high"],
        ])
        ax_d.hlines(yy, low, high, color=COL_LLAMA, lw=2.1, zorder=2)
        ax_d.plot([low, low], [yy - 0.10, yy + 0.10], color=COL_LLAMA, lw=0.9)
        ax_d.plot([high, high], [yy - 0.10, yy + 0.10], color=COL_LLAMA, lw=0.9)
        ax_d.scatter(estimate, yy, s=28, color=COL_LLAMA, edgecolors="white",
                     linewidths=0.6, zorder=3)
        ax_d.text(estimate, yy + 0.20,
                  f"{estimate:.3f} [{low:.3f}, {high:.3f}]",
                  ha="center", va="bottom", fontsize=6.8, color=COL_LLAMA)
        forest.append({"metric": key, "estimate": estimate,
                       "ci_95_low": low, "ci_95_high": high})
    ax_d.axvline(0, color="#b03030", lw=0.9, ls="--", zorder=1)
    ax_d.set_yticks(y_effect)
    ax_d.set_yticklabels([label for _, label in error_specs], fontsize=6.8)
    ax_d.set_xlim(-0.08, 1.08)
    ax_d.set_ylim(-0.55, len(error_specs) - 0.30)
    ax_d.set_xlabel("Improvement over MethylGPT (years)")
    ax_d.set_title("Paired test-set differences", fontsize=7.2, pad=4)
    ax_d.spines["left"].set_visible(False)
    ax_d.tick_params(axis="y", length=0)
    panel_label(ax_d, "d", dx=-0.12)

    # Panel e: nested post-hoc age restrictions of the paired test-set effect.
    ax_e = fig.add_subplot(outer[2, 7:])
    age_rows = [{
        "label": f"All ages (n={len(data):,})",
        "estimate": gap["medae"]["point_estimate"],
        "low": gap["medae"]["ci_95_low"],
        "high": gap["medae"]["ci_95_high"],
    }]
    for key, label in [
        ("adults_age_ge_18", "Age $\\geq$18"),
        ("older_age_ge_40", "Age $\\geq$40"),
    ]:
        result = age_bootstrap["subsets"][key]
        age_rows.append({
            "label": f"{label} (n={result['n']:,})",
            "estimate": result["delta_medae"],
            "low": result["delta_medae_ci_95"][0],
            "high": result["delta_medae_ci_95"][1],
        })
    y_age = np.arange(len(age_rows))[::-1]
    for yy, result in zip(y_age, age_rows):
        ax_e.hlines(yy, result["low"], result["high"], color=COL_LLAMA,
                    lw=2.1, zorder=2)
        ax_e.plot([result["low"], result["low"]], [yy - 0.10, yy + 0.10],
                  color=COL_LLAMA, lw=0.9)
        ax_e.plot([result["high"], result["high"]], [yy - 0.10, yy + 0.10],
                  color=COL_LLAMA, lw=0.9)
        ax_e.scatter(result["estimate"], yy, s=28, color=COL_LLAMA,
                     edgecolors="white", linewidths=0.6, zorder=3)
        ax_e.text(result["estimate"], yy + 0.20,
                  f"{result['estimate']:.3f} [{result['low']:.3f}, {result['high']:.3f}]",
                  ha="center", va="bottom", fontsize=6.8, color=COL_LLAMA)
    ax_e.axvline(0, color="#b03030", lw=0.9, ls="--", zorder=1)
    ax_e.set_yticks(y_age)
    ax_e.set_yticklabels([row["label"] for row in age_rows], fontsize=6.6)
    ax_e.set_xlim(-0.08, 1.08)
    ax_e.set_ylim(-0.55, len(age_rows) - 0.30)
    ax_e.set_xlabel("MedAE improvement over MethylGPT (years)")
    ax_e.set_title("Age-restricted sensitivity", fontsize=7.2, pad=4)
    ax_e.spines["left"].set_visible(False)
    ax_e.tick_params(axis="y", length=0)
    panel_label(ax_e, "e", dx=-0.18)

    save(fig, str(OUT / "fig4_age_benchmark_extended"))

    caption = (
        "\\textbf{Chronological-age prediction on a common held-out cohort.} "
        "\\textbf{a}, Test-set stability across five independently trained MethylLlama "
        "folds; points denote fold-specific values and horizontal lines denote their means. "
        "Each fold used the same held-out test cohort. \\textbf{b}, Profile-level ensemble "
        "predictions for MethylLlama and MethylGPT on the same 2,149 test profiles; dashed lines "
        "indicate identity. \\textbf{c}, Fold-0 validation MedAE trajectories under the "
        "WCED-pretrained and from-scratch training protocols, shown over epochs 20--165. "
        "Thin lines show the observed epoch values and thick lines a centred seven-epoch "
        "rolling median; points mark the prespecified matched comparison at epoch 148. The "
        "runs used the same data split, architecture, prediction head, objective, learning "
        "rates, batch size and epoch budget. This single-fold validation analysis is exploratory. "
        "\\textbf{d}, Paired "
        "MethylGPT-minus-MethylLlama differences in MedAE and MAE with 95\\% "
        "percentile-bootstrap confidence intervals from 10,000 profile-level resamples; "
        "positive values favour MethylLlama. The paired $R^2$ result is reported in Table 1. "
        "\\textbf{e}, Paired MedAE differences for the complete test cohort and after nested "
        "post-hoc restrictions to profiles with age at least 18 or at least 40 years. Points are "
        "observed differences and horizontal lines are 95\\% percentile-bootstrap confidence "
        "intervals from 10,000 profile-level resamples (seed 0). The restricted groups are "
        "sensitivity analyses, not independent age strata."
    )
    (OUT / "fig4_age_benchmark_extended_caption.txt").write_text(caption + "\n")
    (OUT / "fig4_age_benchmark_extended_provenance.json").write_text(json.dumps({
        "panel_a": {
            "source": str(FOLDS.relative_to(REPO)),
            "per_fold": folds.round(6).to_dict(orient="records"),
            "note": "five independently trained folds evaluated on one shared test cohort",
        },
        "panel_b": {
            "source": str(PAIRED.relative_to(REPO)),
            "n_profiles": int(len(data)),
            "methyllama_medae": medae(data["predicted_age_llama"], observed),
            "methylgpt_medae": medae(data["predicted_age_gpt"], observed),
        },
        "panel_c": {
            "pretrained_source": str(PRETRAINED_CURVE.relative_to(REPO)),
            "random_initialization_source": str(RANDOM_CURVE.relative_to(REPO)),
            "display_epoch_window": [CURVE_START, CURVE_LIMIT],
            "shared_recorded_end": shared_end,
            "matched_epoch": MATCHED_EPOCH,
            "matched_epoch_validation": {
                "pretrained_medae": float(pretrained_at_match["val/medae"]),
                "pretrained_r2": float(pretrained_at_match["val/r2"]),
                "random_initialization_medae": float(random_at_match["val/medae"]),
                "random_initialization_r2": float(random_at_match["val/r2"]),
            },
            "full_history_minimum": {
                "pretrained": {
                    "epoch": int(pretrained_best["epoch"]),
                    "medae": float(pretrained_best["val/medae"]),
                },
                "random_initialization": {
                    "epoch": int(random_best["epoch"]),
                    "medae": float(random_best["val/medae"]),
                },
            },
            "smoothing": "centred 7-epoch rolling median over raw observations",
            "interpretation_limit": "single-fold validation diagnostic",
        },
        "panel_d": {
            "source": str(BOOT.relative_to(REPO)),
            "resamples": 10000,
            "displayed": forest,
            "sign_convention": "MethylGPT minus MethylLlama; positive favours MethylLlama",
            "r2_not_displayed": "reported in Table 1",
        },
        "panel_e": {
            "source": str(AGE_BOOT.relative_to(REPO)),
            "resamples": age_bootstrap["n_bootstrap"],
            "seed": age_bootstrap["seed"],
            "displayed": age_rows,
            "note": age_bootstrap["note"].replace("subjects", "profiles"),
        },
        "excluded_from_main_figure": {
            "per_tissue": "post-hoc descriptive analysis without uncertainty estimates",
            "methylgpt_validation_curve": "not comparable to the fold-0 validation cohort",
        },
    }, indent=2))


if __name__ == "__main__":
    main()
