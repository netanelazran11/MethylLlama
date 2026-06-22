#!/usr/bin/env python3
"""
thesis_finetune_comparison_figures.py
======================================
Publication-quality figures + deep statistical analysis comparing two WandB runs.

Pass any two WandB run URLs (or bare run IDs) and the script rebuilds all figures.
WandB URL format accepted:
  https://wandb.ai/<entity>/<project>/runs/<run_id>

Usage:
  # Default hardcoded runs (pretrained vs random-init):
  python scripts/repr_analysis/thesis_finetune_comparison_figures.py

  # Pass new WandB URLs to compare any two runs:
  python scripts/repr_analysis/thesis_finetune_comparison_figures.py \\
      --run1 https://wandb.ai/entity/project/runs/abc123 \\
      --run2 https://wandb.ai/entity/project/runs/def456 \\
      --label1 "My Run A" \\
      --label2 "My Run B"

  # Or just run IDs (uses default entity/project):
  python scripts/repr_analysis/thesis_finetune_comparison_figures.py \\
      --run1 abc123 --run2 def456

  # Custom output directory:
  python scripts/repr_analysis/thesis_finetune_comparison_figures.py \\
      --run1 ... --run2 ... --outdir outputs/my_comparison

Outputs (in --outdir):
  1_learning_curves.png     — Full training dynamics, 2×2 grid, all metrics
  2_early_phase.png         — First 60 epochs zoomed; shows speed advantage
  3_convergence_chart.png   — Epoch when each clinical threshold is first crossed
  4_efficiency.png          — MedAE vs epoch on log-scale with clinical zones
  5_pretraining_story.png   — WHY pretraining helps: architecture + evidence
  6_final_performance.png   — Best metric comparison bar chart
  stats_report.txt          — AUC, asymptote fits, plateau, convergence tables
"""

import argparse
import json
import re
import math
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import curve_fit
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import matplotlib.patches as mpatches
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

warnings.filterwarnings("ignore")

# ────────────────────────────────────────────────────────────────────────────
# Defaults (used when --run1/--run2 are not passed)
# ────────────────────────────────────────────────────────────────────────────
DEFAULT_ENTITY  = "netanelazran11-hebrew-university-of-jerusalem"
DEFAULT_PROJECT = "finetune-llama-small"
DEFAULT_RUNS = {
    "run1": "3t5eve7t",
    "run2": "llama-small-ft-random-init-cls-huber-ep300-wu500-44981091",
}
DEFAULT_LABELS = {
    "run1": "Pretrained (WCED)",
    "run2": "Random Init",
}

RECON_JSON = Path("figures/reconstruction_baselines/reconstruction_baselines.json")
CLS_JSON   = Path("figures/cls_attention/cls_attention_summary.json")

PALETTE = ["#E64B35", "#4DBBD5"]   # run1=coral red, run2=steel blue
SMOOTH = 5
MEDAE_THRESH = [20, 15, 10, 7.5, 5, 4]
R2_THRESH    = [0.2, 0.4, 0.6, 0.8, 0.9]


# ── Runtime context (set once in main, read by all figure functions) ─────────
class _Ctx:
    run_keys: list   = ["run1", "run2"]
    colors:   dict   = {}
    labels:   dict   = {}
    x_axis:   str    = "epoch"   # "epoch" or "steps"

CTX = _Ctx()


# ────────────────────────────────────────────────────────────────────────────
# Run config (built from CLI args)
# ────────────────────────────────────────────────────────────────────────────
def parse_wandb_url(url_or_id: str, default_entity: str, default_project: str):
    """Parse a WandB URL or path → (entity, project, run_id).

    Accepted formats:
      https://wandb.ai/entity/project/runs/run_id
      entity/project/run_id
      run_id   (bare 8-char ID; uses default entity/project)
    """
    s = url_or_id.strip()
    # Full URL
    m = re.match(r"https?://wandb\.ai/([^/]+)/([^/]+)/runs/([^/?#]+)", s)
    if m:
        return m.group(1), m.group(2), m.group(3)
    # entity/project/run_id (exactly 2 slashes, no https prefix)
    parts = s.split("/")
    if len(parts) == 3:
        return parts[0], parts[1], parts[2]
    # Bare run ID — use defaults
    return default_entity, default_project, s


def build_run_config(args):
    """Return (runs_dict, labels_dict, colors_dict, out_dir)."""
    runs   = {}
    labels = {}
    colors = {}
    for i, key in enumerate(("run1", "run2")):
        url = getattr(args, key, None)
        if url:
            entity, project, run_id = parse_wandb_url(
                url, DEFAULT_ENTITY, DEFAULT_PROJECT
            )
            runs[key]   = (entity, project, run_id)
            labels[key] = getattr(args, f"label{i+1}") or f"Run {i+1} ({run_id[:8]})"
        else:
            e, p, r = DEFAULT_ENTITY, DEFAULT_PROJECT, DEFAULT_RUNS[key]
            runs[key]   = (e, p, r)
            labels[key] = DEFAULT_LABELS[key]
        colors[key] = PALETTE[i]

    out_dir = Path(args.outdir) if args.outdir else Path("wandb_run_comparison_outputs/thesis_figures")
    return runs, labels, colors, out_dir


# ────────────────────────────────────────────────────────────────────────────
# Style
# ────────────────────────────────────────────────────────────────────────────
def set_style():
    plt.rcParams.update({
        "font.family":       "sans-serif",
        "font.sans-serif":   ["Arial", "Helvetica Neue", "DejaVu Sans"],
        "font.size":          10,
        "axes.titlesize":     11,
        "axes.titleweight":   "bold",
        "axes.labelsize":     10,
        "xtick.labelsize":    8.5,
        "ytick.labelsize":    8.5,
        "legend.fontsize":    8.5,
        "legend.frameon":     False,
        "axes.spines.top":    False,
        "axes.spines.right":  False,
        "axes.linewidth":     0.9,
        "xtick.major.width":  0.9,
        "ytick.major.width":  0.9,
        "xtick.minor.width":  0.5,
        "lines.linewidth":    2.0,
        "figure.dpi":         150,
        "savefig.dpi":        300,
        "savefig.bbox":       "tight",
        "savefig.facecolor":  "white",
        "axes.axisbelow":     True,
        "grid.alpha":         0.3,
        "grid.linewidth":     0.5,
    })


# ────────────────────────────────────────────────────────────────────────────
# Data loading
# ────────────────────────────────────────────────────────────────────────────
METRIC_CANDIDATES = {
    "train_loss":  ["train/loss_epoch", "train/loss",
                    "train_mse_loss_epoch", "train_mae_loss_epoch"],
    "val_loss":    ["val/loss", "validation/loss",
                    "valid_mse_loss/dataloader_idx_0",
                    "valid_mae_loss/dataloader_idx_0"],
    "val_mae":     ["val/mae", "valid_mae", "validation_mae"],
    "val_medae":   ["val/medae", "valid_medae", "validation_medae"],
    "val_r2":      ["val/r2", "valid_r2", "validation_r2"],
    "lr_pg3":      ["lr-AdamW/pg3"],
    "lr_pg4":      ["lr-AdamW/pg4"],
    # Global gradient-update step (enables step-efficiency comparison)
    "global_step": ["trainer/global_step", "global_step"],
    "epoch":       ["epoch"],
}


def download_run(entity: str, project: str, run_id: str, label: str) -> pd.DataFrame:
    import wandb
    import time
    api = wandb.Api(timeout=600)  # large runs with 900K+ steps need extended timeout
    try:
        run = api.run(f"{entity}/{project}/{run_id}")
    except Exception:
        runs = api.runs(f"{entity}/{project}",
                        filters={"display_name": {"$regex": run_id}})
        run = list(runs)[0]

    print(f"  [{label}] {run.name}  (id={run.id}, state={run.state})")
    n_steps = run.lastHistoryStep or 0
    print(f"  [{label}] {n_steps} total steps ...")

    if n_steps > 50_000:
        # Large run (e.g. 917K steps): server-side key-filter (scan_history(keys=[...]))
        # times out for runs this large because WandB must scan all history server-side.
        # Instead: page through ALL history locally with a large page_size (fast per-page),
        # collect rows that contain any val metric, then merge by epoch proximity.
        # Separate wandb.log() calls per metric → each row has one metric, consecutive steps.
        print(f"  [{label}] large run — chunked local scan (page_size=5000) ...")

        # Build flat set of all WandB column names that are val metrics
        val_col_set: set = set()
        for canonical in ["val_medae", "val_mae", "val_r2", "val_loss"]:
            val_col_set.update(METRIC_CANDIDATES.get(canonical, []))

        val_rows_raw: list = []
        n_pages = 0
        for row in run.scan_history(page_size=5000):
            n_pages += 1
            # Keep rows that carry at least one val metric
            if any(k in row and row[k] is not None for k in val_col_set):
                val_rows_raw.append(row)
            if n_pages % 5000 == 0:
                print(f"    ... scanned {n_pages:,} rows, "
                      f"{len(val_rows_raw)} val rows so far")

        print(f"  [{label}] {len(val_rows_raw)} val-metric rows found in {n_pages} pages")

        if not val_rows_raw:
            raw = pd.DataFrame({"epoch": []})
        else:
            # Rows from the same epoch arrive at consecutive _step values (1-4 apart).
            # Between epochs the _step gap is ≫ 100.  Group rows into epochs by that gap.
            val_rows_raw.sort(key=lambda r: r.get("_step", 0))
            steps = [r.get("_step", 0) for r in val_rows_raw]
            # Assign epoch index: new epoch whenever step gap > 100
            epoch_idx = 0
            epoch_tags = [0]
            for i in range(1, len(steps)):
                if steps[i] - steps[i - 1] > 100:
                    epoch_idx += 1
                epoch_tags.append(epoch_idx)

            # Merge metrics within each epoch group
            from collections import defaultdict
            epoch_data: dict = defaultdict(dict)
            for row, ep in zip(val_rows_raw, epoch_tags):
                epoch_data[ep].update(row)

            raw = pd.DataFrame(list(epoch_data.values()))
            # Synthesise epoch column from group order if not present
            if "epoch" not in raw.columns:
                raw["epoch"] = range(len(raw))
            print(f"  [{label}] assembled {len(raw)} epoch rows")
    else:
        rows = [r for r in run.scan_history()]
        raw  = pd.DataFrame(rows)

    col_map = {}
    for canonical, candidates in METRIC_CANDIDATES.items():
        for c in candidates:
            if c in raw.columns:
                col_map[canonical] = c
                break

    epoch_col = col_map.get("epoch", "epoch")
    if epoch_col not in raw.columns:
        raise ValueError(f"[{label}] 'epoch' column not found in run history")

    raw["epoch"] = pd.to_numeric(raw[epoch_col], errors="coerce")
    raw = raw.dropna(subset=["epoch"])
    raw["epoch"] = raw["epoch"].astype(int)
    ep_df_raw = raw.sort_values("epoch").groupby("epoch").last().reset_index()

    out = pd.DataFrame({"epoch": ep_df_raw["epoch"]})
    for canonical, col in col_map.items():
        if canonical == "epoch":
            continue
        if col in ep_df_raw.columns:
            out[canonical] = pd.to_numeric(ep_df_raw[col], errors="coerce")

    out["run"] = label
    print(f"  [{label}] {len(out)} epochs downloaded")
    return out


def load_data(runs_cfg: dict, out_dir: Path, force_download: bool = False) -> pd.DataFrame:
    """Load epoch-level data, using cached CSV when available."""
    cache_path = out_dir / "epoch_level_data.csv"
    expected_runs = set(runs_cfg.keys())

    if not force_download and cache_path.exists():
        cached = pd.read_csv(cache_path)
        cached_runs = set(cached["run"].unique()) if "run" in cached.columns else set()
        if expected_runs == cached_runs:
            print(f"  Loaded from cache → {cache_path}  ({len(cached)} rows)")
            return cached
        print(f"  Cache exists but run keys differ (have {cached_runs}, need {expected_runs}) — re-downloading")

    print("Downloading from WandB ...")
    epoch_dfs = {}
    for key, (entity, project, run_id) in runs_cfg.items():
        df = download_run(entity, project, run_id, key)
        epoch_dfs[key] = df

    combined = pd.concat(epoch_dfs.values(), ignore_index=True)
    combined.to_csv(cache_path, index=False)
    print(f"  Cached → {cache_path}")
    return combined


# ────────────────────────────────────────────────────────────────────────────
# Statistics
# ────────────────────────────────────────────────────────────────────────────
def smooth(s: pd.Series, w: int = SMOOTH) -> pd.Series:
    return s.rolling(w, min_periods=1, center=True).mean()


def step_fmt(x, _):
    """Formatter for global-step x-axis: '17k', '459k', etc."""
    if x >= 1_000:
        return f"{x / 1000:.0f}k"
    return str(int(x))


# Log-scale tick positions for step axis
_STEP_TICKS = [50, 100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]


def set_log_step_axis(ax, x_min: float = 50, x_max: float = 600_000):
    """Apply log scale + 'Xk' ticks to an axes object's x-axis."""
    ticks = [t for t in _STEP_TICKS if x_min * 0.4 <= t <= x_max * 2.5]
    ax.set_xscale("log")
    ax.set_xlim(x_min, x_max)
    ax.set_xticks(ticks)
    ax.xaxis.set_major_formatter(mticker.FuncFormatter(step_fmt))
    ax.xaxis.set_minor_formatter(mticker.NullFormatter())


def ep_to_step(df: pd.DataFrame, run_key: str, epoch_val) -> "int | None":
    """Look up the global_step value at a given epoch for one run."""
    if epoch_val is None or "global_step" not in df.columns:
        return None
    sub = df[(df["run"] == run_key) & (df["epoch"] == int(epoch_val))]
    if sub.empty or sub["global_step"].isna().all():
        return None
    return int(sub["global_step"].iloc[0])


def exp_decay(t, a, b, k):
    return a + b * np.exp(-k * t)


def fit_asymptote(ep, v):
    try:
        y0 = float(np.nanmean(v[:5]))
        yf = float(np.nanmean(v[-5:])) if len(v) >= 5 else float(np.nanmin(v))
        p0 = [yf, y0 - yf, 0.05]
        popt, _ = curve_fit(exp_decay, ep, v, p0=p0, maxfev=20000,
                            bounds=([0, -300, 0], [300, 300, 10]))
        a, b, k = popt
        resid = v - exp_decay(np.array(ep), *popt)
        r2f   = 1 - np.var(resid) / (np.var(v) + 1e-12)
        return float(a), float(np.log(2) / k if k > 1e-9 else np.inf), float(r2f), True
    except Exception:
        return np.nan, np.nan, np.nan, False


def compute_stats(df: pd.DataFrame) -> dict:
    stats = {}
    for label in df["run"].unique():
        sub = df[df["run"] == label].sort_values("epoch").copy()
        s = {}

        for metric in ["val_medae", "val_mae", "val_r2", "val_loss", "train_loss"]:
            if metric not in sub.columns:
                continue
            vals = sub[metric].dropna()
            if len(vals) < 2:
                continue
            ep = sub.loc[vals.index, "epoch"].values.astype(float)
            v  = vals.values.astype(float)
            lower = metric != "val_r2"
            best_idx = np.argmin(v) if lower else np.argmax(v)

            m = {
                "n_epochs":   int(len(v)),
                "first":      float(v[0]),
                "final":      float(v[-1]),
                "best":       float(v[best_idx]),
                "best_epoch": int(ep[best_idx]),
                "auc":        float(np.trapz(v, ep)),
                "auc_per_ep": float(np.trapz(v, ep) / max(ep[-1] - ep[0], 1)),
                "plateau_mean": float(np.mean(v[-20:])) if len(v) >= 20 else float(np.mean(v)),
                "plateau_std":  float(np.std(v[-20:]))  if len(v) >= 20 else float(np.std(v)),
                "delta_abs": float(v[0] - v[-1]) if lower else float(v[-1] - v[0]),
                "delta_pct": float(100 * (v[0] - v[-1]) / abs(v[0])) if lower and v[0] != 0 else np.nan,
            }

            if lower and len(v) >= 10:
                asym, hl, r2f, ok = fit_asymptote(ep.tolist(), v.tolist())
                m["asymptote"] = asym
                m["halflife"]  = hl
                m["fit_r2"]    = r2f
                m["fit_ok"]    = ok

            thresholds = MEDAE_THRESH if "medae" in metric or ("mae" in metric and "medae" not in metric) else []
            thresholds = R2_THRESH    if "r2" in metric else thresholds
            fpt = {}
            for t in thresholds:
                mask = (v <= t) if lower else (v >= t)
                fpt[t] = int(ep[mask][0]) if mask.any() else None
            m["first_passage"] = fpt

            s[metric] = m

        s["early_speed"] = {}
        if "val_medae" in sub.columns:
            medae = sub["val_medae"].dropna().values
            for k in [5, 10, 20, 50]:
                if len(medae) >= k:
                    s["early_speed"][k] = float(medae[k - 1] - medae[0])

        stats[label] = s
    return stats


def write_stats_report(stats: dict, out_path: Path, labels: dict, run_meta: dict = None):
    run_keys = list(stats.keys())
    title    = " vs ".join(labels.get(k, k) for k in run_keys)
    lines = ["=" * 70, f"STATISTICAL REPORT — {title}",
             "=" * 70, ""]

    for key in run_keys:
        s = stats[key]
        lbl = labels.get(key, key)
        lines += [f"\n{'─'*40}", f"  {lbl}", f"{'─'*40}"]

        for metric in ["val_medae", "val_mae", "val_r2", "val_loss"]:
            if metric not in s:
                continue
            m = s[metric]
            lines.append(f"\n  {metric}:")
            lines.append(f"    Epochs observed : {m['n_epochs']}")
            lines.append(f"    First           : {m['first']:.4f}")
            lines.append(f"    Final           : {m['final']:.4f}")
            lines.append(f"    Best            : {m['best']:.4f}  (ep {m['best_epoch']})")
            lines.append(f"    Improvement     : {m['delta_abs']:+.4f}  ({m['delta_pct']:+.1f}%)")
            lines.append(f"    AUC             : {m['auc']:.2f}")
            lines.append(f"    AUC/epoch       : {m['auc_per_ep']:.4f}")
            lines.append(f"    Plateau (last20): {m['plateau_mean']:.4f} ± {m['plateau_std']:.4f}")
            if m.get("fit_ok"):
                lines.append(f"    Asymptote est.  : {m['asymptote']:.4f}  (half-life={m['halflife']:.1f}ep, fit_R²={m['fit_r2']:.3f})")
            if m.get("first_passage"):
                fp_str = "  ".join([
                    f"<{t}yr@ep{ep}" if ep else f"<{t}yr:NEVER"
                    for t, ep in m["first_passage"].items()
                ])
                lines.append(f"    Convergence     : {fp_str}")

        if s.get("early_speed"):
            lines.append(f"\n  Early learning (MedAE drop):")
            for k, delta in s["early_speed"].items():
                lines.append(f"    First {k:3d} epochs : {delta:+.3f} yr")

    # ── Official test-set metrics (from WandB final-epoch summary) ───────────
    if run_meta:
        lines += ["", "=" * 70,
                  "OFFICIAL REPORTED METRICS (FINAL EPOCH — WANDB SUMMARY)",
                  "=" * 70]
        def _fmt(v, decimals=3):
            return f"{v:.{decimals}f}" if not (v != v) else "N/A"  # nan check
        for key in run_keys:
            m   = run_meta.get(key, {})
            lbl = labels.get(key, key)
            lines.append(f"\n  {lbl}  (ep {m.get('total_epochs',0)}, "
                         f"step {m.get('global_step',0):,}, "
                         f"{m.get('runtime_hrs',0):.1f}h)")
            lines.append(f"  {'Metric':<14} {'Validation':>12} {'Test':>12}")
            lines.append(f"  {'-'*40}")
            for metric, vk, tk in [
                ("MedAE (yr)",  "valid_medae",    "test_medae"),
                ("MAE (yr)",    "valid_mae",       "test_mae"),
                ("RMSE (yr)",   "valid_rmse",      "test_rmse"),
                ("R²",          "valid_r2",        "test_r2"),
                ("Pearson r",   "valid_pearson",   "test_pearson"),
                ("Spearman r",  "valid_spearman",  "test_spearman"),
            ]:
                v = _fmt(m.get(vk, float("nan")))
                t = _fmt(m.get(tk, float("nan")))
                lines.append(f"  {metric:<14} {v:>12} {t:>12}")

        if len(run_keys) >= 2:
            k1, k2 = run_keys[0], run_keys[1]
            m1, m2 = run_meta.get(k1, {}), run_meta.get(k2, {})
            lines += ["", f"  Test-set gap ({labels.get(k1,k1)} vs {labels.get(k2,k2)}):"]
            for metric, tk, lower in [
                ("MedAE", "test_medae", True),
                ("MAE",   "test_mae",   True),
                ("R²",    "test_r2",    False),
            ]:
                v1 = m1.get(tk, float("nan"))
                v2 = m2.get(tk, float("nan"))
                if v1 != v1 or v2 != v2:
                    continue
                gap  = v1 - v2
                pct  = 100 * abs(gap) / (abs(v2) + 1e-9)
                sign = "better" if (gap < 0) == lower else "worse"
                lines.append(f"    {metric}: {labels.get(k1,k1)}={v1:.3f}  "
                             f"{labels.get(k2,k2)}={v2:.3f}  "
                             f"gap={gap:+.3f} ({pct:.1f}% {sign})")

    lines += ["", "=" * 70, "KEY COMPARISON (validation best)", "=" * 70]
    if len(run_keys) >= 2:
        k1, k2 = run_keys[0], run_keys[1]
        for metric, lower in [("val_medae", True), ("val_r2", False), ("val_mae", True)]:
            m1 = stats[k1].get(metric, {})
            m2 = stats[k2].get(metric, {})
            if not m1 or not m2:
                continue
            v1, v2 = m1.get("best", np.nan), m2.get("best", np.nan)
            gap = v1 - v2 if lower else v2 - v1
            pct = 100 * abs(gap) / abs(v2 + 1e-9)
            better = "better" if gap < 0 else "worse"
            lines.append(f"  {metric}: {labels[k1]}={v1:.4f}  {labels[k2]}={v2:.4f}  "
                         f"gap={gap:+.4f} ({pct:.1f}% {better})")

    if run_meta:
        lines += ["", "=" * 70, "TRAINING COST COMPARISON", "=" * 70]
        for key in run_keys:
            m  = run_meta.get(key, {})
            lbl = labels.get(key, key)
            lines.append(f"  {lbl}:")
            lines.append(f"    Gradient steps : {m.get('global_step', 0):,}")
            lines.append(f"    Epochs         : {m.get('total_epochs', 0)}")
            lines.append(f"    Wall-clock time: {m.get('runtime_hrs', 0):.1f} hrs")
        if len(run_keys) >= 2:
            k1, k2 = run_keys[0], run_keys[1]
            s1 = run_meta.get(k1, {}).get("global_step", 0)
            s2 = run_meta.get(k2, {}).get("global_step", 0)
            if s1 and s2:
                ratio = s2 / s1
                lines.append(f"  Step ratio ({labels[k2]}/{labels[k1]}): {ratio:.1f}×")
            # Steps-to-convergence: gradient steps needed per year of MedAE improvement
            # = total_steps / MedAE_improvement.  Lower = more compute-efficient.
            f1 = stats.get(k1, {}).get("val_medae", {}).get("first", np.nan)
            f2 = stats.get(k2, {}).get("val_medae", {}).get("first", np.nan)
            b1 = stats.get(k1, {}).get("val_medae", {}).get("best",  np.nan)
            b2 = stats.get(k2, {}).get("val_medae", {}).get("best",  np.nan)
            imp1 = f1 - b1  # total MedAE improvement (years)
            imp2 = f2 - b2
            if s1 and not np.isnan(imp1) and imp1 > 0:
                lines.append(f"  Steps per yr MedAE improvement {labels[k1]}: "
                             f"{s1/imp1:.0f} steps/yr  (total improvement={imp1:.2f} yr)")
            if s2 and not np.isnan(imp2) and imp2 > 0:
                lines.append(f"  Steps per yr MedAE improvement {labels[k2]}: "
                             f"{s2/imp2:.0f} steps/yr  (total improvement={imp2:.2f} yr)")

    with open(out_path, "w") as f:
        f.write("\n".join(lines))
    print(f"  Saved → {out_path}")


# ────────────────────────────────────────────────────────────────────────────
# Helpers for plotting
# ────────────────────────────────────────────────────────────────────────────
def plot_run(ax, df, label, metric, color, display_label,
             lw=2.0, alpha_raw=0.18, best_marker=True, stats=None, linestyle="-",
             x_col="epoch"):
    sub = df[df["run"] == label].sort_values("epoch")
    vals = sub[metric].dropna() if metric in sub.columns else pd.Series([], dtype=float)
    if len(vals) == 0:
        return
    if x_col == "global_step" and "global_step" in sub.columns:
        x_raw = sub.loc[vals.index, "global_step"]
        ep_fallback = sub.loc[vals.index, "epoch"]
        xv = x_raw.fillna(ep_fallback).values
    else:
        xv = sub.loc[vals.index, "epoch"].values
    raw  = vals.values
    smth = smooth(vals).values
    ax.plot(xv, raw,  color=color, alpha=alpha_raw, linewidth=0.7, zorder=1)
    ax.plot(xv, smth, color=color, linewidth=lw, label=display_label, zorder=2,
            linestyle=linestyle)
    if best_marker and stats and label in stats and metric in stats[label]:
        s    = stats[label][metric]
        bv   = s["best"]
        b_ep = s["best_epoch"]
        if x_col == "global_step" and "global_step" in sub.columns:
            bx = ep_to_step(df, label, b_ep)
            bx = bx if bx is not None else b_ep
        else:
            bx = b_ep
        ax.scatter([bx], [bv], color=color, s=70, zorder=5,
                   marker="*", edgecolors="white", linewidths=0.5)


def legend_handles(run_keys, colors, labels):
    handles = [Line2D([0], [0], color=colors[k], linewidth=2.5, label=labels.get(k, k))
               for k in run_keys]
    handles.append(Line2D([0], [0], color="none", marker="*", markerfacecolor="gray",
                          markersize=8, label="Best epoch"))
    return handles


# ────────────────────────────────────────────────────────────────────────────
# Figure 1: Full learning curves (2×2)
# ────────────────────────────────────────────────────────────────────────────
def fig_learning_curves(df, stats, out_dir):
    set_style()
    use_steps = CTX.x_axis == "steps"
    x_col     = "global_step" if use_steps else "epoch"
    xlabel    = "Gradient Update Steps" if use_steps else "Epoch"

    fig, axes = plt.subplots(2, 2, figsize=(12, 8))
    panels = [
        ("val_medae", "Validation MedAE (years)",   True),
        ("val_r2",    "Validation R²",               False),
        ("val_mae",   "Validation MAE (years)",      True),
        # val_loss intentionally dropped: MethylLlama uses Huber on normalised targets
        # while Baseline uses MSE on raw years (scale 70–290 vs 0.025–0.14) — incomparable.
        ("val_medae_improvement", "MedAE Improvement from Start (years, ↑ better)", False),
    ]

    # Compute global_step range for log axis bounds
    if use_steps and "global_step" in df.columns:
        gs_all = df["global_step"].dropna()
        _gs_min = max(50, int(gs_all.min()))
        _gs_max = int(gs_all.max())
    else:
        _gs_min, _gs_max = 1, 310

    for ax, (metric, ylabel, log_y) in zip(axes.flat, panels):
        if metric == "val_medae_improvement":
            # Improvement = first_val - current_val  (positive = better than start)
            for label in CTX.run_keys:
                sub  = df[df["run"] == label].sort_values("epoch")
                vals = sub["val_medae"].dropna()
                if len(vals) == 0:
                    continue
                first = vals.iloc[0]
                improvement = first - vals
                if use_steps and "global_step" in sub.columns:
                    xv = sub.loc[vals.index, "global_step"].fillna(
                        sub.loc[vals.index, "epoch"]).values
                else:
                    xv = sub.loc[vals.index, "epoch"].values
                smth = smooth(improvement).values
                n_ep = stats.get(label, {}).get("val_medae", {}).get("n_epochs", 300)
                ls   = "--" if n_ep < 200 else "-"
                ax.plot(xv, improvement.values, color=CTX.colors[label], alpha=0.15, lw=0.7)
                ax.plot(xv, smth, color=CTX.colors[label], lw=2.0, linestyle=ls,
                        label=CTX.labels.get(label, label))
            ax.axhline(0, color="#888888", lw=0.7, ls=":")
            if use_steps:
                set_log_step_axis(ax, _gs_min, _gs_max * 1.3)
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.set_title("MedAE Improvement Over Starting Point\n(positive = better than ep-0 baseline)",
                         fontsize=9, loc="left")
        else:
            for label in CTX.run_keys:
                n_ep = stats.get(label, {}).get("val_medae", {}).get("n_epochs", 300)
                ls   = "--" if n_ep < 200 else "-"
                plot_run(ax, df, label, metric, CTX.colors[label],
                         CTX.labels.get(label, label), stats=stats, linestyle=ls,
                         x_col=x_col)

            if use_steps:
                set_log_step_axis(ax, _gs_min, _gs_max * 1.3)

            if metric == "val_medae":
                for t, ls, lbl in [(5, ":", "5 yr"), (7.5, "--", "7.5 yr"), (10, "-.", "10 yr")]:
                    ax.axhline(t, color="#888888", linewidth=0.8, linestyle=ls, alpha=0.7, zorder=0)
                    ax.text(_gs_min * 1.5 if use_steps else 2,
                            t * 1.05, lbl, fontsize=7.5, color="#888888", va="bottom")
            if metric == "val_r2":
                ax.axhline(0.9, color="#888888", linewidth=0.8, linestyle=":", alpha=0.7)
                ax.text(_gs_min * 1.5 if use_steps else 1, 0.91, "R²=0.9",
                        fontsize=7.5, color="#888888")

            if log_y:
                ax.set_yscale("log")
            ax.set_xlabel(xlabel)
            ax.set_ylabel(ylabel)
            ax.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))

        # Vertical marker: where MethylLlama (run1) training ended
        if use_steps:
            k1 = CTX.run_keys[0]
            gs_max = df[df["run"] == k1]["global_step"].max() if "global_step" in df.columns else None
            if gs_max and not np.isnan(gs_max):
                ax.axvline(gs_max, color=CTX.colors[k1], linewidth=0.9,
                           linestyle=":", alpha=0.5)
                ax.text(gs_max * 1.08, ax.get_ylim()[1] * 0.98,
                        f"{CTX.labels.get(k1,'run1')}\nends\n@{step_fmt(gs_max,None)}",
                        fontsize=6.5, color=CTX.colors[k1], va="top", ha="left", alpha=0.7)

    # Subtitle: note any partial runs
    partial_notes = [f"{CTX.labels[k]} ({stats[k]['val_medae']['n_epochs']}/300 ep)"
                     for k in CTX.run_keys
                     if stats.get(k, {}).get("val_medae", {}).get("n_epochs", 300) < 290]
    subtitle = ("  [partial: " + ", ".join(partial_notes) + "]") if partial_notes else ""
    fig.legend(handles=legend_handles(CTX.run_keys, CTX.colors, CTX.labels),
               loc="upper center", ncol=3,
               bbox_to_anchor=(0.5, 1.01), frameon=False, fontsize=9)
    x_note = " (x-axis: gradient steps)" if use_steps else ""
    fig.text(0.5, 1.04,
             f"Fine-tuning Comparison: {' vs. '.join(CTX.labels[k] for k in CTX.run_keys)}"
             f"{subtitle}{x_note}",
             ha="center", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "1_learning_curves.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 2: Early phase zoom (first 60 epochs)
# ────────────────────────────────────────────────────────────────────────────
def fig_early_phase(df, stats, out_dir):
    set_style()
    use_steps = CTX.x_axis == "steps"
    ZOOM_EP   = 65
    ZOOM_STEP = 50_000   # shows all of run1 (17k) + run2's first ~9 epochs

    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    panels = [
        ("val_medae", "Validation MedAE (years)"),
        ("val_r2",    "Validation R²"),
        ("val_mae",   "Validation MAE (years)"),
    ]

    for ax, (metric, ylabel) in zip(axes, panels):
        for label in CTX.run_keys:
            if use_steps and "global_step" in df.columns:
                sub = df[(df["run"] == label) & (df["global_step"] <= ZOOM_STEP)].sort_values("epoch")
            else:
                sub = df[(df["run"] == label) & (df["epoch"] <= ZOOM_EP)].sort_values("epoch")
            vals = sub[metric].dropna() if metric in sub.columns else pd.Series([], dtype=float)
            if len(vals) == 0:
                continue
            if use_steps and "global_step" in sub.columns:
                xv = sub.loc[vals.index, "global_step"].fillna(sub.loc[vals.index, "epoch"]).values
            else:
                xv = sub.loc[vals.index, "epoch"].values
            smth = smooth(vals).values
            ax.plot(xv, vals.values, color=CTX.colors[label], alpha=0.18, linewidth=0.7)
            ax.plot(xv, smth, color=CTX.colors[label], linewidth=2.2, label=CTX.labels.get(label, label))

        # Clinical zones for MedAE
        if metric == "val_medae":
            ax.axhspan(0, 5,   color="#4DBBD5", alpha=0.07)
            ax.axhspan(5, 7.5, color="#91D1C2", alpha=0.07)
            for t, lbl in [(5, "5 yr"), (7.5, "7.5 yr"), (10, "10 yr")]:
                ax.axhline(t, color="#888888", linewidth=0.8, linestyle=":", alpha=0.7)
                ax.text(1 if not use_steps else ZOOM_STEP * 0.01, t * 1.02, lbl,
                        fontsize=7.5, color="#888888")

            # Annotate run1 first convergence to <5yr
            k1 = CTX.run_keys[0]
            pr_m = stats.get(k1, {}).get("val_medae", {})
            if pr_m:
                ep5 = pr_m.get("first_passage", {}).get(5)
                if ep5:
                    if use_steps:
                        x5 = ep_to_step(df, k1, ep5)
                        in_zoom = (x5 is not None and x5 <= ZOOM_STEP)
                    else:
                        x5 = ep5
                        in_zoom = (ep5 <= ZOOM_EP)
                    if in_zoom:
                        sub_pr = df[(df["run"] == k1) & (df["epoch"] == ep5)]
                        if not sub_pr.empty and "val_medae" in sub_pr.columns:
                            yval = float(sub_pr["val_medae"].iloc[0])
                            lbl5 = f"@{step_fmt(x5,None)}: <5yr" if use_steps else f"ep{ep5}: <5yr"
                            offset_x = x5 * 1.1 if use_steps else x5 + 5
                            ax.annotate(lbl5, xy=(x5, yval),
                                        xytext=(offset_x, yval + 2),
                                        arrowprops=dict(arrowstyle="->", color=CTX.colors[k1], lw=1.2),
                                        fontsize=8, color=CTX.colors[k1])

        if use_steps:
            # Find actual min step in zoomed range for log axis lower bound
            _zoom_steps = []
            for lbl in CTX.run_keys:
                sub2 = df[(df["run"] == lbl) & (df["global_step"] <= ZOOM_STEP)]["global_step"].dropna()
                if len(sub2):
                    _zoom_steps.append(int(sub2.min()))
            _z_min = max(50, min(_zoom_steps)) if _zoom_steps else 50
            set_log_step_axis(ax, _z_min, ZOOM_STEP * 1.1)
            ax.set_xlabel("Gradient Update Steps (log scale)")
            # Vertical marker at run1 max step
            k1 = CTX.run_keys[0]
            gs_max = df[df["run"] == k1]["global_step"].max() if "global_step" in df.columns else None
            if gs_max and not np.isnan(gs_max) and gs_max <= ZOOM_STEP:
                ax.axvline(gs_max, color=CTX.colors[k1], lw=0.9, ls=":", alpha=0.55)
                ax.text(gs_max * 1.08, ax.get_ylim()[1] * 0.95,
                        f"{CTX.labels.get(k1,'run1')}\nends",
                        fontsize=6.5, color=CTX.colors[k1], va="top", alpha=0.7)
        else:
            ax.set_xlabel("Epoch")
            ax.set_xlim(0, ZOOM_EP)
        ax.set_ylabel(ylabel)
        ax.set_title(ylabel, fontsize=10, fontweight="bold")

    axes[0].legend(fontsize=8.5)
    if use_steps:
        fig.suptitle(f"Early Learning Dynamics — First {step_fmt(ZOOM_STEP,None)} Steps (log scale)",
                     fontsize=12, fontweight="bold")
    else:
        fig.suptitle("Early Learning Dynamics — First 65 Epochs", fontsize=12, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "2_early_phase.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 3: Convergence milestone chart
# ────────────────────────────────────────────────────────────────────────────
def fig_convergence_chart(stats, out_dir, df=None):
    set_style()
    use_steps = CTX.x_axis == "steps"

    thresholds_medae = [10.0, 7.5, 5.0, 4.0]
    thresholds_r2    = [0.6,  0.8, 0.9]
    all_thresh = (
        [("val_medae", t, f"MedAE < {t} yr")  for t in thresholds_medae] +
        [("val_r2",    t, f"R² ≥ {t}")         for t in thresholds_r2]
    )

    # In steps mode: determine max step for run2 (Baseline) as the right-axis limit
    if use_steps and df is not None:
        max_steps_run2 = df[df["run"] == CTX.run_keys[1]]["global_step"].max() \
            if len(CTX.run_keys) > 1 and "global_step" in df.columns else 460_000
        if np.isnan(max_steps_run2):
            max_steps_run2 = 460_000
        MAX_X = int(max_steps_run2 * 1.05)
        xlabel = "Gradient steps to first convergence"
    else:
        MAX_X  = 310
        xlabel = "Epoch of first convergence"

    fig, ax = plt.subplots(figsize=(10, 6))
    y_positions = list(range(len(all_thresh)))
    height = 0.34

    for yi, (metric, threshold, label_str) in enumerate(all_thresh):
        for xi, run in enumerate(CTX.run_keys):
            s  = stats.get(run, {}).get(metric, {})
            fp = s.get("first_passage", {}).get(threshold)   # this is an epoch number
            y  = yi + (0.18 if xi == 0 else -0.18)

            if use_steps and df is not None and fp is not None:
                fp_x = ep_to_step(df, run, fp)
                fp_x = fp_x if fp_x is not None else fp
            else:
                fp_x = fp

            if fp_x is not None:
                ax.barh(y, fp_x, height=height, color=CTX.colors[run],
                        alpha=0.85, edgecolor="white", linewidth=0.5)
                lbl_text = f"@{step_fmt(fp_x, None)}" if use_steps else f"ep {fp}"
                ax.text(fp_x + MAX_X * 0.01, y, lbl_text, va="center", ha="left",
                        fontsize=8, color=CTX.colors[run], fontweight="bold")
            else:
                ax.barh(y, MAX_X, height=height, color=CTX.colors[run],
                        alpha=0.25, edgecolor=CTX.colors[run], linewidth=1.0,
                        linestyle="--")
                ax.annotate("", xy=(MAX_X, y), xytext=(MAX_X - MAX_X * 0.05, y),
                            arrowprops=dict(arrowstyle="->", color=CTX.colors[run], lw=1.5))
                ax.text(MAX_X + MAX_X * 0.01, y, "not reached", va="center", ha="left",
                        fontsize=8, color=CTX.colors[run], style="italic")

    ax.set_yticks(y_positions)
    ax.set_yticklabels([l for _, _, l in all_thresh], fontsize=9.5)
    ax.set_xlabel(xlabel, fontsize=10)
    ax.set_xlim(0, MAX_X * 1.18)

    if use_steps:
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(step_fmt))
        # Vertical marker at run1 max step
        k1 = CTX.run_keys[0]
        if df is not None and "global_step" in df.columns:
            gs_max = df[df["run"] == k1]["global_step"].max()
            if not np.isnan(gs_max):
                ax.axvline(gs_max, color=CTX.colors[k1], lw=0.9, ls=":", alpha=0.5)
                ax.text(gs_max, len(all_thresh) - 0.3,
                        f"{CTX.labels.get(k1,'run1')} ends",
                        fontsize=7, color=CTX.colors[k1], ha="center")
    else:
        ax.axvline(300, color="gray", linewidth=0.8, linestyle=":", alpha=0.5)
        ax.text(300, len(all_thresh) - 0.3, "max ep", fontsize=7.5, color="gray", ha="center")
    ax.yaxis.set_tick_params(length=0)

    # Divider line between MedAE and R² sections
    ax.axhline(len(thresholds_medae) - 0.5, color="#cccccc", linewidth=0.8)
    ax.text(MAX_X * 0.005, len(thresholds_medae) - 0.35, "MedAE thresholds", fontsize=7.5, color="#888888")
    ax.text(MAX_X * 0.005, len(thresholds_medae) + 0.2, "R² thresholds",    fontsize=7.5, color="#888888")

    legend_els = [mpatches.Patch(color=CTX.colors[k], label=CTX.labels.get(k, k))
                  for k in CTX.run_keys]
    ax.legend(handles=legend_els, loc="lower right", fontsize=9)
    x_note = " (gradient steps)" if use_steps else ""
    ax.set_title(f"Convergence to Clinical Performance Thresholds{x_note}",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "3_convergence_chart.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 4: Training efficiency (log-scale MedAE)
# ────────────────────────────────────────────────────────────────────────────
def fig_efficiency(df, stats, out_dir):
    set_style()
    use_steps = CTX.x_axis == "steps"
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    if use_steps:
        # x-axis = global_step (log scale).  Determine data range.
        all_steps = df["global_step"].dropna() if "global_step" in df.columns else pd.Series([1])
        x_min  = max(1, int(all_steps.min()))
        x_max  = int(all_steps.max()) + 1
        x_fill_max = x_max * 1.1
        step_ticks = [100, 500, 1_000, 5_000, 10_000, 50_000, 100_000, 500_000]
        step_ticks = [t for t in step_ticks if x_min * 0.5 <= t <= x_fill_max * 2]
        xlabel = "Gradient Update Steps (log scale)"
    else:
        x_fill_max = 310
        xlabel = "Epoch (log scale)"

    for ax_idx, (metric, ylabel, legend_loc, ylim) in enumerate([
        ("val_medae", "Validation MedAE (years)", "upper right", (0, 23)),
        ("val_r2",    "Validation R²",             "lower right", (None, None)),
    ]):
        ax = axes[ax_idx]

        if metric == "val_medae":
            ax.fill_between([1, x_fill_max], 0, 5,   color="#2ecc71", alpha=0.08, label="Clinical (<5yr)")
            ax.fill_between([1, x_fill_max], 5, 7.5, color="#f39c12", alpha=0.06, label="Good (<7.5yr)")
            ax.fill_between([1, x_fill_max], 7.5, 25, color="#e74c3c", alpha=0.04, label="Poor (>7.5yr)")
            for t, lbl in [(5, "5 yr"), (7.5, "7.5 yr"), (10, "10 yr")]:
                ax.axhline(t, color="#888888", linewidth=0.7, linestyle=":", alpha=0.6)
                ax.text(1.5 if not use_steps else x_min * 1.2, t * 1.03, lbl, fontsize=7.5, color="#888888")
        else:
            ax.fill_between([1, x_fill_max], 0.9, 1.0, color="#2ecc71", alpha=0.08, label="High quality (R²≥0.9)")
            ax.fill_between([1, x_fill_max], 0.8, 0.9, color="#f39c12", alpha=0.06)
            ax.axhline(0.9, color="#888888", linewidth=0.7, linestyle=":", alpha=0.6)
            ax.text(1.5 if not use_steps else x_min * 1.2, 0.91, "R²=0.9", fontsize=7.5, color="#888888")

        for label in CTX.run_keys:
            sub = df[df["run"] == label].sort_values("epoch")
            vals = sub[metric].dropna() if metric in sub.columns else pd.Series([], dtype=float)
            if len(vals) == 0:
                continue
            if use_steps and "global_step" in sub.columns:
                xv = sub.loc[vals.index, "global_step"].fillna(sub.loc[vals.index, "epoch"]).values
            else:
                xv = sub.loc[vals.index, "epoch"].values
            smth = smooth(vals).values
            ax.plot(xv, vals.values, color=CTX.colors[label], alpha=0.15, linewidth=0.7)
            ax.plot(xv, smth, color=CTX.colors[label], linewidth=2.2, label=CTX.labels[label])

            # Asymptote dashes (MedAE only)
            if metric == "val_medae":
                m = stats.get(label, {}).get("val_medae", {})
                if m.get("fit_ok") and not np.isnan(m.get("asymptote", np.nan)):
                    asym = m["asymptote"]
                    ax.axhline(asym, color=CTX.colors[label], linewidth=1.0,
                               linestyle="--", alpha=0.55)
                    ax.text(x_fill_max * 0.98, asym + 0.2, f"~{asym:.2f}",
                            fontsize=7.5, color=CTX.colors[label], ha="right", alpha=0.8)

        ax.set_xlabel(xlabel, fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        if use_steps:
            set_log_step_axis(ax, x_min * 0.8, x_fill_max)
            # Vertical marker at run1 end
            k1 = CTX.run_keys[0]
            gs_max = df[df["run"] == k1]["global_step"].max() if "global_step" in df.columns else None
            if gs_max and not np.isnan(gs_max):
                ax.axvline(gs_max, color=CTX.colors[k1], lw=0.9, ls=":", alpha=0.5)
        else:
            ax.set_xscale("log")
            ax.set_xlim(1, 350)
            ax.set_xticks([1, 2, 5, 10, 20, 50, 100, 200, 300])
            ax.xaxis.set_major_formatter(mticker.ScalarFormatter())
            ax.xaxis.set_minor_formatter(mticker.NullFormatter())
        if ylim[0] is not None:
            ax.set_ylim(*ylim)
        ax.legend(loc=legend_loc, fontsize=8.5)
        x_note = "Steps" if use_steps else "Epochs"
        ax.set_title(f"Training Efficiency: {ylabel.split('(')[0].strip()} vs. {x_note} (log scale)",
                     fontsize=10, fontweight="bold")

    fig.tight_layout()
    path = out_dir / "4_efficiency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 5: THE STORY FIGURE — Why pretraining helps
# ────────────────────────────────────────────────────────────────────────────
def draw_architecture_diagram(ax):
    """Draw WCED pretraining → CLS bottleneck → fine-tuning diagram."""
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 10)
    ax.axis("off")

    def box(ax, x, y, w, h, text, facecolor="#f0f0f0", edgecolor="#555555",
            fontsize=9, bold=False, text_color="black"):
        rect = FancyBboxPatch((x - w/2, y - h/2), w, h,
                              boxstyle="round,pad=0.1",
                              facecolor=facecolor, edgecolor=edgecolor, linewidth=1.5, zorder=3)
        ax.add_patch(rect)
        fw = "bold" if bold else "normal"
        ax.text(x, y, text, ha="center", va="center", fontsize=fontsize,
                fontweight=fw, color=text_color, zorder=4)

    def arrow(ax, x0, y0, x1, y1, color="#555555", lw=1.5, head=0.3):
        ax.annotate("", xy=(x1, y1), xytext=(x0, y0),
                    arrowprops=dict(arrowstyle=f"->, head_width={head}, head_length={head*0.7}",
                                   color=color, lw=lw),
                    zorder=5)

    # ── WCED Pretraining (top section) ──────────────────────────────────
    ax.text(5, 9.6, "WCED Pretraining", ha="center", fontsize=10,
            fontweight="bold", color="#333333")
    ax.axhline(9.35, color="#cccccc", linewidth=0.8, xmin=0.05, xmax=0.95)

    # Input CpGs
    box(ax, 5, 8.8, 5, 0.65, "CpG methylation β₁, …, βₙ  (19,608 sites)",
        facecolor="#EEF5FF", edgecolor="#3498DB", fontsize=8.5)

    arrow(ax, 5, 8.48, 5, 7.8)

    # Transformer
    box(ax, 5, 7.4, 5.2, 0.75,
        "Transformer Encoder\n4 layers × 256-dim × 4 heads",
        facecolor="#FFF5EE", edgecolor="#E67E22", fontsize=8.5)

    arrow(ax, 5, 7.03, 5, 6.4)

    # CLS bottleneck (highlighted)
    box(ax, 5, 6.05, 3.8, 0.62,
        "[ CLS ]  256-dim bottleneck",
        facecolor="#FDE8E8", edgecolor="#E64B35", fontsize=9, bold=True,
        text_color="#C0392B")

    # Three loss arrows from CLS
    for xdst, ydst, loss_txt, col in [
        (1.5, 4.7, "Reconstruction\nMSE Loss",      "#E74C3C"),
        (5.0, 4.7, "Age Regression\nMSE Loss",       "#E67E22"),
        (8.5, 4.7, "InfoNCE\nContrastive Loss",      "#8E44AD"),
    ]:
        arrow(ax, 5, 5.74, xdst, 5.1, color=col, lw=1.3)
        box(ax, xdst, 4.55, 2.7, 0.8, loss_txt,
            facecolor="white", edgecolor=col, fontsize=8, text_color=col)

    # ── Separator ──────────────────────────────────────────────────────
    ax.axhline(3.85, color="#aaaaaa", linewidth=1.0, xmin=0.05, xmax=0.95, linestyle="--")
    ax.text(5, 3.62, "↓  Transfer to Fine-tuning", ha="center", fontsize=8.5,
            color="#555555", style="italic")

    # ── Fine-tuning (bottom section) ───────────────────────────────────
    ax.text(5, 3.3, "Fine-tuning", ha="center", fontsize=10,
            fontweight="bold", color="#333333")

    box(ax, 5, 2.72, 3.8, 0.65,
        "[ CLS ]  rich biological representation",
        facecolor="#FDE8E8", edgecolor="#E64B35", fontsize=9, bold=True,
        text_color="#C0392B")

    arrow(ax, 5, 2.40, 5, 1.85)

    box(ax, 5, 1.52, 3.5, 0.6, "MLP Head  (2-layer)",
        facecolor="#EFF7EF", edgecolor="#27AE60", fontsize=8.5)

    arrow(ax, 5, 1.22, 5, 0.72)

    box(ax, 5, 0.45, 3.5, 0.6, "Predicted Biological Age",
        facecolor="#EFF7EF", edgecolor="#27AE60", fontsize=9, bold=True,
        text_color="#1E8449")


def fig_pretraining_story(df, stats, out_dir):
    """Figure 5: The WHY figure — 3 columns, thesis narrative."""
    set_style()
    fig = plt.figure(figsize=(18, 10))
    gs  = gridspec.GridSpec(2, 3, figure=fig, width_ratios=[1.1, 1.1, 0.8],
                            hspace=0.38, wspace=0.32)

    # ── Col 0 (full height): Architecture diagram ─────────────────────
    ax_arch = fig.add_subplot(gs[:, 0])
    draw_architecture_diagram(ax_arch)
    ax_arch.set_title("A   WCED Pretraining Architecture",
                      fontsize=11, fontweight="bold", loc="left", pad=6)

    # ── Col 1 top: Learning curves (val/MedAE) ────────────────────────
    use_steps = CTX.x_axis == "steps"
    ax_lc = fig.add_subplot(gs[0, 1])
    ax_lc.fill_between([0, 305], 0, 5, color="#2ecc71", alpha=0.07)
    ax_lc.axhline(5,  color="#888888", lw=0.8, ls=":",  alpha=0.7)
    ax_lc.axhline(10, color="#888888", lw=0.8, ls="-.", alpha=0.5)

    for label in CTX.run_keys:
        sub = df[df["run"] == label].sort_values("epoch")
        vals = sub["val_medae"].dropna() if "val_medae" in sub.columns else pd.Series([], dtype=float)
        if len(vals) == 0:
            continue
        if use_steps and "global_step" in sub.columns:
            xv = sub.loc[vals.index, "global_step"].fillna(sub.loc[vals.index, "epoch"]).values
        else:
            xv = sub.loc[vals.index, "epoch"].values
        smth = smooth(vals).values
        ax_lc.plot(xv, vals.values, color=CTX.colors[label], alpha=0.15, linewidth=0.7)
        ax_lc.plot(xv, smth, color=CTX.colors[label], linewidth=2.2, label=CTX.labels[label])

        m = stats.get(label, {}).get("val_medae", {})
        if m:
            b_ep, bv = m["best_epoch"], m["best"]
            bx = ep_to_step(df, label, b_ep) if use_steps else b_ep
            bx = bx if bx is not None else b_ep
            ax_lc.scatter([bx], [bv], color=CTX.colors[label], s=70, zorder=5,
                          marker="*", edgecolors="white", linewidths=0.5)
            bx_lbl = f"@{step_fmt(bx,None)}" if use_steps else f"ep {b_ep}"
            ax_lc.annotate(f"Best: {bv:.2f} yr\n({bx_lbl})", xy=(bx, bv),
                           xytext=(bx * 1.05 if use_steps else bx + 10, bv + 1.5),
                           fontsize=7.5, color=CTX.colors[label],
                           arrowprops=dict(arrowstyle="->", color=CTX.colors[label], lw=0.9))

    if use_steps and "global_step" in df.columns:
        _gs_all = df["global_step"].dropna()
        _gs_min_lc = max(50, int(_gs_all.min()))
        _gs_max_lc = int(_gs_all.max())
        set_log_step_axis(ax_lc, _gs_min_lc, _gs_max_lc * 1.3)
        ax_lc.text(_gs_min_lc * 1.5, 5.3, "5 yr", fontsize=7.5, color="#888888")
        ax_lc.text(_gs_min_lc * 1.5, 10.3, "10 yr", fontsize=7.5, color="#888888")
        ax_lc.set_xlabel("Gradient Update Steps (log scale)", fontsize=9)
        # Vertical marker
        k1 = CTX.run_keys[0]
        gs_max = df[df["run"] == k1]["global_step"].max()
        if not np.isnan(gs_max):
            ax_lc.axvline(gs_max, color=CTX.colors[k1], lw=0.9, ls=":", alpha=0.5)
    else:
        ax_lc.text(2, 5.3, "5 yr", fontsize=7.5, color="#888888")
        ax_lc.text(2, 10.3, "10 yr", fontsize=7.5, color="#888888")
        ax_lc.set_xlabel("Epoch", fontsize=9)
    ax_lc.set_ylabel("Validation MedAE (years)", fontsize=9)
    ax_lc.set_ylim(0, 22)
    ax_lc.legend(loc="upper right", fontsize=8)
    ax_lc.set_title("B   Fine-tuning Learning Curves", fontsize=11,
                    fontweight="bold", loc="left")

    # ── Col 1 bottom: Early phase zoom ───────────────────────────────────
    ax_early = fig.add_subplot(gs[1, 1])
    ZOOM_EP   = 65
    ZOOM_STEP = 50_000
    ax_early.fill_between([0, ZOOM_STEP if use_steps else ZOOM_EP], 0, 5,
                          color="#2ecc71", alpha=0.10)
    ax_early.axhline(5, color="#888888", lw=0.8, ls=":", alpha=0.7)

    for label in CTX.run_keys:
        if use_steps and "global_step" in df.columns:
            sub = df[(df["run"] == label) & (df["global_step"] <= ZOOM_STEP)].sort_values("epoch")
        else:
            sub = df[(df["run"] == label) & (df["epoch"] <= ZOOM_EP)].sort_values("epoch")
        vals = sub["val_medae"].dropna() if "val_medae" in sub.columns else pd.Series([], dtype=float)
        if len(vals) == 0:
            continue
        if use_steps and "global_step" in sub.columns:
            xv = sub.loc[vals.index, "global_step"].fillna(sub.loc[vals.index, "epoch"]).values
        else:
            xv = sub.loc[vals.index, "epoch"].values
        smth = smooth(vals).values
        ax_early.plot(xv, vals.values, color=CTX.colors[label], alpha=0.15, lw=0.7)
        ax_early.plot(xv, smth, color=CTX.colors[label], lw=2.2, label=CTX.labels[label])

    if use_steps and "global_step" in df.columns:
        _zoom_steps2 = []
        for lbl in CTX.run_keys:
            sub2 = df[(df["run"] == lbl) & (df["global_step"] <= ZOOM_STEP)]["global_step"].dropna()
            if len(sub2):
                _zoom_steps2.append(int(sub2.min()))
        _z_min2 = max(50, min(_zoom_steps2)) if _zoom_steps2 else 50
        set_log_step_axis(ax_early, _z_min2, ZOOM_STEP * 1.1)
        ax_early.text(_z_min2 * 1.5, 5.3, "5 yr", fontsize=7.5, color="#888888")
        ax_early.set_xlabel(f"Gradient Steps — first {step_fmt(ZOOM_STEP,None)} (log scale)", fontsize=9)
        # Marker where run1 ends
        k1 = CTX.run_keys[0]
        gs_max = df[df["run"] == k1]["global_step"].max()
        if not np.isnan(gs_max) and gs_max <= ZOOM_STEP:
            ax_early.axvline(gs_max, color=CTX.colors[k1], lw=0.9, ls=":", alpha=0.55)
    else:
        ax_early.set_xlim(0, ZOOM_EP)
        ax_early.text(1, 5.3, "5 yr", fontsize=7.5, color="#888888")
        ax_early.set_xlabel("Epoch (first 65)", fontsize=9)
    ax_early.set_ylabel("Validation MedAE (years)", fontsize=9)
    ax_early.set_ylim(0, 23)
    ax_early.legend(loc="upper right", fontsize=8)
    c_title = "C   Early Phase — First 50k Steps (log scale)" if use_steps else "C   Early Learning Phase (Zoom)"
    ax_early.set_title(c_title, fontsize=11, fontweight="bold", loc="left")

    # ── Col 2 top: Reconstruction baselines ───────────────────────────
    ax_recon = fig.add_subplot(gs[0, 2])
    try:
        with open(RECON_JSON) as f:
            recon = json.load(f)
        labels_r = ["Model\n(real CLS)", "B3\n(shuffled CLS)", "B4\n(random CLS)", "B1\n(pop. mean)"]
        keys_r   = ["model_mse", "b3_mse", "b4_mse", "b1_mse"]
        vals_r   = [recon[k]["mean"] for k in keys_r]
        colors_r = ["#E64B35", "#F39B7F", "#B09C85", "#8491B4"]
        bars = ax_recon.bar(labels_r, vals_r, color=colors_r, edgecolor="white",
                            linewidth=1.0, width=0.6)
        for bar, v in zip(bars, vals_r):
            ax_recon.text(bar.get_x() + bar.get_width()/2, v + 0.001,
                          f"{v:.4f}", ha="center", va="bottom", fontsize=7.5, fontweight="bold")
        ax_recon.set_ylabel("Reconstruction MSE", fontsize=9)
        ax_recon.set_title("D   CLS Reconstruction Probe\n(Does CLS carry signal?)",
                           fontsize=11, fontweight="bold", loc="left")
        ax_recon.set_ylim(0, max(vals_r) * 1.2)
        ax_recon.set_yscale("log")
        ax_recon.yaxis.set_major_formatter(mticker.FormatStrFormatter("%.3g"))
        # Ratio annotation
        ratio = recon["ratio_model_vs_b3"]
        ax_recon.text(0.5, 0.92,
                      f"Model MSE = {ratio:.2f}× shuffled CLS MSE\n"
                      f"→ CLS carries sample-specific signal",
                      ha="center", va="top", transform=ax_recon.transAxes,
                      fontsize=7.5, color="#333333",
                      bbox=dict(facecolor="#FFF9E6", edgecolor="#F39C12", linewidth=0.8,
                                boxstyle="round,pad=0.3"))
    except FileNotFoundError:
        ax_recon.text(0.5, 0.5, "Reconstruction data\nnot found", ha="center",
                      va="center", transform=ax_recon.transAxes, fontsize=9, color="#888")
        ax_recon.axis("off")

    # ── Col 2 bottom: Key numbers summary ─────────────────────────────
    ax_num = fig.add_subplot(gs[1, 2])
    ax_num.axis("off")
    ax_num.set_title("E   Performance Summary", fontsize=11, fontweight="bold", loc="left")

    k1, k2 = CTX.run_keys[0], CTX.run_keys[1]
    pr_m  = stats.get(k1, {}).get("val_medae", {})
    ri_m  = stats.get(k2, {}).get("val_medae", {})
    pr_r2 = stats.get(k1, {}).get("val_r2",    {})
    ri_r2 = stats.get(k2, {}).get("val_r2",    {})
    ri_ep = ri_m.get("n_epochs", "?")

    def _fp5(m): return f"ep {m.get('first_passage', {}).get(5)}" if m.get("first_passage", {}).get(5) else "not reached"
    def _fp9(m): return f"ep {m.get('first_passage', {}).get(0.9)}" if m.get("first_passage", {}).get(0.9) else "not reached"

    rows = [
        ["Metric",                CTX.labels.get(k1, "Run 1"),     CTX.labels.get(k2, "Run 2")],
        ["─" * 14,               "─" * 12,                         "─" * 12],
        ["Best MedAE",           f"{pr_m.get('best', np.nan):.2f} yr",  f"{ri_m.get('best', np.nan):.2f} yr"],
        ["Best R²",              f"{pr_r2.get('best', np.nan):.3f}",     f"{ri_r2.get('best', np.nan):.3f}"],
        ["MedAE < 5yr at ep",   _fp5(pr_m),                        _fp5(ri_m)],
        ["R² ≥ 0.9 at ep",      _fp9(pr_r2),                       _fp9(ri_r2)],
        ["Speed (Δ5ep MedAE)",  f"{stats.get(k1,{}).get('early_speed',{}).get(5, np.nan):+.1f} yr",
                                 f"{stats.get(k2,{}).get('early_speed',{}).get(5, np.nan):+.1f} yr"],
        ["─" * 14,               "─" * 12,                         "─" * 12],
        [f"ep shown: {ri_ep}",   "",                                ""],
    ]

    y0 = 0.97
    dy = 0.105
    for i, row in enumerate(rows):
        y = y0 - i * dy
        ax_num.text(0.01, y, row[0], transform=ax_num.transAxes,
                    fontsize=8.5, va="top", fontfamily="monospace",
                    color="#333333" if i not in (0, 1, 7) else "#555555")
        ax_num.text(0.45, y, row[1], transform=ax_num.transAxes,
                    fontsize=8.5, va="top", fontfamily="monospace",
                    color=CTX.colors.get("run1", "#E64B35") if i > 1 and i not in (7, 8) else "#555555",
                    fontweight="bold" if i > 1 and i not in (7, 8) else "normal")
        ax_num.text(0.72, y, row[2], transform=ax_num.transAxes,
                    fontsize=8.5, va="top", fontfamily="monospace",
                    color=CTX.colors.get("run2", "#4DBBD5") if i > 1 and i not in (7, 8) else "#555555",
                    fontweight="bold" if i > 1 and i not in (7, 8) else "normal")

    fig.suptitle(
        "The Value of WCED Pretraining: Architecture, Evidence, and Downstream Results",
        fontsize=13, fontweight="bold", y=1.01
    )

    path = out_dir / "5_pretraining_story.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 6: Final performance bar chart
# ────────────────────────────────────────────────────────────────────────────
def fig_final_performance(stats, out_dir):
    set_style()
    fig, axes = plt.subplots(1, 3, figsize=(13, 5))

    metrics = [
        ("val_medae", "Validation MedAE (years)",  True),
        ("val_mae",   "Validation MAE (years)",     True),
        ("val_r2",    "Validation R²",              False),
    ]

    for ax, (metric, ylabel, lower) in zip(axes, metrics):
        for xpos, label in enumerate(CTX.run_keys):
            m = stats.get(label, {}).get(metric, {})
            if not m:
                continue
            best   = m.get("best", np.nan)
            plat_m = m.get("plateau_mean", np.nan)
            plat_s = m.get("plateau_std",  np.nan)
            n_ep   = m.get("n_epochs", 300)

            ax.bar(xpos, best, color=CTX.colors[label], alpha=0.9,
                   edgecolor="white", linewidth=1.0, width=0.55, zorder=3)
            ax.errorbar(xpos, plat_m, yerr=plat_s, fmt="none",
                        ecolor="black", capsize=5, capthick=1.2, elinewidth=1.2, zorder=4)
            ax.text(xpos, best * (0.95 if lower else 1.03), f"{best:.3f}",
                    ha="center", va="top" if lower else "bottom",
                    fontsize=9, fontweight="bold", color=CTX.colors[label])
            if n_ep < 200:
                ax.text(xpos, best * (1.02 if lower else 0.98),
                        f"(ep {n_ep})", ha="center", va="bottom" if lower else "top",
                        fontsize=7.5, color="#888888", style="italic")

        ax.set_xticks(list(range(len(CTX.run_keys))))
        ax.set_xticklabels([CTX.labels.get(k, k) for k in CTX.run_keys], fontsize=9)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.tick_params(axis="x", length=0)

        # Gap annotation (first vs second run)
        if len(CTX.run_keys) >= 2:
            v1 = stats.get(CTX.run_keys[0], {}).get(metric, {}).get("best", np.nan)
            v2 = stats.get(CTX.run_keys[1], {}).get(metric, {}).get("best", np.nan)
            if not np.isnan(v1) and not np.isnan(v2):
                gap_pct = 100 * abs(v1 - v2) / abs(v2 + 1e-9)
                better  = "lower" if lower else "higher"
                ax.text(0.5, 0.97, f"Gap: {abs(v1 - v2):.3f} ({gap_pct:.1f}% {better})",
                        ha="center", va="top", transform=ax.transAxes,
                        fontsize=8.5, color="#333333",
                        bbox=dict(facecolor="#F8F8F8", edgecolor="#cccccc",
                                  linewidth=0.8, boxstyle="round,pad=0.3"))

        ax.set_title(ylabel, fontsize=10, fontweight="bold")

    fig.suptitle("Best Performance Comparison\n(error bars = last-20-epoch plateau std)",
                 fontsize=11, fontweight="bold")
    fig.tight_layout()
    path = out_dir / "6_final_performance.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Figure 7: Gradient-step efficiency
# ────────────────────────────────────────────────────────────────────────────
def fig_step_efficiency(df: pd.DataFrame, stats: dict, run_meta: dict, out_dir: Path):
    """Performance vs. gradient-update steps — shows compute efficiency."""
    set_style()
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))

    total_steps = {k: run_meta.get(k, {}).get("global_step", 0) for k in CTX.run_keys}
    total_epochs = {k: max(1, run_meta.get(k, {}).get("total_epochs", 1)) for k in CTX.run_keys}

    for ax_idx, (metric, ylabel, lower) in enumerate([
        ("val_medae", "Validation MedAE (years)", True),
        ("val_r2",    "Validation R²",             False),
    ]):
        ax = axes[ax_idx]

        if metric == "val_medae":
            ax.fill_between([0, 5e5], 0, 5, color="#2ecc71", alpha=0.08)
            ax.axhline(5,   color="#888888", lw=0.8, ls=":",  alpha=0.7)
            ax.text(1000, 5.3, "5 yr", fontsize=7.5, color="#888888")
            ax.axhline(7.5, color="#888888", lw=0.8, ls="-.", alpha=0.5)
            ax.text(1000, 7.8, "7.5 yr", fontsize=7.5, color="#888888")
        if metric == "val_r2":
            ax.fill_between([0, 5e5], 0.9, 1.0, color="#2ecc71", alpha=0.08)
            ax.axhline(0.9, color="#888888", lw=0.8, ls=":", alpha=0.7)
            ax.text(1000, 0.91, "R²=0.9", fontsize=7.5, color="#888888")

        for key in CTX.run_keys:
            sub  = df[df["run"] == key].sort_values("epoch")
            vals = sub[metric].dropna() if metric in sub.columns else pd.Series([], dtype=float)
            if len(vals) == 0:
                continue
            ep    = sub.loc[vals.index, "epoch"].values
            v     = vals.values
            n_ep  = total_epochs[key]
            n_st  = total_steps[key]
            # Use actual per-epoch global_step from data (downloaded from WandB).
            # Fall back to linear interpolation only if the column is missing/all-NaN.
            if "global_step" in sub.columns:
                gs_raw = sub.loc[vals.index, "global_step"]
                if gs_raw.notna().any():
                    steps_axis = gs_raw.fillna(
                        pd.Series((ep / max(n_ep, 1)) * n_st, index=gs_raw.index)
                    ).values
                else:
                    steps_axis = (ep / max(n_ep, 1)) * n_st
            else:
                steps_axis = (ep / max(n_ep, 1)) * n_st

            smth = smooth(vals).values
            ax.plot(steps_axis, v,    color=CTX.colors[key], alpha=0.15, lw=0.7)
            ax.plot(steps_axis, smth, color=CTX.colors[key], lw=2.2,
                    label=CTX.labels.get(key, key))

            # Mark best point
            m = stats.get(key, {}).get(metric, {})
            if m:
                best_ep = m.get("best_epoch", 0)
                best_v  = m.get("best", np.nan)
                best_step = (best_ep / max(n_ep, 1)) * n_st
                ax.scatter([best_step], [best_v], color=CTX.colors[key],
                           s=80, zorder=5, marker="*", edgecolors="white", lw=0.5)
                ax.annotate(f"  {best_v:.2f} yr\n  @{best_step/1000:.0f}k steps",
                            xy=(best_step, best_v),
                            xytext=(best_step + n_st * 0.05, best_v + (1.5 if lower else -0.06)),
                            fontsize=7.5, color=CTX.colors[key],
                            arrowprops=dict(arrowstyle="->", color=CTX.colors[key], lw=0.8))

        ax.set_xlabel("Gradient Update Steps", fontsize=10)
        ax.set_ylabel(ylabel, fontsize=10)
        ax.xaxis.set_major_formatter(mticker.FuncFormatter(
            lambda x, _: f"{int(x/1000)}k" if x >= 1000 else str(int(x))))
        ax.legend(fontsize=8.5)
        ax.set_title(f"{ylabel} vs. Total Gradient Steps\n"
                     "(step-efficiency: same quality with fewer updates = better WCED initialisation)",
                     fontsize=9.5, fontweight="bold")

    # Add step counts as a text box
    step_info = "\n".join([
        f"{CTX.labels.get(k, k)}: {total_steps[k]:,} steps  ({total_epochs[k]} ep)"
        for k in CTX.run_keys
    ])
    fig.text(0.5, 0.00, step_info, ha="center", va="bottom", fontsize=8.5,
             color="#555555", style="italic",
             bbox=dict(facecolor="#F8F8F8", edgecolor="#cccccc",
                       linewidth=0.8, boxstyle="round,pad=0.3"))

    fig.suptitle("Training Compute Efficiency: Performance vs. Gradient Update Steps",
                 fontsize=12, fontweight="bold")
    fig.tight_layout(rect=[0, 0.06, 1, 1])
    path = out_dir / "7_step_efficiency.png"
    fig.savefig(path)
    plt.close(fig)
    print(f"  Saved → {path}")


# ────────────────────────────────────────────────────────────────────────────
# Main
# ────────────────────────────────────────────────────────────────────────────
def main():
    p = argparse.ArgumentParser(
        description="Generate thesis-quality figures comparing two WandB fine-tuning runs.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Default hardcoded runs (pretrained vs random-init):
  python thesis_finetune_comparison_figures.py

  # Compare any two WandB URLs:
  python thesis_finetune_comparison_figures.py \\
      --run1 https://wandb.ai/myentity/myproject/runs/abc123 \\
      --run2 https://wandb.ai/myentity/myproject/runs/def456 \\
      --label1 "WCED pretrained" --label2 "Random init"

  # Bare run IDs (uses default entity/project from script):
  python thesis_finetune_comparison_figures.py --run1 abc123 --run2 def456

  # Custom output dir:
  python thesis_finetune_comparison_figures.py --run1 ... --run2 ... \\
      --outdir outputs/my_comparison
""")
    p.add_argument("--run1",   type=str, default=None,
                   help="WandB URL or run ID for the first run (default: pretrained WCED)")
    p.add_argument("--run2",   type=str, default=None,
                   help="WandB URL or run ID for the second run (default: random-init)")
    p.add_argument("--label1", type=str, default=None,
                   help="Display label for run1 (auto-generated if omitted)")
    p.add_argument("--label2", type=str, default=None,
                   help="Display label for run2 (auto-generated if omitted)")
    p.add_argument("--outdir", type=str, default=None,
                   help="Output directory (default: wandb_run_comparison_outputs/thesis_figures)")
    p.add_argument("--x_axis", type=str, default="epoch", choices=["epoch", "steps"],
                   help="X-axis for time-series figures: 'epoch' (default) or 'steps' (global_step)")
    args = p.parse_args()

    # ── Build run config and populate CTX ─────────────────────────────
    runs_cfg, labels, colors, out_dir = build_run_config(args)
    CTX.run_keys = list(runs_cfg.keys())
    CTX.colors   = colors
    CTX.labels   = labels
    CTX.x_axis   = args.x_axis

    set_style()
    out_dir.mkdir(parents=True, exist_ok=True)

    print("=" * 60)
    print("Thesis Figure Generator — WandB Run Comparison")
    print("=" * 60)
    for key, (entity, project, run_id) in runs_cfg.items():
        print(f"  {key}: {labels[key]}  ({entity}/{project}/{run_id})")
    print(f"  Output → {out_dir.resolve()}")

    # ── Download data ──────────────────────────────────────────────────
    df = load_data(runs_cfg, out_dir)
    print(f"\nData loaded: {len(df)} total epoch rows")
    for key in CTX.run_keys:
        sub = df[df["run"] == key]
        print(f"  {labels[key]}: {len(sub)} epochs "
              f"(ep {sub['epoch'].min()}–{sub['epoch'].max()})")

    # ── Collect run-level metadata (total steps, runtime) ─────────────
    run_meta = {}
    import wandb as _wandb
    _api = _wandb.Api(timeout=60)
    for key, (entity, project, run_id) in runs_cfg.items():
        try:
            _r = _api.run(f"{entity}/{project}/{run_id}")
            _s = _r.summary
            def _sf(*keys):
                for k in keys:
                    if k in _s and _s[k] is not None:
                        return float(_s[k])
                return float("nan")
            run_meta[key] = {
                "global_step":   int(_s.get("trainer/global_step", 0)),
                "total_epochs":  int(_s.get("epoch", 0)),
                "runtime_hrs":   float(_s.get("_runtime", 0)) / 3600,
                # test-set metrics — try slash format (MethylLlama) then underscore (Baseline)
                "test_medae":    _sf("test/medae",    "test_medae"),
                "test_mae":      _sf("test/mae",      "test_mae"),
                "test_rmse":     _sf("test/rmse",     "test_rmse"),
                "test_r2":       _sf("test/r2",       "test_r2"),
                "test_pearson":  _sf("test/p_r",      "test_p_r"),
                "test_spearman": _sf("test/s_r",      "test_s_r"),
                # validation metrics — try slash format then underscore
                "valid_medae":   _sf("val/medae",     "valid_medae"),
                "valid_mae":     _sf("val/mae",       "valid_mae"),
                "valid_rmse":    _sf("val/rmse",      "valid_rmse"),
                "valid_r2":      _sf("val/r2",        "valid_r2"),
                "valid_pearson": _sf("val/p_r",       "valid_p_r"),
                "valid_spearman":_sf("val/s_r",       "valid_s_r"),
            }
        except Exception:
            run_meta[key] = {"global_step": 0, "total_epochs": 0, "runtime_hrs": 0}
        m = run_meta[key]
        print(f"  {labels.get(key, key):30s}  "
              f"gradient steps={m['global_step']:,}  "
              f"epochs={m['total_epochs']}  "
              f"runtime={m['runtime_hrs']:.1f}h")

    # ── Statistical analysis ───────────────────────────────────────────
    print("\nRunning statistical analysis ...")
    stats = compute_stats(df)
    write_stats_report(stats, out_dir / "stats_report.txt", labels, run_meta)

    for key, s in stats.items():
        m = s.get("val_medae", {})
        r = s.get("val_r2",    {})
        print(f"  {labels.get(key, key):30s}  "
              f"MedAE best={m.get('best', np.nan):.3f}yr  "
              f"R²={r.get('best', np.nan):.4f}  "
              f"asym={m.get('asymptote', np.nan):.3f}yr")

    # ── Generate figures ───────────────────────────────────────────────
    print("\nGenerating figures ...")
    fig_learning_curves(df, stats, out_dir)
    fig_early_phase(df, stats, out_dir)
    fig_convergence_chart(stats, out_dir, df=df)
    fig_efficiency(df, stats, out_dir)
    fig_pretraining_story(df, stats, out_dir)
    fig_final_performance(stats, out_dir)
    fig_step_efficiency(df, stats, run_meta, out_dir)

    print(f"\n{'='*60}")
    print(f"All outputs saved to: {out_dir.resolve()}")
    print(f"{'='*60}")


if __name__ == "__main__":
    main()
