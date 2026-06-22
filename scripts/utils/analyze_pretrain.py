#!/usr/bin/env python3
"""
analyze_pretrain.py — Download and compare MethylLlama pretraining runs from WandB.

Works for:
  - Architecture sweep:      --project arch-sweep
  - Smoke test runs:         --project pretrain-llama-smoke
  - Full pretrain runs:      --project pretrain-llama-wced

Usage:
    python scripts/utils/analyze_pretrain.py
    python scripts/utils/analyze_pretrain.py --project arch-sweep
    python scripts/utils/analyze_pretrain.py --project pretrain-llama-smoke --out_dir figures/smoke

Outputs (in --out_dir):
    summary.csv                  one row per run, all key metrics
    summary.txt                  ranked table printed to terminal + file
    learning_curves_pcc.png      val/pcc over epochs, all runs
    learning_curves_loss.png     val/loss over epochs, all runs
    train_curves_pcc.png         train/pcc over epochs (overfitting check)
    heatmap_val_pcc.png          hidden_size × num_layers → best val pcc   (arch-sweep only)
    heatmap_val_loss.png         hidden_size × num_layers → best val loss  (arch-sweep only)
"""

import argparse
import re
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

try:
    import wandb
except ImportError:
    sys.exit("wandb not installed.  Run: pip install wandb")

# ─────────────────────────────────────────────────────────────────────────────
ENTITY = "netanelazran11-hebrew-university-of-jerusalem"
COLORS = plt.cm.tab10.colors

# Metric keys logged by pretrain_llama.py  (try multiple names for robustness)
VAL_PCC_KEYS  = ["validation/pcc",  "val/pcc",  "validation/beta_values_pcc"]
VAL_LOSS_KEYS = ["validation/loss", "val/loss", "validation/beta_values_mse"]
TRN_PCC_KEYS  = ["train/pcc",  "training/pcc",  "train/beta_values_pcc_epoch"]
TRN_LOSS_KEYS = ["train/loss", "training/loss", "train/loss_epoch"]


# ─────────────────────────────────────────────────────────────────────────────
# helpers
# ─────────────────────────────────────────────────────────────────────────────

def first_val(d: dict, keys: list):
    for k in keys:
        v = d.get(k)
        if v is not None:
            try:
                return float(v)
            except (ValueError, TypeError):
                pass
    return None


def parse_arch(run_name: str) -> tuple[int | None, int | None]:
    """Extract (hidden_size, num_layers) from names like 'h256_l4-...'."""
    m = re.search(r"h(\d+)_l(\d+)", run_name)
    if m:
        return int(m.group(1)), int(m.group(2))
    return None, None


def approx_params_M(h: int, l: int, vocab: int = 49161) -> float:
    attn = 4 * h * h
    ffn  = 3 * h * (4 * h)   # SwiGLU: 3 weight matrices
    emb  = vocab * h
    return (l * (attn + ffn) + emb) / 1e6


def variant_label(h, l, params=None):
    s = f"h{h}_l{l}"
    if params:
        s += f" (~{params:.0f}M)"
    return s


# ─────────────────────────────────────────────────────────────────────────────
# WandB fetch
# ─────────────────────────────────────────────────────────────────────────────

def fetch_all_runs(project: str, entity: str = ENTITY):
    api = wandb.Api(timeout=120)
    print(f"Connecting to WandB: {entity}/{project} ...")
    runs = list(api.runs(f"{entity}/{project}"))
    print(f"  Found {len(runs)} runs")
    return runs, api


def run_summary_row(run) -> dict:
    name = run.name or run.id
    h, l = parse_arch(name)
    s    = run.summary or {}

    row = {
        "run_id":      run.id,
        "run_name":    name,
        "state":       run.state,
        "hidden_size": h,
        "num_layers":  l,
        "params_M":    approx_params_M(h, l) if (h and l) else None,
        "val_pcc":     first_val(s, VAL_PCC_KEYS),
        "val_loss":    first_val(s, VAL_LOSS_KEYS),
        "train_pcc":   first_val(s, TRN_PCC_KEYS),
        "train_loss":  first_val(s, TRN_LOSS_KEYS),
        "epoch":       s.get("epoch"),
        "runtime_h":   (s.get("_wandb") or {}).get("runtime", 0) / 3600,
    }
    row["pcc_gap"] = (
        row["train_pcc"] - row["val_pcc"]
        if (row["train_pcc"] is not None and row["val_pcc"] is not None)
        else None
    )
    return row


def fetch_history(run) -> pd.DataFrame:
    # Fetch ALL columns — don't filter by keys (filtering returns empty if names differ)
    try:
        df = run.history(pandas=True, samples=5000)
        if df.empty:
            # scan_history is slower but more reliable for finished runs
            rows = list(run.scan_history())
            df = pd.DataFrame(rows)
        return df.reset_index(drop=True) if not df.empty else pd.DataFrame()
    except Exception as e:
        print(f"    [warn] history failed for {run.name}: {e}")
        return pd.DataFrame()


# ─────────────────────────────────────────────────────────────────────────────
# normalise history columns to canonical names
# ─────────────────────────────────────────────────────────────────────────────

def canonical(df: pd.DataFrame) -> pd.DataFrame:
    rename = {}
    for target, keys in [
        ("val_pcc",   VAL_PCC_KEYS),
        ("val_loss",  VAL_LOSS_KEYS),
        ("train_pcc", TRN_PCC_KEYS),
        ("train_loss",TRN_LOSS_KEYS),
    ]:
        for k in keys:
            if k in df.columns and target not in df.columns:
                rename[k] = target
    return df.rename(columns=rename)


# ─────────────────────────────────────────────────────────────────────────────
# plots
# ─────────────────────────────────────────────────────────────────────────────

def plot_curves(histories: dict, metric: str, ylabel: str, title: str, out: Path):
    """One line per (hidden, layers) variant."""
    fig, ax = plt.subplots(figsize=(11, 6))
    keys = sorted(histories.keys())
    for i, key in enumerate(keys):
        df = canonical(histories[key])
        if metric not in df.columns:
            continue
        data = df[metric].dropna()
        if data.empty:
            continue
        x = np.arange(len(data))
        h, l = key
        params = approx_params_M(h, l) if (h and l) else None
        label  = variant_label(h, l, params) if (h and l) else str(key)
        ax.plot(x, data.values, color=COLORS[i % len(COLORS)],
                linewidth=2, marker="o", markersize=3, label=label)

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel(ylabel, fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")
    ax.legend(fontsize=8, framealpha=0.9, loc="best")
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_train_val_overlay(histories: dict, out: Path):
    """For each variant: train_pcc (dashed) vs val_pcc (solid)."""
    fig, ax = plt.subplots(figsize=(11, 6))
    keys = sorted(histories.keys())
    for i, key in enumerate(keys):
        df = canonical(histories[key])
        color = COLORS[i % len(COLORS)]
        h, l = key
        label = variant_label(h, l)
        if "val_pcc" in df.columns:
            data = df["val_pcc"].dropna()
            ax.plot(np.arange(len(data)), data.values,
                    color=color, linewidth=2, label=f"{label} val")
        if "train_pcc" in df.columns:
            data = df["train_pcc"].dropna()
            ax.plot(np.arange(len(data)), data.values,
                    color=color, linewidth=1, linestyle="--", alpha=0.6, label=f"{label} train")

    ax.set_xlabel("Epoch", fontsize=12)
    ax.set_ylabel("PCC", fontsize=12)
    ax.set_title("Train (dashed) vs Val (solid) PCC — Overfitting Check", fontsize=13, fontweight="bold")
    ax.legend(fontsize=7, framealpha=0.9, loc="best", ncol=2)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


def plot_heatmap(df: pd.DataFrame, metric: str, title: str, out: Path, higher_better: bool):
    sub = df[df["hidden_size"].notna() & df["num_layers"].notna() & df[metric].notna()].copy()
    if sub.empty:
        return
    sub["hidden_size"] = sub["hidden_size"].astype(int)
    sub["num_layers"]  = sub["num_layers"].astype(int)

    pivot = sub.pivot_table(
        index="num_layers", columns="hidden_size", values=metric,
        aggfunc="max" if higher_better else "min"
    )

    fig, ax = plt.subplots(figsize=(7, 5))
    cmap = "RdYlGn" if higher_better else "RdYlGn_r"
    vmin, vmax = pivot.values[~np.isnan(pivot.values)].min(), pivot.values[~np.isnan(pivot.values)].max()
    im = ax.imshow(pivot.values.astype(float), cmap=cmap, aspect="auto",
                   vmin=vmin, vmax=vmax)
    plt.colorbar(im, ax=ax, label=metric)

    ax.set_xticks(range(len(pivot.columns)))
    ax.set_xticklabels([f"h{c}" for c in pivot.columns], fontsize=11)
    ax.set_yticks(range(len(pivot.index)))
    ax.set_yticklabels([f"l{r}" for r in pivot.index], fontsize=11)
    ax.set_xlabel("Hidden size", fontsize=12)
    ax.set_ylabel("Num layers", fontsize=12)
    ax.set_title(title, fontsize=13, fontweight="bold")

    for r in range(len(pivot.index)):
        for c in range(len(pivot.columns)):
            val = pivot.values[r, c]
            if not np.isnan(float(val)):
                ax.text(c, r, f"{val:.3f}", ha="center", va="center",
                        fontsize=11, fontweight="bold", color="black")

    fig.tight_layout()
    fig.savefig(out, dpi=150)
    plt.close(fig)
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# summary table
# ─────────────────────────────────────────────────────────────────────────────

def print_and_save_summary(df: pd.DataFrame, out: Path):
    cols = ["run_name", "hidden_size", "num_layers", "params_M",
            "val_pcc", "val_loss", "train_pcc", "pcc_gap", "epoch", "state"]
    sub  = df[[c for c in cols if c in df.columns]].copy()
    sub  = sub.sort_values("val_pcc", ascending=False, na_position="last")

    header = "\n" + "=" * 100 + "\n"
    header += f"  RESULTS — sorted by val_pcc\n"
    header += "=" * 100
    print(header)

    with pd.option_context("display.max_columns", 20, "display.width", 120,
                           "display.float_format", "{:.4f}".format):
        table_str = sub.to_string(index=False)
        print(table_str)

    print("=" * 100)

    if not sub.empty and sub["val_pcc"].notna().any():
        best = sub[sub["val_pcc"].notna()].iloc[0]
        h = int(best["hidden_size"]) if pd.notna(best.get("hidden_size")) else "?"
        l = int(best["num_layers"])  if pd.notna(best.get("num_layers"))  else "?"
        p = f"{best['params_M']:.1f}M" if pd.notna(best.get("params_M")) else "?"
        print(f"\n★  Best: h{h}_l{l} ({p})  "
              f"val_pcc={best['val_pcc']:.4f}  val_loss={best['val_loss']:.4f}  "
              f"gap={best.get('pcc_gap', float('nan')):.4f}")

    with open(out, "w") as f:
        f.write(header + "\n" + table_str + "\n")
    print(f"  Saved: {out}")


# ─────────────────────────────────────────────────────────────────────────────
# main
# ─────────────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--project",    default="arch-sweep",
                        help="WandB project (default: arch-sweep)")
    parser.add_argument("--entity",     default=ENTITY)
    parser.add_argument("--out_dir",    default="figures/arch_sweep")
    parser.add_argument("--min_epochs", type=int, default=3,
                        help="Skip runs with fewer completed epochs")
    args = parser.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # ── 1. fetch run summaries ─────────────────────────────────────────────
    runs, _ = fetch_all_runs(args.project, args.entity)
    if not runs:
        sys.exit("No runs found.")

    rows = [run_summary_row(r) for r in runs]
    df   = pd.DataFrame(rows)
    print(f"  Parsed {len(df)} runs")

    # filter runs that haven't started properly
    if "epoch" in df.columns:
        mask = df["epoch"].isna() | (df["epoch"] >= args.min_epochs)
        n_skip = (~mask).sum()
        if n_skip:
            print(f"  Skipping {n_skip} runs with < {args.min_epochs} epochs")
        df = df[mask]

    # ── 2. save CSV ────────────────────────────────────────────────────────
    csv_path = out_dir / "summary.csv"
    df.to_csv(csv_path, index=False)
    print(f"  Saved: {csv_path}")

    # ── 3. summary table ───────────────────────────────────────────────────
    print_and_save_summary(df, out_dir / "summary.txt")

    # ── 4. fetch per-epoch histories ───────────────────────────────────────
    print("\nFetching epoch histories ...")
    run_map   = {r.id: r for r in runs}
    histories = {}

    first = True
    for _, row in df.iterrows():
        rid = row["run_id"]
        r   = run_map.get(rid)
        if r is None:
            continue
        key = (row["hidden_size"], row["num_layers"])
        print(f"  {row['run_name']} ({row['state']}) ...", end=" ", flush=True)
        hist = fetch_history(r)
        if not hist.empty:
            histories[key] = hist
            print(f"{len(hist)} rows")
            if first:
                metric_cols = [c for c in hist.columns if not c.startswith("_")]
                print(f"    Available metrics: {metric_cols[:20]}")
                first = False
        else:
            print("empty")

    if not histories:
        print("No history data available yet — runs may still be starting.")
        return

    # ── 5. learning curve plots ────────────────────────────────────────────
    print("\nGenerating plots ...")
    plot_curves(histories, "val_pcc",  "Validation PCC",
                f"Val PCC — {args.project}",
                out_dir / "learning_curves_pcc.png")

    plot_curves(histories, "val_loss", "Validation Loss (MSE)",
                f"Val Loss — {args.project}",
                out_dir / "learning_curves_loss.png")

    plot_curves(histories, "train_pcc", "Train PCC",
                f"Train PCC — {args.project}",
                out_dir / "train_curves_pcc.png")

    plot_train_val_overlay(histories, out_dir / "overfitting_check.png")

    # ── 6. heatmaps (arch sweep only — need hidden/layer grid) ────────────
    has_arch = df["hidden_size"].notna().any() and df["num_layers"].notna().any()
    if has_arch:
        if df["val_pcc"].notna().any():
            plot_heatmap(df, "val_pcc",  "Best Val PCC by Architecture",
                         out_dir / "heatmap_val_pcc.png",  higher_better=True)
        if df["val_loss"].notna().any():
            plot_heatmap(df, "val_loss", "Best Val Loss by Architecture",
                         out_dir / "heatmap_val_loss.png", higher_better=False)

    print(f"\nAll outputs → {out_dir.resolve()}")


if __name__ == "__main__":
    main()
