#!/usr/bin/env python3
"""
Collect the pilot evaluation into one table (CSV + JSON) and apply the pre-registered decision rule.

Reads, per arm directory under --eval-root (from evaluate_pilot.sh):
  view_consistency/per_run/{overlap,disjoint}_seed0.json   retrieval on the full held-out partition
  hard_negative.json                                        hard_negative_eval.py --out (run first, see below)
  reconstruction/reconstruction_withheld_summary.json       withheld-CpG reconstruction
  frozen_probes/analysis_summary.json                       geometry + frozen probes (AltumAge)

Run from the repository root:
  for d in outputs/objective_pilot/pilot_eval/*/; do
    python -m objective_pilot.hard_negative_eval --emb-dir "$d/view_consistency/per_run/" \
        --out "$d/hard_negative.json" --per-query-out "$d/hard_negative_queries.npz"
  done
  python -m objective_pilot.summarize_pilot

Design (decided 2026-10-18): one seed (42), a λ dose series 0 / 0.01 / 0.05 / 0.2 from identical
initial weights and data order. Evidence of an InfoNCE effect comes from the dose-response, not from
seed replicates.

Decision rule (fixed before the results). For each per-profile metric, every λ>0 arm is compared with
λ=0 on the SAME held-out profiles (paired), 95 % percentile bootstrap over profiles (B=2000):
  effect     the λ=0.2 − λ=0 difference has a CI excluding 0 AND the arm values are monotone in λ
  no effect  every λ>0 − λ=0 difference has a CI including 0
  unclear    anything else (e.g. a non-monotone pattern) -> add seed 43 for λ=0 and λ=0.2 only
Primary metric: disjoint-view hard-negative top-1, same study + tissue (hn_study_tissue).
The bootstrap captures evaluation noise (which profiles), not training noise (which seed); a
borderline result is therefore resolved with a seed replicate, not by reading the CI as final.
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--eval-root", default="outputs/objective_pilot/pilot_eval")
ap.add_argument("--out", default="outputs/objective_pilot/pilot_summary")
a = ap.parse_args()


def load(p):
    p = Path(p)
    return json.loads(p.read_text()) if p.exists() else None


rows = []
for d in sorted(Path(a.eval_root).iterdir()):
    if not d.is_dir():
        continue
    r = {"arm": d.name}
    for cond in ("overlap", "disjoint"):
        v = load(d / "view_consistency" / "per_run" / f"{cond}_seed0.json")
        if v:
            p = v["pools"]["all"]
            r[f"{cond}_top1"] = p["retrieval_at1"]
            r[f"{cond}_matched_cos"] = p["matched_cos"]
            r[f"{cond}_unmatched_cos"] = p["unmatched_cos"]
            if "fully_measured" in v["pools"]:
                r[f"{cond}_top1_fully_measured"] = v["pools"]["fully_measured"]["retrieval_at1"]
    h = load(d / "hard_negative.json")
    if h and "disjoint_seed0" in h:
        pools = h["disjoint_seed0"]["pools"]
        for key, col in [("random", "hn_random"), ("same study", "hn_study"),
                         ("same study+tissue", "hn_study_tissue"),
                         ("same study+tissue+sex", "hn_study_tissue_sex"),
                         ("K nearest neighbours of target", "hn_knn")]:
            if key in pools:
                r[f"{col}_top1"] = pools[key]["top1"]
                if pools[key].get("median_margin") is not None:
                    r[f"{col}_margin"] = pools[key]["median_margin"]
    rc = load(d / "reconstruction" / "reconstruction_withheld_summary.json")
    if rc:
        r["recon_pcc"] = rc["pearson_withheld_obs_vs_pred"]
        r["recon_raw_mse"] = rc["raw_mse_model"]["mean"]
        r["recon_ratio_vs_mean_baseline"] = rc["ratio_model_vs_b_mean"]
    fp = load(d / "frozen_probes" / "analysis_summary.json")
    if fp:
        g = fp["geometry_cls"]
        r["eff_rank"] = g["effective_rank"]
        r["anisotropy"] = g["anisotropy_mean_cos"]
        r["age_ridge_r2"] = fp["age_probe_cls"]["linear_ridge_r2"]
        r["age_ridge_medae"] = fp["age_probe_cls"]["linear_ridge_medae"]
        for c, v in fp.get("class_probes_cls", {}).items():
            r[f"{c}_bal_acc"] = v["balanced_acc"]
    rows.append(r)

df = pd.DataFrame(rows).set_index("arm")
out = Path(a.out)
out.mkdir(parents=True, exist_ok=True)
df.to_csv(out / "pilot_table.csv")
pd.set_option("display.width", 200)
print(df.T.to_string(float_format=lambda x: f"{x:.4f}"))

# ── Decision rule: dose-response over λ with paired bootstrap against λ = 0 ──────────────────────
B, rng = 2000, np.random.default_rng(0)


def lam_of(arm):
    if not arm.startswith("lam"):
        return None
    return float(arm[3:].split("_")[0])


def per_profile(d):
    """Per-profile vectors keyed by metric (higher = better for all, MSE negated)."""
    out = {}
    v = Path(d) / "view_consistency" / "per_run" / "disjoint_seed0.npz"
    if v.exists():
        z = np.load(v)
        out["disjoint_top1"] = (z["all_rank"] == 0).astype(float)
    q = Path(d) / "hard_negative_queries.npz"
    if q.exists():
        z = np.load(q)
        for pool, col in [("same study+tissue", "hn_study_tissue"), ("same study", "hn_study"),
                          ("K nearest neighbours of target", "hn_knn")]:
            k = f"disjoint_seed0|{pool}"
            if f"{k}|hit" in z.files:
                out[col] = pd.Series(z[f"{k}|hit"].astype(float), index=z[f"{k}|q"])
    r = Path(d) / "reconstruction" / "per_sample_metrics.csv"
    if r.exists():
        out["neg_recon_mse"] = -pd.read_csv(r)["model_mse"].values
    return out


arms = {d.name: (lam_of(d.name), per_profile(d)) for d in sorted(Path(a.eval_root).iterdir())
        if d.is_dir() and lam_of(d.name) is not None}
seeds = sorted({n.split("_seed")[1].split("_")[0] for n in arms})
verdict = {"design": "single-seed λ dose series", "seeds_present": seeds, "B": B, "metrics": {}}
base = [n for n, (lam, _) in arms.items() if lam == 0.0 and "_seed42" in n]
if not base:
    verdict["note"] = "no λ=0 seed-42 arm evaluated — rule not applied"
else:
    base = base[0]
    for metric in ("hn_study_tissue", "hn_study", "hn_knn", "disjoint_top1", "neg_recon_mse"):
        if metric not in arms[base][1]:
            continue
        x0 = arms[base][1][metric]
        series, diffs = [], {}
        for n, (lam, pp) in sorted(arms.items(), key=lambda t: t[1][0]):
            if "_seed42" not in n or metric not in pp:
                continue
            x = pp[metric]
            if isinstance(x0, pd.Series):  # align on the same queries
                common = x0.index.intersection(x.index)
                a0, a1 = x0.loc[common].values, x.loc[common].values
            else:
                a0, a1 = x0, x
            series.append((lam, float(a1.mean())))
            if lam == 0.0:
                continue
            dvec = a1 - a0
            idx = rng.integers(0, len(dvec), size=(B, len(dvec)))
            boot = dvec[idx].mean(axis=1)
            lo, hi = np.percentile(boot, [2.5, 97.5])
            diffs[str(lam)] = {"diff_vs_lambda0": float(dvec.mean()), "ci95": [float(lo), float(hi)],
                               "ci_excludes_0": bool(lo > 0 or hi < 0), "n_profiles": int(len(dvec))}
        vals = [v for _, v in series]
        steps = np.sign(np.diff(vals))
        monotone = bool(len(vals) > 2 and (np.all(steps >= 0) or np.all(steps <= 0)))
        top = diffs.get(str(max(l for l, _ in series))) if diffs else None
        if top and top["ci_excludes_0"] and monotone:
            call = "effect"
        elif diffs and not any(dd["ci_excludes_0"] for dd in diffs.values()):
            call = "no effect"
        else:
            call = "unclear"
        verdict["metrics"][metric] = {"by_lambda": series, "vs_lambda0": diffs, "monotone": monotone, "call": call}
    prim = verdict["metrics"].get("hn_study_tissue", {}).get("call")
    verdict["primary_call"] = prim
    verdict["next_step"] = {
        "effect": "InfoNCE changes identity at pilot scale -> run the full-corpus recon-only arm to measure it at scale",
        "no effect": "no measurable InfoNCE effect at pilot scale -> full run optional; do not attribute identity to InfoNCE",
        "unclear": "add seed 43 for λ=0 and λ=0.2 before deciding",
        None: "primary metric missing (run hard_negative_eval with --per-query-out)",
    }[prim]
(out / "pilot_verdict.json").write_text(json.dumps(verdict, indent=1, default=float))
print("\nDecision rule:", json.dumps({k: verdict[k] for k in verdict if k != "metrics"}, indent=1, default=float))
for m, r in verdict.get("metrics", {}).items():
    print(f"  {m:16s} call={r['call']:9s} monotone={r['monotone']}  " +
          "  ".join(f"λ={l}:{v:.4f}" for l, v in r["by_lambda"]))
