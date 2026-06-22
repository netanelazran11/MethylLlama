#!/usr/bin/env python3
"""
compare_datasets_fingerprint.py
=================================
Rigorous cross-dataset comparison:
  Dataset A — MethylLlama  : 19,608-CpG h5ad (train/valid/test)
  Dataset B — AltumAge 21k : 21,368-CpG h5ad (train/valid/test)

Two levels of sample identity check:
  1. Direct ID match  — if both datasets use real GSM/TCGA IDs
  2. Methylation fingerprint — cosine similarity over ALL shared CpG sites
     (cosine >= 0.9999 = identical biological sample)

Outputs (--outdir):
  fingerprint_report.html
  fingerprint_summary.txt
"""

import argparse, sys
from pathlib import Path

import numpy as np
import pandas as pd

# ─────────────────────────────────────────────────────────────────────────────
_BASE = "/sci/labs/benjamin.yakir/netanel.azran"
_DATA = f"{_BASE}/data"

LLAMA_H5AD = (
    f"{_DATA}/data_methyl_finetune_19k_h5ad/"
    "finetuning_19608_clean_stratified_no_outliers.h5ad"
)
ALT_H5AD = f"{_DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"


# ─────────────────────────────────────────────────────────────────────────────
# Loader (works for any h5ad)
# ─────────────────────────────────────────────────────────────────────────────
def load_h5ad(path: str, label: str):
    """
    Returns:
      X_by_split  : dict split -> np.ndarray (n, n_cpgs) float32
      ids_by_split: dict split -> list[str]
      ages_by_split: dict split -> np.ndarray float32
      cpg_ids     : list[str]
    """
    print(f"\n[{label}] loading {path}")
    import scipy.sparse

    try:
        import scanpy as sc
        adata = sc.read_h5ad(path)
    except Exception as e:
        print(f"  scanpy failed ({e}), trying h5py…")
        import h5py, anndata as ad
        with h5py.File(path, "r") as f:
            X_grp = f["X"]
            if isinstance(X_grp, h5py.Dataset):
                X = X_grp[()].astype(np.float32)
            else:
                data    = X_grp["data"][()]
                indices = X_grp["indices"][()]
                indptr  = X_grp["indptr"][()]
                n_obs   = len(f["obs"]["_index"])
                n_vars  = len(f["var"]["_index"])
                X = scipy.sparse.csr_matrix(
                    (data, indices, indptr), shape=(n_obs, n_vars)
                ).toarray().astype(np.float32)

            def _read_grp(grp, n):
                idx_key = "_index" if "_index" in grp else list(grp.keys())[0]
                idx = [x.decode() if isinstance(x, bytes) else str(x)
                       for x in grp[idx_key][:]]
                cols = {}
                for k in grp.keys():
                    if k == idx_key:
                        continue
                    try:
                        v = grp[k]
                        if isinstance(v, h5py.Dataset) and v.ndim == 1 and len(v) == n:
                            raw = v[()]
                            cols[k] = np.array([x.decode() if isinstance(x, bytes) else x
                                                for x in raw])
                    except Exception:
                        pass
                return idx, pd.DataFrame(cols, index=idx)

            obs_idx, obs_df = _read_grp(f["obs"], X.shape[0])
            var_idx, var_df = _read_grp(f["var"], X.shape[1])
        adata = ad.AnnData(X=X, obs=obs_df, var=var_df)

    print(f"  shape: {adata.n_obs:,} × {adata.n_vars:,}")

    obs = adata.obs.copy()
    obs.index = obs.index.astype(str)
    cpg_ids = list(adata.var.index.astype(str))

    if scipy.sparse.issparse(adata.X):
        X_dense = adata.X.toarray().astype(np.float32)
    else:
        X_dense = np.asarray(adata.X, dtype=np.float32)

    split_col = next((c for c in ("split", "Split", "set") if c in obs.columns), None)
    print(f"  split column: {split_col}")
    print(f"  obs columns:  {obs.columns.tolist()}")
    print(f"  sample IDs sample: {obs.index[:5].tolist()}")

    X_by_split, ids_by_split, ages_by_split = {}, {}, {}
    for sp in ("train", "valid", "test"):
        if split_col:
            mask = (obs[split_col] == sp).values
        else:
            mask = np.ones(len(obs), dtype=bool)
        X_by_split[sp]    = X_dense[mask]
        ids_by_split[sp]  = obs.index[mask].tolist()
        ages_by_split[sp] = (
            pd.to_numeric(obs["age"][mask], errors="coerce").values
            if "age" in obs.columns else np.full(mask.sum(), np.nan)
        )
        print(f"  {label} {sp}: {mask.sum():,} samples")

    return X_by_split, ids_by_split, ages_by_split, cpg_ids


# ─────────────────────────────────────────────────────────────────────────────
# CpG overlap
# ─────────────────────────────────────────────────────────────────────────────
def analyze_cpg_overlap(ll_cpg_ids, alt_cpg_ids):
    ll_set  = set(ll_cpg_ids)
    alt_set = set(alt_cpg_ids)
    shared  = ll_set & alt_set
    print(f"\n[CpG overlap]")
    print(f"  MethylLlama : {len(ll_set):,}")
    print(f"  AltumAge 21k: {len(alt_set):,}")
    print(f"  Shared      : {len(shared):,}  ({100*len(shared)/max(len(ll_set),1):.1f}% of LL)")
    print(f"  Only LL     : {len(ll_set-alt_set):,}")
    print(f"  Only Alt    : {len(alt_set-ll_set):,}")
    return {
        "ll_n":          len(ll_set),
        "alt_n":         len(alt_set),
        "shared_n":      len(shared),
        "only_ll":       len(ll_set - alt_set),
        "only_alt":      len(alt_set - ll_set),
        "ll_subset_alt": len(ll_set - alt_set) == 0,
        "pct_ll":        100*len(shared)/max(len(ll_set), 1),
        "pct_alt":       100*len(shared)/max(len(alt_set), 1),
        "shared_ids":    shared,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Direct ID match
# ─────────────────────────────────────────────────────────────────────────────
def direct_id_match(ll_ids, alt_ids, split):
    ll_set  = set(ll_ids)
    alt_set = set(alt_ids)
    shared  = ll_set & alt_set
    pct_ll  = 100 * len(shared) / max(len(ll_set), 1)
    pct_alt = 100 * len(shared) / max(len(alt_set), 1)
    print(f"  [ID/{split}] LL={len(ll_set):,}  Alt={len(alt_set):,}  "
          f"shared={len(shared):,} ({pct_ll:.1f}% of LL, {pct_alt:.1f}% of Alt)")
    return {
        "ll_n":       len(ll_set),
        "alt_n":      len(alt_set),
        "shared_n":   len(shared),
        "only_ll":    len(ll_set - alt_set),
        "only_alt":   len(alt_set - ll_set),
        "pct_ll":     pct_ll,
        "pct_alt":    pct_alt,
        "fully_same": len(shared) == len(ll_set) == len(alt_set),
        "shared_ids_sample": sorted(shared)[:10],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Fingerprint matching
# ─────────────────────────────────────────────────────────────────────────────
def cosine_nn(A: np.ndarray, B: np.ndarray, batch: int = 256) -> tuple:
    """For each row in A find nearest row in B. Returns (best_sim, best_idx)."""
    A = np.nan_to_num(A, nan=0.0).astype(np.float32)
    B = np.nan_to_num(B, nan=0.0).astype(np.float32)
    A /= (np.linalg.norm(A, axis=1, keepdims=True) + 1e-12)
    B /= (np.linalg.norm(B, axis=1, keepdims=True) + 1e-12)
    best_sim = np.empty(len(A), dtype=np.float32)
    best_idx = np.empty(len(A), dtype=np.int32)
    for s in range(0, len(A), batch):
        e = min(s + batch, len(A))
        S = A[s:e] @ B.T
        best_sim[s:e] = S.max(axis=1)
        best_idx[s:e] = S.argmax(axis=1)
    return best_sim, best_idx


def fingerprint_match(ll_X, alt_X, ll_ids, alt_ids, ll_ages, alt_ages, split, threshold=0.9999):
    print(f"\n  [{split}] fingerprint: LL={len(ll_ids):,} × Alt={len(alt_ids):,}")
    if not len(ll_ids) or not len(alt_ids):
        return {"note": "empty split", "ll_n": len(ll_ids), "alt_n": len(alt_ids)}

    best_sim, best_idx = cosine_nn(ll_X, alt_X)
    best_alt_id  = [alt_ids[i] for i in best_idx]
    best_alt_age = alt_ages[best_idx]

    exact   = int((best_sim >= threshold).sum())
    near    = int(((best_sim >= 0.999) & (best_sim < threshold)).sum())
    partial = int(((best_sim >= 0.99)  & (best_sim < 0.999)).sum())
    poor    = int((best_sim < 0.99).sum())

    print(f"  [{split}] sim: min={best_sim.min():.6f} mean={best_sim.mean():.6f} max={best_sim.max():.6f}")
    print(f"  [{split}] exact(>={threshold}): {exact:,}/{len(ll_ids):,} ({100*exact/max(len(ll_ids),1):.1f}%)")
    print(f"  [{split}] near [0.999,{threshold}): {near:,}  partial [0.99,0.999): {partial:,}  poor <0.99: {poor:,}")

    bins   = [0.0, 0.9, 0.99, 0.999, 0.9999, 1.0001]
    labels = ["<0.9", "0.9–0.99", "0.99–0.999", "0.999–0.9999", "≥0.9999"]
    counts, _ = np.histogram(best_sim, bins=bins)

    return {
        "split":           split,
        "ll_n":            len(ll_ids),
        "alt_n":           len(alt_ids),
        "sim_min":         float(best_sim.min()),
        "sim_mean":        float(best_sim.mean()),
        "sim_max":         float(best_sim.max()),
        "sim_median":      float(np.median(best_sim)),
        "exact_n":         exact,
        "exact_pct":       float(100*exact/max(len(ll_ids),1)),
        "near_n":          near,
        "partial_n":       partial,
        "poor_n":          poor,
        "threshold":       threshold,
        "hist_counts":     counts.tolist(),
        "hist_labels":     labels,
        "verdict": (
            f"IDENTICAL — {exact:,}/{len(ll_ids):,} samples matched exactly"
            if exact == len(ll_ids) else
            f"PARTIAL OVERLAP — {exact:,}/{len(ll_ids):,} exact matches"
            if exact > 0 else
            "NO OVERLAP — no sample reached cosine ≥ 0.9999"
        ),
        "sample_detail": [
            {"ll_id": ll_ids[i], "alt_id": best_alt_id[i],
             "sim": float(best_sim[i]),
             "ll_age":  (float(ll_ages[i])  if not np.isnan(ll_ages[i])  else None),
             "alt_age": (float(best_alt_age[i]) if not np.isnan(best_alt_age[i]) else None)}
            for i in range(min(20, len(ll_ids)))
        ],
    }


# ─────────────────────────────────────────────────────────────────────────────
# Report
# ─────────────────────────────────────────────────────────────────────────────
def build_txt(cpg, id_results, fp_results):
    lines = ["="*70,
             "DATASET FINGERPRINT COMPARISON",
             "MethylLlama (19k)  vs  AltumAge 21k h5ad",
             "="*70, ""]

    lines += ["CpG OVERLAP", "-"*40,
              f"  MethylLlama : {cpg['ll_n']:,}",
              f"  AltumAge 21k: {cpg['alt_n']:,}",
              f"  Shared      : {cpg['shared_n']:,} ({cpg['pct_ll']:.1f}% of LL)",
              f"  LL ⊆ Alt    : {cpg['ll_subset_alt']}", ""]

    for sp in ("valid", "test"):
        r_id = id_results.get(sp, {})
        r_fp = fp_results.get(sp, {})
        lines += [f"{sp.upper()} SPLIT", "-"*40,
                  f"  LL samples              : {r_id.get('ll_n','?'):,}",
                  f"  Direct ID match (all Alt): {r_id.get('shared_n_all','?'):,} / {r_id.get('ll_n','?'):,}"
                  f"  ({r_id.get('pct_ll',0):.1f}%)",
                  f"  Of matched → same split : {r_id.get('same_split_n',0):,}"
                  f"  ({r_id.get('same_split_pct',0):.1f}% of LL)",
                  f"  Alt split distribution  : {r_id.get('matched_alt_splits',{})}",
                  f"  Fingerprint exact match : {r_fp.get('exact_n','?'):,} / {r_fp.get('ll_n','?'):,}"
                  f"  ({r_fp.get('exact_pct',0):.1f}%)",
                  f"    same split in Alt     : {r_fp.get('same_split_n',0):,}",
                  f"    different split in Alt: {r_fp.get('diff_split_n',0):,}",
                  f"  Sim min/median/mean/max : "
                  f"{r_fp.get('sim_min',0):.6f} / {r_fp.get('sim_median',0):.6f} / "
                  f"{r_fp.get('sim_mean',0):.6f} / {r_fp.get('sim_max',0):.6f}",
                  f"  VERDICT: {r_fp.get('verdict','?')}", ""]

    lines += ["="*70]
    return "\n".join(lines)


def build_html(cpg, id_results, fp_results):
    def sc(pct):
        return "sc-green" if pct >= 99 else ("sc-amber" if pct >= 50 else "sc-red")

    def callout(pct):
        return "callout-ok" if pct >= 99 else ("callout-warn" if pct >= 50 else "callout-bad")

    def hist_bars(counts, labels, total):
        colors = ["b-red","b-red","b-amber","b-amber","b-green"]
        mx = max(counts) or 1
        html = ""
        for c, lbl, col in zip(counts, labels, colors):
            w = 100*c/mx
            pct = 100*c/max(total,1)
            html += (f'<div class="bar-row">'
                     f'<span class="bar-label">{lbl}</span>'
                     f'<div class="bar-track"><div class="bar-fill {col}" style="width:{w:.1f}%"></div></div>'
                     f'<span class="bar-val">{c:,} ({pct:.1f}%)</span></div>')
        return html

    css = """
* { box-sizing:border-box; margin:0; padding:0; }
body { font-family:'Segoe UI',Arial,sans-serif; background:#eef0f4; color:#1e2535; font-size:14px; }
.slide { width:1280px; min-height:720px; margin:40px auto; background:#fff; border-radius:16px;
         padding:46px 56px; box-shadow:0 4px 24px rgba(0,0,0,.10); border:1px solid #dde1ea; }
.slide-title { font-size:24px; font-weight:700; margin-bottom:28px; color:#1a2340;
               border-bottom:2px solid #dde3f0; padding-bottom:12px; }
.grid2 { display:grid; grid-template-columns:1fr 1fr; gap:20px; }
.grid3 { display:grid; grid-template-columns:1fr 1fr 1fr; gap:16px; }
.panel { background:#f7f9fc; border:1px solid #dde3ef; border-radius:10px; padding:18px 20px; }
.panel-title { font-size:12px; font-weight:700; text-transform:uppercase; letter-spacing:.8px;
               color:#5a6888; margin-bottom:14px; }
.stat-card { border-radius:10px; padding:18px 14px; text-align:center; margin-bottom:0; }
.s-val { font-size:30px; font-weight:800; line-height:1; margin-bottom:6px; }
.s-label { font-size:11px; text-transform:uppercase; letter-spacing:.8px; font-weight:600; }
.s-sub { font-size:11px; margin-top:4px; opacity:.75; }
.sc-green  { background:#edf7ed; border:1.5px solid #4a9a4a; color:#1a5a1a; }
.sc-blue   { background:#eef3ff; border:1.5px solid #7a9fe0; color:#1a3a80; }
.sc-purple { background:#f4eeff; border:1.5px solid #8a60c8; color:#3a1a70; }
.sc-amber  { background:#fff8ee; border:1.5px solid #cc9040; color:#6a3a00; }
.sc-red    { background:#fff0f0; border:1.5px solid #cc4040; color:#6a0a0a; }
.callout-ok   { background:#edf7ed; border-left:4px solid #4a9a4a; color:#1a4a1a;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
.callout-warn { background:#fff8ee; border-left:4px solid #cc9040; color:#5a3a00;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
.callout-bad  { background:#fff0f0; border-left:4px solid #cc4040; color:#5a0a0a;
                border-radius:8px; padding:12px 16px; font-size:13px; margin-top:14px; }
table { width:100%; border-collapse:collapse; font-size:13px; }
th { background:#f0f3f8; color:#4a5a78; font-size:11px; text-transform:uppercase;
     letter-spacing:.6px; padding:7px 10px; font-weight:700; }
td { padding:6px 10px; border-bottom:1px solid #edf0f6; }
tr:last-child td { border-bottom:none; }
.td-r { text-align:right; } .td-c { text-align:center; }
.mono { font-family:'Courier New',monospace; font-size:12px; }
.bar-row { display:flex; align-items:center; gap:8px; margin-bottom:6px; }
.bar-label { font-size:11px; color:#4a5a78; width:100px; text-align:right; flex-shrink:0; font-family:monospace; }
.bar-track { flex:1; height:18px; background:#eef1f8; border-radius:4px; overflow:hidden; }
.bar-fill  { height:100%; border-radius:4px; }
.bar-val   { font-size:11px; color:#6a7890; width:90px; flex-shrink:0; font-family:monospace; }
.b-green { background:#4a9a60; } .b-blue { background:#5a8de0; }
.b-amber { background:#cc8830; } .b-red  { background:#cc5040; }
"""

    # Slide 1: CpG overlap
    s1 = f"""
<div class="slide">
  <div class="slide-title">CpG Site Overlap</div>
  <div class="grid3" style="margin-bottom:22px">
    <div class="stat-card sc-blue"><div class="s-val">{cpg['ll_n']:,}</div>
      <div class="s-label">MethylLlama CpGs</div></div>
    <div class="stat-card sc-purple"><div class="s-val">{cpg['alt_n']:,}</div>
      <div class="s-label">AltumAge 21k CpGs</div></div>
    <div class="stat-card sc-green"><div class="s-val">{cpg['shared_n']:,}</div>
      <div class="s-label">Shared (fingerprint basis)</div>
      <div class="s-sub">{cpg['pct_ll']:.1f}% of LL · {cpg['pct_alt']:.1f}% of Alt</div></div>
  </div>
  <div class="panel">
    <p>LL ⊆ AltumAge 21k: <strong>{'✓ Yes' if cpg['ll_subset_alt'] else '✗ No'}</strong>
       &nbsp;·&nbsp; Only in LL: <strong>{cpg['only_ll']:,}</strong>
       &nbsp;·&nbsp; Only in Alt: <strong>{cpg['only_alt']:,}</strong></p>
    <p style="margin-top:8px">All {cpg['shared_n']:,} shared CpGs were used as the methylation
       fingerprint for sample identity verification.</p>
  </div>
</div>"""

    # Slides per split
    split_slides = ""
    for sp in ("valid", "test"):
        r_id = id_results.get(sp, {})
        r_fp = fp_results.get(sp, {})
        ep   = r_fp.get("exact_pct", 0)
        rows = ""
        for s in r_fp.get("sample_detail", []):
            sim_c = "#1a7a1a" if s["sim"] >= r_fp.get("threshold",0.9999) else (
                    "#cc8830" if s["sim"] >= 0.999 else "#cc5040")
            icon  = "✓" if s["sim"] >= r_fp.get("threshold",0.9999) else ("~" if s["sim"] >= 0.999 else "✗")
            rows += (f'<tr><td class="mono">{s["ll_id"][:35]}</td>'
                     f'<td class="mono">{s["alt_id"][:35]}</td>'
                     f'<td class="td-r" style="color:{sim_c};font-weight:700">{s["sim"]:.6f}</td>'
                     f'<td class="td-c">{icon}</td>'
                     f'<td class="td-r">{s["ll_age"] or "—"}</td>'
                     f'<td class="td-r">{s["alt_age"] or "—"}</td></tr>')
        split_slides += f"""
<div class="slide">
  <div class="slide-title">{sp.capitalize()} Split — Sample Identity Check</div>
  <div class="grid2" style="margin-bottom:20px">
    <div>
      <div class="grid2" style="gap:12px;margin-bottom:14px">
        <div class="stat-card sc-blue"><div class="s-val">{r_fp.get('ll_n',0):,}</div>
          <div class="s-label">MethylLlama samples</div></div>
        <div class="stat-card sc-purple"><div class="s-val">{r_fp.get('alt_n',0):,}</div>
          <div class="s-label">AltumAge samples</div></div>
      </div>
      <div class="grid2" style="gap:12px;margin-bottom:14px">
        <div class="stat-card sc-{'green' if r_id.get('pct_ll',0)>=99 else 'amber' if r_id.get('pct_ll',0)>0 else 'red'}">
          <div class="s-val">{r_id.get('shared_n',0):,}</div>
          <div class="s-label">Direct ID match</div>
          <div class="s-sub">{r_id.get('pct_ll',0):.1f}% of LL</div></div>
        <div class="stat-card {sc(ep)}">
          <div class="s-val">{r_fp.get('exact_n',0):,}</div>
          <div class="s-label">Fingerprint exact</div>
          <div class="s-sub">cosine ≥ {r_fp.get('threshold',0.9999)} · {ep:.1f}%</div></div>
      </div>
      <div class="{callout(ep)}"><strong>Verdict:</strong> {r_fp.get('verdict','?')}</div>
    </div>
    <div class="panel">
      <div class="panel-title">Similarity distribution</div>
      {hist_bars(r_fp.get('hist_counts',[0]*5), r_fp.get('hist_labels',['']*5), r_fp.get('ll_n',1))}
      <p style="margin-top:10px;font-size:12px;color:#5a6888">
        min={r_fp.get('sim_min',0):.6f} · median={r_fp.get('sim_median',0):.6f}
        · mean={r_fp.get('sim_mean',0):.6f} · max={r_fp.get('sim_max',0):.6f}</p>
    </div>
  </div>
  <div class="panel">
    <div class="panel-title">First 20 LL samples — nearest AltumAge neighbour</div>
    <table>
      <tr><th>MethylLlama ID</th><th>Best AltumAge match</th>
          <th class="td-r">Cosine sim</th><th class="td-c">Match?</th>
          <th class="td-r">LL age</th><th class="td-r">Alt age</th></tr>
      {rows}
    </table>
  </div>
</div>"""

    # Summary slide
    pv = fp_results.get("valid", {}).get("exact_pct", 0)
    pt = fp_results.get("test",  {}).get("exact_pct", 0)
    s_summary = f"""
<div class="slide">
  <div class="slide-title">Summary — Are the Evaluation Sets the Same?</div>
  <div class="grid2" style="margin-bottom:20px">
    <div class="panel"><div class="panel-title">Validation</div>
      <p style="font-size:22px;font-weight:800">{pv:.1f}% exact match</p>
      <p style="margin-top:8px">{fp_results.get('valid',{}).get('verdict','?')}</p></div>
    <div class="panel"><div class="panel-title">Test</div>
      <p style="font-size:22px;font-weight:800">{pt:.1f}% exact match</p>
      <p style="margin-top:8px">{fp_results.get('test',{}).get('verdict','?')}</p></div>
  </div>
  <div class="{callout(min(pv,pt))}">
    <strong>Comparability of MedAE scores:</strong><br>
    {'Both evaluation sets contain the same samples → MedAE scores are directly comparable.'
     if min(pv,pt) >= 99 else
     'Evaluation sets differ → MethylLlama and MethylGPT MedAE scores are NOT on the same samples. Direct comparison must be qualified.'}
  </div>
  <div class="panel" style="margin-top:20px">
    <div class="panel-title">Method</div>
    <p>For each MethylLlama sample, cosine similarity was computed against every AltumAge
       sample in the same split over all <strong>{cpg['shared_n']:,} shared CpG sites</strong>.
       The nearest neighbour was taken as the candidate match.</p>
    <p style="margin-top:6px">Cosine ≥ 0.9999 = identical biological sample.
       Direct ID matching was also attempted using the sample index names.</p>
  </div>
</div>"""

    return (f"<!DOCTYPE html><html><head><meta charset='UTF-8'>"
            f"<title>Dataset Fingerprint Comparison</title>"
            f"<style>{css}</style></head><body>"
            f"{s1}{split_slides}{s_summary}</body></html>")


# ─────────────────────────────────────────────────────────────────────────────
# Main
# ─────────────────────────────────────────────────────────────────────────────
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--llama_h5ad",  default=LLAMA_H5AD)
    ap.add_argument("--alt_h5ad",    default=ALT_H5AD)
    ap.add_argument("--outdir",      default="dataset_fingerprint_outputs")
    ap.add_argument("--threshold",   type=float, default=0.9999)
    ap.add_argument("--splits",      default="valid,test")
    args = ap.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)
    splits = [s.strip() for s in args.splits.split(",")]

    # Load both datasets
    ll_X,  ll_ids,  ll_ages,  ll_cpgs  = load_h5ad(args.llama_h5ad, "MethylLlama")
    alt_X, alt_ids, alt_ages, alt_cpgs = load_h5ad(args.alt_h5ad,   "AltumAge21k")

    # CpG overlap + shared column indices
    cpg = analyze_cpg_overlap(ll_cpgs, alt_cpgs)
    shared = sorted(cpg["shared_ids"])
    ll_pos  = {c: i for i, c in enumerate(ll_cpgs)}
    alt_pos = {c: i for i, c in enumerate(alt_cpgs)}
    ll_cols  = np.array([ll_pos[c]  for c in shared], dtype=np.int32)
    alt_cols = np.array([alt_pos[c] for c in shared], dtype=np.int32)
    print(f"\n  Fingerprinting on {len(shared):,} shared CpGs")

    # ── Build full Alt arrays (ALL splits combined) for cross-split search ──
    # The 19k was derived from the 21k by filtering, so splits were assigned
    # independently. A 19k test sample might be a 21k train sample.
    # Correct check: match each 19k sample against ALL 21k samples, then
    # report which 21k split the match falls in.
    all_alt_X   = np.concatenate([alt_X[s]   for s in ("train","valid","test")], axis=0)
    all_alt_ids = sum([alt_ids[s]             for s in ("train","valid","test")], [])
    all_alt_ages= np.concatenate([alt_ages[s] for s in ("train","valid","test")], axis=0)
    # Build a lookup: alt_id -> split label
    alt_id_to_split = {}
    for s in ("train","valid","test"):
        for aid in alt_ids[s]:
            alt_id_to_split[aid] = s

    all_alt_X_shared = all_alt_X[:, alt_cols]
    print(f"  Alt full dataset: {len(all_alt_ids):,} samples")

    id_results, fp_results = {}, {}
    for sp in splits:
        print(f"\n{'─'*50}\n  Split: {sp}\n{'─'*50}")

        ll_X_sp   = ll_X[sp][:, ll_cols]
        ll_ids_sp = ll_ids[sp]
        ll_ages_sp= ll_ages[sp]

        # Direct ID match against ALL alt samples
        print("\n  --- Direct ID matching (vs ALL AltumAge samples) ---")
        ll_set  = set(ll_ids_sp)
        alt_all_set = set(all_alt_ids)
        shared_ids_all = ll_set & alt_all_set
        # For matched samples, check which alt split they landed in
        split_dest = {}
        for sid in shared_ids_all:
            dest = alt_id_to_split.get(sid, "unknown")
            split_dest[dest] = split_dest.get(dest, 0) + 1
        print(f"  LL {sp} vs ALL Alt: {len(shared_ids_all):,}/{len(ll_set):,} "
              f"({100*len(shared_ids_all)/max(len(ll_set),1):.1f}%) matched")
        print(f"  Of those, their Alt split: {split_dest}")
        id_results[sp] = {
            "ll_n":              len(ll_set),
            "alt_total_n":       len(all_alt_ids),
            "shared_n_all":      len(shared_ids_all),
            "pct_ll":            100*len(shared_ids_all)/max(len(ll_set),1),
            "matched_alt_splits":split_dest,
            "same_split_n":      split_dest.get(sp, 0),
            "same_split_pct":    100*split_dest.get(sp,0)/max(len(ll_set),1),
        }

        # Fingerprint against ALL alt samples
        print("\n  --- Methylation fingerprint (vs ALL AltumAge samples) ---")
        best_sim, best_idx = cosine_nn(ll_X_sp, all_alt_X_shared)
        best_alt_ids  = [all_alt_ids[i]  for i in best_idx]
        best_alt_ages = all_alt_ages[best_idx]
        best_alt_splits = [alt_id_to_split.get(aid, "unknown") for aid in best_alt_ids]

        exact   = int((best_sim >= args.threshold).sum())
        near    = int(((best_sim >= 0.999) & (best_sim < args.threshold)).sum())
        partial = int(((best_sim >= 0.99)  & (best_sim < 0.999)).sum())
        poor    = int((best_sim < 0.99).sum())

        # Among exact matches, how many land in the same split vs different?
        exact_mask = best_sim >= args.threshold
        same_split_fp = int(sum(1 for i,m in enumerate(exact_mask)
                                if m and best_alt_splits[i] == sp))
        diff_split_fp = exact - same_split_fp

        print(f"  sim: min={best_sim.min():.6f} mean={best_sim.mean():.6f} max={best_sim.max():.6f}")
        print(f"  exact(>={args.threshold}): {exact:,}/{len(ll_ids_sp):,} ({100*exact/max(len(ll_ids_sp),1):.1f}%)")
        print(f"    → same split '{sp}' in Alt: {same_split_fp:,}  different split: {diff_split_fp:,}")
        print(f"  near: {near:,}  partial: {partial:,}  poor: {poor:,}")

        bins   = [0.0, 0.9, 0.99, 0.999, 0.9999, 1.0001]
        labels = ["<0.9", "0.9–0.99", "0.99–0.999", "0.999–0.9999", "≥0.9999"]
        counts, _ = np.histogram(best_sim, bins=bins)

        fp_results[sp] = {
            "split":          sp,
            "ll_n":           len(ll_ids_sp),
            "alt_n":          len(all_alt_ids),
            "sim_min":        float(best_sim.min()),
            "sim_mean":       float(best_sim.mean()),
            "sim_max":        float(best_sim.max()),
            "sim_median":     float(np.median(best_sim)),
            "exact_n":        exact,
            "exact_pct":      float(100*exact/max(len(ll_ids_sp),1)),
            "same_split_n":   same_split_fp,
            "diff_split_n":   diff_split_fp,
            "near_n":         near,
            "partial_n":      partial,
            "poor_n":         poor,
            "threshold":      args.threshold,
            "hist_counts":    counts.tolist(),
            "hist_labels":    labels,
            "verdict": (
                f"ALL {exact:,} matched samples found in Alt — "
                f"{same_split_fp:,} in same split, {diff_split_fp:,} in different split"
                if exact == len(ll_ids_sp) else
                f"{exact:,}/{len(ll_ids_sp):,} exact matches — "
                f"{same_split_fp:,} same split, {diff_split_fp:,} different split"
            ),
            "sample_detail": [
                {"ll_id": ll_ids_sp[i], "alt_id": best_alt_ids[i],
                 "alt_split": best_alt_splits[i],
                 "sim": float(best_sim[i]),
                 "ll_age":  (float(ll_ages_sp[i])   if not np.isnan(ll_ages_sp[i])   else None),
                 "alt_age": (float(best_alt_ages[i]) if not np.isnan(best_alt_ages[i]) else None)}
                for i in range(min(20, len(ll_ids_sp)))
            ],
        }

    # Write outputs
    txt  = build_txt(cpg, id_results, fp_results)
    html = build_html(cpg, id_results, fp_results)
    (outdir / "fingerprint_summary.txt").write_text(txt)
    (outdir / "fingerprint_report.html").write_text(html)

    print(f"\n{'='*60}")
    print(f"Outputs → {outdir}/")
    print(f"{'='*60}")
    print(txt)


if __name__ == "__main__":
    main()
