"""
view_consistency_heldout_full.py
==================================
FINAL cross-view consistency evaluation for Fig. 2a/b: the ENTIRE held-out
partition of the pretraining corpus (16,912 profiles), not a subsample.

Why the full partition
  Top-1 retrieval depends on the number of candidates (chance = 1/N). Evaluating
  every held-out profile removes the arbitrary choice of a 2,000-profile subsample
  and makes the test as hard as the data allow (chance 0.006%).

What is reported, per (condition, seed)
  pool "all"            every held-out profile is a candidate
  pool "fully_measured" only profiles with >= 99% of the 49,156 CpGs measured.
                        All candidates then offer the same CpGs, so a profile cannot
                        be recognised by WHICH CpGs it has -- only by their values.
  pool "n2000"          a fixed random 2,000-profile subset, for continuity with the
                        first held-out run (job 46246524) and the AltuMAge analysis.
  For each pool: mean matched cosine, mean unmatched cosine, retrieval@1/5/10.
  The embeddings of both views are saved for every run BEFORE any statistic is computed,
  so any other pool or threshold can be evaluated later without encoding again.

Conditions
  overlap       two independent 50% views (the pretraining construction)
  disjoint      the two views partition the measured CpGs and share none
  pattern_only  NEGATIVE CONTROL. Each profile keeps its own set of measured CpGs, but every
                methylation value is replaced by the per-CpG mean of the evaluated profiles.
                The views then carry WHICH CpGs a profile has and nothing about their values.
                If retrieval stays at chance here, sample identity comes from the values.

One (condition, seed) per invocation, so the runs can be a SLURM array.
`--merge` combines the per-run files into one summary; it imports neither torch
nor the model and can run on the login node.

INFERENCE ONLY. Same checkpoint and the same encode_views() as
view_design_eval.py. Writes only under --outdir.
"""
import argparse
import json
from pathlib import Path

import numpy as np

TOPK = (1, 5, 10)
BINS = np.linspace(-1.0, 1.0, 201)     # the full cosine range, so no pair is ever dropped from the histogram


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--merge", action="store_true", help="combine per-run files into summary.json and exit")
    p.add_argument("--checkpoint")
    p.add_argument("--data", help="the PRETRAINING h5ad")
    p.add_argument("--split", default="test")
    p.add_argument("--tokenizer")
    p.add_argument("--genomic_rank", help="cpg_genomic_rank.npy (49,156 entries)")
    p.add_argument("--condition", choices=["overlap", "disjoint", "pattern_only"])
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--n_samples", type=int, default=0, help="0 = the whole held-out partition")
    p.add_argument("--measured_min", type=float, default=0.99)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--input_ratio", type=float, default=0.5)
    p.add_argument("--outdir", default="results/pretrained_cls/view_consistency_heldout_full")
    p.add_argument("--device", default=None)
    return p.parse_args()


def pool_stats(e1, e2, idx):
    """Consistency statistics with the candidate pool restricted to rows `idx`."""
    if len(idx) < 2:                                # nothing to retrieve from; recorded, not computed
        return {"n_candidates": int(len(idx))}, None, None, None
    a, b = e1[idx], e2[idx]
    sim = a @ b.T                                   # rows: view 1, columns: view 2 candidates
    n = sim.shape[0]
    pos = np.diag(sim).copy()
    rank = (sim > pos[:, None]).sum(axis=1)         # 0 = the partner is the nearest candidate
    off_sum = sim.sum() - pos.sum()
    hist = np.histogram(sim[~np.eye(n, dtype=bool)], bins=BINS)[0]
    out = {"n_candidates": int(n), "chance_top1": 1.0 / n,
           "matched_cos": float(pos.mean()), "unmatched_cos": float(off_sum / (n * n - n)),
           **{f"retrieval_at{k}": float((rank < k).mean()) for k in TOPK}}
    return out, pos.astype(np.float32), rank.astype(np.int32), hist


def encode(a):
    import torch
    import sys
    sys.path.insert(0, str(Path(__file__).parent))
    from view_design_eval import encode_views                      # identical encoder call
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    if a.split == "train":
        raise SystemExit("refusing to run on split=train")
    a.device = a.device or ("cuda" if torch.cuda.is_available() else "cpu")
    out = Path(a.outdir) / "per_run"; out.mkdir(parents=True, exist_ok=True)
    tag = f"{a.condition}_seed{a.seed}"
    if (out / f"{tag}.json").exists():
        raise SystemExit(f"{out / (tag + '.json')} exists -- nothing is overwritten")

    print(f"[1/4] checkpoint: {a.checkpoint}", flush=True)
    encoder = load_wced_llama_checkpoint(a.checkpoint).encoder.to(a.device).eval()

    print(f"[2/4] held-out partition of {a.data}", flush=True)
    ds = MethylationDataset(h5ad_path=a.data, split=a.split, normalize_age=False)
    n_split = len(ds)
    rows = np.arange(n_split) if a.n_samples in (0, n_split) else np.sort(
        np.random.default_rng(0).choice(n_split, size=a.n_samples, replace=False))
    X = ds.adata.X if len(rows) == n_split else ds.adata.X[rows]
    betas = np.array(X.toarray() if hasattr(X, "toarray") else X, dtype=np.float32)   # our own copy
    valid = np.isfinite(betas)
    betas[~valid] = 0.0
    sample_ids = np.asarray(ds.adata.obs_names)[rows].astype(str)
    n_cpg = betas.shape[1]
    frac = valid.sum(1) / n_cpg
    print(f"      partition rows={n_split:,} evaluated={len(rows):,} cpgs={n_cpg:,} | measured fraction: "
          f"median {np.median(frac):.3f}, >= {a.measured_min}: {(frac >= a.measured_min).sum():,} profiles", flush=True)

    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    ct = tok.tokenizers["cpg_sites"]; vocab = ct.get_vocab()
    ids = np.array([vocab.get(c, ct.unk_token_id) for c in ds.cpg_sites], dtype=np.int64)
    assert (ids != ct.unk_token_id).all(), "pretraining CpGs map to [UNK]: wrong tokenizer"
    rank_g = np.load(a.genomic_rank)
    assert len(rank_g) == n_cpg, f"genomic rank has {len(rank_g)} entries, panel has {n_cpg}"

    if a.condition == "pattern_only":
        mu = (betas.sum(0, dtype=np.float64) / np.maximum(valid.sum(0), 1)).astype(np.float32)   # per-CpG mean
        np.copyto(betas, np.broadcast_to(mu, betas.shape), where=valid)    # same measured set, no individual values
        print("      pattern_only: methylation values replaced by per-CpG means", flush=True)

    print(f"[3/4] encoding {a.condition}, seed {a.seed}", flush=True)
    e1, e2 = encode_views(encoder, betas, valid, ids, rank_g, ct.cls_token_id, ct.pad_token_id,
                          -2.0, -3.0, a.condition == "disjoint", a.seed, a)
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-9)

    n = len(rows)
    assert e1.shape == e2.shape == (n, e1.shape[1]) and np.isfinite(e1).all() and np.isfinite(e2).all()
    np.savez_compressed(out / f"{tag}_embeddings.npz", e1=e1.astype(np.float32), e2=e2.astype(np.float32),
                        rows=rows, sample_ids=sample_ids, measured_fraction=frac.astype(np.float32))
    print(f"      embeddings saved -> {out}/{tag}_embeddings.npz", flush=True)

    print("[4/4] statistics", flush=True)
    pools = {"all": np.arange(n),
             "fully_measured": np.where(frac >= a.measured_min)[0],
             "n2000": np.sort(np.random.default_rng(0).choice(n, size=min(2000, n), replace=False))}
    res = {"condition": a.condition, "seed": a.seed, "checkpoint": a.checkpoint, "data": a.data,
           "split": a.split, "partition_rows": int(n_split), "n_profiles": int(n), "n_cpgs_panel": int(n_cpg),
           "input_ratio": a.input_ratio, "measured_min": a.measured_min, "pools": {}}
    keep = {}
    for name, idx in pools.items():
        s, pos, rk, hist = pool_stats(e1, e2, idx)
        res["pools"][name] = s
        if pos is None:
            print(f"      {name:15s} N={s['n_candidates']:>6,}  fewer than 2 candidates, no statistics", flush=True)
            continue
        keep[f"{name}_matched_cos"], keep[f"{name}_rank"], keep[f"{name}_unmatched_hist"] = pos, rk, hist
        print(f"      {name:15s} N={s['n_candidates']:>6,}  matched {s['matched_cos']:.4f}  unmatched "
              f"{s['unmatched_cos']:.4f}  top-1 {100 * s['retrieval_at1']:.2f}%  top-10 "
              f"{100 * s['retrieval_at10']:.2f}%  (chance {100 * s['chance_top1']:.4f}%)", flush=True)
    np.savez_compressed(out / f"{tag}.npz", rows=rows, measured_fraction=frac.astype(np.float32),
                        hist_bins=BINS, **keep)
    (out / f"{tag}.json").write_text(json.dumps(res, indent=2))
    print(f"Saved -> {out}/{tag}.json, .npz, _embeddings.npz", flush=True)


def merge(a):
    out = Path(a.outdir)
    runs = [json.loads(f.read_text()) for f in sorted((out / "per_run").glob("*.json"))]
    if not runs:
        raise SystemExit(f"no per-run files in {out / 'per_run'}")
    metrics = ["matched_cos", "unmatched_cos"] + [f"retrieval_at{k}" for k in TOPK]
    summary = {"population": "held-out partition of the pretraining corpus (never used for optimisation)",
               "checkpoint": runs[0]["checkpoint"], "n_profiles": runs[0]["n_profiles"],
               "n_cpgs_panel": runs[0]["n_cpgs_panel"], "conditions": {}}
    for cond in ("overlap", "disjoint", "pattern_only"):
        rc = [r for r in runs if r["condition"] == cond]
        if not rc:
            continue
        summary["conditions"][cond] = {"seeds": sorted(r["seed"] for r in rc), "pools": {}}
        for pool in rc[0]["pools"]:
            if "matched_cos" not in rc[0]["pools"][pool]:
                continue
            v = {m: np.array([r["pools"][pool][m] for r in rc]) for m in metrics}
            summary["conditions"][cond]["pools"][pool] = {
                "n_candidates": rc[0]["pools"][pool]["n_candidates"],
                "chance_top1": rc[0]["pools"][pool]["chance_top1"],
                **{m: {"mean": float(x.mean()), "sd": float(x.std(ddof=1)) if len(x) > 1 else 0.0,
                       "min": float(x.min()), "max": float(x.max())} for m, x in v.items()}}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"{'condition':<14}{'pool':<16}{'N':>7}  matched  unmatched   top-1 (min-max)      top-10")
    for cond, c in summary["conditions"].items():
        for pool, s in c["pools"].items():
            t = s["retrieval_at1"]
            print(f"{cond:<14}{pool:<16}{s['n_candidates']:>7,}  {s['matched_cos']['mean']:.4f}   "
                  f"{s['unmatched_cos']['mean']:.4f}    {100 * t['mean']:.2f}% ({100 * t['min']:.2f}-{100 * t['max']:.2f})   "
                  f"{100 * s['retrieval_at10']['mean']:.2f}%")
    print(f"Saved -> {out}/summary.json  ({len(runs)} runs)")


if __name__ == "__main__":
    args = parse_args()
    merge(args) if args.merge else encode(args)
