"""
view_design_eval.py
=====================
Evaluate how the *construction of the two views* affects cross-view consistency
of the pretrained encoder. Motivated by scConcept (Bahrami et al. 2025), whose
views are disjoint gene panels, whereas our pretraining draws two 50% CpG
subsets independently (so they overlap by ~50% in expectation). Overlap makes
view alignment an easier task, which could inflate our consistency numbers.

This is an INFERENCE-ONLY probe of the existing checkpoint -- it does not
retrain anything. It answers two questions the manuscript currently cannot:

  1. Disjoint-view evaluation. If the two views are forced to be disjoint
     (no shared CpG), does consistency hold up? A large drop would mean our
     reported alignment partly reflects shared input rather than a shared
     sample representation.
  2. Seed stability. Consistency and retrieval are currently reported from a
     single random view draw. Repeating over several seeds gives a mean and
     spread, so the manuscript can report a range instead of one number.

Conditions (same profiles, same checkpoint, same encoder call):
  overlap   -- two independent 50% draws (the pretraining-time construction)
  disjoint  -- a random 50/50 partition: view 1 and view 2 share no CpG
  (each run for --n_seeds different random draws)

Outputs (--outdir):
  view_design_summary.json   per-condition mean/SD over seeds
  view_design_per_seed.csv   one row per (condition, seed)
  simmatrix_<cond>_seed0.npy similarity matrix of the first seed, for plotting

Usage (cluster, GPU):
  python scripts/evaluation/view_design_eval.py \
      --checkpoint checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt --data <21k h5ad> \
      --tokenizer tokenizer \
      --genomic_rank resources/cpg_genomic_rank_finetune.npy \
      --n_samples 2000 --n_seeds 5 \
      --outdir results/pretrained_cls/view_design
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True)
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--input_ratio", type=float, default=0.5)
    p.add_argument("--outdir", default="results/pretrained_cls/view_design")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def build_views(valid_idx, n_input, disjoint, rng):
    """Return (idx_v1, idx_v2) as arrays of column indices."""
    if disjoint:
        perm = rng.permutation(valid_idx)
        # a 50/50 partition: the two views share no CpG by construction
        return np.sort(perm[:n_input]), np.sort(perm[n_input:2 * n_input])
    return (np.sort(rng.choice(valid_idx, size=n_input, replace=False)),
            np.sort(rng.choice(valid_idx, size=n_input, replace=False)))


def encode_views(encoder, betas_mat, valid_mat, cpg_vocab_ids, genomic_rank,
                 cls_id, pad_id, cls_beta, pad_beta, disjoint, seed, a):
    """Encode both views for every profile; returns two [N, D] float arrays."""
    rng = np.random.default_rng(seed)
    n_prof = betas_mat.shape[0]
    embs = [[], []]
    for start in range(0, n_prof, a.batch_size):
        stop = min(start + a.batch_size, n_prof)
        batch_idx = range(start, stop)
        views = [[], []]
        for i in batch_idx:
            valid_idx = np.where(valid_mat[i])[0]
            n_input = int(len(valid_idx) * a.input_ratio)
            if disjoint:
                n_input = min(n_input, len(valid_idx) // 2)
            views[0].append(None); views[1].append(None)
            v1, v2 = build_views(valid_idx, n_input, disjoint, rng)
            views[0][-1], views[1][-1] = v1, v2

        for v in (0, 1):
            L = max(len(x) for x in views[v]) + 1
            B = stop - start
            ids = torch.full((B, L), pad_id, dtype=torch.long)
            bet = torch.full((B, L), pad_beta, dtype=torch.float32)
            att = torch.zeros(B, L, dtype=torch.long)
            pos = torch.zeros(B, L, dtype=torch.long)
            for b, i in enumerate(batch_idx):
                sel = views[v][b]
                sel = sel[np.argsort(genomic_rank[sel])]      # genomic order
                k = len(sel)
                ids[b, 0] = cls_id; bet[b, 0] = cls_beta
                ids[b, 1:k + 1] = torch.from_numpy(cpg_vocab_ids[sel].astype(np.int64))
                bet[b, 1:k + 1] = torch.from_numpy(betas_mat[i, sel].astype(np.float32))
                att[b, :k + 1] = 1
                pos[b, 1:k + 1] = torch.from_numpy((genomic_rank[sel] + 1).astype(np.int64))
            with torch.no_grad():
                inp = torch.stack([ids.float(), bet], dim=1).to(a.device)
                out = encoder(input_ids=inp, attention_mask=att.to(a.device),
                              position_ids=pos.to(a.device))
            embs[v].append(out.pooler_output.cpu().float().numpy())
    return np.concatenate(embs[0]), np.concatenate(embs[1])


def consistency_stats(e1, e2):
    e1 = e1 / (np.linalg.norm(e1, axis=1, keepdims=True) + 1e-9)
    e2 = e2 / (np.linalg.norm(e2, axis=1, keepdims=True) + 1e-9)
    sim = e1 @ e2.T
    n = sim.shape[0]
    pos = np.diag(sim)
    off = sim[~np.eye(n, dtype=bool)]
    ranks = (sim > pos[:, None]).sum(axis=1)
    return {
        "pos_cos": float(pos.mean()),
        "neg_cos": float(off.mean()),
        "alignment_gap": float(pos.mean() - off.mean()),
        "retrieval_at1": float((ranks == 0).mean()),
        "retrieval_at5": float((ranks < 5).mean()),
        "retrieval_at10": float((ranks < 10).mean()),
    }, sim


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    print(f"[1/4] Loading checkpoint: {a.checkpoint}", flush=True)
    module = load_wced_llama_checkpoint(a.checkpoint)
    encoder = module.encoder.to(a.device).eval()

    print(f"[2/4] Loading data: {a.data}", flush=True)
    ds = MethylationDataset(h5ad_path=a.data, split=None, normalize_age=False)
    X = ds.adata.X
    X = X.toarray() if hasattr(X, "toarray") else np.asarray(X)
    n_total = X.shape[0]
    sel_prof = np.random.default_rng(0).choice(
        n_total, size=min(a.n_samples, n_total), replace=False)
    sel_prof.sort()
    betas_mat = X[sel_prof].astype(np.float32)
    valid_mat = np.isfinite(betas_mat)
    betas_mat = np.where(valid_mat, betas_mat, 0.0)
    print(f"      profiles={betas_mat.shape[0]} cpgs={betas_mat.shape[1]}", flush=True)

    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    cpg_tok = tok.tokenizers["cpg_sites"]
    vocab = cpg_tok.get_vocab()
    cpg_vocab_ids = np.array([vocab.get(c, cpg_tok.unk_token_id) for c in ds.cpg_sites],
                             dtype=np.int64)
    genomic_rank = np.load(a.genomic_rank)
    assert len(genomic_rank) == len(ds.cpg_sites), "genomic_rank length mismatch"

    print(f"[3/4] Encoding: 2 conditions x {a.n_seeds} seeds", flush=True)
    recs = []
    for cond, disjoint in [("overlap", False), ("disjoint", True)]:
        for seed in range(a.n_seeds):
            e1, e2 = encode_views(encoder, betas_mat, valid_mat, cpg_vocab_ids,
                                  genomic_rank, cpg_tok.cls_token_id,
                                  cpg_tok.pad_token_id, -2.0, -3.0,
                                  disjoint, seed, a)
            stats, sim = consistency_stats(e1, e2)
            stats.update(condition=cond, seed=seed)
            recs.append(stats)
            print(f"      {cond:9s} seed={seed}  pos={stats['pos_cos']:.4f} "
                  f"neg={stats['neg_cos']:.4f} top1={stats['retrieval_at1']:.3f}",
                  flush=True)
            if seed == 0:
                np.save(outdir / f"simmatrix_{cond}_seed0.npy", sim.astype(np.float32))

    print("[4/4] Writing outputs", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(outdir / "view_design_per_seed.csv", index=False)

    metrics = ["pos_cos", "neg_cos", "alignment_gap",
               "retrieval_at1", "retrieval_at5", "retrieval_at10"]
    summary = {"checkpoint": a.checkpoint, "data": a.data,
               "n_profiles": int(betas_mat.shape[0]), "n_seeds": a.n_seeds,
               "input_ratio": a.input_ratio, "conditions": {}}
    for cond in ["overlap", "disjoint"]:
        sub = df[df.condition == cond]
        summary["conditions"][cond] = {
            m: {"mean": float(sub[m].mean()), "sd": float(sub[m].std(ddof=1)),
                "min": float(sub[m].min()), "max": float(sub[m].max())}
            for m in metrics}
    o, d = summary["conditions"]["overlap"], summary["conditions"]["disjoint"]
    summary["disjoint_minus_overlap"] = {
        m: round(d[m]["mean"] - o[m]["mean"], 4) for m in metrics}

    with open(outdir / "view_design_summary.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary["conditions"], indent=2))
    print("\ndisjoint - overlap:", json.dumps(summary["disjoint_minus_overlap"], indent=2))
    print(f"\nSaved -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
