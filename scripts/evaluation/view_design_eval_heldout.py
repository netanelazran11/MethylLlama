"""
view_design_eval_heldout.py
============================
Cross-view consistency of the pretrained encoder on profiles that were HELD OUT
of pretraining.

Why this file exists. The numbers in Fig. 2a/b (matched 0.984, unmatched 0.499,
top-1 retrieval 73.8%, disjoint 25.3%) were computed by view_design_eval.py on
2,000 AltuMAge profiles, most of which are inside the pretraining corpus. The
manuscript calls them "held-out". This script repeats the identical analysis on
the pretraining corpus' own held-out partition, so that the word is true.

What is identical to view_design_eval.py: checkpoint, encoder call, view
construction (overlap / disjoint), seeds, statistics -- the functions are
imported from that file, not copied.
What differs: the profiles (held-out rows of the pretraining h5ad), the CpG
panel (49,156 instead of 21,368) and therefore the genomic-rank file
(cpg_genomic_rank.npy instead of cpg_genomic_rank_finetune.npy).

The held-out rows are obtained through MethylationDataset(split="test"), the
same code path the training DataModule and reconstruction_withheld_eval.py use
(auto-split, rng(42).permutation, last 10%).  INFERENCE ONLY; nothing is trained
or modified.

Outputs (--outdir):
  view_design_summary.json, view_design_per_seed.csv, simmatrix_<cond>_seed0.npy
  selected_rows.json   the pretraining row indices evaluated, for provenance
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).parent))
from view_design_eval import consistency_stats, encode_views  # noqa: E402


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True, help="the PRETRAINING h5ad")
    p.add_argument("--split", default="test", help="held-out partition of the pretraining h5ad")
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True, help="cpg_genomic_rank.npy (49,156 entries)")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--n_seeds", type=int, default=5)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--input_ratio", type=float, default=0.5)
    p.add_argument("--outdir", default="results/pretrained_cls/view_design_heldout")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def main():
    a = parse_args()
    if a.split == "train":
        raise SystemExit("refusing to run on split=train: this script exists to evaluate held-out profiles")
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)

    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    print(f"[1/4] Loading checkpoint: {a.checkpoint}", flush=True)
    encoder = load_wced_llama_checkpoint(a.checkpoint).encoder.to(a.device).eval()

    print(f"[2/4] Loading held-out partition: {a.data} (split={a.split})", flush=True)
    ds = MethylationDataset(h5ad_path=a.data, split=a.split, normalize_age=False)
    n_split = len(ds)
    X = ds.adata.X
    sel = np.sort(np.random.default_rng(0).choice(n_split, size=min(a.n_samples, n_split), replace=False))
    betas_mat = (X[sel].toarray() if hasattr(X, "toarray") else np.asarray(X[sel])).astype(np.float32)
    valid_mat = np.isfinite(betas_mat)
    betas_mat = np.where(valid_mat, betas_mat, 0.0)
    n_valid = valid_mat.sum(1)
    print(f"      partition rows={n_split:,}  evaluated={betas_mat.shape[0]:,}  cpgs={betas_mat.shape[1]:,}  "
          f"measured CpGs per profile: median {int(np.median(n_valid)):,} (min {int(n_valid.min()):,})", flush=True)

    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    cpg_tok = tok.tokenizers["cpg_sites"]
    vocab = cpg_tok.get_vocab()
    cpg_vocab_ids = np.array([vocab.get(c, cpg_tok.unk_token_id) for c in ds.cpg_sites], dtype=np.int64)
    n_unk = int((cpg_vocab_ids == cpg_tok.unk_token_id).sum())
    assert n_unk == 0, f"{n_unk} pretraining CpGs map to [UNK] -- wrong tokenizer for this panel"
    genomic_rank = np.load(a.genomic_rank)
    assert len(genomic_rank) == len(ds.cpg_sites), (
        f"genomic_rank has {len(genomic_rank)} entries but the panel has {len(ds.cpg_sites)} CpGs -- "
        "pass cpg_genomic_rank.npy, not the fine-tuning rank file")

    print(f"[3/4] Encoding: 2 conditions x {a.n_seeds} seeds", flush=True)
    recs = []
    for cond, disjoint in [("overlap", False), ("disjoint", True)]:
        for seed in range(a.n_seeds):
            e1, e2 = encode_views(encoder, betas_mat, valid_mat, cpg_vocab_ids, genomic_rank,
                                  cpg_tok.cls_token_id, cpg_tok.pad_token_id, -2.0, -3.0,
                                  disjoint, seed, a)
            stats, sim = consistency_stats(e1, e2)
            stats.update(condition=cond, seed=seed)
            recs.append(stats)
            print(f"      {cond:9s} seed={seed}  pos={stats['pos_cos']:.4f} neg={stats['neg_cos']:.4f} "
                  f"top1={stats['retrieval_at1']:.3f}", flush=True)
            if seed == 0:
                np.save(outdir / f"simmatrix_{cond}_seed0.npy", sim.astype(np.float32))

    print("[4/4] Writing outputs", flush=True)
    df = pd.DataFrame(recs)
    df.to_csv(outdir / "view_design_per_seed.csv", index=False)
    metrics = ["pos_cos", "neg_cos", "alignment_gap", "retrieval_at1", "retrieval_at5", "retrieval_at10"]
    summary = {"checkpoint": a.checkpoint, "data": a.data, "split": a.split,
               "population": "held-out partition of the pretraining corpus (never used for optimisation)",
               "partition_rows": int(n_split), "n_profiles": int(betas_mat.shape[0]),
               "n_cpgs_panel": int(betas_mat.shape[1]), "median_measured_cpgs": int(np.median(n_valid)),
               "n_seeds": a.n_seeds, "input_ratio": a.input_ratio, "conditions": {}}
    for cond in ["overlap", "disjoint"]:
        sub = df[df.condition == cond]
        summary["conditions"][cond] = {m: {"mean": float(sub[m].mean()), "sd": float(sub[m].std(ddof=1)),
                                           "min": float(sub[m].min()), "max": float(sub[m].max())} for m in metrics}
    (outdir / "view_design_summary.json").write_text(json.dumps(summary, indent=2))
    (outdir / "selected_rows.json").write_text(json.dumps(
        {"note": "indices are positions within the held-out partition as returned by "
                 "MethylationDataset(split=...), sampled with rng(0)", "rows": sel.tolist()}))
    print(json.dumps(summary["conditions"], indent=2))
    print("reference (AltuMAge, in-corpus): overlap pos 0.984 / neg 0.499 / top-1 0.738 ; disjoint top-1 0.253")
    print(f"Saved -> {outdir}/", flush=True)


if __name__ == "__main__":
    main()
