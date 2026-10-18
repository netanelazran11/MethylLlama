"""
Extract CONTEXTUALIZED CpG embeddings (post-transformer) — for MethylGPT Fig 2.

MethylGPT Fig 2 plots CpG embeddings that clustered by genomic context. Their
embeddings are POST-transformer (contextualized), NOT the raw token table. Your
earlier near-orthogonal result used the raw table — correct for that table, but
not comparable to Fig 2. This extracts the contextualized version.

Method: run the encoder over many samples with genomic ordering (input_ratio=1.0,
so all CpGs present, sorted by genomic rank → sequence position i is the SAME CpG
for every sample). Average last_hidden_state at each CpG position across samples
→ one contextualized embedding per CpG. This is what Fig 2 UMAPs.

Outputs (--outdir, default results/cpg_context/):
  contextual_cpg_emb.npy   mean contextualized CpG embedding [n_cpg, 256]
  cpg_order.csv            sequence-position → cpg_name → genomic_rank (join key)

Run on cluster (GPU):
  python scripts/rope_pilot/extract_contextual_cpg.py \
     --checkpoint checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt --data <h5ad> --tokenizer <tok> \
     --genomic_rank <rank.npy> --max_samples 512 --outdir results/cpg_context
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True)
    p.add_argument("--outdir", default="results/cpg_context")
    p.add_argument("--max_samples", type=int, default=512)
    p.add_argument("--batch_size", type=int, default=16)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument(
        "--feed_position_ids", type=lambda s: s.lower() != "false", default=True,
        help="Default True (unchanged existing behavior): pass the genomic-rank "
             "position_ids to the encoder, matching a model trained WITH genomic "
             "RoPE. Set --feed_position_ids false for a model trained WITHOUT "
             "genomic RoPE (position_ids=None during training) -- --genomic_rank "
             "is still used to fix a consistent CpG order across samples, but the "
             "resulting position_ids are discarded before calling the encoder, so "
             "it sees position_ids=None (its actual trained fallback), not "
             "out-of-distribution genomic-rank values it never learned to use.")
    return p.parse_args()


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    print(f"[1/5] Loading checkpoint: {a.checkpoint}", flush=True)
    module = load_wced_llama_checkpoint(a.checkpoint)
    encoder = module.encoder.to(a.device).eval()
    print(f"[2/5] Encoder on {a.device}, loading tokenizer: {a.tokenizer}", flush=True)
    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    print(f"[3/5] Loading dataset: {a.data}", flush=True)
    ds = MethylationDataset(h5ad_path=a.data, split=None, normalize_age=False)
    cpg_sites = ds.cpg_sites
    print(f"[4/5] Dataset loaded ({len(ds)} samples, {len(cpg_sites)} CpGs). Building collator "
          f"(feed_position_ids={a.feed_position_ids})", flush=True)
    collator = WCEDCollator(tokenizer=tok, cpg_sites=cpg_sites,
                            vocab_size=len(cpg_sites), input_ratio=1.0,
                            contrastive=False, genomic_rank_path=a.genomic_rank)
    loader = DataLoader(ds, batch_size=a.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=0)
    print(f"[5/5] Collator + loader ready. Starting extraction loop...", flush=True)

    # Genomic ordering is deterministic → sequence position (after CLS) maps to a
    # fixed CpG for all samples. Recover that mapping from the genomic rank.
    gr = np.load(a.genomic_rank)[: len(cpg_sites)]
    seq_order = np.argsort(gr)                       # position p in seq -> column idx
    cpg_names_in_order = np.array(cpg_sites)[seq_order]

    hid_sum = None
    n_seen = 0
    with torch.no_grad():
        for batch in loader:
            if n_seen >= a.max_samples:
                break
            cpg_ids = batch["cpg_ids"].to(a.device)
            beta = batch["beta_values"].to(a.device)
            attn = batch["attention_mask"].to(a.device)
            pos = batch.get("position_ids") if a.feed_position_ids else None
            pos = pos.to(a.device) if pos is not None else None
            input_ids = torch.stack([cpg_ids.float(), beta], dim=1)
            out = encoder(input_ids=input_ids, attention_mask=attn, position_ids=pos)
            h = out.last_hidden_state[:, 1:, :].float()   # [B, n_cpg, D] drop CLS
            s = h.sum(0).cpu().numpy()
            hid_sum = s if hid_sum is None else hid_sum + s
            n_seen += h.shape[0]
            if n_seen % 128 == 0:
                print(f"  {n_seen} samples", flush=True)
    ctx = hid_sum / max(n_seen, 1)                       # [n_cpg, D] in sequence order
    np.save(outdir / "contextual_cpg_emb.npy", ctx.astype(np.float32))
    pd.DataFrame({
        "seq_position": np.arange(len(cpg_names_in_order)),
        "cpg_name": cpg_names_in_order,
        "genomic_rank": gr[seq_order],
    }).to_csv(outdir / "cpg_order.csv", index=False)
    print(f"Contextualized CpG embeddings: {ctx.shape} (over {n_seen} samples)")
    print(f"Saved → {outdir}/contextual_cpg_emb.npy, cpg_order.csv")

    with open(outdir / "contextual_cpg_meta.json", "w") as f:
        json.dump({"n_samples": int(n_seen), "max_samples_requested": a.max_samples,
                   "checkpoint": a.checkpoint, "fed_position_ids": bool(a.feed_position_ids)},
                  f, indent=2)


if __name__ == "__main__":
    main()
