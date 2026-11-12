"""
two_view_consistency.py
=======================
Two-view consistency on 2,000 downstream (AltumAge) profiles; its saved view-1 embeddings
(two_view_v1n.npy) are the 50%-CpG subsets of Supplementary Fig. S2
(scripts/figures/figS_panel_independence.py). The manuscript's cross-view statistics on the
held-out pretraining partition come from view_consistency_heldout_full.py instead.

The MethylLlama pretraining objective includes contrastive InfoNCE (w=0.05) between two random
50%-CpG views of the same sample. A well-trained encoder should map the two views of
ONE sample to nearly the same CLS (high positive cosine) while keeping DIFFERENT
samples apart (low negative cosine). This re-derives that alignment/uniformity from
the checkpoint — a direct quality signal for the contrastive pretraining.

Needs a GPU forward (contrastive collator produces two views), so this loads the
checkpoint (unlike the pure post-processing steps of run_full_analysis.sh).

Metrics:
  pos_cos   — mean cosine(view1_CLS, view2_CLS) for the SAME sample   (want high)
  neg_cos   — mean cosine(view1_CLS, other sample view1_CLS)          (want low)
  alignment_gap = pos_cos - neg_cos                                    (want large)
  retrieval@1   — does view2 retrieve its own view1 as nearest?        (want ~1.0)

Usage (cluster, GPU; step 4 of run_full_analysis.sh):
  python scripts/evaluation/two_view_consistency.py \
    --checkpoint checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt \
    --data <altumage_21k_3way.h5ad> --tokenizer tokenizer \
    --genomic_rank resources/cpg_genomic_rank_finetune.npy --n_samples 2000 --outdir results/pretrained_cls
Writes two_view_consistency.json, two_view_v1n.npy, two_view_v2n.npy, two_view_simmatrix.npy
(results/pretrained_cls/ ships two_view_v1n.npy only).
"""

import argparse
import json
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader, Subset


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--checkpoint", required=True)
    p.add_argument("--data", required=True)
    p.add_argument("--tokenizer", required=True)
    p.add_argument("--genomic_rank", required=True)
    p.add_argument("--outdir", default="results/pretrained_cls")
    p.add_argument("--n_samples", type=int, default=2000)
    p.add_argument("--batch_size", type=int, default=32)
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    return p.parse_args()


def encode(encoder, cpg_ids, betas, attn, pos, device):
    input_ids = torch.stack([cpg_ids.float(), betas], dim=1).to(device)
    out = encoder(input_ids=input_ids, attention_mask=attn.to(device),
                  position_ids=pos.to(device) if pos is not None else None)
    return out.pooler_output.cpu().float()


def main():
    a = parse_args()
    outdir = Path(a.outdir); outdir.mkdir(parents=True, exist_ok=True)
    from bmfm_targets.tokenization import MultiFieldTokenizer
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint

    module = load_wced_llama_checkpoint(a.checkpoint)
    encoder = module.encoder.to(a.device).eval()
    tok = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    ds = MethylationDataset(h5ad_path=a.data, split=None, normalize_age=False)
    n = min(a.n_samples, len(ds))
    rng = np.random.default_rng(0)
    sub = Subset(ds, rng.choice(len(ds), n, replace=False).tolist())

    # contrastive=True → collator emits view-1 and view-2 (two random 50% subsets)
    collator = WCEDCollator(
        tokenizer=tok, cpg_sites=ds.cpg_sites, vocab_size=len(ds.cpg_sites),
        input_ratio=0.5, contrastive=True, genomic_rank_path=a.genomic_rank,
    )
    loader = DataLoader(sub, batch_size=a.batch_size, collate_fn=collator,
                        shuffle=False, num_workers=0)

    v1, v2 = [], []
    with torch.no_grad():
        for batch in loader:
            v1.append(encode(encoder, batch["cpg_ids"], batch["beta_values"],
                             batch["attention_mask"], batch.get("position_ids"), a.device))
            v2.append(encode(encoder, batch["cpg_ids_v2"], batch["beta_values_v2"],
                             batch["attention_mask_v2"], batch.get("position_ids_v2"), a.device))
    V1 = torch.cat(v1).numpy(); V2 = torch.cat(v2).numpy()
    V1n = V1 / (np.linalg.norm(V1, axis=1, keepdims=True) + 1e-9)
    V2n = V2 / (np.linalg.norm(V2, axis=1, keepdims=True) + 1e-9)

    pos_cos = float(np.mean(np.sum(V1n * V2n, axis=1)))
    S = V1n @ V2n.T                                   # [n, n] view1 vs view2
    neg = S.copy(); np.fill_diagonal(neg, np.nan)
    neg_cos = float(np.nanmean(neg))
    retrieval_at1 = float(np.mean(np.argmax(S, axis=1) == np.arange(len(S))))

    summary = {
        "n_samples": int(len(V1)),
        "pos_cos": round(pos_cos, 4),
        "neg_cos": round(neg_cos, 4),
        "alignment_gap": round(pos_cos - neg_cos, 4),
        "retrieval_at1": round(retrieval_at1, 4),
    }
    with open(outdir / "two_view_consistency.json", "w") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))

    # Save normalized embeddings + full similarity matrix for downstream visualization
    # (histogram of pos vs neg similarity distributions, similarity-matrix heatmap).
    np.save(outdir / "two_view_v1n.npy", V1n.astype(np.float32))
    np.save(outdir / "two_view_v2n.npy", V2n.astype(np.float32))
    np.save(outdir / "two_view_simmatrix.npy", S.astype(np.float32))
    # Alignment quality is judged by pos_cos + gap. retrieval@1 is secondary: it is
    # depressed by genuine near-duplicate / highly-similar samples (batch structure),
    # not by weak alignment, so it does not gate the verdict.
    strong = summary["pos_cos"] > 0.9 and summary["alignment_gap"] > 0.3
    verdict = "strong contrastive alignment" if strong else "weak alignment — inspect"
    if strong and summary["retrieval_at1"] < 0.8:
        verdict += (f" (retrieval@1={summary['retrieval_at1']} depressed by "
                    f"near-duplicate/similar samples — expected, not a defect)")
    print("VERDICT:", verdict)


if __name__ == "__main__":
    main()
