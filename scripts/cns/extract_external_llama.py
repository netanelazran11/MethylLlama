"""
extract_external_llama.py  (CNS task, stage 6 - MethylLlama on the external cohort)
===================================================================================
Frozen MethylLlama CLS of the GSE215240 profiles (build_external_cns.py), full profile: the input path and encoder
call of extract_sparse_llama.py at level 100 % (imported unchanged), every measured CpG in genomic order.

Gate before the cohort: the first 64 Capper profiles go through the same code and must match the existing
cls_methylllama embeddings (min cosine > 0.9999), so the external CLS are comparable with the ones the probe trains on.

Checks (the script stops on a failure): the gate; tokens per profile == measured CpGs; position_ids present;
finite embeddings.
Outputs  outputs/cns/external/emb_llama/  embeddings_cls.npy, embeddings_mean.npy, n_tokens.npy,
         metadata.csv (sample_id = GSM, row order of the h5ad), summary.json
GPU (methyllama_env, configs/paths.sh), from the repository root:
  python scripts/cns/extract_external_llama.py
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
from extract_sparse_llama import CKPT, REPO, TOKENS_PER_BATCH, MaskedProfiles, encode, log  # noqa: E402


def embed(encoder, collator, base, device, rows=None):
    ds = MaskedProfiles(base, None, None)
    rows = np.arange(len(base)) if rows is None else rows
    expect = np.isfinite(np.asarray(base.adata.X[rows], dtype=np.float32)).sum(1).astype(np.int64)
    idx = rows[np.argsort(-expect, kind="stable")]
    bs = int(max(1, min(512, TOKENS_PER_BATCH // (int(expect.max()) + 1))))
    pos = {r: i for i, r in enumerate(rows)}
    cls = np.zeros((len(rows), encoder.config.hidden_size), np.float32)
    mean, ntok = np.zeros_like(cls), np.zeros(len(rows), np.int64)
    with torch.no_grad():
        for b0 in range(0, len(idx), bs):
            ii = idx[b0:b0 + bs]
            batch = collator([ds[i] for i in ii])
            if "position_ids" not in batch:
                raise SystemExit("position_ids missing: genomic RoPE not active")
            c, m_, t = encode(encoder, batch, device)
            jj = [pos[i] for i in ii]
            cls[jj], mean[jj], ntok[jj] = c, m_, t
    if not np.array_equal(ntok, expect):
        raise SystemExit(f"tokens != measured CpGs for {(ntok != expect).sum()} profiles")
    if not (np.isfinite(cls).all() and np.isfinite(mean).all()):
        raise SystemExit("non-finite embeddings")
    return cls, mean, ntok


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "outputs/cns/external/GSE215240.h5ad"))
    p.add_argument("--capper", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--reference", default=str(REPO / "outputs/cns/cls_methylllama"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/external/emb_llama"))
    p.add_argument("--checkpoint", default=str(REPO / CKPT))
    p.add_argument("--tokenizer", default=str(REPO / "tokenizer"))
    p.add_argument("--genomic_rank", default=str(REPO / "resources/cpg_genomic_rank.npy"))
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    a = p.parse_args()
    from bmfm_methylation.llama.finetune_llama import load_wced_llama_checkpoint
    from bmfm_methylation.shared.data_module import MethylationDataset, WCEDCollator
    from bmfm_targets.tokenization import MultiFieldTokenizer

    encoder = load_wced_llama_checkpoint(a.checkpoint).encoder.to(a.device).eval()
    tokenizer = MultiFieldTokenizer.from_pretrained(a.tokenizer)
    S = {}

    cap = MethylationDataset(h5ad_path=a.capper, split=None, normalize_age=False)
    col = WCEDCollator(tokenizer=tokenizer, cpg_sites=cap.cpg_sites, vocab_size=len(cap.cpg_sites),
                       input_ratio=1.0, contrastive=False, genomic_rank_path=a.genomic_rank)
    ref_meta = pd.read_csv(Path(a.reference) / "metadata.csv")
    if ref_meta.sample_id.astype(str).tolist() != list(cap.adata.obs.index.astype(str)):
        raise SystemExit("Capper row order differs from the existing embeddings")
    c64, _, _ = embed(encoder, col, cap, a.device, np.arange(64))
    R = np.load(Path(a.reference) / "embeddings_cls.npy")[:64]
    cos = (c64 * R).sum(1) / (np.linalg.norm(c64, axis=1) * np.linalg.norm(R, axis=1))
    S["gate_capper64"] = {"cos_min": float(cos.min()), "maxabs": float(np.abs(c64 - R).max())}
    if cos.min() <= 0.9999:
        raise SystemExit(f"gate failed: {S['gate_capper64']}")

    t0 = time.time()
    ext = MethylationDataset(h5ad_path=a.data, split=None, normalize_age=False)
    if ext.cpg_sites != cap.cpg_sites:
        raise SystemExit("external column order differs from Capper")
    col = WCEDCollator(tokenizer=tokenizer, cpg_sites=ext.cpg_sites, vocab_size=len(ext.cpg_sites),
                       input_ratio=1.0, contrastive=False, genomic_rank_path=a.genomic_rank)
    cls, mean, ntok = embed(encoder, col, ext, a.device)
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings_cls.npy", cls)
    np.save(out / "embeddings_mean.npy", mean)
    np.save(out / "n_tokens.npy", ntok)
    pd.DataFrame({"sample_id": ext.adata.obs.index.astype(str)}).to_csv(out / "metadata.csv", index=False)
    S.update({"n": len(cls), "tokens_min": int(ntok.min()), "tokens_max": int(ntok.max()),
              "tokens_equal_measured": True, "seconds": round(time.time() - t0, 1)})
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    log.info(json.dumps(S))


if __name__ == "__main__":
    main()
