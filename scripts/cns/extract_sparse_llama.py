"""
extract_sparse_llama.py  (CNS task, stage 3 - MethylLlama on sparse profiles)
=============================================================================
Frozen MethylLlama embeddings of the Capper profiles when each profile shows only the CpGs kept by the sparse
masks (make_sparse_masks.py). The input path is the one extract_pretrain_cls.py uses for the full-panel
embeddings, unchanged: MethylationDataset -> WCEDCollator(input_ratio=1.0, genomic_rank_path) -> encoder with
position_ids. The collator already drops missing CpGs, so setting the unselected CpGs to NaN is the whole
intervention: only the kept CpGs enter, in genomic order.

The one addition is speed: the collator pads every batch to 49,157 tokens; here each batch is cut to its longest
real sequence (attention_mask = 0 beyond it) and profiles are batched in order of length.

Checks (in summary.json; the script stops on the first three)
  - tokens per profile == kept CpGs (min(k, measured)); at 100% == measured
  - position_ids present (genomic RoPE active)
  - finite embeddings
  - first batch: trimmed vs padded to 49,157 (max |diff|), and the same batch run twice (max |diff|)
  - level 100%: cosine and max |diff| against the existing cls_methylllama embeddings (gate: min cosine > 0.9999)

Outputs  outputs/cns/sparse/emb_llama/<L100 | L5_s0 | ...>/
  embeddings_cls.npy, embeddings_mean.npy, n_tokens.npy, metadata.csv, summary.json

GPU (methyllama_env, configs/paths.sh), from the repository root:
  python scripts/cns/extract_sparse_llama.py --levels 100,5 --seeds 0
"""
import argparse
import json
import logging
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
from make_sparse_masks import keep_columns, load_masks, n_keep, tag  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")
log = logging.getLogger("sparse_llama")
CKPT = "checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt"   # released checkpoint (scripts/download_checkpoint.sh)
TOKENS_PER_BATCH = 16 * 49157          # the memory the full-panel run used (batch 16 x full length)


class MaskedProfiles(torch.utils.data.Dataset):
    """MethylationDataset items with every CpG outside the kept set set to NaN (level 100%: unchanged)."""

    def __init__(self, base, order, k):
        self.base, self.order, self.k = base, order, k

    def __len__(self):
        return len(self.base)

    def __getitem__(self, i):
        from bmfm_targets.tokenization.multifield_instance import MultiFieldInstance
        inst = self.base[i]
        if self.order is None:
            return inst
        x = np.asarray(inst.data["beta_values"], dtype=np.float32)
        keep = keep_columns(self.order[i], self.k)
        if not np.isfinite(x[keep]).all():
            raise SystemExit(f"profile {i}: mask keeps a CpG that is not measured")
        xm = np.full_like(x, np.nan)
        xm[keep] = x[keep]
        return MultiFieldInstance(data={"beta_values": xm.tolist(), "valid_mask": np.isfinite(xm).tolist()},
                                  metadata=inst.metadata)


def encode(encoder, batch, device, trim=True):
    attn = batch["attention_mask"]
    L = int(attn.sum(1).max()) if trim else attn.shape[1]
    cpg_ids = batch["cpg_ids"][:, :L].to(device)
    beta = batch["beta_values"][:, :L].to(device)
    attn = attn[:, :L].to(device)
    pos = batch["position_ids"][:, :L].to(device)
    out = encoder(input_ids=torch.stack([cpg_ids.float(), beta], dim=1), attention_mask=attn, position_ids=pos)
    hidden = out.last_hidden_state[:, 1:, :].float()
    m = attn[:, 1:].float().unsqueeze(-1)
    mean = (hidden * m).sum(1) / m.sum(1).clamp(min=1)
    return out.pooler_output.float().cpu().numpy(), mean.cpu().numpy(), (attn.sum(1) - 1).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="100,5")
    p.add_argument("--seeds", default="0")
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--masks", default=str(REPO / "outputs/cns/sparse/masks"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/sparse/emb_llama"))
    p.add_argument("--reference", default=str(REPO / "outputs/cns/cls_methylllama"))
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
    base = MethylationDataset(h5ad_path=a.data, split=None, normalize_age=False)
    collator = WCEDCollator(tokenizer=tokenizer, cpg_sites=base.cpg_sites, vocab_size=len(base.cpg_sites),
                            input_ratio=1.0, contrastive=False, genomic_rank_path=a.genomic_rank)
    info, n_meas = load_masks(a.masks)
    meta = pd.read_csv(Path(a.masks) / "metadata.csv")
    if list(base.adata.obs.index.astype(str)) != meta.sample_id.astype(str).tolist():
        raise SystemExit("dataset row order differs from the mask file")
    ref_meta = pd.read_csv(Path(a.reference) / "metadata.csv")
    if ref_meta.sample_id.astype(str).tolist() != meta.sample_id.astype(str).tolist():
        raise SystemExit("row order differs from the existing full-panel embeddings")
    n = len(base)

    for level in [float(x) for x in a.levels.split(",")]:
        for seed in ([0] if level == 100 else [int(s) for s in a.seeds.split(",")]):
            t0 = time.time()
            name = tag(level, seed)
            out = Path(a.outdir) / name
            out.mkdir(parents=True, exist_ok=True)
            if level == 100:
                order, k, expect = None, None, n_meas.astype(np.int64)
            else:
                order = np.load(Path(a.masks) / f"order_s{seed}.npy", mmap_mode="r")
                k = n_keep(level, info["panel"])
                expect = np.minimum(k, n_meas).astype(np.int64)
            ds = MaskedProfiles(base, order, k)
            idx = np.argsort(-expect, kind="stable")                 # longest first: full batches, short padding
            bs = int(max(1, min(512, TOKENS_PER_BATCH // (int(expect.max()) + 1))))
            cls = np.zeros((n, encoder.config.hidden_size), np.float32)
            mean, ntok = np.zeros_like(cls), np.zeros(n, np.int64)
            S = {"level": level, "seed": seed, "k": k, "batch_size": bs}
            with torch.no_grad():
                for b0 in range(0, n, bs):
                    ii = idx[b0:b0 + bs]
                    batch = collator([ds[i] for i in ii])
                    if "position_ids" not in batch:
                        raise SystemExit("position_ids missing: genomic RoPE not active")
                    c, m_, t = encode(encoder, batch, a.device)
                    cls[ii], mean[ii], ntok[ii] = c, m_, t
                    if b0 == 0:   # 16 profiles: padded to 49,157 they fit in the memory of the full-panel run
                        sub = collator([ds[i] for i in ii[:16]])
                        c16, _, _ = encode(encoder, sub, a.device)
                        c_pad, _, _ = encode(encoder, sub, a.device, trim=False)
                        c_again, _, _ = encode(encoder, sub, a.device)
                        S["first16_trim_vs_padded_maxabs"] = float(np.abs(c16 - c_pad).max())
                        S["first16_rerun_maxabs"] = float(np.abs(c16 - c_again).max())
                        S["first16_batchsize_effect_maxabs"] = float(np.abs(c16 - c[:16]).max())
                    if (b0 // bs + 1) % 20 == 0:
                        log.info(f"{name}: {b0 + len(ii)}/{n}  {time.time() - t0:.0f}s")
            if not np.array_equal(ntok, expect):
                bad = np.flatnonzero(ntok != expect)
                raise SystemExit(f"{name}: tokens != kept CpGs for {len(bad)} profiles (first {bad[:5]})")
            if not (np.isfinite(cls).all() and np.isfinite(mean).all()):
                raise SystemExit(f"{name}: non-finite embeddings")
            S.update({"n": n, "tokens_min": int(ntok.min()), "tokens_max": int(ntok.max()),
                      "tokens_equal_kept": True, "seconds": round(time.time() - t0, 1)})
            if level == 100:
                R = np.load(Path(a.reference) / "embeddings_cls.npy")
                cos = (cls * R).sum(1) / (np.linalg.norm(cls, axis=1) * np.linalg.norm(R, axis=1))
                S["vs_existing_cls"] = {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                                        "maxabs": float(np.abs(cls - R).max()), "gate_cos_min_gt_0.9999":
                                        bool(cos.min() > 0.9999)}
            else:
                F = np.load(Path(a.outdir) / "L100" / "embeddings_cls.npy") if (Path(a.outdir) / "L100").exists() \
                    else np.load(Path(a.reference) / "embeddings_cls.npy")
                cos = (cls * F).sum(1) / (np.linalg.norm(cls, axis=1) * np.linalg.norm(F, axis=1))
                S["cos_to_full"] = {"mean": float(cos.mean()), "p05": float(np.percentile(cos, 5)),
                                    "min": float(cos.min())}
            np.save(out / "embeddings_cls.npy", cls)
            np.save(out / "embeddings_mean.npy", mean)
            np.save(out / "n_tokens.npy", ntok)
            meta.to_csv(out / "metadata.csv", index=False)
            json.dump(S, open(out / "summary.json", "w"), indent=1)
            log.info(f"{name}: {json.dumps(S)}")


if __name__ == "__main__":
    main()
