"""
extract_cls_methylgpt.py  (CNS task, stage 3 - MethylGPT side)
==============================================================
Frozen MethylGPT-medium sample embeddings for the Capper profiles, in the same order and format as
extract_pretrain_cls.py writes for MethylLlama, so probe_cns.py treats both identically.

  embeddings_cls.npy    cell_emb = encoder output at the <cls> position (MethylGPT's own sample embedding)
  embeddings_mean.npy   mean of the encoder outputs over the 49,156 CpG positions
  metadata.csv          sample_id (GSM), set, methylation_class - row order of the embeddings
  extract_summary.json  load counts, input checks, timing

Safeguards (from the loading audit of 2026-10-25)
  - The model is built as finetuning_age_models.from_pretrained builds it but loaded with strict=True: a partial
    load (the silent bare-except fallback) stops the script.
  - Batch size is fixed at 1: with flash_attn 2.x, FlashTransformerEncoderLayer takes `self_attn(...)[0]`, which for
    batch > 1 gives every sample the attention output of sample 0. With batch 1 the result is correct.
  - Input as in MethylGPT's own runs: NaN -> -2 (pad_value), <cls> appended, include_zero_gene, bf16 autocast.
  - Column order of the h5ad must equal MethylGPT's probe list (token ids follow that order).
  - After extraction: no NaN, and no two profiles may have bit-identical CLS vectors (the batch>1 signature).

GPU, in the MethylGPT environment (methylgpt_env, configs/paths.sh; flash-attn required), from the repository root:
  python scripts/cns/extract_cls_methylgpt.py
"""
import argparse  # noqa: E402
import json  # noqa: E402
import sys  # noqa: E402
import time  # noqa: E402
from pathlib import Path  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import torch  # noqa: E402

GPT_DIR = Path(__file__).resolve().parents[1] / "methylgpt"   # scripts/methylgpt: fine-tuning helpers and configs
sys.path.insert(0, str(GPT_DIR / "finetuning"))
import methylgpt  # noqa: E402,F401
from scgpt.model import model as scgpt_model  # noqa: E402
from scgpt.model.model import TransformerModel  # noqa: E402
from scgpt.tokenizer import tokenize_and_pad_batch  # noqa: E402
from finetuning_age_datasets import CollatableVocab  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/cls_methylgpt"))
    p.add_argument("--args_json", default=str(GPT_DIR / "configs/args.json"))
    p.add_argument("--ckpt", default=str(REPO / "pretrained/methylgpt/medium-best_model_epoch6.pt"))
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    S = {"flash_attn_available": bool(scgpt_model.flash_attn_available), "torch": torch.__version__}
    if not S["flash_attn_available"]:
        raise SystemExit("flash_attn not available: the fast-transformer weights would not match")

    with open(a.args_json) as f:
        margs = json.load(f)
    margs.update({"mask_ratio": 0, "mask_seed": 0, "dropout": 0})
    vocab = CollatableVocab(margs)
    model = TransformerModel(
        len(vocab.special_tokens) + len(vocab.CpG_ids), margs["layer_size"], margs["nhead"], margs["layer_size"],
        margs["nlayers"], vocab=vocab.vocab, dropout=margs["dropout"], pad_token=vocab.pad_token,
        pad_value=vocab.pad_value, do_mvc=True, do_dab=False, use_batch_labels=False, num_batch_labels=None,
        domain_spec_batchnorm=False, n_input_bins=None, ecs_threshold=margs["ecs_thres"], explicit_zero_prob=False,
        use_fast_transformer=margs["fast_transformer"], pre_norm=margs["pre_norm"],
    )
    model.load_state_dict(torch.load(a.ckpt, map_location="cpu"), strict=True)   # raises on any partial load
    S["load"] = "strict OK"
    model = model.to("cuda").eval()
    pad_id = vocab.vocab[vocab.pad_token]

    import anndata as ad
    A = ad.read_h5ad(a.data)
    cpg = A.var["cpg_id"].astype(str).tolist()
    probes = pd.read_csv(margs["probe_id_dir"])["illumina_probe_id"].astype(str).tolist()
    if cpg != probes:
        raise SystemExit("h5ad column order differs from MethylGPT's probe list")
    X = np.asarray(A.X, dtype=np.float32)
    n = len(X)
    cls = np.zeros((n, margs["layer_size"]), dtype=np.float32)
    mean = np.zeros_like(cls)
    with torch.no_grad():
        for i in range(n):                                  # batch size 1 (see header)
            tok = tokenize_and_pad_batch(np.nan_to_num(X[i:i + 1], nan=vocab.pad_value), vocab.CpG_ids,
                                         max_len=vocab.max_seq_len, vocab=vocab.vocab, pad_token=vocab.pad_token,
                                         pad_value=vocab.pad_value, append_cls=True, include_zero_gene=True)
            g, v = tok["genes"].to("cuda"), tok["values"].to("cuda").float()
            with torch.autocast("cuda", dtype=torch.bfloat16):
                h = model._encode(g, v, src_key_padding_mask=g.eq(pad_id), batch_labels=None).float()
            cls[i] = h[0, 0].cpu().numpy()
            mean[i] = h[0, 1:].mean(0).cpu().numpy()
            if (i + 1) % 500 == 0:
                print(f"{i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)

    if not (np.isfinite(cls).all() and np.isfinite(mean).all()):
        raise SystemExit("non-finite embeddings")
    rng = np.random.default_rng(0)
    idx = rng.choice(n, min(500, n), replace=False)
    Cn = cls[idx] / np.linalg.norm(cls[idx], axis=1, keepdims=True)
    G = Cn @ Cn.T
    off = G[np.triu_indices(len(idx), 1)]
    S.update({"n": n, "seq_len": int(g.shape[1]), "cls_pairwise_cos_mean": float(off.mean()),
              "cls_pairwise_cos_max": float(off.max()), "seconds": round(time.time() - t0, 1)})
    S["cls_unique_rows"] = int(np.unique(cls, axis=0).shape[0])
    if S["cls_unique_rows"] < n:   # the batch>1 bug gives bit-identical vectors; near-duplicate profiles do not
        raise SystemExit(f"{n - S['cls_unique_rows']} CLS vectors are bit-identical to another profile's")
    np.save(out / "embeddings_cls.npy", cls)
    np.save(out / "embeddings_mean.npy", mean)
    obs = A.obs
    pd.DataFrame({"sample_id": obs.index.astype(str), "set": obs.set.astype(str).values,
                  "methylation_class": obs.methylation_class.astype(str).values}).to_csv(out / "metadata.csv", index=False)
    json.dump(S, open(out / "extract_summary.json", "w"), indent=1)
    print(json.dumps(S, indent=1))


if __name__ == "__main__":
    main()
