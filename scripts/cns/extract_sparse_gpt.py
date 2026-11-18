"""
extract_sparse_gpt.py  (CNS task, stage 3 - MethylGPT on sparse profiles)
=========================================================================
Frozen MethylGPT-medium embeddings of the Capper profiles when each profile shows only the CpGs kept by the sparse
masks (make_sparse_masks.py). Model build, input and safeguards are those of extract_cls_methylgpt.py (copied, so
that file stays untouched): strict load, batch size 1 (flash_attn 2.x batch>1 bug), NaN -> -2 (pad_value), <cls>
appended, include_zero_gene, bf16 autocast. MethylGPT always reads all 49,156 positions; unselected CpGs become
-2, its own missing-value convention.

Checks (in summary.json; the script stops on the first two)
  - finite embeddings; no two profiles with bit-identical CLS (the batch>1 signature)
  - first 8 profiles run twice (max |diff|)
  - level 100%: cosine and max |diff| against the existing cls_methylgpt embeddings (gate: min cosine > 0.9999)

Outputs  outputs/cns/sparse/emb_gpt/<L100 | L5_s0 | ...>/
  embeddings_cls.npy, embeddings_mean.npy, n_measured_input.npy, metadata.csv, summary.json

GPU, in the MethylGPT environment (methylgpt_env, configs/paths.sh; flash-attn required), from the repository root:
  python scripts/cns/extract_sparse_gpt.py --levels 100,5 --seeds 0
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
sys.path.insert(0, str(Path(__file__).resolve().parent))
import methylgpt  # noqa: E402,F401
from scgpt.model import model as scgpt_model  # noqa: E402
from scgpt.model.model import TransformerModel  # noqa: E402
from scgpt.tokenizer import tokenize_and_pad_batch  # noqa: E402
from finetuning_age_datasets import CollatableVocab  # noqa: E402
from make_sparse_masks import keep_columns, load_masks, n_keep, tag  # noqa: E402

REPO = Path(__file__).resolve().parents[2]


def build_model(args_json, ckpt):
    """As extract_cls_methylgpt.py: from_pretrained's construction, strict load."""
    if not scgpt_model.flash_attn_available:
        raise SystemExit("flash_attn not available: the fast-transformer weights would not match")
    with open(args_json) as f:
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
    model.load_state_dict(torch.load(ckpt, map_location="cpu"), strict=True)
    return model.to("cuda").eval(), vocab, margs


def encode_one(model, vocab, x):
    pad_id = vocab.vocab[vocab.pad_token]
    tok = tokenize_and_pad_batch(np.nan_to_num(x[None], nan=vocab.pad_value), vocab.CpG_ids,
                                 max_len=vocab.max_seq_len, vocab=vocab.vocab, pad_token=vocab.pad_token,
                                 pad_value=vocab.pad_value, append_cls=True, include_zero_gene=True)
    g, v = tok["genes"].to("cuda"), tok["values"].to("cuda").float()
    with torch.autocast("cuda", dtype=torch.bfloat16):
        h = model._encode(g, v, src_key_padding_mask=g.eq(pad_id), batch_labels=None).float()
    return h[0, 0].cpu().numpy(), h[0, 1:].mean(0).cpu().numpy()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--levels", default="100,5")
    p.add_argument("--seeds", default="0")
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--masks", default=str(REPO / "outputs/cns/sparse/masks"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/sparse/emb_gpt"))
    p.add_argument("--reference", default=str(REPO / "outputs/cns/cls_methylgpt"))
    p.add_argument("--args_json", default=str(GPT_DIR / "configs/args.json"))
    p.add_argument("--ckpt", default=str(REPO / "pretrained/methylgpt/medium-best_model_epoch6.pt"))
    a = p.parse_args()
    model, vocab, margs = build_model(a.args_json, a.ckpt)

    import anndata as ad
    A = ad.read_h5ad(a.data)
    probes = pd.read_csv(margs["probe_id_dir"])["illumina_probe_id"].astype(str).tolist()
    if A.var["cpg_id"].astype(str).tolist() != probes:
        raise SystemExit("h5ad column order differs from MethylGPT's probe list")
    X = np.asarray(A.X, dtype=np.float32)
    n = len(X)
    info, n_meas = load_masks(a.masks)
    meta = pd.read_csv(Path(a.masks) / "metadata.csv")
    if A.obs.index.astype(str).tolist() != meta.sample_id.astype(str).tolist():
        raise SystemExit("h5ad row order differs from the mask file")

    for level in [float(x) for x in a.levels.split(",")]:
        for seed in ([0] if level == 100 else [int(s) for s in a.seeds.split(",")]):
            t0 = time.time()
            name = tag(level, seed)
            out = Path(a.outdir) / name
            out.mkdir(parents=True, exist_ok=True)
            order = None if level == 100 else np.load(Path(a.masks) / f"order_s{seed}.npy", mmap_mode="r")
            k = None if level == 100 else n_keep(level, info["panel"])
            cls = np.zeros((n, margs["layer_size"]), np.float32)
            mean, nin = np.zeros_like(cls), np.zeros(n, np.int64)
            S = {"level": level, "seed": seed, "k": k}
            def profile(i):
                if order is None:
                    return X[i]
                keep = keep_columns(order[i], k)
                xm = np.full_like(X[i], np.nan)
                xm[keep] = X[i, keep]
                return xm

            with torch.no_grad():
                for i in range(n):                                   # batch size 1 (see header)
                    x = profile(i)
                    nin[i] = int(np.isfinite(x).sum())
                    cls[i], mean[i] = encode_one(model, vocab, x)
                    if i == 7:
                        again = np.stack([encode_one(model, vocab, profile(j))[0] for j in range(8)])
                        S["first8_rerun_maxabs"] = float(np.abs(again - cls[:8]).max())
                    if (i + 1) % 500 == 0:
                        print(f"{name}: {i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
            expect = n_meas if order is None else np.minimum(k, n_meas)
            if not np.array_equal(nin, expect):
                raise SystemExit(f"{name}: measured inputs != kept CpGs")
            if not (np.isfinite(cls).all() and np.isfinite(mean).all()):
                raise SystemExit(f"{name}: non-finite embeddings")
            uniq = int(np.unique(cls, axis=0).shape[0])
            if uniq < n:
                raise SystemExit(f"{name}: {n - uniq} CLS vectors are bit-identical to another profile's")
            S.update({"n": n, "cls_unique_rows": uniq, "inputs_equal_kept": True,
                      "seconds": round(time.time() - t0, 1)})
            R = np.load(Path(a.reference) / "embeddings_cls.npy")
            cos = (cls * R).sum(1) / (np.linalg.norm(cls, axis=1) * np.linalg.norm(R, axis=1))
            if level == 100:
                S["vs_existing_cls"] = {"cos_min": float(cos.min()), "cos_mean": float(cos.mean()),
                                        "maxabs": float(np.abs(cls - R).max()),
                                        "gate_cos_min_gt_0.9999": bool(cos.min() > 0.9999)}
            else:
                S["cos_to_full"] = {"mean": float(cos.mean()), "p05": float(np.percentile(cos, 5)),
                                    "min": float(cos.min())}
            np.save(out / "embeddings_cls.npy", cls)
            np.save(out / "embeddings_mean.npy", mean)
            np.save(out / "n_measured_input.npy", nin)
            meta.to_csv(out / "metadata.csv", index=False)
            json.dump(S, open(out / "summary.json", "w"), indent=1)
            print(name, json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
