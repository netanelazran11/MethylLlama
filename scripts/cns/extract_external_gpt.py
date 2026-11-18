"""
extract_external_gpt.py  (CNS task, stage 6 - MethylGPT on the external cohort)
===============================================================================
Frozen MethylGPT-medium CLS of the GSE215240 profiles (build_external_cns.py), full profile: model build and
encode_one of extract_sparse_gpt.py (imported unchanged: strict load, batch size 1, NaN -> -2, bf16).

Gate before the cohort: the first 32 Capper profiles must match the existing cls_methylgpt embeddings
(min cosine > 0.9999).
Checks (the script stops on a failure): the gate; finite embeddings; no two bit-identical CLS rows.
Outputs  outputs/cns/external/emb_gpt/  embeddings_cls.npy, embeddings_mean.npy, n_measured_input.npy,
         metadata.csv, summary.json
GPU, in the MethylGPT environment (methylgpt_env, configs/paths.sh), from the repository root:
  python scripts/cns/extract_external_gpt.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import extract_sparse_gpt as G  # noqa: E402  (model build and MethylGPT imports)

import argparse  # noqa: E402
import json  # noqa: E402
import time  # noqa: E402

import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402


@G.torch.no_grad()
def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(G.REPO / "outputs/cns/external/GSE215240.h5ad"))
    p.add_argument("--capper", default=str(G.REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--reference", default=str(G.REPO / "outputs/cns/cls_methylgpt"))
    p.add_argument("--outdir", default=str(G.REPO / "outputs/cns/external/emb_gpt"))
    p.add_argument("--args_json", default=str(G.GPT_DIR / "configs/args.json"))
    p.add_argument("--ckpt", default=str(G.REPO / "pretrained/methylgpt/medium-best_model_epoch6.pt"))
    a = p.parse_args()
    import anndata as ad
    model, vocab, margs = G.build_model(a.args_json, a.ckpt)
    probes = pd.read_csv(margs["probe_id_dir"])["illumina_probe_id"].astype(str).tolist()
    S = {}

    C = ad.read_h5ad(a.capper, backed="r")
    if pd.read_csv(Path(a.reference) / "metadata.csv").sample_id.astype(str).tolist() != list(C.obs.index.astype(str)):
        raise SystemExit("Capper row order differs from the existing embeddings")
    Xc = np.asarray(C.X[:32], dtype=np.float32)
    c32 = np.stack([G.encode_one(model, vocab, x)[0] for x in Xc])
    R = np.load(Path(a.reference) / "embeddings_cls.npy")[:32]
    cos = (c32 * R).sum(1) / (np.linalg.norm(c32, axis=1) * np.linalg.norm(R, axis=1))
    S["gate_capper32"] = {"cos_min": float(cos.min()), "maxabs": float(np.abs(c32 - R).max())}
    if cos.min() <= 0.9999:
        raise SystemExit(f"gate failed: {S['gate_capper32']}")

    t0 = time.time()
    A = ad.read_h5ad(a.data)
    if A.var["cpg_id"].astype(str).tolist() != probes:
        raise SystemExit("h5ad column order differs from MethylGPT's probe list")
    X = np.asarray(A.X, dtype=np.float32)
    n = len(X)
    cls = np.zeros((n, margs["layer_size"]), np.float32)
    mean, nin = np.zeros_like(cls), np.isfinite(X).sum(1).astype(np.int64)
    for i in range(n):                                       # batch size 1 (see extract_sparse_gpt.py)
        cls[i], mean[i] = G.encode_one(model, vocab, X[i])
        if (i + 1) % 200 == 0:
            print(f"{i + 1}/{n}  {time.time() - t0:.0f}s", flush=True)
    if not (np.isfinite(cls).all() and np.isfinite(mean).all()):
        raise SystemExit("non-finite embeddings")
    uniq = int(np.unique(cls, axis=0).shape[0])
    if uniq < n:
        raise SystemExit(f"{n - uniq} CLS vectors are bit-identical to another profile's")
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    np.save(out / "embeddings_cls.npy", cls)
    np.save(out / "embeddings_mean.npy", mean)
    np.save(out / "n_measured_input.npy", nin)
    pd.DataFrame({"sample_id": A.obs.index.astype(str)}).to_csv(out / "metadata.csv", index=False)
    S.update({"n": n, "cls_unique_rows": uniq, "measured_min": int(nin.min()), "measured_max": int(nin.max()),
              "seconds": round(time.time() - t0, 1)})
    json.dump(S, open(out / "summary.json", "w"), indent=1)
    print(json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
