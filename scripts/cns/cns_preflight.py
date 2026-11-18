"""
cns_preflight.py
================
Pre-flight checks for the CNS-tumour classification task (Capper et al. 2018, GSE90496 reference +
GSE109379 validation), before any data is downloaded or any model is touched.

Answers, from the input files:
  1. Which CpG list does each model really use?
       MethylLlama  tokenizer/tokenizers/cpg_sites/vocab.txt  (+ cpg_genomic_rank.npy)
       MethylGPT    probe_ids_type3.csv (illumina_probe_id)
       corpus       var of methylgpt_pretrain_type3.h5ad
     Same set? Same order? How many are HM450 probes (GSE90496/GSE109379 are HM450)?
  2. Are the Capper samples in the pretraining corpus, under which pretraining split, and how complete
     are their corpus profiles (NaN fraction on the 49,156 CpGs)?
  3. Are there duplicate / same-sample profiles inside or across the two Capper sets (leakage
     between CV folds, or between reference and validation)?
  4. Optionally saves the corpus copy of those profiles (float16) so that it can later be compared
     with the betas downloaded from GEO (same-measurement check, as for E-GEOD-56105).

Read-only on every input. h5py only, no torch. Run on a compute node:

  python scripts/cns/cns_preflight.py --save_corpus_rows        (CPU node, 24 GB, about 1 h)

Writes outputs/cns/preflight/{summary.json, capper_in_corpus.csv, cpg_lists_diff.csv,
capper_duplicate_pairs.csv, corpus_capper_betas.npz (with --save_corpus_rows)}.
"""
import argparse
import hashlib
import os
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "data"))
from tcga_same_participant_check import cpg_names, row_names  # noqa: E402

REPO = Path(__file__).resolve().parents[2]
DATA = Path(os.environ.get("METHYLLAMA_DATA", "data"))            # configs/paths.sh
GPT_DATA = Path(os.environ.get("METHYLGPT_DATA", "data/methylgpt"))  # configs/paths.sh
DEFAULTS = dict(
    pretrain=f"{DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad",
    llama_vocab=f"{REPO}/tokenizer/tokenizers/cpg_sites/vocab.txt",
    llama_rank=f"{REPO}/resources/cpg_genomic_rank.npy",
    llama_ckpt=f"{REPO}/checkpoints/pretrained/methyllama_pretrained_epoch85.ckpt",
    gpt_probes=f"{GPT_DATA}/cpg_ids/probe_ids_type3.csv",
    gpt_ckpt=f"{REPO}/pretrained/methylgpt/medium-best_model_epoch6.pt",
    hm450=f"{DATA}/manifests/HM450.hg38.manifest.tsv",
    samples=f"{REPO}/resources/cns/capper_samples.csv",
    pretrain_ids=f"{REPO}/outputs/pretrain_overlap/pretrain_ids.csv.gz",
    outdir=f"{REPO}/outputs/cns/preflight",
)
SPECIAL = {"[PAD]", "[UNK]", "[CLS]", "[SEP]", "[MASK]", "<pad>", "<cls>", "<eoc>"}


def file_info(p):
    p = Path(p)
    if not p.exists():
        return {"path": str(p), "exists": False}
    info = {"path": str(p), "exists": True, "bytes": p.stat().st_size}
    if p.is_file() and p.stat().st_size < 50e6:
        info["md5"] = hashlib.md5(p.read_bytes()).hexdigest()
    return info


def compare_lists(name_a, a, name_b, b):
    sa, sb = set(a), set(b)
    return {
        "pair": f"{name_a} vs {name_b}",
        "n_a": len(a), "n_b": len(b), "unique_a": len(sa), "unique_b": len(sb),
        "intersection": len(sa & sb), "only_a": len(sa - sb), "only_b": len(sb - sa),
        "same_order": bool(len(a) == len(b) and all(x == y for x, y in zip(a, b))),
        "examples_only_a": sorted(sa - sb)[:5], "examples_only_b": sorted(sb - sa)[:5],
    }


def main():
    p = argparse.ArgumentParser()
    for k, v in DEFAULTS.items():
        p.add_argument(f"--{k}", default=v)
    p.add_argument("--save_corpus_rows", action="store_true")
    p.add_argument("--dup_r", type=float, default=0.99)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    S = {"files": {k: file_info(getattr(a, k)) for k in DEFAULTS if k != "outdir"}}

    # ---- 1. CpG lists ---------------------------------------------------------------------------
    lists = {}
    if Path(a.llama_vocab).exists():
        toks = [t.strip() for t in open(a.llama_vocab) if t.strip()]
        S["llama_vocab_first_tokens"] = toks[:8]
        lists["llama_vocab"] = [t for t in toks if t not in SPECIAL]
        S["llama_vocab_total_lines"] = len(toks)
    if Path(a.llama_rank).exists():
        r = np.load(a.llama_rank)
        S["llama_rank"] = {"len": int(len(r)), "min": int(r.min()), "max": int(r.max()),
                           "unique": int(len(np.unique(r)))}
    if Path(a.gpt_probes).exists():
        g = pd.read_csv(a.gpt_probes)
        S["gpt_probes_columns"] = list(g.columns)
        lists["methylgpt_probes"] = g["illumina_probe_id"].astype(str).tolist()
    with h5py.File(a.pretrain, "r") as f:
        n_pre, c_pre = f["X"].shape
        S["pretrain_shape"] = [int(n_pre), int(c_pre)]
        S["pretrain_X_dtype"] = str(f["X"].dtype)
        S["pretrain_X_chunks"] = f["X"].chunks
        lists["corpus_var"] = list(cpg_names(f["var"], c_pre))
        pre_ids, id_key = row_names(f["obs"], n_pre)
        S["pretrain_obs_name_key"] = id_key
    if Path(a.hm450).exists():
        hm = pd.read_csv(a.hm450, sep="\t", usecols=["probeID"])["probeID"].astype(str).tolist()
        lists["hm450"] = hm

    names = list(lists)
    S["cpg_list_comparisons"] = [compare_lists(x, lists[x], y, lists[y])
                                 for i, x in enumerate(names) for y in names[i + 1:]]
    model_lists = [k for k in lists if k != "hm450"]
    union = sorted(set().union(*[set(lists[k]) for k in model_lists]))
    member = pd.DataFrame({"cpg": union, **{k: np.isin(union, lists[k]) for k in lists}})
    member[~member[model_lists].all(axis=1)].to_csv(out / "cpg_lists_diff.csv", index=False)
    if "hm450" in lists:
        S["model_cpgs_on_hm450"] = {k: int(np.isin(lists[k], lists["hm450"]).sum()) for k in model_lists}

    # ---- 2. Capper samples in the corpus -------------------------------------------------------
    s = pd.read_csv(a.samples)
    pos = pd.Series(np.arange(n_pre), index=pre_ids)
    s["corpus_row"] = s.gsm.map(pos)
    if Path(a.pretrain_ids).exists():
        sp = pd.read_csv(a.pretrain_ids, dtype={"sample_id": str}).set_index("sample_id")["pretrain_split"]
        s["pretrain_split"] = s.gsm.map(sp)
    found = s[s.corpus_row.notna()].copy()
    S["capper_in_corpus"] = {
        st: {"n": int((s.set == st).sum()), "in_corpus": int(found.set.eq(st).sum()),
             "pretrain_split": found[found.set == st].get("pretrain_split", pd.Series(dtype=str))
             .value_counts(dropna=False).to_dict()}
        for st in ("reference", "validation")}

    # NaN fraction and value range per profile, on the corpus columns
    rows = found.corpus_row.astype(int).to_numpy()
    order = np.argsort(rows)
    nanfrac = np.empty(len(rows)); lo = np.empty(len(rows)); hi = np.empty(len(rows)); mu = np.empty(len(rows))
    keep = np.empty((len(rows), c_pre), dtype=np.float32)
    with h5py.File(a.pretrain, "r") as f:
        X = f["X"]
        for j in order:
            v = X[int(rows[j])].astype(np.float32)
            ok = np.isfinite(v)
            nanfrac[j] = 1 - ok.mean()
            lo[j], hi[j], mu[j] = (v[ok].min(), v[ok].max(), v[ok].mean()) if ok.any() else (np.nan,) * 3
            keep[j] = v

    # ---- 3. Duplicate / same-sample profiles inside the Capper sets -----------------------------
    # Two criteria: the corpus duplicate rule (cosine >= 0.9999, NaN -> 0, as analyze_full_dataset.py)
    # and Pearson r on row-centred profiles (NaN -> row mean). Every profile's nearest neighbour is
    # reported, so the r distribution of "different tumours of the same class" is visible too.
    def unit(M):
        return M / (np.linalg.norm(M, axis=1, keepdims=True) + 1e-12)
    C = unit(np.nan_to_num(keep, nan=0.0))
    rm = np.nanmean(keep, axis=1, keepdims=True)
    P = unit(np.where(np.isfinite(keep), keep, rm) - rm)
    n = len(keep)
    nn_r, nn_j, pairs = np.empty(n), np.empty(n, dtype=int), []
    for s0 in range(0, n, 256):
        R = P[s0:s0 + 256] @ P.T
        Cs = C[s0:s0 + 256] @ C.T
        for i in range(R.shape[0]):
            gi = s0 + i
            R[i, gi] = -1
            nn_j[gi] = int(R[i].argmax()); nn_r[gi] = R[i, nn_j[gi]]
            for j in np.where((R[i] >= a.dup_r) | (Cs[i] >= 0.9999))[0]:
                if j > gi:
                    pairs.append((gi, int(j), float(R[i, j]), float(Cs[i, j])))
    g, cls, st = found.gsm.to_numpy(), found.methylation_class.to_numpy(), found.set.to_numpy()
    found["nn_gsm"], found["nn_r"] = g[nn_j], nn_r
    found["nn_same_class"] = cls[nn_j] == cls
    dp = pd.DataFrame([(g[i], g[j], st[i], st[j], cls[i], cls[j], r, c) for i, j, r, c in pairs],
                      columns=["gsm_a", "gsm_b", "set_a", "set_b", "class_a", "class_b", "pearson_r", "cosine"])
    dp.to_csv(out / "capper_duplicate_pairs.csv", index=False)
    S["duplicates"] = {
        "criterion": f"pearson_r >= {a.dup_r} or cosine >= 0.9999",
        "pairs": int(len(dp)),
        "pairs_cosine_0_9999": int((dp.cosine >= 0.9999).sum()),
        "pairs_within_reference": int(((dp.set_a == "reference") & (dp.set_b == "reference")).sum()),
        "pairs_reference_validation": int((dp.set_a != dp.set_b).sum()),
        "pairs_different_class": int((dp.class_a != dp.class_b).sum()),
        "nearest_neighbour_r_quantiles": {q: float(np.quantile(nn_r, q)) for q in (0.01, 0.1, 0.5, 0.9, 0.99)},
        "nearest_neighbour_same_class_frac": float(found.nn_same_class.mean()),
    }
    found["corpus_nan_frac"], found["corpus_min"], found["corpus_max"], found["corpus_mean"] = nanfrac, lo, hi, mu
    found.to_csv(out / "capper_in_corpus.csv", index=False)
    S["corpus_profile_quality"] = {
        "nan_frac_median": float(np.median(nanfrac)), "nan_frac_max": float(nanfrac.max()),
        "n_nan_frac_over_5pct": int((nanfrac > 0.05).sum()),
        "values_outside_0_1": int(((lo < 0) | (hi > 1)).sum()),
        "mean_beta_median": float(np.nanmedian(mu)),
    }
    if a.save_corpus_rows:
        np.savez(out / "corpus_capper_betas.npz", betas=keep.astype(np.float16), gsm=found.gsm.to_numpy(),
                 cpg=np.array(lists["corpus_var"]))

    json.dump(S, open(out / "summary.json", "w"), indent=2, default=str)
    print(json.dumps(S, indent=2, default=str))


if __name__ == "__main__":
    main()
