"""
pretrain_heldout_duplicates.py
==============================
Are there held-out pretraining profiles whose methylation values duplicate a TRAINING profile
(re-uploads, technical replicates), and duplicates inside the held-out partition itself?

The question is value identity, the same one the downstream duplicate filter asks
(Methods: cosine >= 0.9999); it is not a claim about which person a profile comes from.

Two stages
  1. Screen.  Every profile is reduced to SCREEN_CPGS CpGs (rarely missing, highest
     variance; a missing value is set to the CpG mean), centred per CpG and L2-normalised. For each held-out profile the TOP_K most
     similar training profiles and the TOP_K most similar other held-out profiles are found
     (exact search). Several candidates are kept because profiles of one tissue are all close.
  2. Verify.  Every screened pair with similarity >= SCREEN_MIN is re-compared on ALL CpGs
     measured in both profiles: Pearson r and mean absolute difference.
     duplicate            mean |difference| <= 0.005 and r >= 0.9999
     near-duplicate       mean |difference| <= 0.02  and r >= 0.995   (same sample, other pipeline,
                          or a replicate array; reported separately, not called identical)

The matrix is streamed twice in row blocks. Memory: the screening columns (~1.4 GB) and the
rows to verify, stored as float16 (< 20 GB).
No torch, no GPU.

  python scripts/data/pretrain_heldout_duplicates.py        (CPU node, 64 GB, 16 cores, about 4 h)

Writes outputs/data_audit/heldout_duplicates_{pairs.csv,summary.json}. results/data_audit/ ships the
summary and heldout_duplicates_flagged_pairs.csv: the duplicate and near-duplicate pairs only, with
the columns comparison, id_a, id_b, duplicate, near_duplicate (the negative-exclusion list of
scripts/evaluation/hard_negative_heldout.py).
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tcga_same_participant_check import PRE, row_names      # noqa: E402


def stats(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 1000:
        return n, np.nan, np.nan
    x, y = a[ok].astype(np.float64), b[ok].astype(np.float64)
    return n, float(np.corrcoef(x, y)[0, 1]), float(np.abs(x - y).mean())


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", default=PRE)
    p.add_argument("--metadata", default=f'{os.environ.get("METHYLLAMA_DATA", "data")}/pretrain_metadata.csv.gz',
                   help="annotation table that accompanies the pretraining corpus (configs/paths.example.sh)")
    p.add_argument("--outdir", default="outputs/data_audit")
    p.add_argument("--screen_cpgs", type=int, default=2000)
    p.add_argument("--screen_min", type=float, default=0.90)
    p.add_argument("--block", type=int, default=4096)
    p.add_argument("--top_k", type=int, default=10)
    a = p.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    for f in ("heldout_duplicates_pairs.csv", "heldout_duplicates_summary.json"):
        if (out / f).exists():
            raise SystemExit(f"{out / f} exists -- nothing is overwritten")
    t0 = time.time()

    with h5py.File(a.pretrain, "r") as f:
        X = f["X"]; n, c = X.shape
        ids, _ = row_names(f["obs"], n)
        perm = np.random.default_rng(42).permutation(n)
        split = np.empty(n, dtype=object)
        split[perm[:int(0.8 * n)]] = "train"
        split[perm[int(0.8 * n):int(0.9 * n)]] = "valid"
        split[perm[int(0.9 * n):]] = "heldout"
        print(f"{n:,} x {c:,} | " + ", ".join(f"{k} {int((split == k).sum()):,}" for k in ("train", "valid", "heldout")), flush=True)

        # choose the screening CpGs from a random sample of rows
        rs = np.sort(np.random.default_rng(0).choice(n, size=min(4000, n), replace=False))
        S = np.stack([X[int(r)] for r in rs]).astype(np.float32)
        miss = np.isnan(S).mean(0)
        var = np.nanvar(S, axis=0)
        for thr in (0.01, 0.02, 0.05, 0.10, 0.20):                     # the least missingness that leaves a choice
            if (miss <= thr).sum() >= 2 * a.screen_cpgs:
                break
        else:
            raise SystemExit("fewer than 2 x screen_cpgs CpGs have <= 20% missing values")
        var[miss > thr] = -1
        cols = np.sort(np.argsort(var)[::-1][:a.screen_cpgs])
        assert (var[cols] > 0).all()
        mu = np.nanmean(S[:, cols], axis=0).astype(np.float32)
        print(f"screening CpGs: {len(cols):,} of {int((miss <= thr).sum()):,} with <= {thr:.0%} missing values "
              f"(variance {var[cols].min():.4f}-{var[cols].max():.4f})", flush=True)
        del S

        Z = np.empty((n, len(cols)), dtype=np.float32)
        for s in range(0, n, a.block):
            b = X[s:s + a.block][:, cols].astype(np.float32)
            b = np.where(np.isnan(b), mu, b) - mu
            Z[s:s + a.block] = b / (np.linalg.norm(b, axis=1, keepdims=True) + 1e-9)
            if (s // a.block) % 5 == 0:
                print(f"   read {min(s + a.block, n):,} rows ({time.time() - t0:.0f} s)", flush=True)

        ho, tr = np.where(split == "heldout")[0], np.where(split == "train")[0]
        Zt = Z[tr]
        cand = []
        for s in range(0, len(ho), 512):
            q = ho[s:s + 512]
            for name, pool, Zp in (("heldout vs train", tr, Zt), ("heldout vs heldout", ho, Z[ho])):
                sim = Z[q] @ Zp.T
                if name == "heldout vs heldout":
                    sim[np.arange(len(q)), s + np.arange(len(q))] = -2   # not itself
                top = np.argpartition(-sim, a.top_k - 1, axis=1)[:, :a.top_k]
                for i, h in enumerate(q):
                    for rank, k in enumerate(top[i][np.argsort(-sim[i, top[i]])]):
                        cand.append((name, int(h), int(pool[k]), float(sim[i, k]), rank + 1))
        cand = pd.DataFrame(cand, columns=["comparison", "row_a", "row_b", "screen_similarity", "screen_rank"])
        print(f"screen done ({time.time() - t0:.0f} s)", flush=True)

        hist = {k: np.histogram(g[g.screen_rank == 1].screen_similarity, bins=np.linspace(-1, 1, 201))[0].tolist()
                for k, g in cand.groupby("comparison")}
        ver = cand[cand.screen_similarity >= a.screen_min].copy()
        print(f"pairs to verify on all CpGs (screen similarity >= {a.screen_min}): {len(ver):,}", flush=True)
        need = np.array(sorted(set(ver.row_a) | set(ver.row_b)), dtype=np.int64)
        where = {int(r): i for i, r in enumerate(need)}
        R = np.empty((len(need), c), dtype=np.float16)
        print(f"rows to read for verification: {len(need):,} ({R.nbytes / 1e9:.1f} GB)", flush=True)
        for s in range(0, n, a.block):
            sel = need[(need >= s) & (need < s + a.block)]
            if len(sel):
                R[[where[int(r)] for r in sel]] = X[s:s + a.block][sel - s].astype(np.float16)
        print(f"second pass done ({time.time() - t0:.0f} s)", flush=True)
    ver[["n_cpgs", "pearson_r", "mean_abs_diff"]] = [
        stats(R[where[x]].astype(np.float32), R[where[y]].astype(np.float32)) for x, y in zip(ver.row_a, ver.row_b)]
    ver["id_a"], ver["id_b"] = ids[ver.row_a.to_numpy()], ids[ver.row_b.to_numpy()]
    ver["duplicate"] = (ver.mean_abs_diff <= 0.005) & (ver.pearson_r >= 0.9999)
    ver["near_duplicate"] = ~ver.duplicate & (ver.mean_abs_diff <= 0.02) & (ver.pearson_r >= 0.995)
    if Path(a.metadata).exists():
        m = pd.read_csv(a.metadata, usecols=["GSM_ID", "dataset", "tissue", "PATIENT_ID"], dtype=str).set_index("GSM_ID")
        for s in ("a", "b"):
            for col in ("dataset", "tissue", "PATIENT_ID"):
                ver[f"{col}_{s}"] = m[col].reindex(ver[f"id_{s}"]).to_numpy()
        ver["same_study"] = ver.dataset_a.notna() & (ver.dataset_a == ver.dataset_b)
    ver.sort_values(["comparison", "mean_abs_diff"]).to_csv(out / "heldout_duplicates_pairs.csv", index=False)

    summ = {"heldout_profiles": int(len(ho)), "train_profiles": int(len(tr)), "screen_cpgs": int(len(cols)),
            "screen_min": a.screen_min, "criteria": {"duplicate": "mean_abs_diff <= 0.005 and r >= 0.9999",
                                                     "near_duplicate": "mean_abs_diff <= 0.02 and r >= 0.995, not duplicate"},
            "screen_similarity_histogram_bins": "np.linspace(-1, 1, 201)", "screen_similarity_histogram": hist, "comparisons": {}}
    for k, g in ver.groupby("comparison"):
        d = {"pairs_verified": int(len(g)),
             "heldout_profiles_with_a_duplicate": int(g[g.duplicate].row_a.nunique()),
             "heldout_profiles_with_a_near_duplicate": int(g[g.near_duplicate].row_a.nunique())}
        if "same_study" in g:
            d["duplicates_within_one_study"] = int((g.duplicate & g.same_study).sum())
            d["duplicates_across_studies_or_unannotated"] = int((g.duplicate & ~g.same_study).sum())
        summ["comparisons"][k] = d
    (out / "heldout_duplicates_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps({k: v for k, v in summ.items() if k != "screen_similarity_histogram"}, indent=1))
    print(f"done in {time.time() - t0:.0f} s")


if __name__ == "__main__":
    main()
