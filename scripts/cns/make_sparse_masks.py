"""
make_sparse_masks.py  (CNS task, stage 3 - sparse-input masks)
==============================================================
One fixed set of CpG masks for the sparse-input experiment, written once and read by every representation
(MethylLlama, MethylGPT, raw), so that all of them see exactly the same CpGs of each profile.

Level L keeps k(L) = round(L/100 * 49,156) CpGs, drawn at random among the CpGs MEASURED in that profile (capped at
the number measured). Masks are nested: per profile and seed one random order of its measured CpGs is drawn, and
level L keeps the first k(L) of that order, so 0.5% c 1% c ... c 50%. Level 100% needs no mask (all measured CpGs).

Outputs  outputs/cns/sparse/masks/
  order_s<seed>.npy   uint16 [n_profiles, k(50%)]  column indices in keep-priority order; 65535 = none (profile has
                      fewer measured CpGs than k(50%))
  n_measured.npy      int32 [n_profiles]
  levels.json         {level: k}, seeds, panel size, md5 of each order file
  metadata.csv        sample_id, set, methylation_class - row order (same as the h5ad and both extractions)

  python scripts/cns/make_sparse_masks.py            (CPU, ~2 min, ~0.6 GB for 3 seeds)
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
LEVELS = (100, 50, 20, 10, 5, 2, 1, 0.5)
SEEDS = (0, 1, 2)
NONE = np.uint16(65535)


def n_keep(level, n_cpg):
    return int(round(level / 100 * n_cpg))


def tag(level, seed):
    """Directory name of one (level, seed); level 100 has no mask, so no seed."""
    return "L100" if level == 100 else f"L{level:g}_s{seed}"


def load_masks(maskdir):
    maskdir = Path(maskdir)
    info = json.load(open(maskdir / "levels.json"))
    return info, np.load(maskdir / "n_measured.npy")


def keep_columns(order_row, k):
    """Column indices kept for one profile at a level with k CpGs (from its row of order_s<seed>.npy)."""
    keep = order_row[:k]
    return keep[keep != NONE].astype(np.int64)


def masked_row(x, order_row, k):
    """Profile x (NaN = not measured) with every CpG outside the kept set set to NaN."""
    keep = keep_columns(order_row, k)
    out = np.full_like(x, np.nan)
    out[keep] = x[keep]
    return out


def md5(path):
    h = hashlib.md5()
    with open(path, "rb") as f:
        for b in iter(lambda: f.read(1 << 22), b""):
            h.update(b)
    return h.hexdigest()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/sparse/masks"))
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    import anndata as ad
    A = ad.read_h5ad(a.data)
    X = np.asarray(A.X, dtype=np.float32)
    n, n_cpg = X.shape
    if n_cpg >= int(NONE):
        raise SystemExit("panel too large for uint16 indices")
    valid = np.isfinite(X)
    n_meas = valid.sum(1).astype(np.int32)
    K = n_keep(max(lv for lv in LEVELS if lv < 100), n_cpg)
    info = {"panel": int(n_cpg), "n_profiles": int(n), "levels": {f"{lv:g}": n_keep(lv, n_cpg) for lv in LEVELS},
            "seeds": list(SEEDS), "k_stored": K, "n_capped_at_50": int((n_meas < K).sum()), "md5": {}}

    for s in SEEDS:
        rng = np.random.default_rng(s)
        order = np.full((n, K), NONE, dtype=np.uint16)
        for i in range(n):
            perm = rng.permutation(np.flatnonzero(valid[i]))[:K]
            order[i, :len(perm)] = perm
        # checks: only measured CpGs, no repeats, nesting holds by construction (prefixes of one order)
        for i in rng.choice(n, 50, replace=False):
            kk = keep_columns(order[i], K)
            assert valid[i, kk].all() and len(np.unique(kk)) == len(kk)
        f = out / f"order_s{s}.npy"
        np.save(f, order)
        info["md5"][f.name] = md5(f)
        print(f"seed {s}: saved {f.name} {order.shape}", flush=True)

    np.save(out / "n_measured.npy", n_meas)
    pd.DataFrame({"sample_id": A.obs.index.astype(str), "set": A.obs.set.astype(str).values,
                  "methylation_class": A.obs.methylation_class.astype(str).values}).to_csv(out / "metadata.csv",
                                                                                            index=False)
    # seeds must give different masks
    o0, o1 = np.load(out / "order_s0.npy", mmap_mode="r"), np.load(out / "order_s1.npy", mmap_mode="r")
    k5 = info["levels"]["5"]
    ov = np.mean([len(np.intersect1d(keep_columns(o0[i], k5), keep_columns(o1[i], k5))) / k5 for i in range(20)])
    info["seed0_vs_seed1_overlap_at_5pct"] = round(float(ov), 4)   # expected about 0.05
    json.dump(info, open(out / "levels.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
