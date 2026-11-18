"""
make_sparse_masks_high.py  (CNS task, stage 3 - masks above 50 %)
================================================================
make_sparse_masks.py stores each profile's random order of measured CpGs only up to k(50 %). To place the crossover
between 50 % and complete profiles, levels 60-90 % need longer orders. This script draws the SAME orders (same
generator, same seed, same loop; rng.permutation draws the whole permutation before truncation) and stores them up to
k(90 %). So every level is nested with the existing ones: 0.5 % c ... c 50 % c 60 % c ... c 90 %.

Gate (the script stops on a failure): for every seed, the first k(50 %) columns must equal the existing order_s<seed>.npy
exactly, i.e. every existing level keeps the same CpGs.

Outputs  outputs/cns/sparse/masks_high/  order_s<seed>.npy (uint16 [n, k(90)], 65535 = none), n_measured.npy,
         metadata.csv (copied), levels.json (levels 0.5-90, md5, gate result)
  python scripts/cns/make_sparse_masks_high.py            (CPU, a few minutes, ~1 GB for 3 seeds)
"""
import argparse
import json
import shutil
from pathlib import Path

import numpy as np

from make_sparse_masks import LEVELS, NONE, SEEDS, md5, n_keep

REPO = Path(__file__).resolve().parents[2]
HIGH = (60, 70, 80, 90)


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--data", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--existing", default=str(REPO / "outputs/cns/sparse/masks"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/sparse/masks_high"))
    a = p.parse_args()
    out, old = Path(a.outdir), Path(a.existing)
    out.mkdir(parents=True, exist_ok=True)
    import anndata as ad
    A = ad.read_h5ad(a.data)
    X = np.asarray(A.X, dtype=np.float32)
    n, n_cpg = X.shape
    valid = np.isfinite(X)
    n_meas = valid.sum(1).astype(np.int32)
    if not np.array_equal(n_meas, np.load(old / "n_measured.npy")):
        raise SystemExit("measured counts differ from the existing masks")
    K = n_keep(max(HIGH), n_cpg)
    K50 = n_keep(50, n_cpg)
    levels = sorted(set(LEVELS) | set(HIGH))
    info = {"panel": int(n_cpg), "n_profiles": int(n), "levels": {f"{lv:g}": n_keep(lv, n_cpg) for lv in levels},
            "seeds": list(SEEDS), "k_stored": K, "n_capped_at_90": int((n_meas < K).sum()), "md5": {},
            "gate_prefix_equals_existing": {}}
    for s in SEEDS:
        rng = np.random.default_rng(s)                       # as make_sparse_masks.py
        order = np.full((n, K), NONE, dtype=np.uint16)
        for i in range(n):
            perm = rng.permutation(np.flatnonzero(valid[i]))[:K]
            order[i, :len(perm)] = perm
        prev = np.load(old / f"order_s{s}.npy", mmap_mode="r")
        same = bool(np.array_equal(order[:, :K50], prev[:, :K50]))
        info["gate_prefix_equals_existing"][f"s{s}"] = same
        if not same:
            raise SystemExit(f"seed {s}: first k(50%) columns differ from the existing masks")
        f = out / f"order_s{s}.npy"
        np.save(f, order)
        info["md5"][f.name] = md5(f)
        print(f"seed {s}: saved {f.name} {order.shape}, prefix gate passed", flush=True)
    np.save(out / "n_measured.npy", n_meas)
    shutil.copy(old / "metadata.csv", out / "metadata.csv")
    json.dump(info, open(out / "levels.json", "w"), indent=1)
    print(json.dumps(info, indent=1))


if __name__ == "__main__":
    main()
