#!/usr/bin/env python3
"""
Build the fixed data subset for the InfoNCE-weight pilot. Read-only on the source.

Partitions are those of the published pretraining run, recomputed exactly as
MethylationDataset's auto-split does it (bmfm_methylation/shared/data_module.py): perm =
default_rng(42).permutation(n); train = perm[:0.8n], valid = perm[0.8n:0.9n], held-out = the rest.
No row is filtered before that split in pretraining (135,296 + 16,912 + 16,912 = 169,120 = n), and
the script asserts these counts. If outputs/pretrain_overlap/pretrain_ids.csv.gz exists (row,
sample_id, pretrain_split) it is used as an independent cross-check, row by row.

    train  N_TRAIN profiles drawn from the published TRAIN partition
    valid  N_VALID profiles drawn from the published VALID partition
    test   N_TEST  profiles drawn from the published HELD-OUT partition

The three groups are written to obs["split"], which MethylationDataset uses instead of its
auto-split, so every arm trains and validates on exactly these profiles. Nothing in the pilot
ever trains on the published held-out partition; the pilot evaluation uses that partition.

Run from the repository root (CPU job, loads the full h5ad):
    python -m objective_pilot.create_pilot_subset --source <pretrain h5ad> --outdir outputs/objective_pilot/pilot_data
"""

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from bmfm_methylation.shared.data_module import _read_h5ad_robust

ap = argparse.ArgumentParser()
ap.add_argument("--source", required=True)
ap.add_argument("--ids", default="outputs/pretrain_overlap/pretrain_ids.csv.gz", help="optional cross-check")
ap.add_argument("--n_train", type=int, default=9000)
ap.add_argument("--n_valid", type=int, default=1000)
ap.add_argument("--n_test", type=int, default=1000)
ap.add_argument("--seed", type=int, default=2026)
ap.add_argument("--outdir", default="outputs/objective_pilot/pilot_data")
a = ap.parse_args()

out = Path(a.outdir)
out.mkdir(parents=True, exist_ok=True)
tag = f"pilot_{a.n_train // 1000}k_seed{a.seed}"
h5_path, ids_path, meta_path = out / f"{tag}.h5ad", out / f"{tag}_ids.csv", out / f"{tag}.json"
if h5_path.exists():
    raise SystemExit(f"Exists, not overwriting: {h5_path} (delete it manually to rebuild)")

print(f"Loading (read-only): {a.source}", flush=True)
adata = _read_h5ad_robust(a.source)
n = adata.n_obs
print(f"  {n:,} profiles x {adata.n_vars:,} CpGs", flush=True)
if "split" in adata.obs.columns:
    raise SystemExit("Source has an obs['split'] column — the published run would not have auto-split; stop")
obs_ids = adata.obs_names.astype(str).values

# Published partitions, same code as MethylationDataset's auto-split
perm = np.random.default_rng(42).permutation(n)
train_end, val_end = int(0.8 * n), int(0.9 * n)
part = np.empty(n, dtype=object)
part[perm[:train_end]] = "train"
part[perm[train_end:val_end]] = "valid"
part[perm[val_end:]] = "heldout"
counts = {k: int((part == k).sum()) for k in ("train", "valid", "heldout")}
print("  published partitions:", counts, flush=True)
if n == 169120 and counts != {"train": 135296, "valid": 16912, "heldout": 16912}:
    raise SystemExit("Partition sizes differ from the published run")

cross_checked = False
if Path(a.ids).exists():
    ids = pd.read_csv(a.ids)
    ok = (len(ids) == n and (ids["row"].values == np.arange(n)).all()
          and (ids["pretrain_split"].values == part).all()
          and (ids["sample_id"].astype(str).values == obs_ids).all())
    if not ok:
        raise SystemExit(f"Recomputed partitions / ids disagree with {a.ids}")
    cross_checked = True
    print(f"  cross-check vs {a.ids}: identical (rows, ids, partitions)", flush=True)
else:
    print(f"  ({a.ids} not found — no independent cross-check)", flush=True)

rng = np.random.default_rng(a.seed)
pick = {}
for name, src, k in [("train", "train", a.n_train), ("valid", "valid", a.n_valid), ("test", "heldout", a.n_test)]:
    pool = np.where(part == src)[0]
    pick[name] = np.sort(rng.choice(pool, size=k, replace=False))
rows = np.concatenate([pick["train"], pick["valid"], pick["test"]])
split = np.array(["train"] * a.n_train + ["valid"] * a.n_valid + ["test"] * a.n_test)
order = np.argsort(rows)  # keep source row order
rows, split = rows[order], split[order]

sub = adata[rows].copy()
sub.obs["split"] = pd.Categorical(split)
assert sub.n_vars == adata.n_vars
sub.write_h5ad(h5_path)

pd.DataFrame({"source_row": rows, "sample_id": obs_ids[rows], "split": split,
              "published_partition": part[rows]}).to_csv(ids_path, index=False)
summary = {
    "source": a.source, "n_cpgs": int(sub.n_vars), "seed": a.seed,
    "counts": {k: int(len(v)) for k, v in pick.items()},
    "drawn_from": {"train": "published train", "valid": "published valid", "test": "published held-out"},
    "published_partition_counts": counts, "cross_checked_with_ids_file": cross_checked,
    "h5ad": str(h5_path), "ids": str(ids_path),
}
meta_path.write_text(json.dumps(summary, indent=1))
print(json.dumps(summary, indent=1))
