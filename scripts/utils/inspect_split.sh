#!/bin/bash -l
#SBATCH --job-name=inspect-split
#SBATCH --partition=glacier
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G
#SBATCH --time=0:10:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/inspect_split_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-smoke/inspect_split_%j.err

H5AD="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"

source /etc/profile.d/modules.sh 2>/dev/null || true
cd /sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl
source bmfm_methyl_env/bin/activate

python3 - <<'PY'
import h5py, os
import numpy as np

H5AD = os.environ.get("H5AD", "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad")

with h5py.File(H5AD, "r") as f:
    n_obs, n_var = f["X"].shape
    print(f"\n{'='*60}")
    print(f"Dataset shape: {n_obs} samples × {n_var} CpGs")
    print(f"{'='*60}")

    print(f"\n--- obs columns (sample metadata) ---")
    for key in sorted(f["obs"].keys()):
        item = f["obs"][key]
        if isinstance(item, h5py.Dataset):
            print(f"  {key}: shape={item.shape}, dtype={item.dtype}")
            if item.shape == (n_obs,):
                sample = item[:5]
                if item.dtype.kind in ("S", "O"):
                    sample = sample.astype(str)
                print(f"    first 5: {sample.tolist()}")
        elif isinstance(item, h5py.Group) and "codes" in item and "categories" in item:
            cats = item["categories"][:].astype(str)
            codes = item["codes"][:]
            print(f"  {key}: categorical, {len(cats)} categories, shape=({n_obs},)")
            print(f"    categories: {cats[:10].tolist()}{'...' if len(cats)>10 else ''}")
            # Value counts
            unique, counts = np.unique(codes[codes >= 0], return_counts=True)
            top = sorted(zip(counts, unique), reverse=True)[:8]
            print(f"    top values: { {cats[u]: int(c) for c,u in top} }")

    # Check split column specifically
    print(f"\n--- split column check ---")
    if "split" in f["obs"]:
        item = f["obs"]["split"]
        if isinstance(item, h5py.Dataset):
            vals = item[:].astype(str)
        elif isinstance(item, h5py.Group):
            cats = item["categories"][:].astype(str)
            codes = item["codes"][:]
            vals = np.where(codes >= 0, cats[np.clip(codes, 0, len(cats)-1)], "")
        unique, counts = np.unique(vals, return_counts=True)
        print(f"  split values: { {str(u): int(c) for u,c in zip(unique, counts)} }")
    else:
        print("  No 'split' column found — will use auto 80/10/10 random split")

print(f"\n{'='*60}")
PY
