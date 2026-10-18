"""
verify_elasticnet_test_set.py
================================
One-off check: does elasticnet_age.py's live re-derivation of the test split
(h5ad split column + age-outlier filter + duplicate_pairs.csv exclusion)
produce the *exact same set of GSM IDs* as the official fixed test set
(outputs/kfold_splits/test_ids.npy) used to evaluate MethylLlama and
MethylGPT?

Deliberately does NOT import bmfm_methylation (that pulls in torch/lightning/
wandb/bmfm_targets and can OOM a memory-capped login node on its own). The
dedup logic below is a verbatim copy of
bmfm_methylation.shared.data_module._compute_dedup_exclusions -- keep the two
in sync if that function ever changes.

Opens the h5ad in backed="r" mode so the methylation matrix (several GB) is
never loaded -- only .obs metadata is read.

Usage:
    python scripts/baselines/verify_elasticnet_test_set.py
"""

import argparse
from collections import defaultdict, deque

import anndata
import numpy as np
import pandas as pd

import os
DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh
REPO = os.environ.get("METHYLLAMA_REPO", ".")             # scripts run from the repository root
METHYLGPT = os.environ.get("METHYLGPT_DIR", "MethylGPT")  # MethylGPT pipeline (comparison only)


def compute_dedup_exclusions(pairs_csv: str) -> set:
    """Verbatim copy of bmfm_methylation.shared.data_module._compute_dedup_exclusions."""
    df = pd.read_csv(pairs_csv)
    if df.empty or "id_a" not in df.columns or "id_b" not in df.columns:
        return set()
    adj: dict = defaultdict(set)
    all_nodes: set = set()
    for _, row in df.iterrows():
        a, b = str(row["id_a"]), str(row["id_b"])
        adj[a].add(b)
        adj[b].add(a)
        all_nodes.add(a)
        all_nodes.add(b)
    visited: set = set()
    to_exclude: set = set()
    for start in sorted(all_nodes):
        if start in visited:
            continue
        component: list = []
        queue: deque = deque([start])
        while queue:
            node = queue.popleft()
            if node in visited:
                continue
            visited.add(node)
            component.append(node)
            for neighbor in adj[node]:
                if neighbor not in visited:
                    queue.append(neighbor)
        component.sort()
        to_exclude.update(component[1:])
    print(f"Dedup: {len(df)} pairs -> {len(all_nodes)} unique samples -> "
          f"keeping {len(all_nodes) - len(to_exclude)}, excluding {len(to_exclude)}")
    return to_exclude


def parse_args():
    p = argparse.ArgumentParser()
    p.add_argument("--h5ad", default=DATA + "/"
                                       "data_methyl_21k_h5ad/altumage_21k_3way.h5ad")
    p.add_argument("--test_ids", default="outputs/kfold_splits/test_ids.npy")
    p.add_argument("--duplicate_pairs_csv", default="dataset_fingerprint_outputs/duplicate_pairs.csv")
    return p.parse_args()


def main():
    a = parse_args()

    print(f"[1/4] Loading official test_ids: {a.test_ids}")
    official = set(np.load(a.test_ids, allow_pickle=True).astype(str))
    print(f"      n = {len(official)}")

    print(f"[2/4] Opening h5ad in backed mode (metadata only): {a.h5ad}")
    adata = anndata.read_h5ad(a.h5ad, backed="r")

    print("[3/4] Applying age-outlier filter (age<0 or age>120 removed) + dedup exclusion")
    age = adata.obs["age"].astype(float)
    keep_age = (age >= 0) & (age <= 120)
    exclude = compute_dedup_exclusions(a.duplicate_pairs_csv)
    keep_dedup = ~adata.obs_names.isin(exclude)
    obs = adata.obs[keep_age.values & keep_dedup]
    elasticnet_test = set(obs.index[obs["split"] == "test"].tolist())

    print("[4/4] Comparing sets")
    print(f"      official n:   {len(official)}")
    print(f"      elasticnet n: {len(elasticnet_test)}")
    identical = official == elasticnet_test
    print(f"      IDENTICAL SETS: {identical}")
    print(f"      only in official:   {len(official - elasticnet_test)}")
    print(f"      only in elasticnet: {len(elasticnet_test - official)}")
    if not identical:
        print(f"      examples only in official:   {sorted(official - elasticnet_test)[:5]}")
        print(f"      examples only in elasticnet: {sorted(elasticnet_test - official)[:5]}")


if __name__ == "__main__":
    main()
