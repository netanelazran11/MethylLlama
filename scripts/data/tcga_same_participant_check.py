"""
tcga_same_participant_check.py
==============================
Downstream TCGA profiles are named by participant (TCGA-BP-4976); the pretraining corpus
names TCGA profiles by aliquot barcode (TCGA-BP-4976-11A-01D-1303-05). Matching by accession
therefore cannot tell whether a downstream TCGA profile is in the pretraining corpus.

This script answers it by MEASUREMENT, but only for pairs that the identifiers already
designate: each downstream TCGA profile is compared with the corpus profiles of THE SAME
PARTICIPANT, at the CpGs measured in both. No nearest-neighbour search is made.

For calibration, each downstream profile is also compared with one corpus normal-tissue
profile of a DIFFERENT participant (fixed seed), so that "identical" can be read against
"another person, same kind of sample".

A pair is called the same measurement when  mean |difference| <= 0.005  and  r >= 0.9999
(the duplicate criterion of the downstream corpus is cosine >= 0.9999).

Reads rows through h5py / backed anndata; the 33 GB matrix is never loaded, torch is not
imported. Runs in minutes on a CPU node:

  python scripts/data/tcga_same_participant_check.py        (CPU node, 16 GB, about 1 h)

Writes outputs/data_audit/tcga_pairs.csv, tcga_same_measurement_ids.csv and tcga_pairs_summary.json
(the summary is shipped in results/data_audit/).
"""
import argparse
import json
import os
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

DATA = os.environ.get("METHYLLAMA_DATA", "data")          # see configs/paths.example.sh
PRE = f"{DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad"
DOWN = f"{DATA}/data_methyl_21k_h5ad/altumage_21k_3way.h5ad"
MAD_MAX, R_MIN = 0.005, 0.9999


def strings(ds):
    a = ds[:]
    return np.array([x.decode() if isinstance(x, bytes) else str(x) for x in a])


def column(grp, key):
    """A per-row string column of an h5ad obs/var group, plain or categorical."""
    item = grp[key]
    if isinstance(item, h5py.Group):
        return strings(item["categories"])[item["codes"][:]]
    return strings(item)


def row_names(grp, n):
    """The per-row string column that holds the names (the `_index` attribute may be wrong)."""
    attr = grp.attrs.get("_index", "_index")
    attr = attr.decode() if isinstance(attr, bytes) else attr
    for k in dict.fromkeys([attr, "_index", "index", *grp.keys()]):
        if k in grp and isinstance(grp[k], h5py.Dataset) and grp[k].shape == (n,) and grp[k].dtype.kind in ("S", "O", "U"):
            return strings(grp[k]), k
    raise SystemExit("no per-row name column found")


def cpg_names(grp, n):
    if "cpg_id" in grp:
        return column(grp, "cpg_id")
    return row_names(grp, n)[0]


def read_rows(X, rows, cols):
    """Rows of a dense h5py matrix, in the order given, restricted to columns `cols`."""
    rows = np.asarray(rows)
    order = np.argsort(rows)
    out = np.empty((len(rows), len(cols)), dtype=np.float32)
    for j in order:                                    # one row at a time: no large temporary
        out[j] = X[int(rows[j])][cols]
    return out


def compare(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    n = int(ok.sum())
    if n < 100:
        return {"n_cpgs": n}
    x, y = a[ok].astype(np.float64), b[ok].astype(np.float64)
    d = np.abs(x - y)
    return {"n_cpgs": n, "pearson_r": float(np.corrcoef(x, y)[0, 1]),
            "cosine": float(x @ y / (np.linalg.norm(x) * np.linalg.norm(y))),
            "mean_abs_diff": float(d.mean()), "max_abs_diff": float(d.max()),
            "frac_abs_diff_below_0.001": float((d < 0.001).mean())}


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", default=PRE)
    p.add_argument("--downstream", default=DOWN)
    p.add_argument("--outdir", default="outputs/data_audit")
    a = p.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)

    with h5py.File(a.pretrain, "r") as fp, h5py.File(a.downstream, "r") as fd:
        assert isinstance(fp["X"], h5py.Dataset) and isinstance(fd["X"], h5py.Dataset), "X must be a dense matrix"
        n_pre, c_pre = fp["X"].shape
        n_dn, c_dn = fd["X"].shape
        pre_ids, k1 = row_names(fp["obs"], n_pre)
        dn_ids, k2 = row_names(fd["obs"], n_dn)
        pre_cpg, dn_cpg = cpg_names(fp["var"], c_pre), cpg_names(fd["var"], c_dn)
        print(f"pretraining {n_pre:,} x {c_pre:,} (names: obs/{k1}) | downstream {n_dn:,} x {c_dn:,} (names: obs/{k2})", flush=True)
        assert len(set(pre_cpg)) == c_pre and len(set(dn_cpg)) == c_dn

        pos = {c: i for i, c in enumerate(pre_cpg)}
        shared = [c for c in dn_cpg if c in pos]
        col_dn = np.array([i for i, c in enumerate(dn_cpg) if c in pos])
        col_pre = np.array([pos[c] for c in shared])
        o_dn, o_pre = np.argsort(col_dn), np.argsort(col_pre)          # h5py wants increasing indices
        print(f"CpGs measured on both panels: {len(shared):,}", flush=True)
        assert len(shared) > 10000

        perm = np.random.default_rng(42).permutation(n_pre)            # the pretraining partition
        split = np.empty(n_pre, dtype=object)
        split[perm[:int(0.8 * n_pre)]] = "train"
        split[perm[int(0.8 * n_pre):int(0.9 * n_pre)]] = "valid"
        split[perm[int(0.9 * n_pre):]] = "heldout"

        pre = pd.DataFrame({"row": np.arange(n_pre), "sample_id": pre_ids, "split": split})
        pre = pre[pre.sample_id.str.startswith("TCGA-")].assign(
            participant=lambda d: d.sample_id.str[:12], sample_type=lambda d: d.sample_id.str[13:15])
        dn = pd.DataFrame({"row": np.arange(n_dn), "sample_id": dn_ids})
        dn = dn[dn.sample_id.str.startswith("TCGA-")].assign(participant=lambda d: d.sample_id.str[:12])
        print(f"TCGA: downstream {len(dn):,} profiles | corpus {len(pre):,} profiles of {pre.participant.nunique():,} participants", flush=True)

        pairs = dn.merge(pre, on="participant", suffixes=("_down", "_corpus"))
        pairs["kind"] = "same participant"
        rng = np.random.default_rng(0)
        normals = pre[pre.sample_type == "11"]
        ctrl = []
        for r in dn.itertuples():
            other = normals[normals.participant != r.participant]
            c = other.iloc[int(rng.integers(len(other)))]
            ctrl.append({"row_down": r.row, "sample_id_down": r.sample_id, "participant": r.participant,
                         "row_corpus": c.row, "sample_id_corpus": c.sample_id, "split": c.split,
                         "sample_type": c.sample_type, "kind": "different participant (calibration)"})
        pairs = pd.concat([pairs, pd.DataFrame(ctrl)], ignore_index=True)
        print(f"pairs to compare: {len(pairs):,}", flush=True)

        def fetch(X, rows, cols, order):
            u = np.unique(rows)
            m = read_rows(X, u, cols[order])
            inv = np.empty_like(order); inv[order] = np.arange(len(order))
            return dict(zip(u, m[:, inv]))                              # columns back in `shared` order
        A = fetch(fd["X"], pairs.row_down.to_numpy(), col_dn, o_dn)
        B = fetch(fp["X"], pairs.row_corpus.to_numpy(), col_pre, o_pre)

    stats = pd.DataFrame([compare(A[r.row_down], B[r.row_corpus]) for r in pairs.itertuples()])
    pairs = pd.concat([pairs, stats], axis=1)
    pairs["same_measurement"] = (pairs.mean_abs_diff <= MAD_MAX) & (pairs.pearson_r >= R_MIN)
    pairs.to_csv(out / "tcga_pairs.csv", index=False)

    same = pairs[pairs.kind == "same participant"]
    hit = same[same.same_measurement]
    summ = {"criterion": {"mean_abs_diff_max": MAD_MAX, "pearson_r_min": R_MIN}, "n_shared_cpgs": len(shared),
            "downstream_tcga_profiles": int(len(dn)),
            "with_a_corpus_profile_of_the_same_participant": int(same.sample_id_down.nunique()),
            "same_measurement_found_in_corpus": int(hit.sample_id_down.nunique()),
            "same_measurement_found_in_pretraining_train": int(hit[hit.split == "train"].sample_id_down.nunique()),
            "same_measurement_by_corpus_sample_type": hit.groupby("sample_type").sample_id_down.nunique().to_dict(),
            "calibration_pairs_called_same_measurement": int(pairs[(pairs.kind != "same participant") & pairs.same_measurement].shape[0])}
    for kind, g in pairs.groupby(["kind", "sample_type"]):
        summ[" | corpus sample type ".join(kind)] = {
            "pairs": int(len(g)),
            "pearson_r": {q: float(g.pearson_r.quantile(v)) for q, v in [("min", 0), ("median", .5), ("max", 1)]},
            "mean_abs_diff": {q: float(g.mean_abs_diff.quantile(v)) for q, v in [("min", 0), ("median", .5), ("max", 1)]}}
    hit[["sample_id_down", "sample_id_corpus", "split"]].to_csv(out / "tcga_same_measurement_ids.csv", index=False)
    (out / "tcga_pairs_summary.json").write_text(json.dumps(summ, indent=2))
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
