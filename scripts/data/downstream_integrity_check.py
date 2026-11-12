"""
downstream_integrity_check.py
=============================
Is every downstream (AltuMAge) profile a valid methylation profile with correctly labelled CpGs?

Trigger: for E-GEOD-56105 the downstream values do not correlate with the same accessions in the
pretraining corpus (r ~ 0.002), and its CLS embeddings sit far from every other blood study.

For every downstream profile, on the CpGs shared with the pretraining panel:
  r_ref       Pearson r with a reference profile (per-CpG median of REF_N random corpus profiles).
              Real methylation profiles of any tissue correlate strongly with it; a profile whose
              CpG labels are scrambled does not.
  r_copy      where the same accession exists in the corpus: r and mean |difference| between copies
  r_sorted    the same for the two SORTED value vectors: ~1 when both copies hold the same values,
              whatever their order (a scramble keeps the values and loses the order)
For E-GEOD-56105 (or --study), the column mapping is recovered: for each downstream CpG, the corpus
CpG whose values correlate best across the study's profiles. If the downstream copy is a column
permutation of the corpus copy, nearly every downstream CpG has one partner with r close to 1,
and that partner is a different CpG.

Streams the corpus matrix in row blocks; loads the downstream matrix (~1 GB). No torch.

  python scripts/data/downstream_integrity_check.py         (CPU node, 32 GB, about 3 h)

Writes outputs/data_audit/downstream_integrity_{per_profile.csv,by_study.csv,summary.json}
and, for the study examined, downstream_integrity_column_map.csv (by_study.csv and summary.json are
shipped in results/data_audit/). Reads outputs/pretrain_overlap/overlap_by_id.csv
(scripts/data/pretrain_overlap_by_id.py) when the downstream h5ad has no study column.
"""
import argparse
import json
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
from tcga_same_participant_check import DOWN, PRE, column, cpg_names, row_names    # noqa: E402

REF_N = 5000


def pearson_rows(A, b):
    """r between each row of A and vector b, using positions finite in both."""
    out = np.full(len(A), np.nan)
    for i, a in enumerate(A):
        ok = np.isfinite(a) & np.isfinite(b)
        if ok.sum() >= 1000:
            out[i] = np.corrcoef(a[ok], b[ok])[0, 1]
    return out


def pair(a, b):
    ok = np.isfinite(a) & np.isfinite(b)
    if ok.sum() < 1000:
        return np.nan, np.nan, np.nan
    x, y = a[ok].astype(np.float64), b[ok].astype(np.float64)
    xs, ys = np.sort(x), np.sort(y)
    return np.corrcoef(x, y)[0, 1], np.abs(x - y).mean(), np.corrcoef(xs, ys)[0, 1]


def colstd(M, min_measured=20):
    """Column-standardise; columns with fewer than `min_measured` values or no variance become 0
    (correlation 0 with everything) and are reported as unusable."""
    ok = np.isfinite(M)
    cnt = ok.sum(0)
    mu = np.where(cnt > 0, np.where(ok, M, 0).sum(0) / np.maximum(cnt, 1), 0.0)
    M = np.where(ok, M, mu) - mu
    sd = M.std(0)
    usable = (cnt >= min_measured) & (sd > 0)
    return np.where(usable, M / np.where(usable, sd, 1.0), 0.0), usable


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--pretrain", default=PRE)
    p.add_argument("--downstream", default=DOWN)
    p.add_argument("--study", default="E-GEOD-56105")
    p.add_argument("--overlap", default="outputs/pretrain_overlap/overlap_by_id.csv")
    p.add_argument("--outdir", default="outputs/data_audit")
    p.add_argument("--block", type=int, default=4096)
    a = p.parse_args()
    out = Path(a.outdir); out.mkdir(parents=True, exist_ok=True)
    names = ["downstream_integrity_per_profile.csv", "downstream_integrity_by_study.csv",
             "downstream_integrity_summary.json", "downstream_integrity_column_map.csv"]
    for f in names:
        if (out / f).exists():
            raise SystemExit(f"{out / f} exists -- nothing is overwritten")

    with h5py.File(a.downstream, "r") as fd:
        n_dn, c_dn = fd["X"].shape
        dn_ids, _ = row_names(fd["obs"], n_dn)
        dn_cpg = cpg_names(fd["var"], c_dn)
        if "dataset" in fd["obs"]:
            study = column(fd["obs"], "dataset")
        else:                                                         # study labels from the overlap table
            ov = pd.read_csv(a.overlap, dtype={"sample_id": str}).set_index("sample_id")
            study = ov.dataset.reindex(dn_ids).fillna("?").to_numpy()
        D = fd["X"][:].astype(np.float32)
    with h5py.File(a.pretrain, "r") as fp:
        X = fp["X"]; n_pre, c_pre = X.shape
        pre_ids, _ = row_names(fp["obs"], n_pre)
        pre_cpg = cpg_names(fp["var"], c_pre)
        pos = {c: i for i, c in enumerate(pre_cpg)}
        shared = np.array([c in pos for c in dn_cpg])
        col_pre = np.array([pos[c] for c in dn_cpg[shared]])
        print(f"downstream {n_dn:,} x {c_dn:,} | corpus {n_pre:,} x {c_pre:,} | shared CpGs {shared.sum():,}", flush=True)

        pre_row = pd.Series(np.arange(n_pre), index=pre_ids)
        copy_row = pre_row.reindex(dn_ids)
        ref_rows = np.sort(np.random.default_rng(0).choice(n_pre, size=min(REF_N, n_pre), replace=False))
        need = np.union1d(ref_rows, copy_row.dropna().astype(int).to_numpy())
        where = {int(r): i for i, r in enumerate(need)}
        Cfull_rows = set(copy_row[study == a.study].dropna().astype(int))      # full width for the column map
        C = np.empty((len(need), int(shared.sum())), dtype=np.float32)
        Cstudy = {}
        for s in range(0, n_pre, a.block):
            sel = need[(need >= s) & (need < s + a.block)]
            if not len(sel):
                continue
            blk = X[s:s + a.block]
            for r in sel:
                row = blk[r - s]
                C[where[int(r)]] = row[col_pre]
                if int(r) in Cfull_rows:
                    Cstudy[int(r)] = row.astype(np.float32)
            if (s // a.block) % 5 == 0:
                print(f"   corpus rows read up to {min(s + a.block, n_pre):,}", flush=True)

    Ds = D[:, shared]
    ref = np.nanmedian(C[[where[int(r)] for r in ref_rows]], axis=0)
    per = pd.DataFrame({"sample_id": dn_ids, "study": study, "r_ref": pearson_rows(Ds, ref),
                        "measured_shared": np.isfinite(Ds).sum(1)})
    rc = [pair(Ds[i], C[where[int(r)]]) if np.isfinite(r) else (np.nan,) * 3 for i, r in enumerate(copy_row.to_numpy())]
    per[["r_copy", "mad_copy", "r_sorted_copy"]] = rc
    per["corpus_copy_r_ref"] = [pearson_rows(C[[where[int(r)]]], ref)[0] if np.isfinite(r) else np.nan
                                for r in copy_row.to_numpy()]
    per.to_csv(out / names[0], index=False)
    bs = per.groupby("study").agg(n=("sample_id", "size"), r_ref_median=("r_ref", "median"), r_ref_min=("r_ref", "min"),
                                  with_copy=("r_copy", "count"), r_copy_median=("r_copy", "median"),
                                  r_sorted_copy_median=("r_sorted_copy", "median"),
                                  corpus_copy_r_ref_median=("corpus_copy_r_ref", "median")).sort_values("r_ref_median")
    bs.to_csv(out / names[1])

    summ = {"n_shared_cpgs": int(shared.sum()), "reference_profiles": int(len(ref_rows)),
            "r_ref_all_profiles": {q: float(per.r_ref.quantile(v)) for q, v in [("p01", .01), ("p05", .05), ("median", .5)]},
            "profiles_r_ref_below_0.8": int((per.r_ref < 0.8).sum()),
            "studies_with_median_r_ref_below_0.8": bs[bs.r_ref_median < 0.8].index.tolist(),
            "profiles_with_corpus_copy": int(per.r_copy.notna().sum()),
            "copies_r_below_0.9": int((per.r_copy < 0.9).sum()),
            "copies_r_below_0.9_by_study": per[per.r_copy < 0.9].study.value_counts().to_dict()}

    # column map for the study
    st = (study == a.study) & copy_row.notna().to_numpy()
    if st.sum() >= 20:
        rows = copy_row[st].astype(int).to_numpy()
        A_, a_ok = colstd(D[st])                                     # n x c_dn
        B_, b_ok = colstd(np.stack([Cstudy[int(r)] for r in rows]))  # n x c_pre
        n = A_.shape[0]
        best, best_r, same_r = np.empty(c_dn, int), np.empty(c_dn), np.full(c_dn, np.nan)
        for j0 in range(0, c_dn, 2000):
            R = A_[:, j0:j0 + 2000].T @ B_ / n
            R[:, ~b_ok] = -np.inf
            best[j0:j0 + 2000] = R.argmax(1)
            best_r[j0:j0 + 2000] = R.max(1)
            for k, j in enumerate(range(j0, min(j0 + 2000, c_dn))):
                if dn_cpg[j] in pos:
                    same_r[j] = R[k, pos[dn_cpg[j]]]
        cm = pd.DataFrame({"downstream_cpg": dn_cpg, "best_corpus_cpg": np.asarray(pre_cpg)[best], "best_r": best_r,
                           "r_with_same_named_cpg": same_r, "usable": a_ok})
        cm.loc[~cm.usable, ["best_corpus_cpg", "best_r", "r_with_same_named_cpg"]] = np.nan
        assert np.isfinite(cm.best_r[cm.usable]).all()
        cm.to_csv(out / names[3], index=False)
        named = cm.r_with_same_named_cpg.notna()
        summ["column_map"] = {
            "study": a.study, "profiles_used": int(n),
            "usable_downstream_cpgs": int(a_ok.sum()), "usable_corpus_cpgs": int(b_ok.sum()),
            "downstream_cpgs_with_best_r_above_0.95": int((cm.best_r > 0.95).sum()),
            "downstream_cpgs_with_best_r_above_0.8": int((cm.best_r > 0.8).sum()),
            "downstream_cpgs": int(c_dn),
            "best_partner_is_the_same_named_cpg": int((cm.downstream_cpg == cm.best_corpus_cpg).sum()),
            "median_r_with_same_named_cpg": float(cm.r_with_same_named_cpg[named].median()),
            "median_best_r": float(cm.best_r[cm.usable].median()),
            "median_best_r_for_reference": "a random column pair of ~280 samples gives |r| around 0.2-0.25 by chance",
            "distinct_best_partners": int(cm.best_corpus_cpg.nunique())}
    (out / names[2]).write_text(json.dumps(summ, indent=2))
    print(bs.head(8).round(4).to_string())
    print(json.dumps(summ, indent=1))


if __name__ == "__main__":
    main()
