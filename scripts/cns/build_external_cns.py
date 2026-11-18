"""
build_external_cns.py  (CNS task, stage 6 - input of the external cohort GSE215240)
===================================================================================
Turns the two processed beta matrices that GSE215240 deposited on GEO (EPIC 937 + 450K 187 profiles) into the
models' input: one row per profile, the 49,156 model CpGs in corpus order, NaN where the array has no probe
(EPIC lacks ~2.4 % of them) or the value is missing. The matrices are streamed from GEO once; only the model CpG
rows are kept, so nothing large is stored.

The profiles come as the authors preprocessed them, not through the pipeline that built the pretraining corpus:
a real change of laboratory and preprocessing, the same for both models and the raw baselines.

Inputs  outputs/cns/external/GSE215240_labels.csv (external_labels.py: GSM, Sentrix ID, label),
        outputs/cns/data/cns_capper.h5ad (column order), the pretraining corpus h5ad (obs names only)
Output  outputs/cns/external/GSE215240.h5ad   X float32 [1,124 x 49,156] (NaN kept), var 'cpg_id' as
        cns_capper.h5ad, obs indexed by GSM: set='external', platform, methylation_class (Capper abbreviation, or
        scoring group), eval_group (empty when the v11b4 score is < 0.9), v11b4_score, nan_frac
        outputs/cns/external/build_summary.json

Checks (the script stops on a failure)
  every matrix column links to exactly one GSM of the label table and every GSM is filled once; probe coverage as
  in a coverage scan of the two matrices (EPIC 47,962, 450K 49,020); values inside [0, 1]; no GSM of the cohort in the pretraining
  corpus (expected: the series is from 2022, after the corpus).

CPU with network (24 GB, about 3 h), from the repository root:
  python scripts/cns/build_external_cns.py
"""
import argparse
import csv
import json
import os
import sys
import time
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(REPO / "scripts" / "data"))
from geo_stream import open_stream, series_dir  # noqa: E402
from tcga_same_participant_check import row_names  # noqa: E402

DATA = Path(os.environ.get("METHYLLAMA_DATA", "data"))  # configs/paths.sh
GSE = "GSE215240"
FILES = {"GPL21145": f"{GSE}_betas_preprocessed_GPL21145.txt.gz", "GPL13534": f"{GSE}_betas_preprocessed_GPL13534.txt.gz"}
COVERAGE = {"GPL21145": 47962, "GPL13534": 49020}       # coverage scan of the two GEO matrices, 2026-11-02


def fail(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--labels", default=str(REPO / "outputs/cns/external/GSE215240_labels.csv"))
    p.add_argument("--capper", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--corpus", default=f"{DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad")
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/external"))
    a = p.parse_args()
    import anndata as ad
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    S = {}

    lab = pd.read_csv(a.labels)
    cap = ad.read_h5ad(a.capper, backed="r")
    cpg = cap.var["cpg_id"].astype(str).tolist()
    col = pd.Series(np.arange(len(cpg)), index=cpg)
    X = np.full((len(lab), len(cpg)), np.nan, dtype=np.float32)
    filled = np.zeros(len(lab), bool)
    row_of = pd.Series(np.arange(len(lab)), index=lab.sentrix.astype(str))

    for plat, fname in FILES.items():
        t0 = time.time()
        reader = csv.reader(open_stream(f"{series_dir(GSE)}/suppl/{fname}", timeout=600), delimiter="\t")
        header = [h.strip('"') for h in next(reader)]
        samples = header[1:]
        missing = [s for s in samples if s not in row_of.index]
        if missing:
            fail(f"{fname}: {len(missing)} columns without a GSM (first {missing[:3]})")
        rows = row_of.loc[samples].to_numpy()
        if filled[rows].any():
            fail(f"{fname}: a profile appears twice")
        if not (lab.platform.to_numpy()[rows] == plat).all():
            fail(f"{fname}: column on the wrong platform")
        seen = 0
        for r in reader:
            j = col.get(r[0].strip('"'))
            if j is None:
                continue
            X[rows, j] = pd.to_numeric(pd.Series(r[1:]), errors="coerce").to_numpy(np.float32)
            seen += 1
        if seen != COVERAGE[plat]:
            fail(f"{fname}: {seen} model CpGs, expected {COVERAGE[plat]}")
        filled[rows] = True
        S[plat] = {"profiles": len(rows), "model_cpgs": seen, "seconds": round(time.time() - t0, 1)}
        print(plat, S[plat], flush=True)
    if not filled.all():
        fail(f"{(~filled).sum()} GSMs not found in either matrix")
    v = X[np.isfinite(X)]
    if v.min() < 0 or v.max() > 1:
        fail(f"values outside [0, 1]: {v.min()} .. {v.max()}")
    nan_frac = np.isnan(X).mean(1)
    S["nan_frac"] = {"min": float(nan_frac.min()), "median": float(np.median(nan_frac)), "max": float(nan_frac.max())}

    with h5py.File(a.corpus, "r") as f:
        ids, _ = row_names(f["obs"], f["X"].shape[0])
    in_corpus = int(lab.gsm.isin(set(map(str, ids))).sum())
    S["gsm_in_pretraining_corpus"] = in_corpus
    if in_corpus:
        fail(f"{in_corpus} profiles of the cohort are in the pretraining corpus")

    obs = pd.DataFrame({"set": "external", "platform": lab.platform.values,
                        "methylation_class": lab.methylation_class.fillna("").values,
                        "eval_group": lab.eval_group.fillna("").values,
                        "v11b4_score": lab.v11b4_score.values, "nan_frac": nan_frac}, index=lab.gsm.astype(str).values)
    A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame({"cpg_id": cpg}, index=[str(i) for i in range(len(cpg))]))
    A.write_h5ad(out / f"{GSE}.h5ad")
    S.update({"profiles": int(len(lab)), "scored": int((obs.eval_group != "").sum()),
              "eval_groups": int(obs.eval_group[obs.eval_group != ""].nunique())})
    json.dump(S, open(out / "build_summary.json", "w"), indent=1)
    print(json.dumps(S), flush=True)


if __name__ == "__main__":
    main()
