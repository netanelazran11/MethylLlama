"""
build_cns_dataset.py  (CNS task, stage 1)
==========================================
Builds the shared input of the Capper CNS-tumour task for MethylLlama, MethylGPT and the raw-data baselines.

Inputs (read-only)
  corpus h5ad      methylgpt_pretrain_type3.h5ad - the profiles exactly as both models saw them in pretraining
  samples          resources/cns/capper_samples.csv (GEO metadata; labels normalised, 91 reference classes)
  preflight        outputs/cns/preflight/{capper_in_corpus.csv, capper_duplicate_pairs.csv,
                   corpus_capper_betas.npz}  (pretraining split, NaN fractions, duplicate pairs, float16 copy)
  probe list       MethylGPT probe_ids_type3.csv (= corpus columns = MethylLlama vocab order, verified 2026-10-06)

Outputs (outputs/cns/data/; hold beta values and are not part of the release)
  cns_capper.h5ad          X float32 [3,902 x 49,156] (NaN kept), var 'cpg_id' in corpus order,
                           obs: gsm, set, methylation_class, prefix_group, material, pretrain_split, nan_frac,
                           val_leak_dup (validation profile with a near-duplicate in the reference)
  folds.csv                reference only: gsm, repeat (0-4), fold (0-4) - StratifiedKFold by class, shuffle, seed=repeat
  fewshot.csv              repeat, fold, k (3, 5), seed (0-4), gsm - k profiles per class from that fold's training part
  methylgpt_cns.parquet    sample_id, set, methylation_class, data (49,156 float32, NaN kept; MethylGPT maps NaN -> -2)
  build_summary.json       every check below with its result

Design decisions (CNS plan, 2026-10-25)
  - Reference control-tissue pairs (CONTR HEMI 11 of 13, CEBM 3 of 8) are NOT grouped into one fold: they are normal
    tissue of one region, not duplicates (no pair has cosine >= 0.9999); grouping would empty the class from training.
  - The 6 validation profiles that have a near-duplicate in the reference are flagged and excluded from the
    validation analysis (train on reference -> test on validation). The one pair inside the validation is not leakage.

Checks (the script stops on a failure)
  corpus columns == probe list; every Capper GSM resolves to exactly one corpus row; float32 rows equal the float16
  npz within 1e-3 with identical NaN positions; NaN fraction equals the preflight value; 91 reference classes,
  validation classes a subset; every fold: each class >= 1 test and >= 6 training profiles; folds partition the
  reference in every repeat; few-shot draws come only from the training part.

CPU only (24 GB, about 1 h), from the repository root:
  python scripts/cns/build_cns_dataset.py
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

import h5py
import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts" / "data"))
from tcga_same_participant_check import cpg_names, row_names  # noqa: E402

DATA = Path(os.environ.get("METHYLLAMA_DATA", "data"))            # configs/paths.sh
GPT_DATA = Path(os.environ.get("METHYLGPT_DATA", "data/methylgpt"))  # configs/paths.sh
PRE = REPO / "outputs/cns/preflight"


def norm_label(s):
    return re.sub(r"\s+", " ", str(s)).strip()


def fail(msg):
    raise SystemExit(f"CHECK FAILED: {msg}")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--corpus", default=f"{DATA}/data_methyl_pretrain_type3_h5ad/methylgpt_pretrain_type3.h5ad")
    p.add_argument("--probes", default=f"{GPT_DATA}/cpg_ids/probe_ids_type3.csv")
    p.add_argument("--samples", default=str(REPO / "resources/cns/capper_samples.csv"))
    p.add_argument("--outdir", default=str(REPO / "outputs/cns/data"))
    p.add_argument("--preflight", default=str(PRE))
    p.add_argument("--repeats", type=int, default=5)
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--shots", default="3,5")
    p.add_argument("--shot_seeds", type=int, default=5)
    a = p.parse_args()
    out = Path(a.outdir)
    out.mkdir(parents=True, exist_ok=True)
    S = {}

    # ---- labels -------------------------------------------------------------------------------------
    s = pd.read_csv(a.samples)
    s["methylation_class"] = s.methylation_class.map(norm_label)
    ref_classes = set(s[s.set == "reference"].methylation_class)
    val_classes = set(s[s.set == "validation"].methylation_class)
    if len(ref_classes) != 91:
        fail(f"{len(ref_classes)} reference classes, expected 91")
    if not val_classes <= ref_classes:
        fail(f"validation classes not in reference: {sorted(val_classes - ref_classes)}")
    S["classes"] = {"reference": len(ref_classes), "validation": len(val_classes)}

    PREF = Path(a.preflight)
    pre = pd.read_csv(PREF / "capper_in_corpus.csv")[["gsm", "pretrain_split", "corpus_nan_frac"]]
    dup = pd.read_csv(PREF / "capper_duplicate_pairs.csv")
    cross = dup[dup.set_a != dup.set_b]
    leak = set(np.where(cross.set_a == "validation", cross.gsm_a, cross.gsm_b))
    S["validation_profiles_with_reference_duplicate"] = len(leak)

    # ---- corpus rows (float32) --------------------------------------------------------------------
    probes = pd.read_csv(a.probes)["illumina_probe_id"].astype(str).tolist()
    with h5py.File(a.corpus, "r") as f:
        n_pre, c_pre = f["X"].shape
        cpg = [str(x) for x in cpg_names(f["var"], c_pre)]
        if cpg != probes:
            fail("corpus columns differ from the probe list (order or names)")
        ids, key = row_names(f["obs"], n_pre)
        pos = pd.Series(np.arange(n_pre), index=ids)
        if pos.index.duplicated().any():
            fail("duplicate sample ids in the corpus")
        s = s[s.gsm.isin(pos.index)].copy()            # 3 reference GSMs are not in the corpus
        s["corpus_row"] = s.gsm.map(pos).astype(int)
        X = np.empty((len(s), c_pre), dtype=np.float32)
        order = np.argsort(s.corpus_row.to_numpy())
        rows = s.corpus_row.to_numpy()
        for j in order:                                 # one row at a time, ascending: no large temporary
            X[j] = f["X"][int(rows[j])].astype(np.float32)
    S["profiles"] = s.set.value_counts().to_dict()
    S["corpus_obs_name_key"] = key

    # ---- integrity checks ---------------------------------------------------------------------------
    z = np.load(PREF / "corpus_capper_betas.npz", allow_pickle=True)
    zi = pd.Series(np.arange(len(z["gsm"])), index=[str(g) for g in z["gsm"]])
    if [str(c) for c in z["cpg"]] != cpg:
        fail("npz CpG order differs from the corpus")
    Z = z["betas"][zi.loc[s.gsm].to_numpy()].astype(np.float32)
    nan_x, nan_z = np.isnan(X), np.isnan(Z)
    if not (nan_x == nan_z).all():
        fail("NaN positions differ between the float32 rows and the npz")
    dmax = float(np.nanmax(np.abs(X - Z)))
    S["max_abs_diff_float32_vs_npz_float16"] = dmax
    if dmax > 1e-3:
        fail(f"float32 rows differ from the npz by {dmax}")
    s["nan_frac"] = nan_x.mean(1)
    s = s.merge(pre, on="gsm", how="left")
    dn = float(np.abs(s.nan_frac - s.corpus_nan_frac).max())
    S["max_abs_diff_nan_frac_vs_preflight"] = dn
    if dn > 1e-9 or s.pretrain_split.isna().any():
        fail("NaN fraction or pretraining split does not match the preflight table")
    finite = X[~nan_x]
    S["beta_range"] = [float(finite.min()), float(finite.max())]
    if finite.min() < 0 or finite.max() > 1:
        fail("values outside [0, 1]")
    s["val_leak_dup"] = s.gsm.isin(leak)

    # ---- folds (reference only) -------------------------------------------------------------------------
    from sklearn.model_selection import StratifiedKFold
    ref = s[s.set == "reference"].reset_index(drop=True)
    y = ref.methylation_class.to_numpy()
    folds, shots = [], []
    ks = [int(k) for k in a.shots.split(",")]
    min_train, min_test = 10**9, 10**9
    for r in range(a.repeats):
        skf = StratifiedKFold(n_splits=a.folds, shuffle=True, random_state=r)
        seen = np.zeros(len(ref), dtype=int)
        for fo, (tr, te) in enumerate(skf.split(np.zeros(len(y)), y)):
            seen[te] += 1
            ctr, cte = pd.Series(y[tr]).value_counts(), pd.Series(y[te]).value_counts()
            if len(ctr) != 91 or len(cte) != 91:
                fail(f"repeat {r} fold {fo}: a class is missing from train ({len(ctr)}) or test ({len(cte)})")
            min_train, min_test = min(min_train, ctr.min()), min(min_test, cte.min())
            folds += [(g, r, fo) for g in ref.gsm.iloc[te]]
            trd = ref.iloc[tr]
            for k in ks:
                for sd in range(a.shot_seeds):
                    rng = np.random.default_rng(1000 * r + 100 * fo + 10 * k + sd)
                    for c, grp in trd.groupby("methylation_class"):
                        pick = rng.choice(grp.gsm.to_numpy(), size=min(k, len(grp)), replace=False)
                        shots += [(r, fo, k, sd, g) for g in pick]
        if not (seen == 1).all():
            fail(f"repeat {r}: folds do not partition the reference")
    if min_train < 6:
        fail(f"a training fold has only {min_train} profiles of some class")
    folds = pd.DataFrame(folds, columns=["gsm", "repeat", "fold"])
    shots = pd.DataFrame(shots, columns=["repeat", "fold", "k", "seed", "gsm"])
    chk = shots.merge(folds.rename(columns={"fold": "test_fold"}), on=["repeat", "gsm"], how="left")
    if chk.test_fold.isna().any() or (chk.test_fold == chk.fold).any():
        fail("a few-shot draw contains a test profile of its own fold")
    S["folds"] = {"repeats": a.repeats, "folds": a.folds, "min_train_per_class": int(min_train),
                  "min_test_per_class": int(min_test)}
    S["fewshot"] = {"k": ks, "seeds": a.shot_seeds, "rows": len(shots),
                    "profiles_per_draw": {str(k): int(shots[shots.k == k].groupby(["repeat", "fold", "seed"]).size().iloc[0])
                                          for k in ks}}

    # ---- write ------------------------------------------------------------------------------------------
    import anndata as ad
    obs = s[["gsm", "set", "methylation_class", "prefix_group", "material", "pretrain_split", "nan_frac",
             "val_leak_dup"]].set_index("gsm")
    obs.index.name = None
    A = ad.AnnData(X=X, obs=obs, var=pd.DataFrame({"cpg_id": cpg}, index=[str(i) for i in range(len(cpg))]))
    A.write_h5ad(out / "cns_capper.h5ad")
    folds.to_csv(out / "folds.csv", index=False)
    shots.to_csv(out / "fewshot.csv", index=False)
    pq = pd.DataFrame({"sample_id": s.gsm.to_numpy(), "set": s.set.to_numpy(),
                       "methylation_class": s.methylation_class.to_numpy(), "data": list(X)})
    pq.to_parquet(out / "methylgpt_cns.parquet", index=False)
    S["written"] = sorted(x.name for x in out.iterdir())
    S["status"] = "OK"
    json.dump(S, open(out / "build_summary.json", "w"), indent=1, default=str)
    print(json.dumps(S, indent=1, default=str))


if __name__ == "__main__":
    main()
