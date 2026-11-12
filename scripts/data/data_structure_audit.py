"""
data_structure_audit.py
=======================
What is a "sample", an "individual", a "tissue" and a "study" in the two corpora, and
which of them are shared between data partitions.

Reads only identifier and annotation tables (no methylation matrix):
  outputs/pretrain_overlap/pretrain_ids.csv.gz      row -> sample id -> pretraining partition
  outputs/pretrain_overlap/overlap_by_id.csv        downstream sample -> study, tissue, partitions
                                                    (both from scripts/data/pretrain_overlap_by_id.py)
  ${METHYLLAMA_DATA}/pretrain_metadata.csv.gz       annotations that accompany the pretraining corpus
                                                    (GSM_ID, PATIENT_ID, tissue, dataset, ...)
  results/pretrained_cls/metadata.csv               downstream age
  outputs/kfold_splits/                             the benchmark partition (scripts/data/create_kfold_splits.py)

Individuals are identified only where an identifier exists:
  GEO            (dataset, PATIENT_ID) from the compiled annotations
  TCGA / TARGET  participant part of the barcode
  HipSci         donor code of the cell-line name
Everything reported about individuals is therefore a LOWER BOUND.

Writes outputs/data_audit/{audit.json, *.csv}. Changes nothing else. audit.json and
downstream_exposure_per_profile.csv (input of the paired-bootstrap sensitivity analyses) are shipped
in results/data_audit/.

Usage:  python scripts/data/data_structure_audit.py      (CPU, a few minutes)
"""
import json
import os
import re
from pathlib import Path

import numpy as np
import pandas as pd

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "outputs/data_audit"
DATA = Path(os.environ.get("METHYLLAMA_DATA", "data"))          # see configs/paths.example.sh
MAX_GROUP = 20      # a (dataset, PATIENT_ID) key shared by more samples than this is treated as a label, not a person


def individual_key(sample_id, dataset, patient_id):
    if isinstance(sample_id, str):
        if sample_id.startswith("TCGA-"):
            return "TCGA|" + sample_id[:12]
        if sample_id.startswith("TARGET-"):
            return "TARGET|" + sample_id[:16]
        if sample_id.startswith("HPSI"):
            return "HPSI|" + sample_id.split("-")[-1].split("_")[0]
    if isinstance(patient_id, str) and isinstance(dataset, str):
        return f"{dataset}|{patient_id}"
    return None


def norm_study(s):
    """E-GEOD-12345 and GSE12345 are the same GEO series."""
    m = re.match(r"^E-GEOD-(\d+)$", s)
    return f"GSE{m.group(1)}" if m else s


def crossing(df, key, split):
    """Groups of `key` with >1 sample, and how many of them span more than one value of `split`."""
    g = df.dropna(subset=[key]).groupby(key)
    size, nsplit = g.size(), g[split].nunique()
    multi = size[size > 1].index
    span = nsplit[multi][nsplit[multi] > 1].index
    return {"groups": int(len(size)), "samples": int(size.sum()),
            "groups_with_more_than_one_sample": int(len(multi)),
            "samples_in_such_groups": int(size[multi].sum()),
            "groups_spanning_partitions": int(len(span)),
            "samples_in_groups_spanning_partitions": int(size[span].sum())}, span


def main():
    OUT.mkdir(parents=True, exist_ok=True)
    res = {}
    meta = pd.read_csv(DATA / "pretrain_metadata.csv.gz", low_memory=False, dtype={"PATIENT_ID": str})
    ids = pd.read_csv(REPO / "outputs/pretrain_overlap/pretrain_ids.csv.gz")
    down = pd.read_csv(REPO / "outputs/pretrain_overlap/overlap_by_id.csv")
    age = pd.read_csv(REPO / "results/pretrained_cls/metadata.csv")[["sample_id", "age"]]

    # ───────────────────────── pretraining corpus ─────────────────────────
    pre = ids.merge(meta, left_on="sample_id", right_on="GSM_ID", how="left")
    assert len(pre) == len(ids) == 169120 and pre.sample_id.is_unique
    pre["source"] = pre.sample_id.str.extract(r"^([A-Za-z]+)")[0]
    pre.loc[~pre.source.isin(["GSM", "TCGA", "TARGET", "TWPID", "ENCSR", "HPSI", "X"]), "source"] = "other"
    pre["annotated"] = pre.GSM_ID.notna()
    pre["individual"] = [individual_key(s, d, p) for s, d, p in zip(pre.sample_id, pre.dataset, pre.PATIENT_ID)]
    gsize = pre.groupby("individual").size()
    label_like = set(gsize[(gsize > MAX_GROUP) & ~gsize.index.str.startswith(("TCGA|", "TARGET|", "HPSI|"))].index)
    pre["individual_reliable"] = pre.individual.where(~pre.individual.isin(label_like))

    res["pretraining"] = p = {}
    p["n_profiles"] = len(pre)
    p["unique_sample_ids"] = int(pre.sample_id.nunique())
    p["partition"] = pre.pretrain_split.value_counts().to_dict()
    p["source_by_id_prefix"] = pre.source.value_counts().to_dict()
    p["annotation_coverage"] = {
        "profiles_with_any_annotation": int(pre.annotated.sum()),
        "with_tissue": int(pre.tissue.notna().sum()), "with_study": int(pre.dataset.notna().sum()),
        "with_age": int(pre.age.notna().sum()), "with_sex": int(pre.sex.notna().sum()),
        "with_patient_id": int(pre.PATIENT_ID.notna().sum()),
        "with_individual_identifier_any_source": int(pre.individual.notna().sum()),
        "with_reliable_individual_identifier": int(pre.individual_reliable.notna().sum())}

    t = pd.crosstab(pre.tissue.fillna("(not annotated)"), pre.pretrain_split)
    t["total"] = t.sum(1); t = t.sort_values("total", ascending=False)
    t.to_csv(OUT / "pretraining_tissue_by_partition.csv")
    ann = pre[pre.tissue.notna()]
    p["tissue"] = {"categories": int(ann.tissue.nunique()), "profiles_annotated": int(len(ann)),
                   "counts": ann.tissue.value_counts().to_dict(),
                   "categories_in_heldout": int(ann[ann.pretrain_split == "heldout"].tissue.nunique()),
                   "heldout_categories_absent_from_train": sorted(
                       set(ann[ann.pretrain_split == "heldout"].tissue) - set(ann[ann.pretrain_split == "train"].tissue)),
                   "in_vitro_profiles": int((pre.vivo_vitro == "in_vitro").sum()),
                   "cell_line_profiles": int(pre.cell_line.notna().sum())}

    st = pre[pre.dataset.notna()]
    s = pd.crosstab(st.dataset, st.pretrain_split)
    s["total"] = s.sum(1); s.sort_values("total", ascending=False).to_csv(OUT / "pretraining_study_by_partition.csv")
    ho = st[st.pretrain_split == "heldout"]
    in_train = set(st[st.pretrain_split == "train"].dataset)
    p["study"] = {"studies": int(st.dataset.nunique()), "profiles_annotated": int(len(st)),
                  "studies_with_heldout_profiles": int(ho.dataset.nunique()),
                  "heldout_profiles_annotated": int(len(ho)),
                  "heldout_profiles_whose_study_is_in_train": int(ho.dataset.isin(in_train).sum()),
                  "heldout_profiles_whose_study_is_not_in_train": int((~ho.dataset.isin(in_train)).sum()),
                  "studies_present_in_all_three_partitions": int(((s.train > 0) & (s.valid > 0) & (s.heldout > 0)).sum()),
                  "profiles_per_study_median": float(s.total.median()), "profiles_per_study_max": int(s.total.max())}

    p["individual"] = {}
    for name, col in [("all_identifiers", "individual"), ("reliable_identifiers", "individual_reliable")]:
        d, span = crossing(pre, col, "pretrain_split")
        k = pre[pre[col].isin(span)]
        tr = set(k[k.pretrain_split == "train"][col])
        hk = pre[(pre.pretrain_split == "heldout") & pre[col].notna()]
        d["heldout_profiles_with_identifier"] = int(len(hk))
        d["heldout_profiles_whose_individual_is_also_in_train"] = int(hk[col].isin(tr).sum())
        p["individual"][name] = d
    p["individual"]["keys_treated_as_labels"] = sorted(label_like)
    by_src = {}
    for src, sub in pre[pre.individual_reliable.notna()].groupby("source"):
        by_src[src] = crossing(sub, "individual_reliable", "pretrain_split")[0]
    p["individual"]["by_source"] = by_src
    # several tissues from one individual
    r = pre[pre.individual_reliable.notna() & pre.tissue.notna()].groupby("individual_reliable").tissue.nunique()
    p["individual"]["individuals_with_more_than_one_tissue_category"] = int((r > 1).sum())
    tc = pre[pre.source == "TCGA"].copy()
    tc["sample_type"] = tc.sample_id.str[13:15]
    p["individual"]["tcga_sample_type_codes"] = tc.sample_type.value_counts().to_dict()

    # cultured cell lines: one line is profiled many times, often in several studies
    cl = pre[pre.cell_line.notna()].assign(line=lambda d: d.cell_line.str.strip().str.lower())
    d, _ = crossing(cl, "line", "pretrain_split")
    hc = cl[cl.pretrain_split == "heldout"]
    d["heldout_profiles_of_a_cell_line"] = int(len(hc))
    d["heldout_profiles_whose_cell_line_is_in_train"] = int(hc.line.isin(set(cl[cl.pretrain_split == "train"].line)).sum())
    d["most_profiled_lines"] = cl.line.value_counts().head(8).to_dict()
    p["cell_lines"] = d
    p["ids_that_are_not_single_sample_accessions"] = sorted(pre[pre.source == "other"].sample_id)

    # ───────────────────────── downstream corpus ─────────────────────────
    down = down.merge(age, on="sample_id", how="left").merge(
        meta[["GSM_ID", "PATIENT_ID", "dataset", "tissue"]].rename(
            columns={"dataset": "geo_dataset", "tissue": "coarse_tissue"}),
        left_on="sample_id", right_on="GSM_ID", how="left")
    assert len(down) == 10988 and down.sample_id.is_unique
    down["study"] = down.dataset.map(norm_study)
    down["age_ok"] = down.age.between(0, 120)
    folds = REPO / "outputs/kfold_splits"
    if (folds / "test_ids.npy").exists():          # the benchmark exactly as run (age filter + duplicate removal)
        test_ids = set(np.load(folds / "test_ids.npy", allow_pickle=True).astype(str))
        pool = set().union(*[set(np.load(folds / f"fold_{k}_{s}.npy", allow_pickle=True).astype(str))
                             for k in range(5) for s in ("train", "val")])
        assert len(test_ids) == 2149 and len(pool) == 8440 and not test_ids & pool
        down["bench"] = np.where(down.sample_id.isin(test_ids), "test",
                                 np.where(down.sample_id.isin(pool), "development", "excluded"))
    else:
        down["bench"] = np.where(down.in_benchmark_test_set, "test",
                                 np.where(down.split.isin(["train", "valid"]) & down.age_ok, "development", "excluded"))
    down["individual"] = [individual_key(s, d, p_) for s, d, p_ in zip(down.sample_id, down.geo_dataset, down.PATIENT_ID)]
    res["downstream"] = q = {}
    q["n_profiles"] = len(down)
    q["source_by_id_prefix"] = down.sample_id.str.extract(r"^([A-Za-z]+)")[0].value_counts().to_dict()
    q["original_partition"] = down.split.value_counts().to_dict()
    q["benchmark_membership"] = down.bench.value_counts().to_dict()
    q["note_development_pool"] = ("from outputs/kfold_splits when present (8,440 = age filter and 71 redundant "
                                  "profiles removed); otherwise original train+valid after the age filter only")
    q["tissue_labels_raw"] = int(down.tissue_type.nunique())
    q["studies_as_named"] = int(down.dataset.nunique())
    q["studies_after_merging_E-GEOD_with_GSE"] = int(down.study.nunique())
    q["profiles_per_study"] = {"median": float(down.study.value_counts().median()),
                               "max": int(down.study.value_counts().max()),
                               "studies_with_fewer_than_10": int((down.study.value_counts() < 10).sum())}
    b = down[down.bench != "excluded"]
    sb = pd.crosstab(b.study, b.bench); sb["total"] = sb.sum(1)
    sb.sort_values("total", ascending=False).to_csv(OUT / "downstream_study_by_benchmark_partition.csv")
    tb = pd.crosstab(b.tissue_type, b.bench); tb["total"] = tb.sum(1)
    tb.sort_values("total", ascending=False).to_csv(OUT / "downstream_tissue_by_benchmark_partition.csv")
    te, dv = b[b.bench == "test"], b[b.bench == "development"]
    q["benchmark"] = {
        "test_profiles": int(len(te)), "development_profiles": int(len(dv)),
        "studies_in_test": int(te.study.nunique()), "studies_in_development": int(dv.study.nunique()),
        "studies_in_both": int(len(set(te.study) & set(dv.study))),
        "test_profiles_whose_study_is_in_development": int(te.study.isin(set(dv.study)).sum()),
        "test_profiles_whose_study_is_not_in_development": int((~te.study.isin(set(dv.study))).sum()),
        "tissue_labels_in_test": int(te.tissue_type.nunique()),
        "test_profiles_whose_tissue_label_is_in_development": int(te.tissue_type.isin(set(dv.tissue_type)).sum()),
        "test_tissue_labels_absent_from_development": sorted(set(te.tissue_type) - set(dv.tissue_type))}
    # studies that contribute several tissue labels (several samples per donor are possible there)
    mt = down.groupby("study").agg(profiles=("sample_id", "size"), tissue_labels=("tissue_type", "nunique"),
                                   labels=("tissue_type", lambda x: "; ".join(sorted(set(x)))))
    mt[mt.tissue_labels > 1].sort_values("profiles", ascending=False).to_csv(OUT / "downstream_studies_with_several_tissues.csv")
    q["studies_with_more_than_one_tissue_label"] = {"studies": int((mt.tissue_labels > 1).sum()),
                                                    "profiles": int(mt[mt.tissue_labels > 1].profiles.sum())}
    di = b[b.individual.notna()]
    gs = di.groupby("individual").size()
    di = di[~di.individual.isin(gs[gs > MAX_GROUP].index)]
    d, span = crossing(di, "individual", "bench")
    dvk = set(di[di.bench == "development"].individual)
    tek = di[di.bench == "test"]
    d["benchmark_profiles_with_identifier"] = int(len(di))
    d["benchmark_profiles_total"] = int(len(b))
    d["test_profiles_with_identifier"] = int(len(tek))
    d["test_profiles_whose_individual_is_also_in_development"] = int(tek.individual.isin(dvk).sum())
    q["individual"] = d
    di[di.individual.isin(span)].sort_values(["individual", "bench"])[
        ["sample_id", "individual", "study", "tissue_type", "age", "bench"]].to_csv(
        OUT / "downstream_individuals_spanning_test_and_development.csv", index=False)

    # ───────────────── pretraining exposure of the benchmark test set ─────────────────
    # Downstream TCGA profiles are named by PARTICIPANT (12 characters); the corpus names TCGA
    # profiles by full aliquot barcode (28 characters). Exact-id matching therefore cannot
    # match any TCGA profile, and TCGA is matched here by participant and sample type.
    pre_tcga = pre[pre.source == "TCGA"].assign(participant=lambda d: d.sample_id.str[:12],
                                                sample_type=lambda d: d.sample_id.str[13:15])
    normal_train = set(pre_tcga[(pre_tcga.pretrain_split == "train") & (pre_tcga.sample_type == "11")].participant)
    any_train = set(pre_tcga[pre_tcga.pretrain_split == "train"].participant)
    any_corpus = set(pre_tcga.participant)
    normal_corpus = set(pre_tcga[pre_tcga.sample_type == "11"].participant)

    pre_train = pre[pre.pretrain_split == "train"]
    train_ind = set(pre_train.individual_reliable.dropna())
    is_tcga = down.sample_id.str.startswith("TCGA-")
    # a study is in pretraining-train if any of its profiles is: from the corpus annotations,
    # from the downstream profiles matched by id, or (TCGA projects) from matched participants
    train_studies = (set(pre_train.dataset.dropna())
                     | set(down[down.pretrain_split == "train"].study)
                     | set(down[is_tcga & down.sample_id.isin(any_train)].study))

    def exposure(r):
        if r.pretrain_split == "train":
            return "1 profile in pretraining train (same accession)"
        if r.sample_id.startswith("TCGA-"):
            if r.sample_id in normal_train:
                return "2 TCGA: normal-tissue profile of the same participant in pretraining train"
            if r.sample_id in any_train:
                return "3 same individual in pretraining train (other sample)"
        elif r.individual in train_ind:
            return "3 same individual in pretraining train (other sample)"
        if r.pretrain_split in ("valid", "heldout"):
            return "4 profile in corpus, validation or held-out partition only"
        if r.sample_id.startswith("TCGA-") and r.sample_id in any_corpus:
            return "4 profile in corpus, validation or held-out partition only"
        if r.study in train_studies:
            return "5 not found in corpus: study in pretraining train"
        return "6 not found in corpus: study not in pretraining train"

    down["exposure"] = [exposure(r) for r in down.itertuples()]
    down["has_individual_identifier"] = down.individual.notna()
    te = down[down.bench == "test"]
    ex = te.groupby("exposure").agg(profiles=("sample_id", "size"),
                                    with_individual_identifier=("has_individual_identifier", "sum"),
                                    studies=("study", "nunique"))
    res["benchmark_test_vs_pretraining"] = {
        "test_profiles": int(len(te)),
        "tcga_in_test": int(te.sample_id.str.startswith("TCGA-").sum()),
        "tcga_id_format": {"downstream": "participant, 12 characters", "corpus": "aliquot barcode, 28 characters"},
        "tcga_downstream_participants": {
            "profiles": int(is_tcga.sum()),
            "participant_in_corpus": int(down[is_tcga].sample_id.isin(any_corpus).sum()),
            "participant_in_corpus_with_normal_tissue_profile": int(down[is_tcga].sample_id.isin(normal_corpus).sum()),
            "participant_in_pretraining_train": int(down[is_tcga].sample_id.isin(any_train).sum()),
            "participant_in_pretraining_train_with_normal_tissue_profile": int(down[is_tcga].sample_id.isin(normal_train).sum())},
        "exposure_of_test_profiles": {k: {c: int(v[c]) for c in ex.columns} for k, v in ex.iterrows()},
        "studies_of_test_profiles_in_class_6": te[te.exposure.str.startswith("6")].study.value_counts().to_dict(),
        "studies_of_test_profiles_in_class_5": te[te.exposure.str.startswith("5")].study.value_counts().to_dict()}
    down[["sample_id", "study", "tissue_type", "age", "split", "bench", "pretrain_split", "individual", "exposure"]].to_csv(
        OUT / "downstream_exposure_per_profile.csv", index=False)

    (OUT / "audit.json").write_text(json.dumps(res, indent=2, default=str))
    print(json.dumps(res, indent=1, default=str))


if __name__ == "__main__":
    main()
