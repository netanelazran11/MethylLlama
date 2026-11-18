"""
external_labels.py  (CNS task, stage 6 - labels of the external cohort GSE215240)
=================================================================================
GSE215240 (Sturm et al., Nat Med 2023; 1,124 paediatric CNS tumours, EPIC 937 + 450K 187) lists only the WHO
histology on GEO. The paper's Supplementary Table 1 gives, per patient, the class of the reference random-forest
classifier v11b4 - the same 91-class system as Capper 2018 - and its calibrated score. That is the label used here,
under the classifier's own reporting rule: a class is assigned only when the calibrated score is >= 0.9.

The table writes the v11b4 classes in WHO-2021 wording; CLASS_MAP turns them into the Capper abbreviations of
cns_capper.h5ad. Scoring groups (EVAL_GROUP) merge classes that the table does not separate:
  MB, G3 + MB, G4                    the table lists 120 "Group 3" tumours and no "Group 4" one, although Group 4 is the
                                     most common paediatric medulloblastoma: the two are reported together
  GBM, RTK I + RTK II + RTK III      "High-grade glioma, pediatric RTK" does not say which of the three
"Pineoblastoma, Group RB" (1 tumour) has no unambiguous v11b4 counterpart and is left out.

Inputs  Supplementary Table 1 (41591_2023_2255_MOESM3_ESM.xlsx; only patient ID, array type, v11b4 class and score
        are read - no clinical column), the two GEO series matrices (GSM, title = patient ID, IDAT Sentrix ID).
Output  outputs/cns/external/GSE215240_labels.csv  (sample-level: never goes to the public repo)
        gsm, title, platform, sentrix, v11b4_class, v11b4_score, confident, methylation_class, eval_group
  python scripts/cns/external_labels.py --supp <xlsx> --series <dir with GPL21145.txt.gz, GPL13534.txt.gz>
"""
import argparse
import gzip
import re
from pathlib import Path

import pandas as pd

REPO = Path(__file__).resolve().parents[2]

CLASS_MAP = {
    "Adamantinomatous craniopharyngioma": "CPH, ADM",
    "Astrocytoma, IDH-mutated": "A IDH",
    "Atypical teratoid/rhabdoid tumor (AT/RT), MYC": "ATRT, MYC",
    "Atypical teratoid/rhabdoid tumor (AT/RT), SHH": "ATRT, SHH",
    "Atypical teratoid/rhabdoid tumor (AT/RT), TYR": "ATRT, TYR",
    "CNS Neuroblastoma, FOXR2-activated": "CNS NB, FOXR2",
    "CNS sarcoma, CIC-altered": "EFT, CIC",
    "CNS tumor with BCOR internal tandem duplication": "HGNET, BCOR",
    "Chordoma": "CHORDM",
    "Choroid plexus tumor, adult type": "PLEX, AD",
    "Choroid plexus tumor, ped. type A": "PLEX, PED A",
    "Choroid plexus tumor, ped. type B": "PLEX, PED B",
    "Control, cerebellar tissue": "CONTR, CEBM",
    "Control, hemispheric tissue": "CONTR, HEMI",
    "Control, inflammatory tissue": "CONTR, INFLAM",
    "Control, pineal tissue": "CONTR, PINEAL",
    "Control, reactive tissue": "CONTR, REACT",
    "Desmoplastic infantile ganglioglioma/astrocytoma": "LGG, DIG/DIA",
    "Diffuse leptomeningeal glioneuronal tumor (DLGNT)": "DLGNT",
    "Diffuse midline glioma, H3 K27-mutated": "DMG, K27",
    "Dysembryoplastic neuroepithelial tumor (DNET)": "LGG, DNT",
    "Embryonal tumor with multilayered rosettes (ETMR)": "ETMR",
    "Ependymoma, ZFTA fusion-positive": "EPN, RELA",
    "Ependymoma, YAP1 fusion-positive": "EPN, YAP",
    "Ependymoma, posterior fossa A": "EPN, PF A",
    "Ependymoma, posterior fossa B": "EPN, PF B",
    "Ependymoma, spinal": "EPN, SPINE",
    "Ewing's sarcoma": "EWS",
    "Ganglioglioma": "LGG, GG",
    "Hemangioblastoma": "HMB",
    "High-grade astrocytoma, IDH-mutated": "A IDH, HG",
    "High-grade glioma, H3 G34-mutated": "GBM, G34",
    "High-grade glioma, MYCN": "GBM, MYCN",
    "High-grade glioma, midline": "GBM, MID",
    "High-grade glioma, pediatric RTK": "GBM, RTK",           # scoring group, see header
    "Infantile hemispheric glioma": "IHG",
    "Low-grade glioma, MYB/MYBL1-altered": "LGG, MYB",
    "Medulloblastoma, Group 3": "MB, G3/G4",                  # scoring group, see header
    "Medulloblastoma, SHH; adult": "MB, SHH CHL AD",
    "Medulloblastoma, SHH; infant": "MB, SHH INF",
    "Medulloblastoma, WNT": "MB, WNT",
    "Meningioma": "MNG",
    "Myxopapillary ependymoma (MPE)": "EPN, MPE",
    "Neuroepithelial tumor, MN1-altered": "HGNET, MN1",
    "Oligodendroglioma, IDH-mutated": "O IDH",
    "PA, midline": "LGG, PA MID",
    "PA/ganglioglioma, hemispheric": "LGG, PA/GG ST",
    "Papillary tumor of pineal region (PTPR), Group B": "PTPR, B",
    "Pilocytic Astrocytoma (PA), posterior fossa": "LGG, PA PF",
    "Pineal parenchymal tumor of intermediate differentiation (PPTID)": "PIN T, PPT",
    "Pineoblastoma, Group B": "PIN T, PB B",
    "Pituicytoma": "PITUI",
    "Pleomorphic xanthoastrocytoma (PXA)": "PXA",
    "Rosette-forming glioneuronal tumor (RGNT)": "LGG, RGNT",
    "Schwannoma": "SCHW",
    "Subependymal giant cell astrocytoma (SEGA)": "LGG, SEGA",
    "Subependymoma, spinal": "SUBEPN, SPINE",
}
UNMAPPED = {"Pineoblastoma, Group RB"}
# Capper class -> scoring group (every other class is its own group)
EVAL_GROUP = {"MB, G3": "MB, G3/G4", "MB, G4": "MB, G3/G4",
              "GBM, RTK I": "GBM, RTK", "GBM, RTK II": "GBM, RTK", "GBM, RTK III": "GBM, RTK"}


def series(path):
    rows = {}
    for line in gzip.open(path, "rt"):
        if line.startswith(("!Sample_geo_accession", "!Sample_title", "!Sample_platform_id",
                            "!Sample_supplementary_file")):
            key = line.split("\t", 1)[0]
            vals = [c.strip('"') for c in line.rstrip("\n").split("\t")[1:]]
            rows.setdefault(key, vals)                      # first supplementary file line (Grn idat) is enough
    sentrix = [re.search(r"_(\d+_R\d\dC\d\d)_", f).group(1) for f in rows["!Sample_supplementary_file"]]
    return pd.DataFrame({"gsm": rows["!Sample_geo_accession"], "title": rows["!Sample_title"],
                         "platform": rows["!Sample_platform_id"], "sentrix": sentrix})


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--supp", required=True)
    p.add_argument("--series", required=True)
    p.add_argument("--capper_classes", default=str(REPO / "outputs/cns/data/cns_capper.h5ad"))
    p.add_argument("--out", default=str(REPO / "outputs/cns/external/GSE215240_labels.csv"))
    a = p.parse_args()

    geo = pd.concat([series(Path(a.series) / f"{g}.txt.gz") for g in ("GPL21145", "GPL13534")], ignore_index=True)
    t = pd.read_excel(a.supp, sheet_name="Supplementary Table 1", header=None, skiprows=4, usecols=[0, 55, 56])
    t.columns = ["title", "v11b4_class", "v11b4_score"]
    t = t[t.title.astype(str).str.startswith("mnp")]
    t["v11b4_score"] = pd.to_numeric(t.v11b4_score, errors="coerce")
    if t.title.duplicated().any() or geo.gsm.duplicated().any() or geo.sentrix.duplicated().any():
        raise SystemExit("duplicate patient ID, GSM or Sentrix ID")
    d = geo.merge(t, on="title", how="left", validate="1:1")
    if d.v11b4_class.isna().any():
        raise SystemExit(f"{d.v11b4_class.isna().sum()} GEO samples without a v11b4 entry")

    d["confident"] = d.v11b4_score >= 0.9
    conf = set(d.v11b4_class[d.confident])
    unknown = conf - set(CLASS_MAP) - UNMAPPED
    if unknown:
        raise SystemExit(f"confident v11b4 classes without a mapping: {sorted(unknown)}")
    d["methylation_class"] = d.v11b4_class.map(CLASS_MAP)
    import anndata as ad
    capper = set(ad.read_h5ad(a.capper_classes, backed="r").obs.methylation_class.astype(str))
    groups = {EVAL_GROUP.get(c, c) for c in capper}
    bad = set(d.methylation_class.dropna()) - groups
    if bad:
        raise SystemExit(f"mapped classes not among the Capper classes / groups: {sorted(bad)}")
    d["eval_group"] = d.methylation_class.where(d.confident)

    out = Path(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    d.to_csv(out, index=False)
    e = d[d.eval_group.notna()]
    print(f"GEO samples {len(d)} (EPIC {(d.platform == 'GPL21145').sum()}, 450K {(d.platform == 'GPL13534').sum()}); "
          f"v11b4 score >= 0.9: {int(d.confident.sum())}; scored: {len(e)} in {e.eval_group.nunique()} groups; "
          f"left out (unmapped): {int((d.confident & d.methylation_class.isna()).sum())}")
    print(e.eval_group.value_counts().to_string())


if __name__ == "__main__":
    main()
