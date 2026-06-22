#!/usr/bin/env python3
"""
verify_figure3_inputs.py
========================
Deep sanity check of all inputs needed for figure3_comparison.py.
Run this interactively on the cluster BEFORE submitting the SLURM job.

Usage:
  python scripts/repr_analysis/verify_figure3_inputs.py
"""

import sys
import numpy as np
import pandas as pd
from pathlib import Path

# ─────────────────────────────────────────────────────────────────────────────
# Paths — edit if needed
# ─────────────────────────────────────────────────────────────────────────────
REPO       = Path("/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl")
BASE_19K   = REPO / "outputs/repr_analysis/cls_probing_44905909"
CLS_NPY    = BASE_19K / "embeddings_cls.npy"
MEAN_NPY   = BASE_19K / "embeddings_mean.npy"
RANDOM_NPY = BASE_19K / "embeddings_random_cls.npy"
METADATA   = BASE_19K / "metadata.csv"
EXT_META   = REPO / "data/pretrain_metadata.csv.gz"
H5AD       = Path("/sci/labs/benjamin.yakir/netanel.azran/data"
                  "/data_methyl_finetune_19k_h5ad"
                  "/finetuning_19608_clean_stratified_no_outliers.h5ad")

PASS = "  [OK]"
FAIL = "  [FAIL]"
WARN = "  [WARN]"

errors   = []
warnings = []

def ok(msg):   print(f"{PASS}  {msg}")
def fail(msg): print(f"{FAIL}  {msg}"); errors.append(msg)
def warn(msg): print(f"{WARN}  {msg}"); warnings.append(msg)

def section(title):
    print()
    print("=" * 60)
    print(f"  {title}")
    print("=" * 60)


# ─────────────────────────────────────────────────────────────────────────────
# 1. File existence
# ─────────────────────────────────────────────────────────────────────────────
section("1. File existence")

required = {
    "CLS embeddings":    CLS_NPY,
    "metadata.csv":      METADATA,
    "pretrain_metadata": EXT_META,
    "h5ad (19k)":        H5AD,
}
optional = {
    "Mean embeddings":   MEAN_NPY,
    "Random embeddings": RANDOM_NPY,
}

for label, path in required.items():
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        ok(f"{label}: {path}  ({size_mb:.1f} MB)")
    else:
        fail(f"{label} NOT FOUND: {path}")

for label, path in optional.items():
    if path.exists():
        size_mb = path.stat().st_size / 1e6
        ok(f"{label}: {path}  ({size_mb:.1f} MB)")
    else:
        warn(f"{label} not found (optional): {path}")


# ─────────────────────────────────────────────────────────────────────────────
# 2. CLS embeddings deep check
# ─────────────────────────────────────────────────────────────────────────────
section("2. CLS embeddings")

cls_emb = None
if CLS_NPY.exists():
    cls_emb = np.load(CLS_NPY)
    print(f"  Shape  : {cls_emb.shape}")
    print(f"  Dtype  : {cls_emb.dtype}")
    print(f"  Range  : [{cls_emb.min():.4f}, {cls_emb.max():.4f}]")
    print(f"  Mean   : {cls_emb.mean():.4f}   Std: {cls_emb.std():.4f}")
    n_nan = np.isnan(cls_emb).sum()
    n_inf = np.isinf(cls_emb).sum()

    if cls_emb.ndim == 2:
        ok(f"ndim=2  N={cls_emb.shape[0]:,}  D={cls_emb.shape[1]}")
    else:
        fail(f"Expected 2D array, got shape {cls_emb.shape}")

    if n_nan == 0:
        ok("No NaN values")
    else:
        fail(f"{n_nan:,} NaN values in CLS embeddings")

    if n_inf == 0:
        ok("No Inf values")
    else:
        fail(f"{n_inf:,} Inf values in CLS embeddings")

    if cls_emb.shape[1] == 256:
        ok("Embedding dimension = 256 (expected MethylLlama-Small CLS)")
    else:
        warn(f"Embedding dimension = {cls_emb.shape[1]}  (expected 256)")

    # Check for all-zero rows (dead embeddings)
    zero_rows = (cls_emb == 0).all(axis=1).sum()
    if zero_rows == 0:
        ok("No all-zero rows")
    else:
        fail(f"{zero_rows:,} all-zero rows (dead embeddings)")

    # Check for constant rows (no variance per sample)
    row_std = cls_emb.std(axis=1)
    dead = (row_std < 1e-6).sum()
    if dead == 0:
        ok("All rows have variance (no constant embeddings)")
    else:
        warn(f"{dead:,} rows with std < 1e-6")


# ─────────────────────────────────────────────────────────────────────────────
# 3. Mean + Random embeddings
# ─────────────────────────────────────────────────────────────────────────────
section("3. Mean / Random embeddings")

for label, path in [("Mean", MEAN_NPY), ("Random", RANDOM_NPY)]:
    if not path.exists():
        warn(f"{label} embeddings not found — skipping")
        continue
    emb = np.load(path)
    print(f"  {label}: shape={emb.shape}  dtype={emb.dtype}"
          f"  range=[{emb.min():.3f}, {emb.max():.3f}]"
          f"  NaN={np.isnan(emb).sum()}")
    if cls_emb is not None:
        if emb.shape == cls_emb.shape:
            ok(f"{label} shape matches CLS shape")
        else:
            fail(f"{label} shape {emb.shape} ≠ CLS shape {cls_emb.shape}")


# ─────────────────────────────────────────────────────────────────────────────
# 4. Metadata.csv
# ─────────────────────────────────────────────────────────────────────────────
section("4. metadata.csv")

meta = None
if METADATA.exists():
    meta = pd.read_csv(METADATA, index_col=0)
    print(f"  Shape  : {meta.shape}")
    print(f"  Columns: {list(meta.columns)}")
    print(f"  Index sample (first 3): {list(meta.index[:3])}")
    print(f"  Index sample (last 3) : {list(meta.index[-3:])}")

    if cls_emb is not None:
        if len(meta) == cls_emb.shape[0]:
            ok(f"Row count matches CLS embeddings: {len(meta):,}")
        else:
            fail(f"Row count mismatch: metadata={len(meta):,}  CLS={cls_emb.shape[0]:,}")

    # Check index looks like GSM IDs
    sample_idx = str(meta.index[0])
    if sample_idx.startswith("GSM"):
        ok(f"Index looks like GSM IDs (e.g. '{sample_idx}')")
    else:
        warn(f"Index does NOT look like GSM IDs: '{sample_idx}' — join with ext_metadata may fail")

    # Check for duplicates
    n_dup = meta.index.duplicated().sum()
    if n_dup == 0:
        ok("No duplicate index values")
    else:
        fail(f"{n_dup:,} duplicate index values")

    # Existing label columns
    for col in ["tissue", "sex", "dataset", "age"]:
        if col in meta.columns:
            n_valid = meta[col].notna().sum()
            print(f"  '{col}' in metadata: {n_valid:,}/{len(meta):,} non-null")
        else:
            print(f"  '{col}' NOT in metadata — will join from ext_metadata")


# ─────────────────────────────────────────────────────────────────────────────
# 5. External metadata join simulation
# ─────────────────────────────────────────────────────────────────────────────
section("5. External metadata join (tissue / sex)")

if EXT_META.exists() and meta is not None:
    ext = pd.read_csv(EXT_META)
    print(f"  ext_metadata shape: {ext.shape}")
    print(f"  ext_metadata columns: {list(ext.columns)}")

    ext_indexed = ext.drop_duplicates(subset="GSM_ID").set_index("GSM_ID")

    # Simulate join
    meta_ids = set(meta.index)
    ext_ids  = set(ext_indexed.index)
    overlap  = meta_ids & ext_ids
    missing  = meta_ids - ext_ids

    print(f"\n  Metadata samples      : {len(meta_ids):,}")
    print(f"  Ext_metadata samples  : {len(ext_ids):,}")
    print(f"  Overlap (can join)    : {len(overlap):,}")
    print(f"  No match in ext_meta  : {len(missing):,}")

    pct = 100 * len(overlap) / len(meta_ids)
    if pct >= 95:
        ok(f"Join coverage: {pct:.1f}% of samples matched")
    elif pct >= 80:
        warn(f"Join coverage: {pct:.1f}% — some samples will have no tissue/sex label")
    else:
        fail(f"Join coverage only {pct:.1f}% — most samples will have no labels")

    if len(missing) > 0:
        print(f"  Missing IDs sample: {list(missing)[:5]}")

    # After join: tissue coverage
    joined = meta.join(ext_indexed[["tissue", "sex"]], how="left")

    for col in ["tissue", "sex"]:
        if col in joined.columns:
            n_valid = joined[col].notna().sum()
            pct_col = 100 * n_valid / len(joined)
            vc = joined[col].value_counts()
            print(f"\n  '{col}' after join: {n_valid:,}/{len(joined):,} ({pct_col:.1f}%) non-null")
            print(f"    Unique values: {joined[col].nunique()}")
            print(f"    Top 10:")
            for val, cnt in vc.head(10).items():
                print(f"      {cnt:5d}  {val}")
            if pct_col >= 90:
                ok(f"'{col}' coverage {pct_col:.1f}% — good")
            elif pct_col >= 70:
                warn(f"'{col}' coverage {pct_col:.1f}% — some unlabeled samples")
            else:
                fail(f"'{col}' coverage only {pct_col:.1f}%")


# ─────────────────────────────────────────────────────────────────────────────
# 5b. Tissue palette coverage check
# ─────────────────────────────────────────────────────────────────────────────
section("5b. Tissue palette coverage")

# Exact palette defined in figure3_comparison.py
TISSUE_COLORS = {
    "Whole Blood":          "#E64B35",
    "Brain":                "#4DBBD5",
    "Other":                "#AAAAAA",
    "Cells":                "#9B59B6",
    "Breast":               "#F39B7F",
    "Lung":                 "#91D1C2",
    "Colon":                "#C0392B",
    "Liver":                "#3C5488",
    "Prostate":             "#7E6148",
    "Skin":                 "#8491B4",
    "Testis":               "#27AE60",
    "Ovary":                "#FF69B4",
    "Stomach":              "#E67E22",
    "Muscle":               "#00A087",
    "Kidney":               "#F4A460",
    "Esophagus":            "#808000",
    "Pancreas":             "#F1C40F",
    "Adipose":              "#B09C85",
    "Bladder":              "#FA8072",
    "Uterus":               "#C39BD3",
    "Cervix":               "#DDA0DD",
    "Thyroid":              "#5B2C6F",
    "Adrenal Gland":        "#D35400",
    "Nerve":                "#F7DC6F",
    "Small Intestine":      "#7DCEA0",
    "Heart":                "#922B21",
    "Minor Salivary Gland": "#708090",
    "Artery":               "#FF4500",
    "Pituitary":            "#98FB98",
    "Fallopian Tube":       "#FF91A4",
    "Spleen":               "#6B8E23",
    "Vagina":               "#FFDAB9",
    "Blood":                "#E64B35",  # alias
}

if EXT_META.exists() and meta is not None:
    ext = pd.read_csv(EXT_META)
    ext_indexed = ext.drop_duplicates(subset="GSM_ID").set_index("GSM_ID")
    joined = meta.join(ext_indexed[["tissue"]], how="left")

    # All tissue values that actually appear in the 19k dataset
    actual_tissues = set(joined["tissue"].dropna().unique())
    palette_tissues = set(TISSUE_COLORS.keys())

    print(f"  Tissue types in the 19k dataset : {len(actual_tissues)}")
    print(f"  Tissue types in TISSUE_COLORS   : {len(palette_tissues)}")

    # In data but NOT in palette → will fall back to auto tab20 color
    missing_from_palette = actual_tissues - palette_tissues
    # In palette but NOT in data → dead palette entries (harmless)
    not_in_data = palette_tissues - actual_tissues

    if not missing_from_palette:
        ok("All tissue types in data are covered by TISSUE_COLORS palette")
    else:
        fail(f"{len(missing_from_palette)} tissue type(s) in data have NO palette color (will get auto-color):")
        for t in sorted(missing_from_palette):
            n = (joined["tissue"] == t).sum()
            print(f"    MISSING  '{t}'  ({n:,} samples)")

    if not_in_data:
        print(f"\n  Palette entries not in this dataset (harmless):")
        for t in sorted(not_in_data):
            print(f"    unused   '{t}'")

    # Print full tissue breakdown with palette status
    print(f"\n  Full tissue breakdown (data count + palette color):")
    vc = joined["tissue"].value_counts()
    for tissue, count in vc.items():
        color = TISSUE_COLORS.get(tissue, "AUTO (no explicit color)")
        status = "OK " if tissue in palette_tissues else "MISSING"
        print(f"    [{status}]  {count:5d}  {tissue:<25s}  {color}")
else:
    warn("Cannot check palette — ext_metadata not found")


# ─────────────────────────────────────────────────────────────────────────────
# 6. h5ad raw methylation check
# ─────────────────────────────────────────────────────────────────────────────
section("6. h5ad raw methylation matrix")

h5ad_obs = None
if H5AD.exists():
    try:
        import anndata
        adata = anndata.read_h5ad(str(H5AD), backed="r")
        h5ad_obs = list(adata.obs_names)
        x_shape  = adata.shape
        print(f"  obs_names count : {len(h5ad_obs):,}")
        print(f"  obs_names sample: {h5ad_obs[:3]}")
        print(f"  X shape         : {x_shape}")
        ok(f"X matrix: {x_shape[0]:,} samples × {x_shape[1]:,} CpGs")

        # Check obs_names format
        sample_id = h5ad_obs[0]
        if sample_id.startswith("GSM") or sample_id.startswith("TCGA"):
            ok(f"obs_names format recognized ('{sample_id}')")
        else:
            warn(f"Unexpected obs_names format: '{sample_id}'")

        # Alignment with metadata
        if meta is not None:
            meta_set  = set(meta.index)
            h5ad_set  = set(h5ad_obs)
            overlap   = meta_set & h5ad_set
            only_meta = meta_set - h5ad_set
            only_h5ad = h5ad_set - meta_set

            print(f"\n  metadata samples  : {len(meta_set):,}")
            print(f"  h5ad samples      : {len(h5ad_set):,}")
            print(f"  overlap           : {len(overlap):,}")
            print(f"  only in metadata  : {len(only_meta):,}")
            print(f"  only in h5ad      : {len(only_h5ad):,}")

            pct = 100 * len(overlap) / len(meta_set)
            if pct >= 99:
                ok(f"h5ad ↔ metadata alignment: {pct:.1f}% — perfect")
            elif pct >= 90:
                warn(f"h5ad ↔ metadata alignment: {pct:.1f}%")
            else:
                fail(f"h5ad ↔ metadata alignment only {pct:.1f}% — check obs_names format")

            # Show obs_names types breakdown
            n_gsm  = sum(1 for s in meta_set if str(s).startswith("GSM"))
            n_tcga = sum(1 for s in meta_set if str(s).startswith("TCGA"))
            n_other = len(meta_set) - n_gsm - n_tcga
            print(f"\n  metadata ID breakdown:")
            print(f"    GSM  IDs : {n_gsm:,}  (can join to pretrain_metadata)")
            print(f"    TCGA IDs : {n_tcga:,}  (no GSM_ID match — will show as 'unknown')")
            print(f"    Other    : {n_other:,}")

    except Exception as e:
        fail(f"Could not read h5ad: {e}")
        import traceback; traceback.print_exc()


# ─────────────────────────────────────────────────────────────────────────────
# Summary
# ─────────────────────────────────────────────────────────────────────────────
section("SUMMARY")

print(f"  Errors   : {len(errors)}")
for e in errors:
    print(f"    [FAIL] {e}")

print(f"  Warnings : {len(warnings)}")
for w in warnings:
    print(f"    [WARN] {w}")

if len(errors) == 0:
    print()
    print("  ALL CHECKS PASSED — safe to run run_figure3.sh")
else:
    print()
    print("  FIX ERRORS ABOVE before running the SLURM job")
    sys.exit(1)
