"""
ElasticNet baseline for age prediction on AltumAge 21k methylation data.

Matches the exact evaluation protocol used for MethylLlama/MethylGPT:
same split column, same duplicate-pair exclusion (dataset_fingerprint_outputs/
duplicate_pairs.csv), same age-outlier filter (age<0 or age>120 removed) --
so the resulting test set is the identical 2,149 samples used in the
paired-bootstrap comparison, not the raw 2,264.

Hyperparameters are chosen by scikit-learn's own ElasticNetCV -- standard
internal k-fold cross-validation with its default automatic alpha path, and
l1_ratio taken verbatim from sklearn's own documented example grid
(https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.ElasticNetCV.html).
Deliberately NOT a hand-designed or iteratively-widened grid: after an earlier
manual grid search (job 45888519) showed ElasticNet competitive with
MethylLlama, widening that grid further to chase a "better" number would be
exactly the kind of post-hoc baseline tuning that undermines a fair
comparison. This runs the standard library tool once, with its own defaults,
and reports whatever comes out -- better than MethylLlama, worse, or a tie.
Matches AltumAge's own paper, which used "the built-in hyperparameter tuning
from Python glmnet" (the R equivalent of ElasticNetCV) rather than a custom
grid.

train+valid are pooled into one set for ElasticNetCV's internal CV (it does
its own k-fold split, so a separate fixed validation set isn't needed). test
is never touched until the single final evaluation.
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNetCV
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import StandardScaler

logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
logger = logging.getLogger(__name__)


def median_absolute_error(y_true, y_pred):
    return float(np.median(np.abs(np.array(y_true) - np.array(y_pred))))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5ad",     required=True)
    parser.add_argument("--outdir",   required=True)
    parser.add_argument("--l1_ratios", type=float, nargs="+",
                         default=[0.1, 0.5, 0.7, 0.9, 0.95, 0.99, 1.0],
                         help="l1_ratio candidates for ElasticNetCV -- verbatim from sklearn's own "
                              "documented example, not hand-tuned for this dataset.")
    parser.add_argument("--n_alphas", type=int, default=100,
                         help="ElasticNetCV's automatic alpha path length (sklearn default=100).")
    parser.add_argument("--cv_folds", type=int, default=5,
                         help="Internal cross-validation folds for ElasticNetCV (standard default).")
    parser.add_argument("--age_col",   default="age")
    parser.add_argument("--split_col", default="split")
    parser.add_argument("--duplicate_pairs_csv",
                         default=None,
                         help="Path to duplicate_pairs.csv -- if set, excludes one ID per "
                              "duplicate pair, matching the MethylLlama/MethylGPT eval protocol.")
    parser.add_argument("--filter_age_outliers", action="store_true", default=True)
    parser.add_argument("--no_filter_age_outliers", dest="filter_age_outliers", action="store_false")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 0. Fail loudly if inputs don't exist -- don't burn a job on a typo'd path ──
    h5ad_path = Path(args.h5ad)
    if not h5ad_path.exists():
        raise FileNotFoundError(f"--h5ad not found: {h5ad_path}")
    if args.duplicate_pairs_csv is not None and not Path(args.duplicate_pairs_csv).exists():
        raise FileNotFoundError(f"--duplicate_pairs_csv not found: {args.duplicate_pairs_csv}")

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    logger.info(f"  Total samples : {adata.shape[0]:,}")
    logger.info(f"  Total CpGs    : {adata.shape[1]:,}")

    # ── 1b. Apply the same exclusions as the MethylLlama/MethylGPT eval set ───
    n_before = adata.shape[0]
    if args.filter_age_outliers and args.age_col in adata.obs.columns:
        age_vals = adata.obs[args.age_col].astype(float)
        keep = (age_vals >= 0) & (age_vals <= 120)
        n_removed = int((~keep).sum())
        adata = adata[keep].copy()
        logger.info(f"Age outlier filter: removed {n_removed} samples (age<0 or age>120)")

    if args.duplicate_pairs_csv is not None:
        from bmfm_methylation.shared.data_module import _compute_dedup_exclusions
        exclude_ids = _compute_dedup_exclusions(args.duplicate_pairs_csv)
        keep_mask = ~adata.obs_names.isin(exclude_ids)
        n_removed = int((~keep_mask).sum())
        adata = adata[keep_mask].copy()
        logger.info(f"Duplicate filter: removed {n_removed} samples ({len(exclude_ids)} IDs in dedup set)")

    logger.info(f"Total after filtering: {adata.shape[0]:,} / {n_before:,} samples")

    # ── 2. Split by existing split column ─────────────────────────────────────
    # train+valid are pooled: ElasticNetCV does its own internal k-fold CV, so a
    # separate fixed validation set isn't part of this method's standard usage.
    # test is kept completely separate and untouched until the final evaluation.
    splits = adata.obs[args.split_col]
    for name in ("train", "valid", "test"):
        logger.info(f"  {name:5s} : {int((splits == name).sum()):,} samples")

    idx_cv   = splits.isin(["train", "valid"])
    idx_test = splits == "test"

    def to_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X.astype(np.float32)

    X_cv   = to_dense(adata[idx_cv].X)
    X_test = to_dense(adata[idx_test].X)

    y_cv   = adata[idx_cv].obs[args.age_col].values.astype(np.float32)
    y_test = adata[idx_test].obs[args.age_col].values.astype(np.float32)

    logger.info(f"CV pool (train+valid): {len(y_cv):,} samples, "
                f"age mean={y_cv.mean():.1f}yr std={y_cv.std():.1f}yr")

    # ── 3. Feature scaling (StandardScaler — required for ElasticNet) ─────────
    scaler = StandardScaler()
    X_cv   = scaler.fit_transform(X_cv)
    X_test = scaler.transform(X_test)

    # ── 4. ElasticNetCV: standard library tool, standard defaults, run once ───
    # alpha: sklearn's own automatic path (n_alphas points, log-spaced, range
    # derived from the data). l1_ratio: sklearn's own documented example list.
    # Neither was hand-tuned for this dataset or adjusted after seeing results.
    logger.info(f"ElasticNetCV: l1_ratio candidates={args.l1_ratios}, "
                f"n_alphas={args.n_alphas} (auto path), cv={args.cv_folds}-fold")
    model = ElasticNetCV(
        l1_ratio=args.l1_ratios,
        n_alphas=args.n_alphas,
        cv=args.cv_folds,
        max_iter=10000,
        tol=1e-4,
        random_state=42,
        selection="random",
        n_jobs=-1,
    )
    model.fit(X_cv, y_cv)
    n_nonzero = int(np.sum(model.coef_ != 0))
    logger.info(f"Selected by internal CV: alpha={model.alpha_:.6g}, l1_ratio={model.l1_ratio_}, "
                f"nonzero={n_nonzero}/{X_cv.shape[1]}")

    # ── 5. Evaluate the selected model ────────────────────────────────────────
    def evaluate(X, y, name):
        pred  = model.predict(X)
        mae   = float(mean_absolute_error(y, pred))
        medae = median_absolute_error(y, pred)
        r2    = float(r2_score(y, pred))
        pcc,_ = pearsonr(y, pred)
        logger.info(f"  {name:5s}  MAE={mae:.2f}yr  MedAE={medae:.2f}yr  R²={r2:.4f}  PCC={pcc:.4f}")
        return {"mae": mae, "medae": medae, "r2": r2, "pcc": float(pcc), "n": int(len(y))}

    logger.info("=" * 60)
    logger.info("RESULTS (ElasticNetCV, single run, standard settings)")
    logger.info("=" * 60)
    results = {
        "model": "ElasticNet",
        "selection_method": "sklearn.linear_model.ElasticNetCV, single run, standard settings "
                             "(l1_ratio grid from sklearn docs, automatic alpha path, "
                             f"{args.cv_folds}-fold internal CV on pooled train+valid)",
        "alpha": float(model.alpha_),
        "l1_ratio": float(model.l1_ratio_),
        "n_features": int(X_cv.shape[1]),
        "n_nonzero_coef": n_nonzero,
        "cv_pool": evaluate(X_cv, y_cv, "cv_pool"),
        "test":  evaluate(X_test,  y_test,  "test"),
    }

    # ── 6. Save ───────────────────────────────────────────────────────────────
    with open(outdir / "elasticnet_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = (
        f"\n{'='*60}\n"
        f"ElasticNet Baseline — ElasticNetCV selected alpha={model.alpha_:.6g}, "
        f"l1_ratio={model.l1_ratio_} (single run, standard settings, not hand-tuned)\n"
        f"Non-zero CpGs : {n_nonzero:,} / {X_cv.shape[1]:,}\n"
        f"{'='*60}\n"
        f"{'Split':<10} {'N':>6} {'MAE':>8} {'MedAE':>8} {'R²':>8}\n"
        f"{'-'*44}\n"
        f"{'cv_pool':<10} {results['cv_pool']['n']:>6} {results['cv_pool']['mae']:>8.2f} {results['cv_pool']['medae']:>8.2f} {results['cv_pool']['r2']:>8.4f}\n"
        f"{'test':<10} {results['test']['n']:>6} {results['test']['mae']:>8.2f} {results['test']['medae']:>8.2f} {results['test']['r2']:>8.4f}\n"
        f"{'='*60}\n"
    )
    print(summary)
    with open(outdir / "elasticnet_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
