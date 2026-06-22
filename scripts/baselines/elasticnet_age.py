"""
ElasticNet baseline for age prediction on AltumAge 21k methylation data.

Runs on the raw data as-is — no age filtering, no deduplication.
Uses the split column from the h5ad directly.
Reports MedAE / MAE / R² for direct comparison with MethylLlama V5/V6.

ElasticNet: alpha=0.01, l1_ratio=0.5 (equal L1+L2, mild regularization).
"""

import argparse
import json
import logging
from pathlib import Path

import numpy as np
import pandas as pd
import scanpy as sc
from scipy.stats import pearsonr
from sklearn.linear_model import ElasticNet
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
    parser.add_argument("--alpha",    type=float, default=0.01)
    parser.add_argument("--l1_ratio", type=float, default=0.5)
    parser.add_argument("--age_col",   default="age")
    parser.add_argument("--split_col", default="split")
    args = parser.parse_args()

    outdir = Path(args.outdir)
    outdir.mkdir(parents=True, exist_ok=True)

    # ── 1. Load raw data ──────────────────────────────────────────────────────
    logger.info(f"Loading {args.h5ad}")
    adata = sc.read_h5ad(args.h5ad)
    logger.info(f"  Total samples : {adata.shape[0]:,}")
    logger.info(f"  Total CpGs    : {adata.shape[1]:,}")

    # ── 2. Split by existing split column ─────────────────────────────────────
    splits = adata.obs[args.split_col]
    for name in ("train", "valid", "test"):
        logger.info(f"  {name:5s} : {int((splits == name).sum()):,} samples")

    idx_train = splits == "train"
    idx_valid = splits == "valid"
    idx_test  = splits == "test"

    def to_dense(X):
        if hasattr(X, "toarray"):
            X = X.toarray()
        return X.astype(np.float32)

    X_train = to_dense(adata[idx_train].X)
    X_valid = to_dense(adata[idx_valid].X)
    X_test  = to_dense(adata[idx_test].X)

    y_train = adata[idx_train].obs[args.age_col].values.astype(np.float32)
    y_valid = adata[idx_valid].obs[args.age_col].values.astype(np.float32)
    y_test  = adata[idx_test].obs[args.age_col].values.astype(np.float32)

    logger.info(f"Train age: mean={y_train.mean():.1f}yr  std={y_train.std():.1f}yr")

    # ── 3. Feature scaling (StandardScaler — required for ElasticNet) ─────────
    scaler = StandardScaler()
    X_train = scaler.fit_transform(X_train)
    X_valid = scaler.transform(X_valid)
    X_test  = scaler.transform(X_test)

    # ── 4. Fit ElasticNet ─────────────────────────────────────────────────────
    logger.info(f"Fitting ElasticNet(alpha={args.alpha}, l1_ratio={args.l1_ratio})")
    model = ElasticNet(
        alpha=args.alpha,
        l1_ratio=args.l1_ratio,
        max_iter=10000,
        tol=1e-4,
        random_state=42,
        selection="random",
    )
    model.fit(X_train, y_train)
    n_nonzero = int(np.sum(model.coef_ != 0))
    logger.info(f"Fitted: {n_nonzero:,} / {X_train.shape[1]:,} non-zero coefficients")

    # ── 5. Evaluate ───────────────────────────────────────────────────────────
    def evaluate(X, y, name):
        pred  = model.predict(X)
        mae   = float(mean_absolute_error(y, pred))
        medae = median_absolute_error(y, pred)
        r2    = float(r2_score(y, pred))
        pcc,_ = pearsonr(y, pred)
        logger.info(f"  {name:5s}  MAE={mae:.2f}yr  MedAE={medae:.2f}yr  R²={r2:.4f}  PCC={pcc:.4f}")
        return {"mae": mae, "medae": medae, "r2": r2, "pcc": float(pcc), "n": int(len(y))}

    logger.info("=" * 60)
    logger.info("RESULTS")
    logger.info("=" * 60)
    results = {
        "model": "ElasticNet",
        "alpha": args.alpha,
        "l1_ratio": args.l1_ratio,
        "n_features": int(X_train.shape[1]),
        "n_nonzero_coef": n_nonzero,
        "train": evaluate(X_train, y_train, "train"),
        "valid": evaluate(X_valid, y_valid, "valid"),
        "test":  evaluate(X_test,  y_test,  "test"),
    }

    # ── 6. Save ───────────────────────────────────────────────────────────────
    with open(outdir / "elasticnet_results.json", "w") as f:
        json.dump(results, f, indent=2)

    summary = (
        f"\n{'='*60}\n"
        f"ElasticNet Baseline — alpha={args.alpha}, l1_ratio={args.l1_ratio}\n"
        f"Non-zero CpGs : {n_nonzero:,} / {X_train.shape[1]:,}\n"
        f"{'='*60}\n"
        f"{'Split':<8} {'N':>6} {'MAE':>8} {'MedAE':>8} {'R²':>8}\n"
        f"{'-'*42}\n"
        f"{'train':<8} {results['train']['n']:>6} {results['train']['mae']:>8.2f} {results['train']['medae']:>8.2f} {results['train']['r2']:>8.4f}\n"
        f"{'valid':<8} {results['valid']['n']:>6} {results['valid']['mae']:>8.2f} {results['valid']['medae']:>8.2f} {results['valid']['r2']:>8.4f}\n"
        f"{'test':<8} {results['test']['n']:>6} {results['test']['mae']:>8.2f} {results['test']['medae']:>8.2f} {results['test']['r2']:>8.4f}\n"
        f"{'='*60}\n"
        f"MethylLlama V5 reference: MedAE=3.65yr, R²=0.905\n"
    )
    print(summary)
    with open(outdir / "elasticnet_summary.txt", "w") as f:
        f.write(summary)


if __name__ == "__main__":
    main()
