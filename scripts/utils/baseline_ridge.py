#!/usr/bin/env python3
"""
Baseline: Ridge + ElasticNet regression on raw beta values for age prediction.

Runs on the clean outlier-free stratified split.
Reports MAE, MedAE, RMSE, R², PCC per split + per-tissue breakdown on test.
"""

import sys
import numpy as np
import scanpy as sc
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from scipy import stats
from scipy.stats import median_abs_deviation


def medae(y_true, y_pred):
    return np.median(np.abs(y_true - y_pred))


def report(name, y_true, y_pred):
    mae  = mean_absolute_error(y_true, y_pred)
    med  = medae(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    r2   = r2_score(y_true, y_pred)
    pcc  = stats.pearsonr(y_true, y_pred)[0]
    print(f"  {name:6s}: MAE={mae:.3f}yr  MedAE={med:.3f}yr  RMSE={rmse:.3f}yr  R²={r2:.4f}  PCC={pcc:.4f}")
    return mae, med, r2


def main():
    if len(sys.argv) < 2:
        print("Usage: python baseline_ridge.py <h5ad_path>")
        sys.exit(1)

    h5ad_path = sys.argv[1]
    print(f"Loading: {h5ad_path}")
    adata = sc.read_h5ad(h5ad_path)
    print(f"  {adata.n_obs} samples × {adata.n_vars} CpGs")

    train_mask = (adata.obs["split"] == "train").values
    val_mask   = (adata.obs["split"] == "valid").values
    test_mask  = (adata.obs["split"] == "test").values
    print(f"  train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")

    import scipy.sparse as sp
    X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)

    y = adata.obs["age"].values.astype(np.float64)

    # Drop NaN ages from train (should be none in clean dataset, but guard)
    train_valid = train_mask & ~np.isnan(y)
    val_valid   = val_mask   & ~np.isnan(y)
    test_valid  = test_mask  & ~np.isnan(y)
    if train_valid.sum() < train_mask.sum():
        print(f"  WARNING: {train_mask.sum()-train_valid.sum()} train samples have NaN age — excluded")

    X_train, y_train = X[train_valid], y[train_valid]
    X_val,   y_val   = X[val_valid],   y[val_valid]
    X_test,  y_test  = X[test_valid],  y[test_valid]

    print(f"\nAge (train): mean={y_train.mean():.1f}  std={y_train.std():.1f}  "
          f"range=[{y_train.min():.0f}, {y_train.max():.0f}]")

    # ── Predict-mean baseline ─────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("BASELINE: Predict Mean Age")
    print(f"{'='*65}")
    report("Test", y_test, np.full_like(y_test, y_train.mean()))

    # ── Ridge — grid search over alpha ───────────────────────────────────────
    print(f"\n{'='*65}")
    print("RIDGE REGRESSION — alpha grid search (select by val/MAE)")
    print(f"{'='*65}")
    best_val_mae = float('inf')
    best_alpha_ridge = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
        ridge = Ridge(alpha=alpha)
        ridge.fit(X_train, y_train)
        val_mae = mean_absolute_error(y_val, ridge.predict(X_val))
        test_mae = mean_absolute_error(y_test, ridge.predict(X_test))
        print(f"  alpha={alpha:>8.2f} | val/MAE={val_mae:.3f}yr | test/MAE={test_mae:.3f}yr")
        if val_mae < best_val_mae:
            best_val_mae = val_mae
            best_alpha_ridge = alpha

    print(f"\nBest alpha={best_alpha_ridge} (val/MAE={best_val_mae:.3f}yr)")
    ridge_best = Ridge(alpha=best_alpha_ridge)
    ridge_best.fit(X_train, y_train)
    print(f"{'='*65}")
    print(f"RIDGE BEST (alpha={best_alpha_ridge})")
    print(f"{'='*65}")
    r_tr_mae, r_tr_med, _ = report("Train", y_train, ridge_best.predict(X_train))
    r_va_mae, r_va_med, _ = report("Val",   y_val,   ridge_best.predict(X_val))
    r_te_mae, r_te_med, r_te_r2 = report("Test",  y_test,  ridge_best.predict(X_test))

    # ── ElasticNet — grid search over alpha and l1_ratio ─────────────────────
    print(f"\n{'='*65}")
    print("ELASTICNET — grid search (select by val/MAE)")
    print(f"{'='*65}")
    best_val_mae_en = float('inf')
    best_params_en  = None

    for alpha in [0.01, 0.1, 1.0, 10.0, 100.0]:
        for l1_ratio in [0.1, 0.5, 0.9]:
            enet = ElasticNet(alpha=alpha, l1_ratio=l1_ratio, max_iter=10000)
            enet.fit(X_train, y_train)
            val_mae = mean_absolute_error(y_val, enet.predict(X_val))
            if val_mae < best_val_mae_en:
                best_val_mae_en = val_mae
                best_params_en  = (alpha, l1_ratio)

    print(f"Best alpha={best_params_en[0]}, l1_ratio={best_params_en[1]} (val/MAE={best_val_mae_en:.3f}yr)")
    enet_best = ElasticNet(alpha=best_params_en[0], l1_ratio=best_params_en[1], max_iter=10000)
    enet_best.fit(X_train, y_train)
    n_nonzero = np.sum(np.abs(enet_best.coef_) > 1e-8)
    print(f"Non-zero coefficients: {n_nonzero} / {len(enet_best.coef_)}")
    print(f"{'='*65}")
    print(f"ELASTICNET BEST (alpha={best_params_en[0]}, l1_ratio={best_params_en[1]})")
    print(f"{'='*65}")
    report("Train", y_train, enet_best.predict(X_train))
    report("Val",   y_val,   enet_best.predict(X_val))
    en_te_mae, en_te_med, en_te_r2 = report("Test",  y_test,  enet_best.predict(X_test))

    # ── Per-tissue breakdown (test set, best Ridge) ───────────────────────────
    if "tissue_type" in adata.obs.columns:
        print(f"\n{'='*65}")
        print("PER-TISSUE BREAKDOWN — Test set (Best Ridge)")
        print(f"{'='*65}")
        test_obs = adata.obs[test_valid].copy()
        test_preds = ridge_best.predict(X_test)
        test_obs["pred_age"] = test_preds
        test_obs["error"]    = np.abs(test_preds - y_test)

        tissue_stats = []
        for tissue in sorted(test_obs["tissue_type"].unique()):
            mask = test_obs["tissue_type"] == tissue
            errs = test_obs.loc[mask, "error"].values
            ages = test_obs.loc[mask, "age"].astype(float).values
            tissue_stats.append((tissue, len(errs), errs.mean(), np.median(errs)))

        tissue_stats.sort(key=lambda x: x[2], reverse=True)
        print(f"  {'Tissue':40s}  {'N':>5}  {'MAE':>7}  {'MedAE':>7}")
        print(f"  {'-'*40}  {'-'*5}  {'-'*7}  {'-'*7}")
        for tissue, n, mae_t, med_t in tissue_stats:
            print(f"  {str(tissue):40s}  {n:5d}  {mae_t:7.2f}  {med_t:7.2f}")

    # ── Final summary ─────────────────────────────────────────────────────────
    print(f"\n{'='*65}")
    print("FINAL SUMMARY — Test set")
    print(f"{'='*65}")
    print(f"  {'Model':30s}  {'MAE':>7}  {'MedAE':>7}  {'R²':>7}")
    print(f"  {'-'*30}  {'-'*7}  {'-'*7}  {'-'*7}")
    print(f"  {'Ridge (best alpha)':30s}  {r_te_mae:7.3f}  {r_te_med:7.3f}  {r_te_r2:7.4f}")
    print(f"  {'ElasticNet (best params)':30s}  {en_te_mae:7.3f}  {en_te_med:7.3f}  {en_te_r2:7.4f}")
    print(f"\n  MethylLlama V4b warmstart (ref):  MAE=5.546  MedAE=3.633  R²=0.904")
    print(f"  MethylLlama V4b scratch   (ref):  MAE=5.691  MedAE=3.750  R²=0.901")


if __name__ == "__main__":
    main()
