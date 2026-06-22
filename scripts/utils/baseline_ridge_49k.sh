#!/bin/bash -l
#SBATCH --job-name=baseline-ridge-49k
#SBATCH --partition=salmon
#SBATCH --gres=gpu:l40s:1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=128G
#SBATCH --time=4:00:00

#SBATCH --output=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.out
#SBATCH --error=/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl/logs_llama-wced/%x_%j.err

set -euo pipefail

REPO="/sci/labs/benjamin.yakir/netanel.azran/repos/BMFM-RNA/methyl"
DATA="/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"

echo "============================================================"
echo "RIDGE / ELASTICNET BASELINE — MethylGPT 49k dataset"
echo "============================================================"
echo "Job: ${SLURM_JOB_ID} | Host: $(hostname) | Time: $(date)"
echo "Data: ${DATA}"
echo "CPUs: ${SLURM_CPUS_PER_TASK}"
echo "============================================================"

source /etc/profile.d/modules.sh 2>/dev/null || true

cd "${REPO}"
source bmfm_methyl_env/bin/activate

export OMP_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export OPENBLAS_NUM_THREADS=${SLURM_CPUS_PER_TASK}
export MKL_NUM_THREADS=${SLURM_CPUS_PER_TASK}

python - <<'PYEOF'
import sys
import numpy as np
import scanpy as sc
import scipy.sparse as sp
from sklearn.linear_model import Ridge, ElasticNet
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from sklearn.model_selection import train_test_split
from scipy import stats

DATA = "/sci/labs/benjamin.yakir/netanel.azran/data/data_methyl_finetune_49k_h5ad/finetuning_49k.h5ad"

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

# ── Load data ─────────────────────────────────────────────────────────────────
print(f"Loading: {DATA}")
adata = sc.read_h5ad(DATA)
print(f"  {adata.n_obs} samples × {adata.n_vars} CpGs")
print(f"  obs columns: {list(adata.obs.columns)}")

X = adata.X.toarray() if sp.issparse(adata.X) else np.array(adata.X, dtype=np.float32)
y = adata.obs["age"].values.astype(np.float64)

nan_frac = np.isnan(X).mean()
print(f"  NaN fraction in X: {nan_frac:.4f} ({nan_frac*100:.1f}%)")

# ── Build splits ──────────────────────────────────────────────────────────────
if "split" in adata.obs.columns:
    print(f"\nUsing existing 'split' column:")
    print(adata.obs["split"].value_counts().to_string())
    train_mask = (adata.obs["split"] == "train").values
    val_mask   = (adata.obs["split"].isin(["valid", "val", "validation"])).values
    test_mask  = (adata.obs["split"] == "test").values
else:
    # MethylGPT paper: train=5461, val=1366, test=4626 (total=13453)
    # Create same-size stratified split for fair comparison
    print(f"\nNo 'split' column — creating stratified split matching MethylGPT paper sizes")
    print(f"  Paper: train=5461, val=1366, test=4626")
    valid_idx = ~np.isnan(y)
    X_v, y_v = X[valid_idx], y[valid_idx]
    n = len(y_v)
    print(f"  Samples with valid age: {n}")

    # Stratified split by age bins (same logic as our data)
    idx = np.arange(n)
    # test ~34.3% of total (4626/13453)
    idx_trainval, idx_test = train_test_split(idx, test_size=4626, random_state=42,
                                               shuffle=True)
    # val ~10.1% of total (1366/13453)
    idx_train, idx_val = train_test_split(idx_trainval, test_size=1366, random_state=42,
                                           shuffle=True)
    train_mask = np.zeros(n, dtype=bool); train_mask[idx_train] = True
    val_mask   = np.zeros(n, dtype=bool); val_mask[idx_val]     = True
    test_mask  = np.zeros(n, dtype=bool); test_mask[idx_test]   = True
    X, y = X_v, y_v
    print(f"  Created: train={train_mask.sum()}  val={val_mask.sum()}  test={test_mask.sum()}")

# Remove NaN ages
train_mask = train_mask & ~np.isnan(y)
val_mask   = val_mask   & ~np.isnan(y)
test_mask  = test_mask  & ~np.isnan(y)

X_train, y_train = X[train_mask], y[train_mask]
X_val,   y_val   = X[val_mask],   y[val_mask]
X_test,  y_test  = X[test_mask],  y[test_mask]

print(f"\nFinal splits: train={len(y_train)}  val={len(y_val)}  test={len(y_test)}")

# ── Mean imputation (train mean → fill all splits) ────────────────────────────
print("Imputing NaN with per-CpG train mean...")
col_means = np.nanmean(X_train, axis=0)
# If a CpG is ALL NaN in train, fill with 0.5 (midpoint of beta scale)
col_means = np.where(np.isnan(col_means), 0.5, col_means)
for arr in [X_train, X_val, X_test]:
    nan_mask = np.isnan(arr)
    arr[nan_mask] = np.take(col_means, np.where(nan_mask)[1])
print(f"  NaN after imputation: {np.isnan(X_train).sum() + np.isnan(X_val).sum() + np.isnan(X_test).sum()}")
print(f"Age (train): mean={y_train.mean():.1f}  std={y_train.std():.1f}  "
      f"range=[{y_train.min():.0f}, {y_train.max():.0f}]")

# ── Predict-mean baseline ─────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("BASELINE: Predict Mean Age")
print(f"{'='*65}")
report("Test", y_test, np.full_like(y_test, y_train.mean()))

# ── Ridge — grid search ───────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("RIDGE REGRESSION — alpha grid search (select by val/MAE)")
print(f"{'='*65}")
best_val_mae = float('inf')
best_alpha_ridge = None

for alpha in [0.01, 0.1, 1.0, 10.0, 100.0, 1000.0, 10000.0]:
    ridge = Ridge(alpha=alpha)
    ridge.fit(X_train, y_train)
    val_mae  = mean_absolute_error(y_val,  ridge.predict(X_val))
    test_mae = mean_absolute_error(y_test, ridge.predict(X_test))
    val_med  = medae(y_val,  ridge.predict(X_val))
    test_med = medae(y_test, ridge.predict(X_test))
    print(f"  alpha={alpha:>8.2f} | val/MAE={val_mae:.3f}  val/MedAE={val_med:.3f} | "
          f"test/MAE={test_mae:.3f}  test/MedAE={test_med:.3f}")
    if val_mae < best_val_mae:
        best_val_mae = val_mae
        best_alpha_ridge = alpha

print(f"\nBest alpha={best_alpha_ridge} (val/MAE={best_val_mae:.3f})")
ridge_best = Ridge(alpha=best_alpha_ridge)
ridge_best.fit(X_train, y_train)
print(f"{'='*65}")
print(f"RIDGE BEST (alpha={best_alpha_ridge})")
print(f"{'='*65}")
report("Train", y_train, ridge_best.predict(X_train))
report("Val",   y_val,   ridge_best.predict(X_val))
r_te_mae, r_te_med, r_te_r2 = report("Test",  y_test,  ridge_best.predict(X_test))

# ── ElasticNet — grid search ──────────────────────────────────────────────────
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

print(f"Best alpha={best_params_en[0]}, l1_ratio={best_params_en[1]} (val/MAE={best_val_mae_en:.3f})")
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

# ── Final summary ─────────────────────────────────────────────────────────────
print(f"\n{'='*65}")
print("FINAL SUMMARY — Test set (MethylGPT 49k dataset)")
print(f"{'='*65}")
print(f"  {'Model':35s}  {'MAE':>7}  {'MedAE':>7}  {'R²':>7}")
print(f"  {'-'*35}  {'-'*7}  {'-'*7}  {'-'*7}")
print(f"  {'Ridge (best alpha, grid search)':35s}  {r_te_mae:7.3f}  {r_te_med:7.3f}  {r_te_r2:7.4f}")
print(f"  {'ElasticNet (best params, grid search)':35s}  {en_te_mae:7.3f}  {en_te_med:7.3f}  {en_te_r2:7.4f}")
print(f"\n  MethylGPT reported (Fig 4e):")
print(f"  {'MethylGPT (transformer)':35s}  {'?':>7}  {'4.59':>7}  {'?':>7}  (test set)")
print(f"  {'ElasticNet (their baseline)':35s}  {'?':>7}  {'5.10':>7}  {'?':>7}  (test set)")
PYEOF

echo "============================================================"
echo "Done: $(date)"
echo "============================================================"
