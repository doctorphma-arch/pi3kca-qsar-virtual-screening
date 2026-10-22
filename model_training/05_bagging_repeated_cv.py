#!/usr/bin/env python3
"""Bagging: repeated cross-validation (10-fold x 5) with out-of-fold ensemble.

Protocol
--------
1. Inner 5-fold CV on the full curated set selects hyperparameters (training-only
   model selection; the held-out single split is never used here).
2. The frozen configuration is evaluated with RepeatedKFold(n_splits=10,
   n_repeats=5) = 50 test folds. Every molecule is predicted exactly five times
   by models that never saw it; the per-molecule mean gives the out-of-fold
   (OOF) ensemble estimate reported in the paper (Table 1).
"""

import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import GridSearchCV, KFold, ParameterGrid, RepeatedKFold

warnings.filterwarnings("ignore")
from sklearn.ensemble import BaggingRegressor
from sklearn.tree import DecisionTreeRegressor

# ----------------------- configuration -----------------------
DATA_PATH = "../data/PIK3CA_ecfp2048.csv"
SEED = 42
N_SPLITS = 10
N_REPEATS = 5
INNER_SPLITS = 5
MODEL_NAME = "Bagging"
OUT_DIR = "5x10cv_results"
os.makedirs(OUT_DIR, exist_ok=True)

# ----------------------- data -----------------------
df = pd.read_csv(DATA_PATH)
_drop = [c for c in df.columns
         if any(k in c.lower() for k in ("smiles", "chembl", "name", "unnamed",
                                         "molecule", "catalog", "id"))]
df.drop(columns=_drop, inplace=True, errors="ignore")
_ycols = [c for c in df.columns if "pic50" in c.lower()]
assert _ycols, "no pIC50 column found in the data"
y_col = _ycols[0]
X = df.drop(columns=[y_col]).select_dtypes(include=[np.number])
y = pd.to_numeric(df[y_col], errors="coerce")
_ok = y.notna() & X.notna().all(axis=1)
X, y = X[_ok].astype(float).reset_index(drop=True), y[_ok].reset_index(drop=True)
print(f"Data: {{X.shape[0]}} compounds x {{X.shape[1]}} features")

# ----------------------- model and grid -----------------------
def _make_bagging():
    """BaggingRegressor: 'estimator' on new sklearn, 'base_estimator' on old."""
    dt = DecisionTreeRegressor(random_state=SEED)
    try:
        return BaggingRegressor(estimator=dt, bootstrap=True,
                                bootstrap_features=False, random_state=SEED)
    except TypeError:
        return BaggingRegressor(base_estimator=dt, bootstrap=True,
                                bootstrap_features=False, random_state=SEED)


ESTIMATOR = _make_bagging()
PARAM_GRID = {
    "n_estimators": [600, 1000],
    "max_samples":  [0.7, 1.0],
    "max_features": [0.3, 0.5],
}

# ------------- step 1: inner 5-fold CV hyperparameter search -------------
inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
n_combos = len(list(ParameterGrid(PARAM_GRID)))
print(f"[{Bagging}] inner {{INNER_SPLITS}}-fold CV: {{n_combos}} hyperparameter combinations ...")
t0 = time.time()
gs = GridSearchCV(ESTIMATOR, PARAM_GRID, scoring="r2", cv=inner_cv, n_jobs=-1)
gs.fit(X, y)
print(f"    best hyperparameters: {{gs.best_params_}}")
print(f"    inner CV mean R2: {{gs.best_score_:.4f}}   (search took {{time.time()-t0:.0f}} s)")

# ------------- step 2: repeated CV (frozen config) + OOF ensemble -------------
rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
_splits = list(rkf.split(X))
_fp = hashlib.sha256(str([tuple(t) for _, t in _splits]).encode()).hexdigest()[:16]
print(f"[{Bagging}] 50-fold split fingerprint: {{_fp}} (must match across models)")

rows, oof = [], {{}}
t0 = time.time()
for i, (tr_idx, te_idx) in enumerate(_splits):
    est = clone(gs.best_estimator_)
    if "n_jobs" in est.get_params():
        est.set_params(n_jobs=-1)
    X_tr, X_te = X.iloc[tr_idx], X.iloc[te_idx]
    y_tr, y_te = y.iloc[tr_idx], y.iloc[te_idx]
    est.fit(X_tr, y_tr)
    yp = est.predict(X_te)
    yt = y_te.to_numpy()
    for j, idx_mol in enumerate(te_idx):
        oof.setdefault(int(idx_mol), []).append((float(yt[j]), float(yp[j])))
    rows.append({
        "repeat_no": i // N_SPLITS + 1,
        "fold_no": i % N_SPLITS + 1,
        "n_train": len(tr_idx), "n_test": len(te_idx),
        "train_r2": est.score(X_tr, y_tr),
        "test_r2": r2_score(y_te, yp),
        "test_rmse": float(np.sqrt(mean_squared_error(y_te, yp))),
    })
    print(f"    fold {{i+1:2d}}/50 (rep {{rows[-1]['repeat_no']}}, fold {{rows[-1]['fold_no']}})"
          f"  R2={{rows[-1]['test_r2']:.4f}} RMSE={{rows[-1]['test_rmse']:.4f}}")

df_folds = pd.DataFrame(rows)
df_folds.to_csv(os.path.join(OUT_DIR, f"{Bagging}_fold_details.csv"), index=False)

# OOF ensemble: mean of the 5 out-of-fold predictions per molecule
ens_y = np.array([v[0][0] for v in oof.values()])
ens_p = np.array([np.mean([p for _, p in v]) for v in oof.values()])
ens_r2 = float(r2_score(ens_y, ens_p))
ens_rmse = float(np.sqrt(mean_squared_error(ens_y, ens_p)))
pd.DataFrame({{"idx": list(oof.keys()), "y": ens_y, "pred_mean": ens_p,
               "n_preds": [len(v) for v in oof.values()]}}).to_csv(
    os.path.join(OUT_DIR, f"{Bagging}_ensemble_preds.csv"), index=False)
print(f"[{Bagging}] OOF ensemble: R2={{ens_r2:.4f}} RMSE={{ens_rmse:.4f}}")


def _ms(v):
    return float(np.mean(v)), float(np.std(v, ddof=1))


r2_m, r2_sd = _ms(df_folds["test_r2"])
rm_m, rm_sd = _ms(df_folds["test_rmse"])
rep_r2 = df_folds.groupby("repeat_no")["test_r2"].mean()
rep_rm = df_folds.groupby("repeat_no")["test_rmse"].mean()
rp_m, rp_sd = _ms(rep_r2.values)
rpm_m, rpm_sd = _ms(rep_rm.values)

summary = pd.DataFrame([{
    "model": MODEL_NAME,
    "best_params": json.dumps(gs.best_params_),
    "inner_cv_r2": gs.best_score_,
    "r2_mean": r2_m, "r2_sd": r2_sd,
    "r2_min": df_folds["test_r2"].min(), "r2_max": df_folds["test_r2"].max(),
    "rmse_mean": rm_m, "rmse_sd": rm_sd,
    "rmse_min": df_folds["test_rmse"].min(), "rmse_max": df_folds["test_rmse"].max(),
    "rep_r2_mean": rp_m, "rep_r2_sd": rp_sd,
    "rep_rmse_mean": rpm_m, "rep_rmse_sd": rpm_sd,
    "ensemble_r2": ens_r2, "ensemble_rmse": ens_rmse,
    "total_seconds": round(time.time() - t0, 1),
}])
summary.to_csv(os.path.join(OUT_DIR, f"{Bagging}_summary.csv"), index=False)
with open(os.path.join(OUT_DIR, f"{Bagging}_best_params.json"), "w") as f:
    json.dump({{"model": MODEL_NAME, "best_params": gs.best_params_,
                "inner_cv_r2": gs.best_score_}}, f, indent=2)

print("\n" + "=" * 64)
print(f"{Bagging}  repeated CV ({{len(df_folds)}} test folds)")
print("=" * 64)
print(f"per-fold R2   = {{r2_m:.4f}} +/- {{r2_sd:.4f}}   (min {{df_folds['test_r2'].min():.4f}}, max {{df_folds['test_r2'].max():.4f}})")
print(f"per-fold RMSE = {{rm_m:.4f}} +/- {{rm_sd:.4f}}")
print(f"OOF ensemble R2 = {{ens_r2:.4f}} | RMSE = {{ens_rmse:.4f}}")
print(f"done in {{time.time()-t0:.0f}} s; results in {{OUT_DIR}}/")
