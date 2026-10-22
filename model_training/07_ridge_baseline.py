#!/usr/bin/env python3
"""Ridge-regression baseline.

Three estimates are reported: (i) a 90:10 holdout split, (ii) the same
repeated 10-fold x 5 CV protocol as the other models, and (iii) the OOF
ensemble. The holdout seed below is the seed used for the ridge holdout and
the Y-randomization runs; the original single split of the paper (seed 42,
verified against data/split_membership.csv) can be reproduced by setting
SEED_SPLIT = 42.
"""

import hashlib
import json
import os
import time

import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import RepeatedKFold
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

# ----------------------- configuration -----------------------
DATA_PATH = "../data/PIK3CA_ecfp2048.csv"
SEED_SPLIT = 17   # holdout seed used for the ridge/Y-randomization runs
TEST_SIZE = 0.1
SEED = 42
N_SPLITS = 10
N_REPEATS = 5
ALPHAS = np.logspace(-2, 6, 40)
MODEL_NAME = "Ridge"
OUT_DIR = "ridge_results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    ycols = [c for c in df.columns if "pic50" in c.lower()]
    assert ycols, "no pIC50 column found in the data"
    y_col = ycols[0]
    bits = [c for c in df.columns if c.startswith("bit_")]
    assert bits, "no bit_* fingerprint columns found"
    X = df[bits]
    y = pd.to_numeric(df[y_col], errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X[ok].astype(float).reset_index(drop=True), y[ok].reset_index(drop=True)
    print(f"Data: {X.shape[0]} compounds x {X.shape[1]} features")
    return X, y


def build_ridge():
    return Pipeline([
        ("scaler", StandardScaler()),
        ("ridge", RidgeCV(alphas=ALPHAS)),
    ])


def main():
    X, y = load_data()

    # ---- holdout estimate ----
    perm = np.random.RandomState(SEED_SPLIT).permutation(len(X))
    n_test = int(np.ceil(TEST_SIZE * len(X)))
    tr_idx, te_idx = perm[n_test:], perm[:n_test]
    est = build_ridge().fit(X.iloc[tr_idx], y.iloc[tr_idx])
    yp = est.predict(X.iloc[te_idx])
    ho_r2 = float(r2_score(y.iloc[te_idx], yp))
    ho_rmse = float(np.sqrt(mean_squared_error(y.iloc[te_idx], yp)))
    alpha_ho = float(est.named_steps["ridge"].alpha_)
    print(f"[{MODEL_NAME}] holdout (seed={SEED_SPLIT}, {1-TEST_SIZE:.0%}/{TEST_SIZE:.0%}) "
          f"R2={ho_r2:.4f}  RMSE={ho_rmse:.4f}  alpha={alpha_ho:.4g}")
    with open(os.path.join(OUT_DIR, "Ridge_holdout_summary.json"), "w") as f:
        json.dump({"protocol": f"holdout seed={SEED_SPLIT}", "n_test": int(n_test),
                   "r2": ho_r2, "rmse": ho_rmse, "alpha": alpha_ho}, f, indent=2)

    # ---- repeated 10-fold x 5 CV (same splits as the other models) ----
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rkf.split(X))
    fp = hashlib.sha256(str([tuple(t) for _, t in splits]).encode()).hexdigest()[:16]
    print(f"[{MODEL_NAME}] repeated-CV 50-fold split fingerprint: {fp} (must match other models)")

    rows, oof = [], {}
    t0 = time.time()
    for i, (tr, te) in enumerate(splits):
        e = clone(build_ridge()).fit(X.iloc[tr], y.iloc[tr])
        p = e.predict(X.iloc[te])
        for j, mol in enumerate(te):
            oof.setdefault(int(mol), []).append(float(p[j]))
        rows.append({
            "repeat_no": i // N_SPLITS + 1, "fold_no": i % N_SPLITS + 1,
            "n_train": len(tr), "n_test": len(te),
            "alpha": float(e.named_steps["ridge"].alpha_),
            "train_r2": float(e.score(X.iloc[tr], y.iloc[tr])),
            "test_r2": float(r2_score(y.iloc[te], p)),
            "test_rmse": float(np.sqrt(mean_squared_error(y.iloc[te], p))),
        })
    df_folds = pd.DataFrame(rows)
    df_folds.to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_fold_details.csv"), index=False)

    ens_y = np.array([y.iloc[k] for k in oof.keys()])
    ens_p = np.array([np.mean(v) for v in oof.values()])
    ens_r2 = float(r2_score(ens_y, ens_p))
    ens_rmse = float(np.sqrt(mean_squared_error(ens_y, ens_p)))
    pd.DataFrame({"idx": list(oof.keys()), "y": ens_y, "pred_mean": ens_p,
                  "n_preds": [len(v) for v in oof.values()]}).to_csv(
        os.path.join(OUT_DIR, f"{MODEL_NAME}_ensemble_preds.csv"), index=False)

    r2_m, r2_sd = df_folds["test_r2"].mean(), df_folds["test_r2"].std(ddof=1)
    rm_m, rm_sd = df_folds["test_rmse"].mean(), df_folds["test_rmse"].std(ddof=1)
    pd.DataFrame([{
        "model": MODEL_NAME,
        "best_params": json.dumps({"alpha_grid": "RidgeCV(logspace(-2,6,40))"}),
        "inner_alpha_median": float(df_folds["alpha"].median()),
        "r2_mean": r2_m, "r2_sd": r2_sd,
        "rmse_mean": rm_m, "rmse_sd": rm_sd,
        "ensemble_r2": ens_r2, "ensemble_rmse": ens_rmse,
        "total_seconds": round(time.time() - t0, 1),
    }]).to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_best_params.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "alpha_holdout": alpha_ho,
                   "alpha_median_5x10": float(df_folds["alpha"].median())}, f, indent=2)

    print(f"[{MODEL_NAME}] per-fold R2   = {r2_m:.4f} +/- {r2_sd:.4f}")
    print(f"[{MODEL_NAME}] per-fold RMSE = {rm_m:.4f} +/- {rm_sd:.4f}")
    print(f"[{MODEL_NAME}] OOF ensemble R2 = {ens_r2:.4f} | RMSE = {ens_rmse:.4f}")
    print(f"done in {time.time()-t0:.0f} s; results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
