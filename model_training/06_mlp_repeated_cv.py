#!/usr/bin/env python3
"""MLP: repeated cross-validation (10-fold x 5) with out-of-fold ensemble.

Protocol
--------
1. Inner 5-fold CV over the hyperparameter grid below selects the configuration
   on training data only.
2. The frozen configuration is evaluated with RepeatedKFold(n_splits=10,
   n_repeats=5) = 50 test folds. Every molecule is scored by five models that
   never saw it; the per-molecule mean gives the OOF ensemble estimate
   (Table 1 of the paper).

Output (./5x5cv_results_10fold/):
  MLP_best_params.json, MLP_fold_details.csv, MLP_summary.csv,
  MLP_ensemble_preds.csv (per-molecule OOF predictions and their mean)
"""

import copy
import hashlib
import json
import os
import time
import warnings

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, Dataset

from sklearn.decomposition import PCA
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.model_selection import KFold, ParameterGrid, RepeatedKFold
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ----------------------- configuration -----------------------
DATA_PATH = "../data/PIK3CA_ecfp2048.csv"
SEED = 42
N_SPLITS = 10
N_REPEATS = 5
INNER_SPLITS = 5
MODEL_NAME = "MLP"
OUT_DIR = "5x5cv_results_10fold"

USE_PCA = False
PCA_COMPONENTS = 512
USE_BN = True

PARAM_GRID = {
    "hidden_sizes":  [[512, 256, 128], [1024, 512, 256]],
    "dropout":       [0.2, 0.4],
    "lr":            [1e-3, 3e-3],
    "weight_decay":  [0.0, 1e-4],
}
MAX_EPOCHS = 150
PATIENCE = 25
BATCH_SIZE = 256
VAL_FRAC = 0.15

os.makedirs(OUT_DIR, exist_ok=True)
_Y_MU, _Y_SD = 0.0, 1.0


def load_data():
    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(f"data file not found: {DATA_PATH}; adjust DATA_PATH")
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
    print(f"Data: {X.shape[0]} compounds x {X.shape[1]} fingerprint features")
    return X.to_numpy(), y.to_numpy()


class QSAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class FNN_QSAR(nn.Module):
    def __init__(self, input_size, hidden_sizes, dropout_rate, use_bn=True):
        super().__init__()
        layers, prev = [], input_size
        for hs in hidden_sizes:
            layers.append(nn.Linear(prev, hs))
            if use_bn:
                layers += [nn.BatchNorm1d(hs), nn.ReLU(), nn.Dropout(dropout_rate)]
            else:
                layers += [nn.ReLU(), nn.Dropout(dropout_rate)]
            prev = hs
        layers.append(nn.Linear(prev, 1))
        self.fc = nn.Sequential(*layers)

    def forward(self, x):
        return self.fc(x).squeeze(-1)


def _fit_fold(X, y, tr_idx, te_idx, params, seed, device):
    """Fit one fold; return fold metrics and raw-scale (y, prediction) for the OOF pool."""
    rng = np.random.RandomState(seed)
    perm = rng.permutation(len(tr_idx))
    n_val = max(1, int(round(VAL_FRAC * len(tr_idx))))
    va_idx, tr2_idx = tr_idx[perm[:n_val]], tr_idx[perm[n_val:]]

    scaler = StandardScaler().fit(X[tr2_idx])
    X_tr2 = scaler.transform(X[tr2_idx])
    pca = None
    if USE_PCA:
        pca = PCA(n_components=min(PCA_COMPONENTS, X_tr2.shape[1], X_tr2.shape[0]))
        X_tr2 = pca.fit_transform(X_tr2)
    X_va = scaler.transform(X[va_idx])
    X_te = scaler.transform(X[te_idx])
    if pca is not None:
        X_va, X_te = pca.transform(X_va), pca.transform(X_te)

    bs = min(BATCH_SIZE, len(X_tr2))
    n_full = len(X_tr2) // bs
    drop_last = n_full >= 2 and (len(X_tr2) % bs) == 1
    tr_loader = DataLoader(QSAR_Dataset(X_tr2, y[tr2_idx]), batch_size=bs,
                           shuffle=True, drop_last=drop_last)
    va_loader = DataLoader(QSAR_Dataset(X_va, y[va_idx]), batch_size=512, shuffle=False)
    te_loader = DataLoader(QSAR_Dataset(X_te, y[te_idx]), batch_size=512, shuffle=False)

    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    model = FNN_QSAR(X_tr2.shape[1], params["hidden_sizes"], params["dropout"],
                     use_bn=USE_BN).to(device)
    optimizer = optim.Adam(model.parameters(), lr=params["lr"],
                           weight_decay=params["weight_decay"])
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="min", factor=0.5,
                                                     patience=max(2, PATIENCE // 2),
                                                     min_lr=1e-6)
    crit = nn.MSELoss()

    best_rmse, best_epoch, bad, best_state = float("inf"), 0, 0, None
    for epoch in range(1, MAX_EPOCHS + 1):
        model.train()
        for Xb, yb in tr_loader:
            Xb, yb = Xb.to(device), yb.to(device)
            optimizer.zero_grad()
            loss = crit(model(Xb), yb)
            loss.backward()
            optimizer.step()
        model.eval()
        yv, yp = [], []
        with torch.no_grad():
            for Xb, _ in va_loader:
                yp.extend(model(Xb.to(device)).cpu().numpy())
            for _, yb in va_loader:
                yv.extend(yb.numpy())
        val_rmse = float(np.sqrt(mean_squared_error(yv, yp)))
        scheduler.step(val_rmse)
        if val_rmse < best_rmse:
            best_rmse, best_epoch, bad = val_rmse, epoch, 0
            best_state = copy.deepcopy(model.state_dict())
        else:
            bad += 1
            if bad >= PATIENCE:
                break
    if best_state is None:
        best_state = copy.deepcopy(model.state_dict())
    model.load_state_dict(best_state)
    model.eval()

    def _pred(loader):
        ps = []
        with torch.no_grad():
            for Xb, _ in loader:
                ps.extend(model(Xb.to(device)).cpu().numpy())
        return np.array(ps)

    yp_tr = _pred(DataLoader(QSAR_Dataset(X_tr2, y[tr2_idx]), batch_size=512, shuffle=False))
    yp_te = _pred(te_loader)
    return {
        "train_r2": float(r2_score(y[tr2_idx], yp_tr)),
        "test_r2": float(r2_score(y[te_idx], yp_te)),
        "test_rmse": float(np.sqrt(mean_squared_error(y[te_idx], yp_te))) * _Y_SD,
        "best_epoch": best_epoch,
        "val_rmse": best_rmse * _Y_SD,
        "te_idx": te_idx,
        "y_te_raw": y[te_idx] * _Y_SD + _Y_MU,
        "pred_te_raw": yp_te * _Y_SD + _Y_MU,
    }


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"[{MODEL_NAME}] device: {device}")
    X, y = load_data()
    X = X.astype(np.float32)
    y = y.astype(np.float64)
    global _Y_MU, _Y_SD
    _Y_MU, _Y_SD = float(y.mean()), float(y.std())
    y = (y - _Y_MU) / _Y_SD
    print(f"[{MODEL_NAME}] y standardised: mean={_Y_MU:.3f}, std={_Y_SD:.3f}")

    combos = list(ParameterGrid(PARAM_GRID))
    print(f"[{MODEL_NAME}] {len(combos)} hyperparameter combinations; "
          f"inner {INNER_SPLITS}-fold search + repeated CV (50 folds) + OOF ensemble")

    # step 1: inner 5-fold hyperparameter search
    inner_cv = KFold(n_splits=INNER_SPLITS, shuffle=True, random_state=SEED)
    best_score, best_params = -np.inf, None
    t0 = time.time()
    for ci, params in enumerate(combos, 1):
        fold_scores = []
        for fi, (tr_idx, te_idx) in enumerate(inner_cv.split(X)):
            r = _fit_fold(X, y, tr_idx, te_idx, params,
                          seed=SEED + fi * 131 + ci, device=device)
            fold_scores.append(r["test_r2"])
        score = float(np.mean(fold_scores))
        print(f"    combo {ci}/{len(combos)} {params} -> inner R2 {score:.4f} "
              f"(elapsed {time.time()-t0:.0f} s)")
        if score > best_score:
            best_score, best_params = score, params
    print(f"[{MODEL_NAME}] best hyperparameters: {best_params} "
          f"(inner R2 = {best_score:.4f}); frozen")

    # step 2: repeated CV + OOF pool
    rkf = RepeatedKFold(n_splits=N_SPLITS, n_repeats=N_REPEATS, random_state=SEED)
    splits = list(rkf.split(X))
    fp = hashlib.sha256(str([tuple(t) for _, t in splits]).encode()).hexdigest()[:16]
    print(f"[{MODEL_NAME}] 50-fold split fingerprint: {fp}")

    rows, oof = [], {}
    t0 = time.time()
    for i, (tr_idx, te_idx) in enumerate(splits):
        r = _fit_fold(X, y, tr_idx, te_idx, best_params,
                      seed=SEED + 500 + i, device=device)
        rows.append({
            "repeat_no": i // N_SPLITS + 1, "fold_no": i % N_SPLITS + 1,
            "n_train": len(tr_idx), "n_test": len(te_idx),
            "train_r2": r["train_r2"], "test_r2": r["test_r2"],
            "test_rmse": r["test_rmse"], "best_epoch": r["best_epoch"],
        })
        for idx, yt, pt in zip(r["te_idx"], r["y_te_raw"], r["pred_te_raw"]):
            oof.setdefault(int(idx), []).append((float(yt), float(pt)))
        print(f"    fold {i+1:2d}/{len(splits)} (rep {rows[-1]['repeat_no']}, "
              f"fold {rows[-1]['fold_no']})  R2={r['test_r2']:.4f} "
              f"RMSE={r['test_rmse']:.4f} (epoch {r['best_epoch']})")

    df_folds = pd.DataFrame(rows)
    df_folds.to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_fold_details.csv"), index=False)

    ens_y = np.array([v[0][0] for v in oof.values()])
    ens_p = np.array([np.mean([p for _, p in v]) for v in oof.values()])
    ens_r2 = float(r2_score(ens_y, ens_p))
    ens_rmse = float(np.sqrt(mean_squared_error(ens_y, ens_p)))
    pd.DataFrame({"idx": list(oof.keys()), "y": ens_y, "pred_mean": ens_p,
                  "pred_n": [len(v) for v in oof.values()]}).to_csv(
        os.path.join(OUT_DIR, f"{MODEL_NAME}_ensemble_preds.csv"), index=False)
    print(f"[{MODEL_NAME}] OOF ensemble (mean of {N_REPEATS} predictions per molecule): "
          f"R2={ens_r2:.4f} RMSE={ens_rmse:.4f}")

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
        "best_params": json.dumps(best_params),
        "use_bn": USE_BN,
        "inner_cv_r2": best_score,
        "r2_mean": r2_m, "r2_sd": r2_sd,
        "r2_min": df_folds["test_r2"].min(), "r2_max": df_folds["test_r2"].max(),
        "rmse_mean": rm_m, "rmse_sd": rm_sd,
        "rmse_min": df_folds["test_rmse"].min(), "rmse_max": df_folds["test_rmse"].max(),
        "rep_r2_mean": rp_m, "rep_r2_sd": rp_sd,
        "rep_rmse_mean": rpm_m, "rep_rmse_sd": rpm_sd,
        "ensemble_r2": ens_r2, "ensemble_rmse": ens_rmse,
        "total_seconds": round(time.time() - t0, 1),
    }])
    summary.to_csv(os.path.join(OUT_DIR, f"{MODEL_NAME}_summary.csv"), index=False)
    with open(os.path.join(OUT_DIR, f"{MODEL_NAME}_best_params.json"), "w") as f:
        json.dump({"model": MODEL_NAME, "best_params": best_params,
                   "inner_cv_r2": best_score}, f, indent=2)

    print("\n" + "=" * 66)
    print(f"{MODEL_NAME}  repeated CV ({len(df_folds)} test folds)")
    print("=" * 66)
    print(f"per-fold R2   = {r2_m:.4f} +/- {r2_sd:.4f}   "
          f"(min {df_folds['test_r2'].min():.4f}, max {df_folds['test_r2'].max():.4f})")
    print(f"per-fold RMSE = {rm_m:.4f} +/- {rm_sd:.4f}")
    print(f"OOF ensemble R2 = {ens_r2:.4f} | RMSE = {ens_rmse:.4f}")
    print(f"done in {time.time()-t0:.0f} s; results in {OUT_DIR}/")


if __name__ == "__main__":
    main()
