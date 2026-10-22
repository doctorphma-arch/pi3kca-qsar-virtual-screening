#!/usr/bin/env python3
"""Y-randomization (label permutation) test for the deployed MLP configuration.

The real model is compared against 200 models trained on randomly permuted
labels. Reported per the paper: permuted R2 mean +/- SD, range, and the
one-sided permutation p-value (real R2 must exceed all permuted runs).

Set CV_MODE = "holdout" to evaluate on a single 90:10 holdout split
(SEED_SPLIT selects the split; the split used for the published run is
seed 17), or "repeated" for the repeated-KFold variant.
"""

import json
import os
import time

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim

# ----------------------- configuration -----------------------
DATA_PATH = "../data/PIK3CA_ecfp2048.csv"
N_ITER = 200
CV_MODE = "holdout"        # "holdout" | "kfold" | "repeated"
SEED_SPLIT = 17            # holdout split seed used for the published run
TEST_SIZE = 0.1
N_SPLITS = 5
N_REPEATS = 5

# MLP configuration (identical to the deployed model)
TORCH_SEED = 42
HIDDEN = [512, 256]
DROPOUT = 0.3
LR = 1e-3
EPOCHS = 50
BATCH_SIZE = 32

OUT_DIR = "yrandom_results"
os.makedirs(OUT_DIR, exist_ok=True)


def load_data():
    df = pd.read_csv(DATA_PATH)
    ycols = [c for c in df.columns if "pic50" in c.lower()]
    assert ycols, "no pIC50 column found"
    y_col = ycols[0]
    bits = [c for c in df.columns if c.startswith("bit_")]
    assert bits, "no bit_* fingerprint columns found"
    X = df[bits]
    y = pd.to_numeric(df[y_col], errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    X, y = X[ok].to_numpy(np.float32), y[ok].to_numpy(np.float64)
    print(f"Data: {X.shape[0]} compounds x {X.shape[1]} features")
    return X, y


def _make_model(input_size):
    layers, prev = [], input_size
    for hs in HIDDEN:
        layers += [nn.Linear(prev, hs), nn.ReLU(), nn.Dropout(DROPOUT)]
        prev = hs
    layers.append(nn.Linear(prev, 1))
    return nn.Sequential(*layers)


def _r2(y_true, y_pred):
    y_true = np.asarray(y_true, float)
    y_pred = np.asarray(y_pred, float)
    ss_res = float(np.sum((y_true - y_pred) ** 2))
    ss_tot = float(np.sum((y_true - y_true.mean()) ** 2))
    return 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0


def fit_predict(X_tr, yz_tr, X_te, device):
    torch.manual_seed(TORCH_SEED)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(TORCH_SEED)
    model = _make_model(X_tr.shape[1]).to(device)
    opt = optim.Adam(model.parameters(), lr=LR)
    crit = nn.MSELoss()
    Xt = torch.as_tensor(X_tr, dtype=torch.float32, device=device)
    yt = torch.as_tensor(yz_tr, dtype=torch.float32, device=device)
    n, bs = len(Xt), BATCH_SIZE
    model.train()
    for _ in range(EPOCHS):
        perm = torch.randperm(n, device=device)
        for i in range(0, n, bs):
            idx = perm[i:i + bs]
            opt.zero_grad()
            loss = crit(model(Xt[idx]).squeeze(-1), yt[idx])
            loss.backward()
            opt.step()
    model.eval()
    Xe = torch.as_tensor(X_te, dtype=torch.float32, device=device)
    with torch.no_grad():
        preds = model(Xe).squeeze(-1).cpu().numpy()
    return preds


def _kfold_indices(n, k, seed):
    idx = np.random.RandomState(seed).permutation(n)
    folds = np.array_split(idx, k)
    for i in range(k):
        te = folds[i]
        tr = np.concatenate([folds[j] for j in range(k) if j != i])
        yield tr, te


def build_splits(n):
    if CV_MODE == "holdout":
        perm = np.random.RandomState(SEED_SPLIT).permutation(n)
        n_test = int(np.ceil(TEST_SIZE * n))
        return [(perm[n_test:], perm[:n_test])]
    if CV_MODE == "kfold":
        return list(_kfold_indices(n, N_SPLITS, SEED_SPLIT))
    # repeated
    splits = []
    for r in range(N_REPEATS):
        splits += list(_kfold_indices(n, N_SPLITS, SEED_SPLIT + r))
    return splits


def eval_once(X, y_raw, splits, device):
    ym, ys = float(y_raw.mean()), float(y_raw.std())
    yz = (y_raw - ym) / ys
    scores = []
    for tr, te in splits:
        pred = fit_predict(X[tr], yz[tr], X[te], device)
        scores.append(_r2(yz[te], pred))
    return float(np.mean(scores))


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    X, y = load_data()
    n = len(y)
    splits = build_splits(n)
    if CV_MODE == "holdout":
        print(f"Protocol: single {1 - TEST_SIZE:.0%}/{TEST_SIZE:.0%} holdout "
              f"(seed={SEED_SPLIT}; test set of {len(splits[0][1])} compounds)")
    else:
        print(f"Protocol: {CV_MODE}, {len(splits)} test folds")

    t0 = time.time()
    r2_real = eval_once(X, y, splits, device)
    t_real = time.time() - t0
    print(f"[Y-rand] real R2 = {r2_real:.4f} (one evaluation ~{t_real:.1f} s)")
    print(f"[Y-rand] estimated total time ~{t_real * N_ITER / 60:.1f} min")

    rng = np.random.RandomState(12345)
    r2_perm = np.empty(N_ITER)
    t0 = time.time()
    for i in range(N_ITER):
        y_perm = rng.permutation(y)
        r2_perm[i] = eval_once(X, y_perm, splits, device)
        if (i + 1) % 10 == 0:
            el = time.time() - t0
            eta = el / (i + 1) * (N_ITER - i - 1)
            print(f"[Y-rand] {i+1:3d}/{N_ITER}  permuted mean {r2_perm[:i+1].mean():+.4f}"
                  f"  (elapsed {el/60:.1f} min, eta {eta/60:.1f} min)")

    p_value = (np.sum(r2_perm >= r2_real) + 1) / (N_ITER + 1)
    cR2p = r2_real * np.sqrt(max(r2_real ** 2 - r2_perm.mean() ** 2, 0.0))

    print("=" * 64)
    print(f"real R2           = {r2_real:.4f}")
    print(f"permuted R2       = {r2_perm.mean():+.4f} +/- {r2_perm.std(ddof=1):.4f}")
    print(f"permuted range    = [{r2_perm.min():+.4f}, {r2_perm.max():+.4f}]")
    print(f"p-value           = {p_value:.4f}")
    print(f"cR2p              = {cR2p:.4f}")

    pd.DataFrame({"iter": np.arange(1, N_ITER + 1), "r2_perm": r2_perm}).to_csv(
        os.path.join(OUT_DIR, "yrandom_scores.csv"), index=False)
    with open(os.path.join(OUT_DIR, "yrandom_summary.json"), "w") as f:
        json.dump({
            "model": "MLP(512-256, dropout0.3, 50 epochs, batch 32)",
            "cv_mode": CV_MODE, "seed_split": SEED_SPLIT, "n_iter": N_ITER,
            "r2_real": float(r2_real),
            "r2_perm_mean": float(r2_perm.mean()),
            "r2_perm_sd": float(r2_perm.std(ddof=1)),
            "r2_perm_min": float(r2_perm.min()),
            "r2_perm_max": float(r2_perm.max()),
            "p_value": float(p_value), "cR2p": float(cR2p),
        }, f, indent=2)

    fig, ax = plt.subplots(figsize=(6, 4))
    ax.hist(r2_perm, bins=25, color="#8fbcd4", edgecolor="white")
    ax.axvline(r2_real, color="crimson", lw=2, label=f"Original R2 = {r2_real:.3f}")
    ax.axvline(r2_perm.mean(), color="gray", ls="--", lw=1.2,
               label=f"Scrambled mean = {r2_perm.mean():+.3f}")
    ax.set_xlabel("R2 (Y-randomized)")
    ax.set_ylabel("Count")
    ax.set_title(f"MLP Y-randomization x {N_ITER}  (p = {p_value:.3f})")
    ax.legend()
    fig.tight_layout()
    fig.savefig(os.path.join(OUT_DIR, "yrandom_plot.png"), dpi=300)
    print(f"\nresults in {OUT_DIR}/")


if __name__ == "__main__":
    main()
