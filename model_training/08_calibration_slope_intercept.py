#!/usr/bin/env python3
"""Calibration slope/intercept of each model's OOF ensemble predictions.

Regresses predicted on experimental pIC50 (pred = slope * exp + intercept)
for every finalist model and writes calibration_slope_intercept.csv.
"""

import csv
import os

import numpy as np

FILES = {
    "SVR": "5x10cv_results/SVR_ensemble_preds.csv",
    "RandomForest": "5x10cv_results/RandomForest_ensemble_preds.csv",
    "LGBM": "5x10cv_results/LGBM_ensemble_preds.csv",
    "HistGB": "5x10cv_results/HistGB_ensemble_preds.csv",
    "Bagging": "5x10cv_results/Bagging_ensemble_preds.csv",
    "MLP": "5x5cv_results_10fold/MLP_ensemble_preds.csv",
    "Ridge": "ridge_results/Ridge_ensemble_preds.csv",
}
OUT = "calibration_slope_intercept.csv"

rows = []
for name, f in FILES.items():
    if not os.path.exists(f):
        print(f"missing: {f}")
        continue
    d = np.genfromtxt(f, delimiter=",", names=True)
    y = np.array(d["y"])
    p = np.array(d["pred_mean"])
    slope, intercept = np.polyfit(y, p, 1)  # p = slope * y + intercept
    ss_res = np.sum((y - p) ** 2)
    ss_tot = np.sum((y - y.mean()) ** 2)
    r2 = 1 - ss_res / ss_tot
    rmse = np.sqrt(np.mean((y - p) ** 2))
    rows.append([name, len(y), round(r2, 4), round(slope, 3),
                 round(intercept, 3), round(rmse, 4)])

with open(OUT, "w", newline="") as fh:
    w = csv.writer(fh)
    w.writerow(["model", "n", "oof_r2", "calib_slope", "calib_intercept", "rmse"])
    w.writerows(rows)

print(f"{'model':<14}{'N':>6}{'R2(OOF)':>10}{'slope':>8}{'intercept':>10}{'RMSE':>8}")
for r in rows:
    print(f"{r[0]:<14}{r[1]:>6}{r[2]:>10.4f}{r[3]:>8.3f}{r[4]:>10.3f}{r[5]:>8.3f}")
print(f"\nwritten: {OUT}")
