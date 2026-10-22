#!/usr/bin/env python3
"""Collect the per-model metrics from the individual result files into one table.

Merges: repeated-CV summaries, OOF ensemble metrics, scaffold-split external
metrics, and calibration slope/intercept into model_metrics_summary.csv.
"""

import csv
import os


def read_last_row(f):
    d = {}
    with open(f) as fh:
        for row in csv.DictReader(fh):
            d = row
    return d


models = ["SVR", "RandomForest", "LGBM", "HistGB", "Bagging"]
rows = {}
for m in models:
    f = f"5x10cv_results/{m}_summary.csv"
    if os.path.exists(f):
        rows[m] = read_last_row(f)
f = "5x5cv_results_10fold/MLP_summary.csv"
if os.path.exists(f):
    rows["MLP"] = read_last_row(f)
f = "ridge_results/Ridge_summary.csv"
if os.path.exists(f):
    rows["Ridge"] = read_last_row(f)

# scaffold-disjoint external validation
ext = {}
f = "results_scaffold_AD/external_metrics_7models.csv"
if not os.path.exists(f):
    f = "results_scaffold_AD/external_metrics_6models.csv"
if os.path.exists(f):
    for row in csv.DictReader(open(f)):
        ext[row["model"]] = row

# selection-stage 5-fold screening benchmark
scr = {}
if os.path.exists("screening_5fold_results.csv"):
    for row in csv.DictReader(open("screening_5fold_results.csv")):
        scr[row["Model"]] = row["R-Squared"]

# calibration
cal = {}
if os.path.exists("calibration_slope_intercept.csv"):
    for row in csv.DictReader(open("calibration_slope_intercept.csv")):
        cal[row["model"]] = row

hdr = ["model", "screening5fold_R2", "cv_r2_mean", "cv_r2_sd", "cv_rmse",
       "ensemble_r2", "ensemble_rmse", "scaffold_ext_r2", "ext_rmse",
       "calib_slope", "calib_intercept"]
out = [hdr]
order = ["SVR", "RandomForest", "LGBM", "HistGB", "Bagging", "MLP", "Ridge"]
for m in order:
    d = rows.get(m, {})
    e = ext.get(m, {})
    c = cal.get(m, {})
    out.append([
        m,
        round(float(scr[m]), 4) if m in scr else "",
        round(float(d.get("r2_mean", "nan")), 4) if d else "",
        round(float(d.get("r2_sd", "nan")), 4) if d else "",
        round(float(d.get("rmse_mean", "nan")), 4) if d else "",
        round(float(d.get("ensemble_r2", "nan")), 4) if d else "",
        round(float(d.get("ensemble_rmse", "nan")), 4) if d else "",
        round(float(e.get("ext_r2", "nan")), 4) if e else "",
        round(float(e.get("ext_rmse", "nan")), 4) if e else "",
        c.get("calib_slope", ""), c.get("calib_intercept", ""),
    ])

with open("model_metrics_summary.csv", "w", newline="") as fh:
    csv.writer(fh).writerows(out)

print(f"{'model':<14}{'5fold':>8}{'5x10CV':>9}{'+/-SD':>8}{'RMSE':>8}"
      f"{'Ens.R2':>8}{'scaffold':>9}{'slope':>7}{'intercpt':>9}")
print("-" * 80)
for r in out[1:]:
    print(f"{r[0]:<14}{str(r[1]):>8}{str(r[2]):>9}{str(r[3]):>8}{str(r[4]):>8}"
          f"{str(r[5]):>8}{str(r[7]):>9}{str(r[9]):>7}{str(r[10]):>9}")
print(f"\nwritten: model_metrics_summary.csv")
