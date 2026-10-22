#!/usr/bin/env python3
"""Ridge baseline on the Bemis-Murcko scaffold-disjoint external split.

The scaffold membership is read from data/split_membership.csv (the same file
deposited with the paper); no scaffold is shared between its train and test
parts. Optionally breaks the external metrics down by applicability domain
(AD) membership when results_scaffold_AD/ad_test_sims.csv is available.
"""

import os

import numpy as np
import pandas as pd
from sklearn.linear_model import RidgeCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

DATA_PATH = "../data/PIK3CA_ecfp2048.csv"
MEMBERSHIP_PATH = "../data/split_membership.csv"
OUT_DIR = "results_scaffold_AD"
SEED = 42
os.makedirs(OUT_DIR, exist_ok=True)

# ---- load fingerprints (same loading convention as the other scripts) ----
df = pd.read_csv(DATA_PATH)
assert "canonical_smiles" in df.columns
drop = [c for c in df.columns
        if any(k in c.lower() for k in ("smiles", "chembl", "name", "unnamed",
                                        "molecule", "catalog", "id"))]
df2 = df.drop(columns=drop, errors="ignore")
ycol = [c for c in df2.columns if "pic50" in c.lower()][0]
X = df2.drop(columns=[ycol]).select_dtypes(include=[np.number])
y = pd.to_numeric(df2[ycol], errors="coerce")
ok = y.notna() & X.notna().all(axis=1)
X = X[ok].astype(float).reset_index(drop=True)
y = y[ok].reset_index(drop=True)

# ---- scaffold split from the deposited membership table ----
mem = pd.read_csv(MEMBERSHIP_PATH)
assert (mem["molecule_chembl_id"].astype(str).values ==
        df.loc[ok, "molecule_chembl_id"].astype(str).values).all(), \
    "row order of the fingerprint file and split_membership.csv must match"
te_mask = (mem["scaffold_split"] == "test").to_numpy()
tr_idx = np.where(~te_mask)[0]
te_idx = np.where(te_mask)[0]
print(f"scaffold split: train {len(tr_idx)} / test {len(te_idx)}")
Xtr, Xte = X.iloc[tr_idx], X.iloc[te_idx]
ytr, yte = y.iloc[tr_idx].to_numpy(), y.iloc[te_idx].to_numpy()

# ---- optional AD breakdown ----
ad_path = os.path.join(OUT_DIR, "ad_test_sims.csv")
in_ad = None
if os.path.exists(ad_path):
    ad = pd.read_csv(ad_path)
    in_ad = ad["in_ad"].astype(bool).to_numpy()
    print(f"AD coverage: {in_ad.mean():.3%}")

# ---- ridge baseline (StandardScaler + RidgeCV; efficient LOO-CV inner loop) ----
est = Pipeline([("scaler", StandardScaler()),
                ("ridge", RidgeCV(alphas=np.logspace(-3, 3, 50)))])
est.fit(Xtr, ytr)
yp = est.predict(Xte)


def metrics(yy, pp):
    return float(r2_score(yy, pp)), float(np.sqrt(mean_squared_error(yy, pp)))


r2_all, rm_all = metrics(yte, yp)
r2_in = rm_in = r2_out = rm_out = float("nan")
if in_ad is not None:
    if in_ad.sum() > 2:
        r2_in, rm_in = metrics(yte[in_ad], yp[in_ad])
    if (~in_ad).sum() > 2:
        r2_out, rm_out = metrics(yte[~in_ad], yp[~in_ad])
print(f"[Ridge] external R2={r2_all:.4f} RMSE={rm_all:.4f} | "
      f"in-AD R2={r2_in:.4f} | out-AD R2={r2_out:.4f}")
print(f"        selected alpha = {est.named_steps['ridge'].alpha_:.1f}")

row = {"model": "Ridge", "ext_r2": r2_all, "ext_rmse": rm_all,
       "inAD_r2": r2_in, "inAD_rmse": rm_in,
       "outAD_r2": r2_out, "outAD_rmse": rm_out,
       "ad_coverage": float(in_ad.mean()) if in_ad is not None else ""}
out_path = os.path.join(OUT_DIR, "external_metrics_ridge.csv")
pd.DataFrame([row]).to_csv(out_path, index=False)
print(f"\nwritten: {out_path}")
