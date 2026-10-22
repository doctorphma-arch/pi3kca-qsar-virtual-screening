#!/usr/bin/env python3
"""Applicability-domain (AD) analysis for the top-ranked screening hits.

For every candidate compound the maximum ECFP4 Tanimoto similarity to the
reference (training) set is computed; compounds with a nearest-neighbour
similarity >= AD_TANIMOTO (0.5) are flagged in-domain. The reference set is
the scaffold-split training part by default, matching the external validation.

Usage: python 13_applicability_domain.py [data_csv candidates_csv output_csv]
"""

import os
import re
import sys

import numpy as np
import pandas as pd
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from rdkit.Chem.Scaffolds import MurckoScaffold

# ----------------------- configuration -----------------------
DATA_PATH = "../data/PIK3CA_ecfp2048.csv"
CAND_PATH = "top10_candidates.csv"
OUT_PATH = "ad_results_top10.csv"

AD_TANIMOTO = 0.5        # in-domain threshold (keep consistent with external validation)
REFERENCE = "scaffold_train"   # "scaffold_train" = 4 of 5 scaffold groups | "all"
SEED = 42
N_GROUPS = 5
TEST_GROUP = 0
RADIUS = 2               # ECFP4
N_BITS = 2048


def load_data(data_path):
    df = pd.read_csv(data_path)
    assert "canonical_smiles" in df.columns, "training data need a canonical_smiles column"
    smi = df["canonical_smiles"].astype(str).values
    drop = [c for c in df.columns
            if any(k in c.lower() for k in ("smiles", "chembl", "name", "unnamed",
                                            "molecule", "catalog", "id"))
            and c != "canonical_smiles"]
    df2 = df.drop(columns=[c for c in drop if c != "canonical_smiles"], errors="ignore")
    ycols = [c for c in df2.columns if "pic50" in c.lower()]
    assert ycols, "no pIC50 column found"
    ycol = ycols[0]
    X = df2.drop(columns=[ycol]).select_dtypes(include=[np.number])
    y = pd.to_numeric(df2[ycol], errors="coerce")
    ok = y.notna() & X.notna().all(axis=1)
    X, y, smi = X[ok].astype(float).reset_index(drop=True), y[ok].reset_index(drop=True), smi[ok]
    bit_cols = [c for c in X.columns if re.fullmatch(r"bit_\d+", c)]
    if len(bit_cols) < 100:
        raise ValueError("no bit_* fingerprint columns found")
    return X, y, smi, bit_cols


def scaffold_groups(smi_list, n_groups):
    scaf_of, failed = [], 0
    for s in smi_list:
        mol = Chem.MolFromSmiles(s)
        if mol is None:
            failed += 1
            scaf_of.append(None)
        else:
            scaf_of.append(MurckoScaffold.GetScaffoldForMol(mol))
    order = [(sc if sc is not None else f"__fail__{i}", i) for i, sc in enumerate(scaf_of)]
    cnt = {}
    for sc, _ in order:
        cnt[sc] = cnt.get(sc, 0) + 1
    order.sort(key=lambda t: (-cnt[t[0]], str(t[0])))
    group_of = {}
    for gi, (sc, idx) in enumerate(order):
        group_of[idx] = gi % n_groups
    return [group_of[i] for i in range(len(smi_list))], scaf_of, failed


def tanimoto_nn(A_bits, B_bits):
    """Max Tanimoto similarity of each row of B against all rows of A."""
    A = np.asarray(A_bits, dtype=np.float32)
    B = np.asarray(B_bits, dtype=np.float32)
    pa = A.sum(1, dtype=np.float32)[None, :]
    pb = B.sum(1, dtype=np.float32)[:, None]
    inter = B @ A.T
    union = pb + pa - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return sim.max(axis=1)


def ecfp_matrix(smiles_list, radius=RADIUS, n_bits=N_BITS):
    gen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits,
                                                    includeChirality=False)
    n = len(smiles_list)
    arr = np.zeros((n, n_bits), dtype=np.int8)
    valid = np.zeros(n, dtype=bool)
    for i, s in enumerate(smiles_list):
        if pd.isna(s) or str(s).strip() == "":
            continue
        mol = Chem.MolFromSmiles(str(s))
        if mol is None:
            continue
        fp = gen.GetFingerprint(mol)
        for b in fp.GetOnBits():
            arr[i, b] = 1
        valid[i] = True
    return arr, valid


def main():
    global DATA_PATH, CAND_PATH, OUT_PATH
    if len(sys.argv) >= 4:
        DATA_PATH, CAND_PATH, OUT_PATH = sys.argv[1], sys.argv[2], sys.argv[3]

    # 1) reference (training) domain
    X, y, smi, bit_cols = load_data(DATA_PATH)
    if REFERENCE == "scaffold_train":
        groups, _, _ = scaffold_groups(smi, N_GROUPS)
        g = np.array(groups)
        ref_mask = (g != TEST_GROUP)
        ref_name = f"scaffold-split training part (seed={SEED}, group {TEST_GROUP} held out)"
    else:
        ref_mask = np.ones(len(smi), dtype=bool)
        ref_name = "all compounds"
    Xref_bits = X.loc[ref_mask, bit_cols].to_numpy()
    print(f"Reference domain: {ref_name} -> {Xref_bits.shape[0]} molecules x {Xref_bits.shape[1]} bits")
    print(f"AD threshold: NN-Tanimoto >= {AD_TANIMOTO}")

    # 2) candidate compounds
    cand = pd.read_csv(CAND_PATH)
    smi_col = next((cand.columns[i] for i, c in enumerate(cand.columns)
                    if re.fullmatch(r"(canonical_)?smiles", c.strip().lower())), None)
    if smi_col is None:
        raise ValueError(f"no SMILES column in the candidate file; columns: {list(cand.columns)}")
    id_col = next((cand.columns[i] for i, c in enumerate(cand.columns)
                   if re.fullmatch(r"(catalog[ _]?id|mol[ _]?id|id|name)", c.strip().lower())), None)
    pred_col = next((cand.columns[i] for i, c in enumerate(cand.columns)
                     if re.search(r"(pred|pic50|score|activity)", c.strip().lower())), None)

    cand_bits, valid = ecfp_matrix(cand[smi_col].values)
    if (~valid).any():
        print(f"warning: {(~valid).sum()} candidate SMILES failed to parse (flagged out-of-domain)")

    # 3) nearest-neighbour Tanimoto + in/out-of-domain flag
    nn_sim = tanimoto_nn(Xref_bits, cand_bits)
    nn_sim[~valid] = 0.0
    in_ad = nn_sim >= AD_TANIMOTO

    # 4) output
    out = pd.DataFrame()
    out["rank"] = range(1, len(cand) + 1)
    out["catalog_id"] = cand[id_col].values if id_col is not None else range(1, len(cand) + 1)
    if pred_col is not None:
        out["predicted"] = cand[pred_col].values
    out["nn_tanimoto"] = np.round(nn_sim, 4)
    out["in_ad"] = np.where(in_ad, "in-domain", "out-of-domain")
    out.to_csv(OUT_PATH, index=False)

    print(f"\nresults ({OUT_PATH}):")
    for _, r in out.iterrows():
        print(f"  {int(r['rank']):2d}  {str(r['catalog_id']):>14s}  "
              f"NN={r['nn_tanimoto']:.3f}  {r['in_ad']}")
    print(f"\nin-domain coverage: {in_ad.mean():.1%} ({int(in_ad.sum())}/{len(cand)})")
    print("note: out-of-domain predictions should be interpreted as extrapolations.")


if __name__ == "__main__":
    main()
