#!/usr/bin/env python3
"""Rank the Enamine library with the deployed MLP and flag applicability domain.

Loads the trained weights (model_training/qsar_model.pth) and y-scaler
(model_training/scaler.pkl), computes 2,048-bit ECFP4 fingerprints for every
library compound, predicts pIC50, ranks the library (descending), and reports
each compound's maximum ECFP4 Tanimoto similarity to the training set
(in-domain threshold 0.5, matching the paper's AD analysis).

Outputs screening_predictions.csv with columns:
rank, catalog_id, smiles, predicted_pIC50, nn_tanimoto_to_train, in_ad
"""

import os

import numpy as np
import pandas as pd
import torch
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator

LIBRARY_CSV = "../data/enamine_immuno_oncology_45760.csv"
TRAIN_ECFP_CSV = "../data/PIK3CA_ecfp2048.csv"
MODEL_PATH = "../model_training/qsar_model.pth"
SCALER_PATH = "../model_training/scaler.pkl"
OUTPUT_CSV = "screening_predictions.csv"
AD_TANIMOTO = 0.5
N_BITS = 2048
BATCH = 2048

GEN = rdFingerprintGenerator.GetMorganGenerator(radius=2, fpSize=N_BITS)


def ecfp(smiles):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return None
    arr = np.zeros((N_BITS,), dtype=np.float32)
    for b in GEN.GetFingerprint(mol).GetOnBits():
        arr[b] = 1.0
    return arr


def max_tanimoto_to_train(cand_bits, train_bits):
    """Max Tanimoto similarity of each candidate to the training fingerprints."""
    A = np.asarray(train_bits, dtype=np.float32)
    B = np.asarray(cand_bits, dtype=np.float32)
    pa = A.sum(1, dtype=np.float32)[None, :]
    pb = B.sum(1, dtype=np.float32)[:, None]
    inter = B @ A.T
    union = pb + pa - inter
    sim = np.where(union > 0, inter / np.maximum(union, 1e-9), 0.0)
    return sim.max(axis=1)


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Device: {device}")

    lib = pd.read_csv(LIBRARY_CSV)
    print(f"Library: {len(lib)} compounds")

    # fingerprints (invalid SMILES -> zero vector, flagged invalid)
    bits = np.zeros((len(lib), N_BITS), dtype=np.float32)
    valid = np.zeros(len(lib), dtype=bool)
    for i, s in enumerate(lib["SMILES"].values):
        b = ecfp(s)
        if b is not None:
            bits[i] = b
            valid[i] = True
    print(f"parsed {valid.sum()} SMILES ({(~valid).sum()} invalid)")

    # deployed model
    input_size = N_BITS
    model = torch.nn.Sequential(
        torch.nn.Linear(input_size, 512), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(512, 256), torch.nn.ReLU(), torch.nn.Dropout(0.3),
        torch.nn.Linear(256, 1),
    )
    model.load_state_dict(torch.load(MODEL_PATH, map_location=device))
    model.to(device).eval()
    import pickle
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)

    preds = np.zeros(len(lib), dtype=np.float64)
    with torch.no_grad():
        for s in range(0, len(lib), BATCH):
            xb = torch.tensor(bits[s:s + BATCH], dtype=torch.float32, device=device)
            p = model(xb).squeeze(-1).cpu().numpy()
            preds[s:s + len(p)] = p
    preds = scaler.inverse_transform(preds.reshape(-1, 1)).flatten()

    # applicability domain vs the training set
    train_df = pd.read_csv(TRAIN_ECFP_CSV)
    bit_cols = [c for c in train_df.columns if c.startswith("bit_")]
    train_bits = train_df[bit_cols].to_numpy()
    nn_sim = max_tanimoto_to_train(bits, train_bits)
    nn_sim[~valid] = 0.0

    out = pd.DataFrame({
        "rank": range(1, len(lib) + 1),
        "catalog_id": lib["Catalog ID"].values,
        "smiles": lib["SMILES"].values,
        "predicted_pIC50": np.round(preds, 4),
        "nn_tanimoto_to_train": np.round(nn_sim, 4),
        "in_ad": np.where(nn_sim >= AD_TANIMOTO, "in-domain", "out-of-domain"),
    }).sort_values("predicted_pIC50", ascending=False).reset_index(drop=True)
    out["rank"] = range(1, len(out) + 1)
    out.to_csv(OUTPUT_CSV, index=False)

    print(f"top-10 predictions:")
    print(out.head(10).to_string(index=False))
    print(f"\nin-domain coverage: {(out['in_ad'] == 'in-domain').mean():.2%} "
          f"of the library; written {OUTPUT_CSV}")
    print("note: out-of-domain predictions are extrapolations and should be "
          "read as ranking hypotheses, not absolute potencies.")


if __name__ == "__main__":
    main()
