#!/usr/bin/env python3
"""Train the deployed MLP QSAR model (2 x 2048-bit ECFP4 -> 512 -> 256 -> 1).

Architecture and training match the paper: two hidden layers (512 and 256
ReLU units), dropout 0.3, MSE loss, Adam (lr 1e-3), 50 epochs, batch size 32.
The 90:10 split is taken from data/split_membership.csv (the split deposited
with the paper; identical to a numpy RandomState(42).permutation split).
Outputs the trained weights (qsar_model.pth), the y-scaler (scaler.pkl),
metric printouts, and Train/Test performance plots.
"""

import os
import pickle

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
from rdkit import Chem
from rdkit.Chem import rdFingerprintGenerator
from scipy.stats import pearsonr
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from torch.utils.data import DataLoader, Dataset

DATA_PATH = "../data/PIK3CA_curated_descriptors.csv"
MEMBERSHIP_PATH = "../data/split_membership.csv"
SEED_FALLBACK = 42       # used only if the membership file is absent
MODEL_PATH = "qsar_model.pth"
SCALER_PATH = "scaler.pkl"


def generate_ecfp(smiles, radius=2, n_bits=2048):
    mol = Chem.MolFromSmiles(smiles)
    if mol is None:
        return np.zeros(n_bits)
    mfpgen = rdFingerprintGenerator.GetMorganGenerator(radius=radius, fpSize=n_bits)
    fp = mfpgen.GetFingerprint(mol)
    return np.array(list(fp))


class QSAR_Dataset(Dataset):
    def __init__(self, X, y):
        self.X = torch.tensor(X, dtype=torch.float32)
        self.y = torch.tensor(y, dtype=torch.float32)

    def __len__(self):
        return len(self.X)

    def __getitem__(self, idx):
        return self.X[idx], self.y[idx]


class QSAR_Model(nn.Module):
    """Deployed architecture: 2048 -> 512 -> 256 -> 1 with dropout 0.3."""

    def __init__(self, input_size, hidden_size=512, dropout_rate=0.3):
        super().__init__()
        self.model = nn.Sequential(
            nn.Linear(input_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, 1),
        )

    def forward(self, x):
        return self.model(x)


def load_split_ids():
    """Return (train_ids, test_ids) from the deposited membership table, or None."""
    if not os.path.exists(MEMBERSHIP_PATH):
        return None
    mem = pd.read_csv(MEMBERSHIP_PATH)
    return (mem.loc[mem["single_split"] == "train", "molecule_chembl_id"].astype(str).values,
            mem.loc[mem["single_split"] == "test", "molecule_chembl_id"].astype(str).values)


def load_data(csv_file):
    df = pd.read_csv(csv_file)
    smiles = df["canonical_smiles"].values
    y = df["pIC50"].values

    X = np.array([generate_ecfp(s) for s in smiles])

    from sklearn.preprocessing import StandardScaler
    scaler = StandardScaler()
    y = scaler.fit_transform(y.reshape(-1, 1)).flatten()

    ids = df["molecule_chembl_id"].astype(str).values if "molecule_chembl_id" in df.columns else None
    split = load_split_ids()
    if split is not None and ids is not None:
        id_set = {m: i for i, m in enumerate(ids)}
        tr_idx = [id_set[m] for m in split[0] if m in id_set]
        te_idx = [id_set[m] for m in split[1] if m in id_set]
        print(f"Split from {MEMBERSHIP_PATH}: train {len(tr_idx)} / test {len(te_idx)}")
    else:
        from sklearn.model_selection import train_test_split
        X_tr, X_te, y_tr, y_te, i_tr, i_te = train_test_split(
            X, np.arange(len(X)), test_size=0.1, random_state=SEED_FALLBACK)
        tr_idx, te_idx = i_tr, i_te
        print(f"Split from train_test_split (random_state={SEED_FALLBACK})")
    return X, y, np.array(tr_idx), np.array(te_idx), scaler


def concordance_correlation_coefficient(y_true, y_pred):
    y_true = np.array(y_true)
    y_pred = np.array(y_pred)
    cor = np.corrcoef(y_true, y_pred)[0][1]
    mean_true = np.mean(y_true)
    mean_pred = np.mean(y_pred)
    var_true = np.var(y_true)
    var_pred = np.var(y_pred)
    numerator = 2 * cor * np.sqrt(var_true * var_pred)
    denominator = var_true + var_pred + (mean_true - mean_pred) ** 2
    return numerator / denominator


def calculate_metrics(y_true, y_pred):
    r2 = r2_score(y_true, y_pred)
    rmse = np.sqrt(mean_squared_error(y_true, y_pred))
    mae = mean_absolute_error(y_true, y_pred)
    ccc = concordance_correlation_coefficient(y_true, y_pred)
    return r2, rmse, mae, ccc, rmse


def train_model(model, train_loader, criterion, optimizer, device, num_epochs=50):
    model.train()
    for epoch in range(num_epochs):
        total_loss = 0
        for X_batch, y_batch in train_loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            optimizer.zero_grad()
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        if (epoch + 1) % 10 == 0:
            print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {total_loss / len(train_loader):.4f}")


def evaluate_set(model, loader, scaler, device, is_train=True, set_name="Set"):
    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for X_batch, y_batch in loader:
            X_batch = X_batch.to(device)
            y_batch = y_batch.to(device)
            outputs = model(X_batch).squeeze()
            y_true.extend(y_batch.cpu().numpy())
            y_pred.extend(outputs.cpu().numpy())

    y_true = scaler.inverse_transform(np.array(y_true).reshape(-1, 1)).flatten()
    y_pred = scaler.inverse_transform(np.array(y_pred).reshape(-1, 1)).flatten()
    metrics = calculate_metrics(y_true, y_pred)
    tag = "R2" if is_train else "Q2"
    print(f"{set_name} {tag}: {metrics[0]:.4f}, RMSE: {metrics[1]:.4f}, "
          f"MAE: {metrics[2]:.4f}, CCC: {metrics[3]:.4f}")
    return y_true, y_pred, metrics


def visualize_predictions(actuals_dict, predictions_dict, metrics_dict,
                          save_stem="model_performance"):
    plt.rcParams["font.family"] = "Times New Roman"
    plt.rcParams["font.size"] = 12

    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle("QSAR Model Performance Evaluation", fontsize=16, fontweight="bold")

    subset_names = ["Train", "Test"]
    colors = ["#1f77b4", "#ff7f0e"]

    for i, (name, color) in enumerate(zip(subset_names, colors)):
        actuals = np.array(actuals_dict[name])
        preds = np.array(predictions_dict[name])
        r2, rmse, mae = metrics_dict[name]

        ax = axes[0, i]
        ax.scatter(actuals, preds, alpha=0.7, color=color, edgecolor="k",
                   linewidth=0.5, s=50)
        min_val = min(actuals.min(), preds.min())
        max_val = max(actuals.max(), preds.max())
        ax.plot([min_val, max_val], [min_val, max_val], "r--", lw=2, label="Ideal")
        ax.set_xlabel("Actual pIC50", fontsize=14, fontweight="bold")
        ax.set_ylabel("Predicted pIC50", fontsize=14, fontweight="bold")
        ax.set_title(f"{name} Set", fontsize=14, fontweight="bold")
        ax.grid(True, linestyle="--", alpha=0.6)

        corr, _ = pearsonr(actuals, preds)
        textstr = (f"R2 = {r2:.4f}\nRMSE = {rmse:.4f}\nMAE = {mae:.4f}\n"
                   f"Pearson r = {corr:.4f}")
        props = dict(boxstyle="round", facecolor="wheat", alpha=0.8)
        ax.text(0.05, 0.95, textstr, transform=ax.transAxes, fontsize=11,
                verticalalignment="top", bbox=props)
        ax.legend(loc="lower right")

    residuals_test = np.array(predictions_dict["Test"]) - np.array(actuals_dict["Test"])
    axes[1, 0].scatter(actuals_dict["Test"], residuals_test, alpha=0.7, color="#2ca02c",
                       edgecolor="k", linewidth=0.5, s=50)
    axes[1, 0].axhline(y=0, color="r", linestyle="--", lw=2)
    axes[1, 0].set_xlabel("Actual pIC50", fontsize=14, fontweight="bold")
    axes[1, 0].set_ylabel("Residuals (Predicted - Actual)", fontsize=14, fontweight="bold")
    axes[1, 0].set_title("Residual Plot (Test Set)", fontsize=14, fontweight="bold")
    axes[1, 0].grid(True, linestyle="--", alpha=0.6)

    axes[1, 1].hist(actuals_dict["Test"], bins=25, alpha=0.6, label="Actual",
                    color="#1f77b4", edgecolor="black")
    axes[1, 1].hist(predictions_dict["Test"], bins=25, alpha=0.6, label="Predicted",
                    color="#ff7f0e", edgecolor="black")
    axes[1, 1].set_xlabel("pIC50 Value", fontsize=14, fontweight="bold")
    axes[1, 1].set_ylabel("Frequency", fontsize=14, fontweight="bold")
    axes[1, 1].set_title("Distribution of Actual vs Predicted (Test Set)",
                         fontsize=14, fontweight="bold")
    axes[1, 1].legend(fontsize=12)
    axes[1, 1].grid(True, linestyle="--", alpha=0.6)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.savefig(f"{save_stem}.png", dpi=300, bbox_inches="tight")
    plt.savefig(f"{save_stem}.pdf", dpi=300, bbox_inches="tight")


def save_model(model, scaler, model_path=MODEL_PATH, scaler_path=SCALER_PATH):
    torch.save(model.state_dict(), model_path)
    with open(scaler_path, "wb") as f:
        pickle.dump(scaler, f)
    print(f"Model saved to {model_path}, scaler saved to {scaler_path}")


def load_model(model_class, input_size, device, model_path=MODEL_PATH,
               scaler_path=SCALER_PATH):
    model = model_class(input_size)
    model.load_state_dict(torch.load(model_path, map_location=device))
    model.to(device)
    model.eval()
    with open(scaler_path, "rb") as f:
        scaler = pickle.load(f)
    print(f"Model loaded from {model_path}")
    return model, scaler


def predict_with_model(model, scaler, smiles_list, device, radius=2, n_bits=2048):
    X = np.array([generate_ecfp(smiles, radius, n_bits) for smiles in smiles_list])
    X_tensor = torch.tensor(X, dtype=torch.float32).to(device)
    model.eval()
    with torch.no_grad():
        predictions = model(X_tensor).squeeze().cpu().numpy()
    predictions = scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
    return predictions


def main():
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    X, y, tr_idx, te_idx, scaler = load_data(DATA_PATH)
    train_dataset = QSAR_Dataset(X[tr_idx], y[tr_idx])
    test_dataset = QSAR_Dataset(X[te_idx], y[te_idx])
    train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
    test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

    input_size = X.shape[1]
    model = QSAR_Model(input_size).to(device)
    criterion = nn.MSELoss()
    optimizer = optim.Adam(model.parameters(), lr=0.001)

    train_model(model, train_loader, criterion, optimizer, device, num_epochs=50)

    actuals_train, preds_train, metrics_train = evaluate_set(
        model, train_loader, scaler, device, is_train=True, set_name="Train")
    actuals_test, preds_test, metrics_test = evaluate_set(
        model, test_loader, scaler, device, is_train=False, set_name="Test")

    actuals_dict = {"Train": actuals_train, "Test": actuals_test}
    predictions_dict = {"Train": preds_train, "Test": preds_test}
    metrics_dict = {
        "Train": [metrics_train[0], metrics_train[1], metrics_train[2]],
        "Test": [metrics_test[0], metrics_test[1], metrics_test[2]],
    }
    visualize_predictions(actuals_dict, predictions_dict, metrics_dict)
    save_model(model, scaler)


if __name__ == "__main__":
    main()
