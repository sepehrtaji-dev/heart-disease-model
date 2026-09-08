"""PyTorch ensemble model for heart disease risk prediction.

Trains a 5-fold x 2-seed ensemble of MLP classifiers on
heart_disease_risk_2026.csv with engineered clinical features, tunes the
decision threshold on out-of-fold predictions, and reports accuracy, AUC,
and a confusion matrix on a held-out test set.

Usage:
    python heart_model.py [path/to/heart_disease_risk_2026.csv]
"""
import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, roc_auc_score
from sklearn.model_selection import StratifiedKFold, train_test_split
from sklearn.preprocessing import StandardScaler

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Morteza\Desktop\projects\AI course\selftrain\Heart illness\heart_disease_risk_2026.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
SEEDS_PER_FOLD = 2
N_FOLDS = 5


def load_data(path):
    df = pd.read_csv(path)
    y = df.pop("has_heart_disease").values.astype(np.float32)
    df = df.drop(columns=["patient_id"])
    age = df["age"].astype(np.float32)
    df["pulse_pressure"] = df["resting_bp_systolic"] - df["resting_bp_diastolic"]
    df["map"] = df["resting_bp_diastolic"] + df["pulse_pressure"] / 3
    df["chol_hdl_ratio"] = df["cholesterol_total"] / df["hdl"]
    df["ldl_hdl_ratio"] = df["ldl"] / df["hdl"]
    df["non_hdl"] = df["cholesterol_total"] - df["hdl"]
    df["trig_hdl_ratio"] = df["triglycerides"] / df["hdl"]
    df["hr_reserve"] = (220 - age) - df["max_heart_rate_achieved"]
    df["hr_ratio"] = df["max_heart_rate_achieved"] / (220 - age)
    df["bmi_x_hba1c"] = df["bmi"] * df["hba1c"]
    df["angina_x_st"] = df["exercise_induced_angina"].astype(np.float32) * df["st_depression"]
    df["fh_x_chol"] = df["family_history"].astype(np.float32) * df["cholesterol_total"]
    cp = pd.get_dummies(df["chest_pain_type"], prefix="cp", dtype=np.float32)
    for c in cp.columns:
        df[c] = cp[c]
        df[f"{c}_x_age"] = cp[c] * age
    df = pd.get_dummies(df, columns=["sex", "chest_pain_type", "smoker_status"], drop_first=False)
    for c in df.select_dtypes(include="bool").columns:
        df[c] = df[c].astype(np.float32)
    return df.astype(np.float32).values, y


class HeartNet(nn.Module):
    def __init__(self, n_features, widths=(256, 128, 64), dropout=0.25):
        super().__init__()
        layers, d = [], n_features
        for h in widths:
            layers += [nn.Linear(d, h), nn.BatchNorm1d(h), nn.ReLU(), nn.Dropout(dropout)]
            d = h
        layers += [nn.Linear(d, 1)]
        self.net = nn.Sequential(*layers)

    def forward(self, x):
        return self.net(x).squeeze(1)


def train_member(Xtr, ytr, Xva, yva, seed, epochs=400, batch_size=128, patience=30):
    torch.manual_seed(seed)
    model = HeartNet(Xtr.shape[1]).to(DEVICE)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    criterion = nn.BCEWithLogitsLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xtr), torch.tensor(ytr)),
        batch_size=batch_size, shuffle=True, drop_last=True,
    )
    Xv = torch.tensor(Xva).to(DEVICE)
    best_auc, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb in dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            opt.zero_grad()
            criterion(model(xb), yb).backward()
            opt.step()
        model.eval()
        with torch.no_grad():
            val_auc = roc_auc_score(yva, torch.sigmoid(model(Xv)).cpu().numpy())
        if val_auc > best_auc:
            best_auc, best_state, bad = val_auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience:
                break
    model.load_state_dict(best_state)
    return model, best_auc


def predict(model, X):
    model.eval()
    with torch.no_grad():
        return torch.sigmoid(model(torch.tensor(X).to(DEVICE))).cpu().numpy()


def main():
    X, y = load_data(DATA_PATH)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, positive rate {y.mean():.3f}")
    print(f"Device: {DEVICE}")

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)

    members, oof = [], np.zeros(len(X_train))
    skf = StratifiedKFold(n_splits=N_FOLDS, shuffle=True, random_state=42)
    for fold, (tr_i, va_i) in enumerate(skf.split(X_train, y_train)):
        scaler = StandardScaler().fit(X_train[tr_i])
        fold_probs = np.zeros(len(va_i))
        for s in range(SEEDS_PER_FOLD):
            model, val_auc = train_member(
                scaler.transform(X_train[tr_i]), y_train[tr_i],
                scaler.transform(X_train[va_i]), y_train[va_i],
                seed=100 * (fold + 1) + s,
            )
            members.append({"state_dict": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_})
            fold_probs += predict(model, scaler.transform(X_train[va_i]))
        oof[va_i] = fold_probs / SEEDS_PER_FOLD
        print(f"fold {fold}: OOF AUC {roc_auc_score(y_train[va_i], oof[va_i]):.4f}")

    thresholds = np.linspace(0.2, 0.8, 241)
    threshold = float(thresholds[np.argmax([accuracy_score(y_train, oof > t) for t in thresholds])])
    print(f"Tuned decision threshold (OOF): {threshold:.3f}  OOF acc {accuracy_score(y_train, oof > threshold):.4f}")

    test_probs = np.zeros(len(X_test))
    for m in members:
        model = HeartNet(X.shape[1]).to(DEVICE)
        model.load_state_dict(m["state_dict"])
        Xs = ((X_test - m["scaler_mean"]) / m["scaler_scale"]).astype(np.float32)
        test_probs += predict(model, Xs)
    test_probs /= len(members)
    preds = (test_probs > threshold).astype(int)

    print(f"\nENSEMBLE TEST  acc {accuracy_score(y_test, preds):.4f}  AUC {roc_auc_score(y_test, test_probs):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(y_test, preds))
    print(classification_report(y_test, preds, target_names=["no disease", "disease"]))

    torch.save({"members": members, "threshold": threshold, "feature_names": None}, "heart_model.pt")
    print(f"Saved {len(members)}-member ensemble to heart_model.pt")


if __name__ == "__main__":
    main()
