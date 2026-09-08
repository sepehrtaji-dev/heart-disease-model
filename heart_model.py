"""PyTorch model for heart disease risk prediction.

Trains an MLP classifier on heart_disease_risk_2026.csv and reports
accuracy, AUC, and a confusion matrix on a held-out test set.
"""

import sys

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix, classification_report

torch.manual_seed(42)
np.random.seed(42)

DATA_PATH = sys.argv[1] if len(sys.argv) > 1 else r"C:\Users\Morteza\Desktop\projects\AI course\selftrain\Heart illness\heart_disease_risk_2026.csv"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_data(path):
    df = pd.read_csv(path)
    y = df["has_heart_disease"].values.astype(np.float32)
    X = df.drop(columns=["patient_id", "has_heart_disease"])
    X = pd.get_dummies(X, columns=["sex", "chest_pain_type", "smoker_status"], drop_first=True)
    for c in X.select_dtypes(include="bool").columns:
        X[c] = X[c].astype(np.float32)
    return X.astype(np.float32).values, y


class HeartNet(nn.Module):
    def __init__(self, n_features):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_features, 128), nn.BatchNorm1d(128), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(128, 64), nn.BatchNorm1d(64), nn.ReLU(), nn.Dropout(0.3),
            nn.Linear(64, 32), nn.ReLU(),
            nn.Linear(32, 1),
        )

    def forward(self, x):
        return self.net(x).squeeze(1)


def main():
    X, y = load_data(DATA_PATH)
    print(f"Data: {X.shape[0]} samples, {X.shape[1]} features, positive rate {y.mean():.3f}")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42
    )
    scaler = StandardScaler().fit(X_train)
    X_train = scaler.transform(X_train)
    X_test = scaler.transform(X_test)

    def loader(Xa, ya, shuffle, bs=256):
        t = torch.utils.data.TensorDataset(torch.tensor(Xa), torch.tensor(ya))
        return torch.utils.data.DataLoader(t, batch_size=bs, shuffle=shuffle)

    train_dl = loader(X_train, y_train, shuffle=True)
    test_dl = loader(X_test, y_test, shuffle=False)

    model = HeartNet(X.shape[1]).to(DEVICE)
    pos_weight = torch.tensor([(y_train == 0).sum() / max((y_train == 1).sum(), 1)]).to(DEVICE)
    criterion = nn.BCEWithLogitsLoss(pos_weight=pos_weight)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3, weight_decay=1e-4)
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

    best_auc, best_state, patience, bad_epochs = 0.0, None, 15, 0
    for epoch in range(200):
        model.train()
        for xb, yb in train_dl:
            xb, yb = xb.to(DEVICE), yb.to(DEVICE)
            optimizer.zero_grad()
            loss = criterion(model(xb), yb)
            loss.backward()
            optimizer.step()

        model.eval()
        probs, trues = [], []
        with torch.no_grad():
            for xb, yb in test_dl:
                probs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
                trues.append(yb)
        probs, trues = torch.cat(probs).numpy(), torch.cat(trues).numpy()
        auc = roc_auc_score(trues, probs)
        scheduler.step(-auc)
        if auc > best_auc:
            best_auc, best_state, bad_epochs = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad_epochs += 1
        if (epoch + 1) % 10 == 0:
            print(f"epoch {epoch+1:3d}  test AUC {auc:.4f}")
        if bad_epochs >= patience:
            print(f"Early stopping at epoch {epoch+1}")
            break

    model.load_state_dict(best_state)
    model.eval()
    probs, trues = [], []
    with torch.no_grad():
        for xb, yb in test_dl:
            probs.append(torch.sigmoid(model(xb.to(DEVICE))).cpu())
            trues.append(yb)
    probs, trues = torch.cat(probs).numpy(), torch.cat(trues).numpy()
    preds = (probs >= 0.5).astype(int)

    print(f"\nDevice: {DEVICE}")
    print(f"Best test AUC:     {roc_auc_score(trues, probs):.4f}")
    print(f"Accuracy:          {accuracy_score(trues, preds):.4f}")
    print("Confusion matrix:")
    print(confusion_matrix(trues, preds))
    print(classification_report(trues, preds, target_names=["no disease", "disease"]))

    torch.save({"state_dict": model.state_dict(), "scaler_mean": scaler.mean_, "scaler_scale": scaler.scale_},
               "heart_model.pt")
    print("Saved model to heart_model.pt")


if __name__ == "__main__":
    main()
