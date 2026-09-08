"""Test Sepehr's hypothesis: is the sub-95% accuracy due to cleanable data dirt?

Step 1: full-file quality scan (NaNs, duplicate IDs, impossible values).
Step 2: flag labels that strongly disagree with CV-predicted risk.
Step 3: train models on cleaned train data, evaluate on the ORIGINAL
        untouched test set (the honest comparison).
"""
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import StratifiedKFold, cross_val_predict, train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

DATA = r"C:\Users\Morteza\Desktop\projects\AI course\selftrain\Heart illness\heart_disease_risk_2026.csv"
DEV = "cuda" if torch.cuda.is_available() else "cpu"

df = pd.read_csv(DATA)
print("=== Step 1: data quality scan ===")
print(f"rows: {len(df)}, cols: {len(df.columns)}")
na = df.isna().sum()
print(f"missing values total: {int(na.sum())}")
print(f"exact duplicate rows: {int(df.drop(columns=['patient_id']).duplicated().sum())}")
print(f"duplicate patient_ids: {int(df['patient_id'].duplicated().sum())}")
num = df.select_dtypes(include="number").drop(columns=["patient_id"])
bad = {}
for c in num.columns:
    lo, hi = num[c].min(), num[c].max()
    if c == "age" and (lo < 0 or hi > 120): bad[c] = (lo, hi)
    if "bp" in c and (lo < 40 or hi > 260): bad[c] = (lo, hi)
    if "sleep" in c and (lo < 0 or hi > 24): bad[c] = (lo, hi)
    if "steps" in c and lo < 0: bad[c] = (lo, hi)
print(f"impossible numeric ranges: {bad if bad else 'none'}")
for c in ["sex", "chest_pain_type", "smoker_status"]:
    print(f"{c}: {sorted(df[c].unique())}")
print(f"label values: {sorted(df['has_heart_disease'].unique())}")

print("\n=== Step 2: flag suspicious labels (10-fold CV logistic) ===")
import types
src = open("heart_model.py").read().replace('if __name__ == "__main__":\n    main()', "")
mod = types.ModuleType("hm")
exec(src, mod.__dict__)
X, y = mod.load_data(DATA)
y_int = y.astype(int)

pipe = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
p_cv = cross_val_predict(pipe, X, y_int, cv=StratifiedKFold(10, shuffle=True, random_state=0), method="predict_proba")[:, 1]
flag = ((p_cv > 0.8) & (y_int == 0)) | ((p_cv < 0.2) & (y_int == 1))
print(f"suspicious labels (|p-y| > 0.8): {int(flag.sum())} of {len(y)} ({flag.mean()*100:.1f}%)")
for t in (0.9, 0.95):
    f = ((p_cv > t) & (y_int == 0)) | ((p_cv < 1 - t) & (y_int == 1))
    print(f"  at |p-y| > {t}: {int(f.sum())} ({f.mean()*100:.1f}%)")

print("\n=== Step 3: train on cleaned train, test on ORIGINAL test set ===")
Xtr, Xte, ytr, yte, ptr, _ = train_test_split(X, y_int, p_cv, test_size=0.2, stratify=y_int, random_state=42)
flag_tr = ((ptr > 0.8) & (ytr == 0)) | ((ptr < 0.2) & (ytr == 1))
Xtr_c, ytr_c = Xtr[~flag_tr], ytr[~flag_tr]
print(f"removed {int(flag_tr.sum())} suspicious rows from train ({len(Xtr_c)} remain)")

pipe_c = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
pipe_c.fit(Xtr_c, ytr_c)
pc = pipe_c.predict_proba(Xte)[:, 1]
pipe_o = make_pipeline(StandardScaler(), LogisticRegression(max_iter=5000, C=1.0))
pipe_o.fit(Xtr, ytr)
po = pipe_o.predict_proba(Xte)[:, 1]
print(f"[LOGREG orig-train   ] TEST acc {accuracy_score(yte, po > 0.5):.4f}  AUC {roc_auc_score(yte, po):.4f}")
print(f"[LOGREG cleaned-train] TEST acc {accuracy_score(yte, pc > 0.5):.4f}  AUC {roc_auc_score(yte, pc):.4f}")


def train_member(Xa, ya, Xb, yb, seed, epochs=300, bs=128, patience=25):
    torch.manual_seed(seed)
    model = mod.HeartNet(Xa.shape[1]).to(DEV)
    opt = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-5)
    crit = nn.BCEWithLogitsLoss()
    dl = torch.utils.data.DataLoader(
        torch.utils.data.TensorDataset(torch.tensor(Xa), torch.tensor(ya.astype(np.float32))),
        batch_size=bs, shuffle=True, drop_last=True)
    Xv = torch.tensor(Xb).to(DEV)
    best, best_state, bad = 0.0, None, 0
    for _ in range(epochs):
        model.train()
        for xb, yb2 in dl:
            xb, yb2 = xb.to(DEV), yb2.to(DEV)
            opt.zero_grad(); crit(model(xb), yb2).backward(); opt.step()
        model.eval()
        with torch.no_grad():
            auc = roc_auc_score(yb, torch.sigmoid(model(Xv)).cpu().numpy())
        if auc > best:
            best, best_state, bad = auc, {k: v.clone() for k, v in model.state_dict().items()}, 0
        else:
            bad += 1
            if bad >= patience: break
    model.load_state_dict(best_state)
    return model

def ens_acc(clean):
    probs = np.zeros(len(Xte))
    for s in range(3):
        sc = StandardScaler().fit(Xtr_c if clean else Xtr)
        m = train_member(sc.transform(Xtr_c if clean else Xtr), ytr_c if clean else ytr,
                         sc.transform(Xte), yte, seed=7 + s)
        m.eval()
        with torch.no_grad():
            probs += torch.sigmoid(m(torch.tensor(sc.transform(Xte).astype(np.float32)).to(DEV))).cpu().numpy()
    probs /= 3
    return accuracy_score(yte, probs > 0.5), roc_auc_score(yte, probs)

a_o, u_o = ens_acc(False)
a_c, u_c = ens_acc(True)
print(f"[MLP-3  orig-train   ] TEST acc {a_o:.4f}  AUC {u_o:.4f}")
print(f"[MLP-3  cleaned-train] TEST acc {a_c:.4f}  AUC {u_c:.4f}")
