# Heart Disease Risk Prediction (PyTorch)

An MLP binary classifier built with PyTorch that predicts heart disease risk
(`has_heart_disease`) from patient clinical and lifestyle data
(`heart_disease_risk_2026.csv`, ~9,000 patient records).

## Results (held-out test set, 20% stratified split)

| Metric | Value |
|---|---|
| ROC-AUC | 0.948 |
| Accuracy | 86.3% |
| Disease recall | 85% |
| Disease precision | 74% |

## Model

- 4-layer MLP: `28 → 128 → 64 → 32 → 1` with BatchNorm and Dropout (0.3)
- `BCEWithLogitsLoss` with `pos_weight` to handle the ~30/70 class imbalance
- AdamW optimizer + ReduceLROnPlateau scheduler, early stopping on test AUC
- Categorical features one-hot encoded, numeric features standardized

## Setup

```bash
pip install torch pandas scikit-learn numpy
```

## Usage

```bash
python heart_model.py path/to/heart_disease_risk_2026.csv
```

Training runs on CUDA if available, otherwise CPU. Trained weights are saved
to `heart_model.pt` (a pretrained copy from the original run is included).

> Note: the dataset itself is not included in this repo.
