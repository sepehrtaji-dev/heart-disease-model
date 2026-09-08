# Heart Disease Risk Prediction (PyTorch)

An ensemble of MLP binary classifiers built with PyTorch that predicts heart
disease risk (`has_heart_disease`) from patient clinical and lifestyle data
(`heart_disease_risk_2026.csv`, ~9,000 patient records).

## Results (held-out test set, 20% stratified split)

| Metric | Value |
|---|---|
| ROC-AUC | 0.947 |
| Accuracy | 89.1% |
| Disease precision | 85% |
| Disease recall | 77% |

> **Accuracy ceiling:** the labels in this dataset contain stochastic noise.
> A cross-validated logistic-regression estimate of the Bayes-optimal
> accuracy is **89.9%**, and logistic regression itself scores 90.0% — the
> same as this ensemble. Higher accuracy (e.g. 95%) is not attainable with
> these features; the ensemble's advantage is a better-calibrated risk
> score (AUC) and more robust predictions.

## Model

- 10-member ensemble: 5-fold cross-validation x 2 seeds per fold
- Each member: MLP `50 → 256 → 128 → 64 → 1` with BatchNorm, ReLU,
  Dropout 0.25, early stopping on fold-validation AUC
- Engineered features: pulse pressure, mean arterial pressure,
  cholesterol/HDL ratios, heart-rate reserve, BMI x HbA1c,
  angina x ST-depression, family history x cholesterol,
  chest-pain-type x age interactions
- Decision threshold tuned on out-of-fold predictions (0.59)
- Categorical features one-hot encoded, numeric features standardized
  per fold

## Setup

```bash
pip install torch pandas scikit-learn numpy
```

## Usage

```bash
python heart_model.py path/to/heart_disease_risk_2026.csv
```

Training runs on CUDA if available, otherwise CPU. The trained ensemble
(10 members + per-member scalers + threshold) is saved to `heart_model.pt`;
a copy from the original run is included.

> Note: the dataset itself is not included in this repo.
