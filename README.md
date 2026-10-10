# Unsupervised Anomaly Detection with Autoencoders

Detect anomalies in multivariate sensor-style data with a deep autoencoder:
train **only on normal data**, then flag samples with high reconstruction
error. Includes threshold calibration, a full precision/recall evaluation
and latent-space visualisation.

## How it works

1. **Data** (`src/data.py`) — synthetic 12-sensor dataset: normal points from
   a Gaussian mixture, anomalies injected as shifted/scaled outliers.
   The training split contains *only* normal data (true unsupervised setup).
2. **Model** (`src/model.py`) — symmetric MLP autoencoder
   (12 → 64 → 32 → 3 → 32 → 64 → 12) with dropout; the tight 3-D bottleneck
   forces the network to learn the normal manifold.
3. **Training** (`src/train.py`) — MSE reconstruction loss, AdamW,
   ReduceLROnPlateau scheduling, early stopping, and threshold calibration
   at the 95th percentile of training reconstruction errors.
4. **Evaluation** (`src/detect.py`) — precision/recall/F1, ROC AUC,
   average precision, plus a threshold sweep.
5. **Visualisation** (`src/visualize.py`) — error distributions and a
   2-D latent-space scatter plot.

## Quickstart

```bash
pip install -r requirements.txt

python src/data.py        # generate data
python src/train.py       # train + calibrate threshold
python src/detect.py      # evaluate on labelled test set
python src/visualize.py   # save plots to checkpoints/
```

## Results (measured on the synthetic test set)

| Metric | Value |
|---|---|
| ROC AUC | 1.000 |
| Average precision | 1.000 |
| Accuracy | 0.962 |
| Anomaly precision / recall / F1 | 0.846 / 1.000 / 0.916 |

The calibrated 95th-percentile threshold catches every injected anomaly
with only a small false-positive rate on normal data.

## Key ideas demonstrated

- **Unsupervised learning**: no anomaly labels are used during training —
  the model learns "normal" and anything else stands out.
- **Threshold calibration** from the training error distribution instead
  of a magic number.
- **Honest evaluation**: ROC AUC / average precision on a held-out
  labelled test set, with a precision–recall trade-off sweep.
