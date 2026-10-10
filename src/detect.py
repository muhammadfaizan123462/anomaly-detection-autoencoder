"""Evaluate detection quality on the labelled test set.

Reports precision/recall/F1 at the calibrated threshold and scans a range
of thresholds to show the trade-off.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (average_precision_score, classification_report,
                             precision_recall_curve, roc_auc_score)

from model import Autoencoder


def main() -> None:
    p = argparse.ArgumentParser(description="Evaluate anomaly detection.")
    p.add_argument("--latent-dim", type=int, default=3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = joblib.load("checkpoints/scaler.joblib")
    threshold = json.loads(Path("checkpoints/threshold.json").read_text())["threshold"]

    df = pd.read_csv("data/test.csv")
    y_true = df["is_anomaly"].to_numpy()
    X = scaler.transform(
        df.drop(columns=["is_anomaly"]).to_numpy(dtype=np.float32)).astype(np.float32)

    model = Autoencoder(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/autoencoder.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        scores = model.reconstruction_error(
            torch.from_numpy(X).to(device)).cpu().numpy()

    y_pred = (scores >= threshold).astype(int)
    print(f"Threshold: {threshold:.5f}")
    print(classification_report(y_true, y_pred, digits=4,
                                target_names=["normal", "anomaly"]))
    print(f"ROC AUC: {roc_auc_score(y_true, scores):.4f}")
    print(f"Average precision: {average_precision_score(y_true, scores):.4f}")

    # Threshold sweep for the report.
    precision, recall, thresholds = precision_recall_curve(y_true, scores)
    f1 = 2 * precision * recall / np.clip(precision + recall, 1e-9, None)
    best_i = int(np.argmax(f1))
    print(f"\nBest F1 {f1[best_i]:.4f} at threshold "
          f"{thresholds[best_i] if best_i < len(thresholds) else float('inf'):.5f} "
          f"(precision {precision[best_i]:.4f}, recall {recall[best_i]:.4f})")


if __name__ == "__main__":
    main()
