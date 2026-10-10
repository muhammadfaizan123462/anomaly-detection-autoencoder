"""Visualise reconstruction-error distributions and the latent space."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import joblib
import matplotlib
import numpy as np
import pandas as pd
import torch

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

from model import Autoencoder  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(description="Plot anomaly detection results.")
    p.add_argument("--latent-dim", type=int, default=3)
    args = p.parse_args()
    device = "cuda" if torch.cuda.is_available() else "cpu"

    scaler = joblib.load("checkpoints/scaler.joblib")
    threshold = json.loads(Path("checkpoints/threshold.json").read_text())["threshold"]
    df = pd.read_csv("data/test.csv")
    y = df["is_anomaly"].to_numpy()
    X = scaler.transform(
        df.drop(columns=["is_anomaly"]).to_numpy(dtype=np.float32)).astype(np.float32)

    model = Autoencoder(input_dim=X.shape[1], latent_dim=args.latent_dim).to(device)
    model.load_state_dict(torch.load("checkpoints/autoencoder.pt",
                                     map_location=device, weights_only=True))
    model.eval()
    with torch.no_grad():
        xt = torch.from_numpy(X).to(device)
        errors = model.reconstruction_error(xt).cpu().numpy()
        latent = model.encode(xt).cpu().numpy()

    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    axes[0].hist(errors[y == 0], bins=60, alpha=0.7, label="normal")
    axes[0].hist(errors[y == 1], bins=60, alpha=0.7, label="anomaly")
    axes[0].axvline(threshold, color="red", linestyle="--", label="threshold")
    axes[0].set_xlabel("Reconstruction error")
    axes[0].set_ylabel("Count")
    axes[0].legend()
    axes[0].set_title("Error distribution")

    sc = axes[1].scatter(latent[:, 0], latent[:, 1], c=y, cmap="coolwarm",
                         s=8, alpha=0.6)
    axes[1].set_xlabel("Latent dim 1")
    axes[1].set_ylabel("Latent dim 2")
    axes[1].set_title("Latent space (first 2 dims)")
    fig.colorbar(sc, ax=axes[1], label="anomaly")
    fig.tight_layout()
    Path("checkpoints").mkdir(exist_ok=True)
    fig.savefig("checkpoints/analysis.png", dpi=120)
    print("Saved checkpoints/analysis.png")


if __name__ == "__main__":
    main()
