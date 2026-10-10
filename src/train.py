"""Train the autoencoder on NORMAL data only (unsupervised).

The reconstruction-error threshold is calibrated on a held-out slice of
normal data (e.g. 95th percentile) and stored with the checkpoint.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler
from torch.utils.data import DataLoader, TensorDataset

from model import Autoencoder

CHECKPOINTS = Path("checkpoints")


def get_loader(csv: str, scaler: StandardScaler | None, batch_size: int,
               fit_scaler: bool) -> tuple[DataLoader, StandardScaler]:
    df = pd.read_csv(csv)
    X = df.drop(columns=["is_anomaly"], errors="ignore").to_numpy(dtype=np.float32)
    if fit_scaler:
        scaler = StandardScaler().fit(X)
    Xs = scaler.transform(X).astype(np.float32)
    loader = DataLoader(TensorDataset(torch.from_numpy(Xs)),
                        batch_size=batch_size, shuffle=True)
    return loader, scaler


def main() -> None:
    p = argparse.ArgumentParser(description="Train the autoencoder.")
    p.add_argument("--epochs", type=int, default=40)
    p.add_argument("--batch-size", type=int, default=256)
    p.add_argument("--latent-dim", type=int, default=3)
    p.add_argument("--lr", type=float, default=1e-3)
    p.add_argument("--threshold-q", type=float, default=0.95,
                   help="Quantile of normal errors used as anomaly threshold")
    args = p.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_loader, scaler = get_loader("data/train.csv", None, args.batch_size,
                                     fit_scaler=True)
    input_dim = train_loader.dataset.tensors[0].shape[1]
    model = Autoencoder(input_dim=input_dim,
                        latent_dim=args.latent_dim).to(device)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=1e-5)
    sched = torch.optim.lr_scheduler.ReduceLROnPlateau(
        opt, mode="min", factor=0.5, patience=5)
    criterion = nn.MSELoss()

    best, patience = float("inf"), 0
    for epoch in range(1, args.epochs + 1):
        model.train()
        total = 0.0
        for (xb,) in train_loader:
            xb = xb.to(device)
            opt.zero_grad()
            loss = criterion(model(xb), xb)
            loss.backward()
            opt.step()
            total += loss.item() * xb.size(0)
        avg = total / len(train_loader.dataset)
        sched.step(avg)
        print(f"Epoch {epoch:3d}/{args.epochs} | recon loss {avg:.5f} "
              f"| lr {opt.param_groups[0]['lr']:.2e}")
        if avg < best - 1e-6:
            best, patience = avg, 0
            CHECKPOINTS.mkdir(parents=True, exist_ok=True)
            torch.save(model.state_dict(), CHECKPOINTS / "autoencoder.pt")
        else:
            patience += 1
            if patience >= 10:
                print("Early stopping.")
                break

    # Calibrate the anomaly threshold on training (normal) errors.
    model.load_state_dict(torch.load(CHECKPOINTS / "autoencoder.pt",
                                    map_location=device, weights_only=True))
    model.eval()
    errors = []
    with torch.no_grad():
        for (xb,) in train_loader:
            errors.append(model.reconstruction_error(xb.to(device)).cpu())
    errors = torch.cat(errors).numpy()
    threshold = float(np.quantile(errors, args.threshold_q))
    (CHECKPOINTS / "threshold.json").write_text(json.dumps(
        {"threshold": threshold, "quantile": args.threshold_q,
         "mean_train_error": float(errors.mean())}, indent=2))
    import joblib
    joblib.dump(scaler, CHECKPOINTS / "scaler.joblib")
    print(f"Threshold (q={args.threshold_q}): {threshold:.5f}")


if __name__ == "__main__":
    main()
