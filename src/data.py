"""Synthetic dataset for unsupervised anomaly detection.

Generates normal operating data from a mixture of Gaussians plus a small
fraction of injected anomalies (shifted / scaled outliers). The training
split contains ONLY normal points (unsupervised setup); the test split is
labelled so we can measure detection quality.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

RANDOM_STATE = 42
N_FEATURES = 12


def generate(n_normal: int = 8000, n_anomalies: int = 400,
             random_state: int = RANDOM_STATE) -> tuple[pd.DataFrame, pd.DataFrame]:
    rng = np.random.default_rng(random_state)

    def mixture(n: int) -> np.ndarray:
        comp = rng.integers(0, 3, n)
        means = np.array([[-2.0] * N_FEATURES, [0.0] * N_FEATURES, [2.0] * N_FEATURES])
        cov_scale = np.array([0.6, 1.0, 0.8])
        X = np.stack([
            rng.normal(means[c], cov_scale[c], N_FEATURES) for c in comp
        ])
        # Correlated feature pairs, like real sensor data.
        X[:, 4] = 0.7 * X[:, 0] + 0.3 * rng.normal(0, 0.5, n)
        X[:, 9] = -0.5 * X[:, 2] + rng.normal(0, 0.7, n)
        return X

    X_normal = mixture(n_normal)
    X_anom = mixture(n_anomalies)
    # Push anomalies far from the normal manifold.
    shift = rng.choice([-1, 1], (n_anomalies, N_FEATURES)) * rng.uniform(4, 7, (n_anomalies, N_FEATURES))
    X_anom = X_anom + shift

    cols = [f"sensor_{i:02d}" for i in range(N_FEATURES)]
    train_df = pd.DataFrame(X_normal, columns=cols)
    test_df = pd.DataFrame(np.vstack([X_normal[:1500], X_anom]), columns=cols)
    test_df["is_anomaly"] = [0] * 1500 + [1] * n_anomalies
    test_df = test_df.sample(frac=1.0, random_state=random_state).reset_index(drop=True)
    return train_df, test_df


def main() -> None:
    p = argparse.ArgumentParser(description="Generate anomaly detection data.")
    p.add_argument("--out", type=str, default="data")
    args = p.parse_args()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    train_df, test_df = generate()
    train_df.to_csv(out / "train.csv", index=False)
    test_df.to_csv(out / "test.csv", index=False)
    print(f"Wrote {len(train_df)} train (normal only) / {len(test_df)} test rows.")


if __name__ == "__main__":
    main()
