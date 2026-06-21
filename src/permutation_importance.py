from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score


def permutation_importance(
    predict_fn,
    x: np.ndarray,
    y: np.ndarray,
    feature_names: list[str],
    n_repeats: int = 3,
    random_state: int = 42,
) -> pd.DataFrame:
    rng = np.random.default_rng(random_state)
    baseline = f1_score(y, predict_fn(x), average="macro")
    rows = []
    for j, name in enumerate(feature_names):
        drops = []
        for _ in range(n_repeats):
            x_perm = x.copy()
            x_perm[:, j] = rng.permutation(x_perm[:, j])
            score = f1_score(y, predict_fn(x_perm), average="macro")
            drops.append(baseline - score)
        rows.append({"feature": name, "importance_mean": np.mean(drops), "importance_std": np.std(drops)})
    return pd.DataFrame(rows).sort_values("importance_mean", ascending=False)


def segment_importance(feature_importance: pd.DataFrame, segment_size: int = 30) -> pd.DataFrame:
    rows = []
    for _, row in feature_importance.iterrows():
        idx = int(row["feature"].lstrip("Xx"))
        segment = f"X{((idx - 1) // segment_size) * segment_size + 1}-X{min(((idx - 1) // segment_size + 1) * segment_size, 178)}"
        rows.append({**row.to_dict(), "segment": segment})
    return (
        pd.DataFrame(rows)
        .groupby("segment", as_index=False)["importance_mean"]
        .sum()
        .sort_values("importance_mean", ascending=False)
    )


def save_importance(df: pd.DataFrame, path: str | Path, title: str) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    top = df.head(20).iloc[::-1]
    plt.figure(figsize=(8, 6))
    plt.barh(top.iloc[:, 0], top["importance_mean"])
    plt.xlabel("Macro F1 drop after permutation")
    plt.title(title)
    plt.tight_layout()
    plt.savefig(path, dpi=180)
    plt.close()
