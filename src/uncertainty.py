from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


def uncertainty_table(y_true: np.ndarray, probs: np.ndarray) -> pd.DataFrame:
    sorted_probs = np.sort(probs, axis=1)
    preds = probs.argmax(axis=1)
    entropy = -np.sum(probs * np.log(np.clip(probs, 1e-12, 1.0)), axis=1)
    return pd.DataFrame(
        {
            "index": np.arange(len(y_true)),
            "true": y_true,
            "pred": preds,
            "correct": preds == y_true,
            "max_probability": probs.max(axis=1),
            "entropy": entropy,
            "margin_top1_top2": sorted_probs[:, -1] - sorted_probs[:, -2],
        }
    )


def save_uncertainty_cases(table: pd.DataFrame, output_dir: str | Path, n: int = 20) -> dict[str, pd.DataFrame]:
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "most_confident_correct": table[table["correct"]].sort_values("max_probability", ascending=False).head(n),
        "most_uncertain_correct": table[table["correct"]].sort_values("entropy", ascending=False).head(n),
        "most_uncertain_wrong": table[~table["correct"]].sort_values("entropy", ascending=False).head(n),
    }
    for name, df in groups.items():
        df.to_csv(output_dir / f"{name}.csv", index=False)
    table.to_csv(output_dir / "uncertainty_all_test_predictions.csv", index=False)
    return groups

