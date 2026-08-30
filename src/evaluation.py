from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd
import seaborn as sns
import torch

matplotlib.use("Agg")

from matplotlib import pyplot as plt
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)
from sklearn.metrics import (
    classification_report as sklearn_classification_report,
)

from src.calibration import expected_calibration_error, multiclass_brier

CLASS_NAMES = [
    "epileptic_seizure",
    "tumor_area_eeg",
    "healthy_area_eeg",
    "eyes_closed",
    "eyes_open",
]


def predict_logits(model: torch.nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)


def metrics_dict(y_true: np.ndarray, probs: np.ndarray) -> dict[str, float]:
    preds = probs.argmax(axis=1)
    values = {
        "accuracy": accuracy_score(y_true, preds),
        "balanced_accuracy": balanced_accuracy_score(y_true, preds),
        "macro_f1": f1_score(y_true, preds, average="macro"),
        "weighted_f1": f1_score(y_true, preds, average="weighted"),
        "nll": log_loss(y_true, probs, labels=list(range(probs.shape[1]))),
        "brier": multiclass_brier(y_true, probs),
        "ece": expected_calibration_error(y_true, probs),
    }
    try:
        values["macro_ovr_auroc"] = roc_auc_score(y_true, probs, multi_class="ovr", average="macro")
    except ValueError:
        values["macro_ovr_auroc"] = float("nan")
    return {key: float(value) for key, value in values.items()}


def save_reliability_diagram(
    y_true: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    figures_dir: Path,
) -> None:
    figures_dir = Path(figures_dir)
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, 11)
    for label, probs in [("before", probs_before), ("after", probs_after)]:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = pred == y_true
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:], strict=False):
            mask = (conf > lo) & (conf <= hi)
            if mask.any():
                xs.append(conf[mask].mean())
                ys.append(correct[mask].mean())
        plt.plot(xs, ys, marker="o", label=label)
    plt.plot([0, 1], [0, 1], linestyle="--", color="black", linewidth=1)
    plt.xlabel("Mean confidence")
    plt.ylabel("Accuracy")
    plt.title("Reliability diagram")
    plt.legend()
    plt.tight_layout()
    plt.savefig(figures_dir / "reliability_diagram.png", dpi=180)
    plt.close()


def save_evaluation(
    y_true: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    metrics_dir: Path,
    figures_dir: Path,
) -> None:
    metrics_dir = Path(metrics_dir)
    figures_dir = Path(figures_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    figures_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    for stage, probs in [("before_calibration", probs_before), ("after_calibration", probs_after)]:
        row = {"stage": stage}
        row.update(metrics_dict(y_true, probs))
        rows.append(row)
    pd.DataFrame(rows).to_csv(metrics_dir / "test_metrics.csv", index=False)

    preds = probs_after.argmax(axis=1)
    report = sklearn_classification_report(y_true, preds, target_names=CLASS_NAMES, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(metrics_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix - calibrated model")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    save_reliability_diagram(y_true, probs_before, probs_after, figures_dir)


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


def save_uncertainty_cases(table: pd.DataFrame, tables_dir: Path, n: int = 20) -> dict[str, pd.DataFrame]:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    groups = {
        "most_confident_correct": table[table["correct"]].sort_values("max_probability", ascending=False).head(n),
        "most_uncertain_correct": table[table["correct"]].sort_values("entropy", ascending=False).head(n),
        "most_uncertain_wrong": table[~table["correct"]].sort_values("entropy", ascending=False).head(n),
    }
    table.to_csv(tables_dir / "uncertainty_all_test_predictions.csv", index=False)
    for name, group in groups.items():
        group.to_csv(tables_dir / f"{name}.csv", index=False)
    return groups
