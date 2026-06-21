from __future__ import annotations

from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    brier_score_loss,
    classification_report,
    confusion_matrix,
    f1_score,
    log_loss,
    roc_auc_score,
)


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == y_true
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:]):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))


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
    return {k: float(v) for k, v in values.items()}


def save_evaluation(
    y_true: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    label_names: list[str],
    metrics_dir: str | Path,
    figures_dir: str | Path,
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
    report = classification_report(y_true, preds, target_names=label_names, output_dict=True, zero_division=0)
    pd.DataFrame(report).T.to_csv(metrics_dir / "classification_report.csv")

    cm = confusion_matrix(y_true, preds)
    plt.figure(figsize=(7, 6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=label_names, yticklabels=label_names)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.title("Confusion matrix - calibrated model")
    plt.tight_layout()
    plt.savefig(figures_dir / "confusion_matrix.png", dpi=180)
    plt.close()

    save_reliability_diagram(y_true, probs_before, probs_after, figures_dir / "reliability_diagram.png")


def save_reliability_diagram(
    y_true: np.ndarray,
    probs_before: np.ndarray,
    probs_after: np.ndarray,
    output_path: str | Path,
    n_bins: int = 10,
) -> None:
    plt.figure(figsize=(7, 5))
    bins = np.linspace(0, 1, n_bins + 1)
    for label, probs in [("before", probs_before), ("after", probs_after)]:
        conf = probs.max(axis=1)
        pred = probs.argmax(axis=1)
        correct = pred == y_true
        xs, ys = [], []
        for lo, hi in zip(bins[:-1], bins[1:]):
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
    plt.savefig(output_path, dpi=180)
    plt.close()
