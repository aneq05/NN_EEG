from __future__ import annotations

import numpy as np
from scipy.optimize import minimize
from sklearn.metrics import log_loss


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = np.asarray(logits)
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> TemperatureScaler:
        logits = np.asarray(logits, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int64)

        def objective(log_t: np.ndarray) -> float:
            temperature = float(np.exp(log_t[0]))
            probs = softmax(logits / temperature)
            return log_loss(y_true, probs, labels=list(range(probs.shape[1])))

        result = minimize(objective, x0=np.array([0.0]), method="Nelder-Mead")
        self.temperature = float(np.exp(result.x[0]))
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) / self.temperature


def expected_calibration_error(y_true: np.ndarray, probs: np.ndarray, n_bins: int = 15) -> float:
    confidences = probs.max(axis=1)
    predictions = probs.argmax(axis=1)
    correct = predictions == y_true
    ece = 0.0
    bins = np.linspace(0, 1, n_bins + 1)
    for lo, hi in zip(bins[:-1], bins[1:], strict=False):
        mask = (confidences > lo) & (confidences <= hi)
        if mask.any():
            ece += mask.mean() * abs(correct[mask].mean() - confidences[mask].mean())
    return float(ece)


def multiclass_brier(y_true: np.ndarray, probs: np.ndarray) -> float:
    y_onehot = np.eye(probs.shape[1])[y_true]
    return float(np.mean(np.sum((probs - y_onehot) ** 2, axis=1)))
