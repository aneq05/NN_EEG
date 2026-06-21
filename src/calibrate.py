from __future__ import annotations

import numpy as np
import torch
from scipy.optimize import minimize
from sklearn.metrics import log_loss


class TemperatureScaler:
    def __init__(self) -> None:
        self.temperature = 1.0

    def fit(self, logits: np.ndarray, y_true: np.ndarray) -> "TemperatureScaler":
        logits = np.asarray(logits, dtype=np.float64)
        y_true = np.asarray(y_true, dtype=np.int64)

        def objective(log_t: np.ndarray) -> float:
            temp = float(np.exp(log_t[0]))
            probs = softmax(logits / temp)
            return log_loss(y_true, probs, labels=list(range(probs.shape[1])))

        result = minimize(objective, x0=np.array([0.0]), method="Nelder-Mead")
        self.temperature = float(np.exp(result.x[0]))
        return self

    def transform_logits(self, logits: np.ndarray) -> np.ndarray:
        return np.asarray(logits) / self.temperature


def softmax(logits: np.ndarray) -> np.ndarray:
    logits = logits - logits.max(axis=1, keepdims=True)
    exp = np.exp(logits)
    return exp / exp.sum(axis=1, keepdims=True)


def predict_logits(model: torch.nn.Module, x: np.ndarray, batch_size: int = 512) -> np.ndarray:
    model.eval()
    outputs = []
    with torch.no_grad():
        for start in range(0, len(x), batch_size):
            xb = torch.tensor(x[start : start + batch_size], dtype=torch.float32)
            outputs.append(model(xb).detach().cpu().numpy())
    return np.concatenate(outputs, axis=0)

