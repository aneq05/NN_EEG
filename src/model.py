from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import lightning as L
import numpy as np
import torch
from torch import nn
from torch.utils.data import DataLoader, TensorDataset
from torchmetrics.classification import MulticlassF1Score

ARCHITECTURES = {
    "small": [128, 64],
    "medium": [256, 128, 64],
    "large": [512, 256, 128],
}


@dataclass
class ModelConfig:
    input_dim: int = 178
    num_classes: int = 5
    hidden_dims: list[int] = field(default_factory=lambda: [256, 128, 64])
    dropout: float = 0.3
    learning_rate: float = 0.001
    weight_decay: float = 0.0001


def seed_everything(seed: int) -> None:
    L.seed_everything(seed, workers=True)
    torch.use_deterministic_algorithms(True, warn_only=True)


class EEGDataModule(L.LightningDataModule):
    def __init__(self, splits: Any, batch_size: int = 128):
        super().__init__()
        self.splits = splits
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        self.train_ds = self._dataset(self.splits.x_train, self.splits.y_train)
        self.val_ds = self._dataset(self.splits.x_val, self.splits.y_val)
        self.test_ds = self._dataset(self.splits.x_test, self.splits.y_test)

    @staticmethod
    def _dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(torch.tensor(x, dtype=torch.float32), torch.tensor(y, dtype=torch.long))

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)


class EEGMLP(L.LightningModule):
    def __init__(self, cfg: ModelConfig):
        super().__init__()
        self.save_hyperparameters()
        layers: list[nn.Module] = []
        in_dim = cfg.input_dim
        for hidden_dim in cfg.hidden_dims:
            layers.extend(
                [
                    nn.Linear(in_dim, hidden_dim),
                    nn.BatchNorm1d(hidden_dim),
                    nn.ReLU(),
                    nn.Dropout(cfg.dropout),
                ]
            )
            in_dim = hidden_dim
        layers.append(nn.Linear(in_dim, cfg.num_classes))
        self.net = nn.Sequential(*layers)
        self.loss_fn = nn.CrossEntropyLoss()
        self.learning_rate = cfg.learning_rate
        self.weight_decay = cfg.weight_decay
        self.val_f1 = MulticlassF1Score(num_classes=cfg.num_classes, average="macro")

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", acc, prog_bar=False)
        return loss

    def validation_step(self, batch: tuple[torch.Tensor, torch.Tensor], batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.val_f1.update(preds, y)
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_macro_f1", self.val_f1, on_step=False, on_epoch=True, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(self.parameters(), lr=self.learning_rate, weight_decay=self.weight_decay)
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode="max", factor=0.5, patience=3)
        return {"optimizer": optimizer, "lr_scheduler": {"scheduler": scheduler, "monitor": "val_macro_f1"}}


def make_model(overrides: dict[str, Any] | None = None) -> EEGMLP:
    overrides = overrides or {}
    cfg = ModelConfig(
        input_dim=overrides.get("input_dim", 178),
        num_classes=overrides.get("num_classes", 5),
        hidden_dims=overrides.get("hidden_dims", [256, 128, 64]),
        dropout=overrides.get("dropout", 0.3),
        learning_rate=overrides.get("learning_rate", 0.001),
        weight_decay=overrides.get("weight_decay", 0.0001),
    )
    return EEGMLP(cfg)
