from __future__ import annotations

from dataclasses import dataclass

import lightning as L
import torch
from sklearn.metrics import f1_score
from torch import nn


@dataclass
class ModelConfig:
    input_dim: int
    num_classes: int
    hidden_dims: list[int]
    dropout: float
    learning_rate: float
    weight_decay: float


class EEGMLP(L.LightningModule):
    def __init__(self, cfg: ModelConfig) -> None:
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)

    def training_step(self, batch, batch_idx: int) -> torch.Tensor:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        self.log("train_loss", loss, prog_bar=False)
        self.log("train_acc", acc, prog_bar=False)
        return loss

    def validation_step(self, batch, batch_idx: int) -> None:
        x, y = batch
        logits = self(x)
        loss = self.loss_fn(logits, y)
        preds = logits.argmax(dim=1)
        acc = (preds == y).float().mean()
        macro_f1 = f1_score(y.detach().cpu().numpy(), preds.detach().cpu().numpy(), average="macro")
        self.log("val_loss", loss, prog_bar=True)
        self.log("val_acc", acc, prog_bar=True)
        self.log("val_macro_f1", macro_f1, prog_bar=True)

    def configure_optimizers(self):
        optimizer = torch.optim.AdamW(
            self.parameters(),
            lr=self.learning_rate,
            weight_decay=self.weight_decay,
        )
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.5,
            patience=3,
        )
        return {
            "optimizer": optimizer,
            "lr_scheduler": {"scheduler": scheduler, "monitor": "val_macro_f1"},
        }

