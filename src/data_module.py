from __future__ import annotations

import lightning as L
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset


class EEGDataModule(L.LightningDataModule):
    def __init__(
        self,
        x_train: np.ndarray,
        y_train: np.ndarray,
        x_val: np.ndarray,
        y_val: np.ndarray,
        x_test: np.ndarray,
        y_test: np.ndarray,
        batch_size: int = 128,
    ) -> None:
        super().__init__()
        self.arrays = x_train, y_train, x_val, y_val, x_test, y_test
        self.batch_size = batch_size

    def setup(self, stage: str | None = None) -> None:
        x_train, y_train, x_val, y_val, x_test, y_test = self.arrays
        self.train_ds = self._dataset(x_train, y_train)
        self.val_ds = self._dataset(x_val, y_val)
        self.test_ds = self._dataset(x_test, y_test)

    @staticmethod
    def _dataset(x: np.ndarray, y: np.ndarray) -> TensorDataset:
        return TensorDataset(
            torch.tensor(x, dtype=torch.float32),
            torch.tensor(y, dtype=torch.long),
        )

    def train_dataloader(self) -> DataLoader:
        return DataLoader(self.train_ds, batch_size=self.batch_size, shuffle=True, num_workers=0)

    def val_dataloader(self) -> DataLoader:
        return DataLoader(self.val_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

    def test_dataloader(self) -> DataLoader:
        return DataLoader(self.test_ds, batch_size=self.batch_size, shuffle=False, num_workers=0)

