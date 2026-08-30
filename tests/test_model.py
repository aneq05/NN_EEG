import pytest
import torch
from sklearn.metrics import f1_score

from src.model import make_model


def test_model_output_shape():
    model = make_model({"input_dim": 178, "num_classes": 5, "hidden_dims": [16], "dropout": 0.0})
    logits = model(torch.zeros((4, 178), dtype=torch.float32))

    assert logits.shape == (4, 5)


def test_validation_macro_f1_is_computed_over_full_epoch_not_batch_mean():
    model = make_model({"input_dim": 2, "num_classes": 3, "hidden_dims": [4], "dropout": 0.0})
    batch_1_preds = torch.tensor([0, 0])
    batch_1_y = torch.tensor([0, 1])
    batch_2_preds = torch.tensor([1, 2])
    batch_2_y = torch.tensor([1, 2])

    model.val_f1.update(batch_1_preds, batch_1_y)
    model.val_f1.update(batch_2_preds, batch_2_y)

    epoch_f1 = model.val_f1.compute().item()
    full_dataset_f1 = f1_score(
        torch.cat([batch_1_y, batch_2_y]).numpy(),
        torch.cat([batch_1_preds, batch_2_preds]).numpy(),
        average="macro",
    )
    batch_mean_f1 = (
        f1_score(batch_1_y.numpy(), batch_1_preds.numpy(), average="macro")
        + f1_score(batch_2_y.numpy(), batch_2_preds.numpy(), average="macro")
    ) / 2

    assert epoch_f1 == pytest.approx(full_dataset_f1)
    assert epoch_f1 != batch_mean_f1
