import torch

from src.model import make_model


def test_model_output_shape():
    model = make_model({"input_dim": 178, "num_classes": 5, "hidden_dims": [16], "dropout": 0.0})
    logits = model(torch.zeros((4, 178), dtype=torch.float32))

    assert logits.shape == (4, 5)

