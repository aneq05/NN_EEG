import numpy as np
import pandas as pd

from src.data import clean_dataframe, compute_clip_bounds, split_and_scale


def test_clean_dataframe_removes_identifier_and_maps_labels():
    raw = pd.DataFrame(
        {
            "Unnamed": [10, 11, 12, 13, 14],
            "X1": [1, 2, 3, 4, 5],
            "X2": [5, 4, 3, 2, 1],
            "y": [1, 2, 3, 4, 5],
        }
    )

    cleaned = clean_dataframe(raw)

    assert "Unnamed" not in cleaned.columns
    assert cleaned["target"].tolist() == [0, 1, 2, 3, 4]


def test_clip_bounds_are_computed_from_training_data_only():
    train = pd.DataFrame({"X1": [0.0, 1.0, 2.0, 100.0], "X2": [-5.0, 0.0, 5.0, 10.0]})
    validation_with_extremes = pd.DataFrame({"X1": [-999.0, 999.0], "X2": [-999.0, 999.0]})

    low, high = compute_clip_bounds(train, clip_quantiles=(0.25, 0.75))

    assert np.isclose(low["X1"], train["X1"].quantile(0.25))
    assert np.isclose(high["X1"], train["X1"].quantile(0.75))
    assert low["X1"] != validation_with_extremes["X1"].min()
    assert high["X1"] != validation_with_extremes["X1"].max()


def test_split_and_scale_keeps_stratified_shapes_and_train_scaler_fit():
    rows = []
    for label in range(1, 6):
        for idx in range(10):
            rows.append({"X1": label * 10 + idx, "X2": label - idx, "y": label, "target": label - 1})
    df = pd.DataFrame(rows)

    splits = split_and_scale(df, test_size=0.2, val_size=0.2, random_state=7, clip_quantiles=(0.0, 1.0))

    assert splits.x_train.shape == (30, 2)
    assert splits.x_val.shape == (10, 2)
    assert splits.x_test.shape == (10, 2)
    assert set(splits.y_train) == {0, 1, 2, 3, 4}
    assert np.allclose(splits.x_train.mean(axis=0), 0.0, atol=1e-6)

