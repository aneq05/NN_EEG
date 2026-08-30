import numpy as np
import pandas as pd
import pytest

from src.data import EXPECTED_FEATURES, clean_dataframe, compute_clip_bounds, feature_columns, split_and_scale


def valid_raw_dataframe() -> pd.DataFrame:
    rows = []
    for label in range(1, 6):
        row = {feature: float(index + label) for index, feature in enumerate(EXPECTED_FEATURES)}
        row["Unnamed"] = label
        row["y"] = label
        rows.append(row)
    return pd.DataFrame(rows, columns=["Unnamed", *EXPECTED_FEATURES, "y"])


def test_clean_dataframe_removes_identifier_and_maps_labels():
    raw = pd.DataFrame(
        {
            "Unnamed": [10, 11, 12, 13, 14],
            "X1": [1, 2, 3, 4, 5],
            "X2": [5, 4, 3, 2, 1],
            "y": [1, 2, 3, 4, 5],
        }
    )

    cleaned = clean_dataframe(raw, strict_schema=False)

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

    splits = split_and_scale(
        df,
        test_size=0.2,
        val_size=0.2,
        random_state=7,
        clip_quantiles=(0.0, 1.0),
        strict_schema=False,
    )

    assert splits.x_train.shape == (30, 2)
    assert splits.x_val.shape == (10, 2)
    assert splits.x_test.shape == (10, 2)
    assert set(splits.y_train) == {0, 1, 2, 3, 4}
    assert np.allclose(splits.x_train.mean(axis=0), 0.0, atol=1e-6)


def test_invalid_target_labels_are_rejected():
    raw = valid_raw_dataframe()
    raw.loc[0, "y"] = 6

    with pytest.raises(ValueError, match="labels 1-5"):
        clean_dataframe(raw)


def test_missing_values_are_rejected():
    raw = valid_raw_dataframe()
    raw.loc[0, "X1"] = np.nan

    with pytest.raises(ValueError, match="missing values"):
        clean_dataframe(raw)


def test_feature_schema_is_x1_to_x178():
    raw = valid_raw_dataframe()
    cleaned = clean_dataframe(raw)

    assert feature_columns(cleaned) == EXPECTED_FEATURES


def test_feature_schema_rejects_wrong_order():
    raw = valid_raw_dataframe()
    wrong_order = raw[["Unnamed", "X2", "X1", *EXPECTED_FEATURES[2:], "y"]]

    with pytest.raises(ValueError, match="exactly ordered"):
        clean_dataframe(wrong_order)


def test_feature_schema_rejects_non_numeric_features():
    raw = valid_raw_dataframe()
    raw["X10"] = "not numeric"

    with pytest.raises(ValueError, match="numeric"):
        clean_dataframe(raw)
