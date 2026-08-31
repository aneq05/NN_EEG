from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

RAW_DATASET = "harunshimanto/epileptic-seizure-recognition"
DEFAULT_SEED = 42
DEFAULT_TEST_SIZE = 0.15
DEFAULT_VAL_SIZE = 0.15
DEFAULT_CLIP_QUANTILES = (0.001, 0.999)
EXPECTED_FEATURES = [f"X{i}" for i in range(1, 179)]
EXPECTED_TARGET_VALUES = {1, 2, 3, 4, 5}


@dataclass(frozen=True)
class SplitData:
    x_train: np.ndarray
    y_train: np.ndarray
    x_val: np.ndarray
    y_val: np.ndarray
    x_test: np.ndarray
    y_test: np.ndarray
    feature_names: list[str]
    scaler: StandardScaler
    clip_low: pd.Series
    clip_high: pd.Series


def ensure_raw_data(raw_path: Path, dataset: str = RAW_DATASET) -> Path:
    """Return the raw CSV path, downloading it with kagglehub when absent."""
    raw_path = Path(raw_path)
    if raw_path.exists():
        return raw_path

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError(
            "Raw data is missing and kagglehub is not installed. "
            "Install project requirements or place data.csv in data/raw/."
        ) from exc

    raw_path.parent.mkdir(parents=True, exist_ok=True)
    dataset_dir = Path(kagglehub.dataset_download(dataset))
    candidates = sorted(dataset_dir.rglob("*.csv"))
    if not candidates:
        raise FileNotFoundError(f"No CSV file found in downloaded dataset: {dataset_dir}")

    raw_path.write_bytes(candidates[0].read_bytes())
    return raw_path


def load_raw_data(raw_path: Path) -> pd.DataFrame:
    return pd.read_csv(ensure_raw_data(raw_path))


def validate_raw_schema(df: pd.DataFrame) -> None:
    if "y" not in df.columns:
        raise ValueError("Raw dataframe must contain target column 'y'.")

    features_in_data = [column for column in df.columns if column.upper().startswith("X")]
    if features_in_data != EXPECTED_FEATURES:
        missing = [feature for feature in EXPECTED_FEATURES if feature not in features_in_data]
        unexpected = [feature for feature in features_in_data if feature not in EXPECTED_FEATURES]
        raise ValueError(
            "Raw dataframe must contain exactly ordered EEG features X1 through X178. "
            f"Missing: {missing[:5]}, unexpected: {unexpected[:5]}."
        )

    non_numeric = [column for column in EXPECTED_FEATURES if not pd.api.types.is_numeric_dtype(df[column])]
    if non_numeric:
        raise ValueError(f"EEG feature columns must be numeric. Non-numeric columns: {non_numeric[:5]}.")

    if df[EXPECTED_FEATURES + ["y"]].isna().any().any():
        raise ValueError("Raw dataframe contains missing values in EEG features or target.")

    target_values = set(df["y"].unique())
    if target_values != EXPECTED_TARGET_VALUES:
        raise ValueError(f"Target column 'y' must contain exactly labels 1-5. Found: {sorted(target_values)}.")


def clean_dataframe(df: pd.DataFrame, strict_schema: bool = True) -> pd.DataFrame:
    if strict_schema:
        validate_raw_schema(df)

    cleaned = df.copy()
    unnamed_columns = [column for column in cleaned.columns if column.lower().startswith("unnamed")]
    cleaned = cleaned.drop(columns=unnamed_columns, errors="ignore")
    cleaned = cleaned.drop_duplicates()
    cleaned["target"] = cleaned["y"].astype(int) - 1
    return cleaned


def feature_columns(df: pd.DataFrame, strict_schema: bool = True) -> list[str]:
    features = [column for column in df.columns if column.upper().startswith("X")]
    if strict_schema and features != EXPECTED_FEATURES:
        raise ValueError("Feature columns must be exactly ordered as X1 through X178.")
    return features


def prepare_model_inputs(df: pd.DataFrame, strict_schema: bool = True) -> tuple[pd.DataFrame, pd.Series, list[str]]:
    features = feature_columns(df, strict_schema=strict_schema)
    x = df[features].astype("float32")
    y = df["target"].astype("int64")
    return x, y, features


def split_train_val_test(
    x: pd.DataFrame,
    y: pd.Series,
    test_size: float = 0.15,
    val_size: float = 0.15,
    random_state: int = DEFAULT_SEED,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.Series, pd.Series, pd.Series]:
    holdout_size = val_size + test_size
    if not 0 < holdout_size < 1:
        raise ValueError("test_size + val_size must be in the interval (0, 1)")

    x_train, x_holdout, y_train, y_holdout = train_test_split(
        x,
        y,
        test_size=holdout_size,
        stratify=y,
        random_state=random_state,
    )

    test_share_in_holdout = test_size / holdout_size
    x_val, x_test, y_val, y_test = train_test_split(
        x_holdout,
        y_holdout,
        test_size=test_share_in_holdout,
        stratify=y_holdout,
        random_state=random_state,
    )
    return x_train, x_val, x_test, y_train, y_val, y_test


def compute_clip_bounds(
    x_train: pd.DataFrame,
    clip_quantiles: tuple[float, float] = (0.001, 0.999),
) -> tuple[pd.Series, pd.Series]:
    low_q, high_q = clip_quantiles
    if not 0 <= low_q < high_q <= 1:
        raise ValueError("clip_quantiles must satisfy 0 <= low < high <= 1")
    return x_train.quantile(low_q), x_train.quantile(high_q)


def apply_clip_bounds(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
    low: pd.Series,
    high: pd.Series,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    return (
        x_train.clip(low, high, axis=1),
        x_val.clip(low, high, axis=1),
        x_test.clip(low, high, axis=1),
    )


def scale_features(
    x_train: pd.DataFrame,
    x_val: pd.DataFrame,
    x_test: pd.DataFrame,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, StandardScaler]:
    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train).astype("float32")
    x_val_scaled = scaler.transform(x_val).astype("float32")
    x_test_scaled = scaler.transform(x_test).astype("float32")
    return x_train_scaled, x_val_scaled, x_test_scaled, scaler


def split_and_scale(
    df: pd.DataFrame,
    test_size: float = DEFAULT_TEST_SIZE,
    val_size: float = DEFAULT_VAL_SIZE,
    random_state: int = DEFAULT_SEED,
    clip_quantiles: tuple[float, float] = DEFAULT_CLIP_QUANTILES,
    strict_schema: bool = True,
) -> SplitData:
    x, y, features = prepare_model_inputs(df, strict_schema=strict_schema)
    x_train, x_val, x_test, y_train, y_val, y_test = split_train_val_test(
        x,
        y,
        test_size=test_size,
        val_size=val_size,
        random_state=random_state,
    )
    low, high = compute_clip_bounds(x_train, clip_quantiles)
    x_train, x_val, x_test = apply_clip_bounds(x_train, x_val, x_test, low, high)
    x_train_scaled, x_val_scaled, x_test_scaled, scaler = scale_features(x_train, x_val, x_test)
    return SplitData(
        x_train=x_train_scaled,
        y_train=y_train.to_numpy(dtype="int64"),
        x_val=x_val_scaled,
        y_val=y_val.to_numpy(dtype="int64"),
        x_test=x_test_scaled,
        y_test=y_test.to_numpy(dtype="int64"),
        feature_names=features,
        scaler=scaler,
        clip_low=low,
        clip_high=high,
    )


def save_processed_splits(splits: SplitData, processed_dir: Path) -> None:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    for split_name in ["train", "val", "test"]:
        pd.DataFrame(getattr(splits, f"x_{split_name}"), columns=splits.feature_names).assign(
            target=getattr(splits, f"y_{split_name}")
        ).to_csv(processed_dir / f"{split_name}.csv", index=False)

    joblib.dump(splits.scaler, processed_dir / "scaler.joblib")
    joblib.dump({"low": splits.clip_low, "high": splits.clip_high}, processed_dir / "outlier_clip_bounds.joblib")
    joblib.dump(splits.feature_names, processed_dir / "feature_names.joblib")


def save_data_summary(cleaned: pd.DataFrame, feature_names: list[str], tables_dir: Path) -> None:
    tables_dir = Path(tables_dir)
    tables_dir.mkdir(parents=True, exist_ok=True)
    cleaning_summary = pd.DataFrame(
        {
            "n_rows": [len(cleaned)],
            "n_features": [len(feature_names)],
            "missing_values": [int(cleaned[feature_names + ["y"]].isna().sum().sum())],
            "duplicates": [int(cleaned.duplicated().sum())],
        }
    )
    cleaning_summary.to_csv(tables_dir / "data_cleaning_summary.csv", index=False)
    cleaned["y"].value_counts().sort_index().rename_axis("class").reset_index(name="count").to_csv(
        tables_dir / "class_distribution.csv", index=False
    )
    cleaned[feature_names].describe().T.to_csv(tables_dir / "feature_describe.csv")
