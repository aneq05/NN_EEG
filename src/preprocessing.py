from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler


def load_raw_data(path: str | Path) -> pd.DataFrame:
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Missing raw data at {path}. Run download_data.py first.")
    return pd.read_csv(path)


def clean_dataframe(df: pd.DataFrame, id_columns: list[str], remove_duplicates: bool = True) -> pd.DataFrame:
    cleaned = df.drop(columns=[c for c in id_columns if c in df.columns], errors="ignore").copy()
    cleaned = cleaned.drop(columns=[c for c in cleaned.columns if c.lower().startswith("unnamed")], errors="ignore")
    if remove_duplicates:
        cleaned = cleaned.drop_duplicates()
    cleaned["target"] = cleaned["y"].astype(int) - 1
    return cleaned


def feature_columns(df: pd.DataFrame) -> list[str]:
    return [c for c in df.columns if c.upper().startswith("X")]


def split_and_scale(
    df: pd.DataFrame,
    processed_dir: str | Path,
    test_size: float,
    val_size: float,
    random_state: int,
    clip_quantiles: tuple[float, float] | list[float] | None,
) -> dict[str, np.ndarray | list[str] | StandardScaler]:
    processed_dir = Path(processed_dir)
    processed_dir.mkdir(parents=True, exist_ok=True)
    features = feature_columns(df)
    x = df[features].astype("float32")
    y = df["target"].astype("int64")

    x_train, x_temp, y_train, y_temp = train_test_split(
        x, y, test_size=test_size + val_size, stratify=y, random_state=random_state
    )
    relative_test = test_size / (test_size + val_size)
    x_val, x_test, y_val, y_test = train_test_split(
        x_temp, y_temp, test_size=relative_test, stratify=y_temp, random_state=random_state
    )

    if clip_quantiles:
        low_q, high_q = clip_quantiles
        low = x_train.quantile(low_q)
        high = x_train.quantile(high_q)
        x_train = x_train.clip(low, high, axis=1)
        x_val = x_val.clip(low, high, axis=1)
        x_test = x_test.clip(low, high, axis=1)
        joblib.dump({"low": low, "high": high}, processed_dir / "outlier_clip_bounds.joblib")

    scaler = StandardScaler()
    x_train_scaled = scaler.fit_transform(x_train)
    x_val_scaled = scaler.transform(x_val)
    x_test_scaled = scaler.transform(x_test)

    splits = {
        "x_train": x_train_scaled.astype("float32"),
        "y_train": y_train.to_numpy(dtype="int64"),
        "x_val": x_val_scaled.astype("float32"),
        "y_val": y_val.to_numpy(dtype="int64"),
        "x_test": x_test_scaled.astype("float32"),
        "y_test": y_test.to_numpy(dtype="int64"),
        "x_train_raw": x_train.to_numpy(dtype="float32"),
        "x_val_raw": x_val.to_numpy(dtype="float32"),
        "x_test_raw": x_test.to_numpy(dtype="float32"),
        "feature_names": features,
        "scaler": scaler,
    }
    joblib.dump(scaler, processed_dir / "scaler.joblib")
    for name in ["train", "val", "test"]:
        pd.DataFrame(splits[f"x_{name}"], columns=features).assign(target=splits[f"y_{name}"]).to_csv(
            processed_dir / f"{name}.csv", index=False
        )
    return splits

