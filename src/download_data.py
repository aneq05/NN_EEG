from __future__ import annotations

import argparse
import shutil
from pathlib import Path

import pandas as pd

from src.config import load_config


def download_dataset(config_path: str = "configs/config.yaml") -> Path:
    cfg = load_config(config_path)
    raw_path = Path(cfg["paths"]["raw_data"])
    raw_path.parent.mkdir(parents=True, exist_ok=True)
    if raw_path.exists():
        return raw_path

    try:
        import kagglehub
    except ImportError as exc:
        raise RuntimeError("Install kagglehub or place the CSV manually in data/raw/data.csv") from exc

    downloaded_dir = Path(kagglehub.dataset_download(cfg["data"]["kaggle_slug"]))
    csv_files = list(downloaded_dir.glob("*.csv"))
    if not csv_files:
        raise FileNotFoundError(f"No CSV file found in {downloaded_dir}")
    shutil.copy2(csv_files[0], raw_path)

    # Quick sanity check, fail early if a wrong file was downloaded.
    df = pd.read_csv(raw_path, nrows=5)
    if "y" not in df.columns:
        raise ValueError(f"Downloaded file does not contain target column 'y': {raw_path}")
    return raw_path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="configs/config.yaml")
    args = parser.parse_args()
    print(download_dataset(args.config))


if __name__ == "__main__":
    main()

