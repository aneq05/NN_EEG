from __future__ import annotations

from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    with Path(path).open("r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    return cfg


def ensure_dirs(cfg: dict[str, Any]) -> None:
    for key, value in cfg["paths"].items():
        if key.endswith("_dir"):
            Path(value).mkdir(parents=True, exist_ok=True)


def project_path(cfg: dict[str, Any], key: str) -> Path:
    return Path(cfg["paths"][key])

