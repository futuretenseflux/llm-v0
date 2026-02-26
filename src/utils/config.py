from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any, Dict, Optional, Union

import yaml


def _repo_root() -> Path:
    return Path(__file__).resolve().parent.parent.parent


@lru_cache(maxsize=None)
def load_lm_config(config_path: Optional[Union[str, Path]] = None) -> Dict[str, Any]:
    path = Path(config_path) if config_path is not None else (_repo_root() / "configs" / "lm.yaml")
    if not path.is_absolute():
        path = _repo_root() / path

    try:
        with open(path, "r") as f:
            data = yaml.safe_load(f)
    except FileNotFoundError as e:
        raise FileNotFoundError(f"Could not find config file at: {path}") from e

    if not isinstance(data, dict):
        raise ValueError(f"Expected {path} to contain a YAML mapping at the top level, got: {type(data).__name__}")

    return data
