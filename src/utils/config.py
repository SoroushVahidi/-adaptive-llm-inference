"""Configuration loading utilities."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import yaml


def load_config(path: str | Path) -> dict[str, Any]:
    """Load a YAML or JSON config file.

    Dispatches on file extension (``.yaml`` / ``.yml`` → YAML, else JSON).
    """
    path = Path(path)
    text = path.read_text()
    if path.suffix in (".yaml", ".yml"):
        return yaml.safe_load(text)
    return json.loads(text)
