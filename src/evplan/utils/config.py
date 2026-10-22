"""Configuration loader — reads config.yaml and schema.yaml with path resolution."""

from __future__ import annotations

import os
from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml


def _find_root() -> Path:
    """Walk up from cwd until we find configs/config.yaml."""
    here = Path(__file__).resolve()
    for parent in [here, *here.parents]:
        if (parent / "configs" / "config.yaml").exists():
            return parent
    # fallback: cwd
    return Path.cwd()


PROJECT_ROOT: Path = _find_root()


@lru_cache(maxsize=1)
def load_config(path: str | None = None) -> dict[str, Any]:
    """Load and cache project config.yaml."""
    cfg_path = Path(path) if path else PROJECT_ROOT / "configs" / "config.yaml"
    with open(cfg_path, "r", encoding="utf-8") as f:
        cfg = yaml.safe_load(f)
    # Resolve relative paths to absolute
    for key, val in cfg.get("paths", {}).items():
        cfg["paths"][key] = str(PROJECT_ROOT / val)
    return cfg


@lru_cache(maxsize=1)
def load_schema(path: str | None = None) -> dict[str, Any]:
    """Load and cache schema.yaml."""
    schema_path = Path(path) if path else PROJECT_ROOT / "configs" / "schema.yaml"
    with open(schema_path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def get_path(key: str) -> Path:
    """Return an absolute Path for a named path key from config."""
    cfg = load_config()
    return Path(cfg["paths"][key])


def get_param(section: str, key: str, default: Any = None) -> Any:
    """Safely retrieve a config parameter."""
    cfg = load_config()
    return cfg.get(section, {}).get(key, default)
