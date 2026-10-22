"""Data loaders — read raw files, apply schema mapping, return standardized DataFrames."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evplan.data.schema import map_schema, audit_required_columns
from evplan.utils.config import load_config, PROJECT_ROOT
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def _read_file(path: Path) -> Optional[pd.DataFrame]:
    """Auto-detect file format and read."""
    suffix = path.suffix.lower()
    try:
        if suffix == ".csv":
            return pd.read_csv(path, low_memory=False)
        elif suffix in (".xls", ".xlsx"):
            return pd.read_excel(path)
        elif suffix == ".json":
            return pd.read_json(path)
        elif suffix == ".parquet":
            return pd.read_parquet(path)
        else:
            logger.debug("Skipping unsupported file: %s", path)
            return None
    except Exception as exc:
        logger.warning("Failed to read %s: %s", path, exc)
        return None


def _load_directory(directory: Path) -> Optional[pd.DataFrame]:
    """Load all CSV/Excel/JSON files from a directory and concatenate."""
    files = sorted(directory.rglob("*.csv")) + sorted(directory.rglob("*.xlsx"))
    frames = []
    for f in files:
        df = _read_file(f)
        if df is not None and not df.empty:
            df["_source_file"] = f.name
            frames.append(df)
            logger.info("Loaded %s: %d rows x %d cols", f.name, len(df), len(df.columns))
    if not frames:
        return None
    combined = pd.concat(frames, ignore_index=True)
    logger.info("Combined dataset: %d rows x %d cols", len(combined), len(combined.columns))
    return combined


def load_demand_dataset() -> Optional[pd.DataFrame]:
    """Load and schema-map the EV charging demand dataset (Dataset A)."""
    cfg = load_config()
    data_dir = PROJECT_ROOT / cfg["datasets"]["demand"]["local_dir"]
    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.warning("Demand dataset not found at %s. Run download first.", data_dir)
        return None
    raw = _load_directory(data_dir)
    if raw is None:
        return None
    df = map_schema(raw, "demand")
    ok, missing = audit_required_columns(df, "demand")
    if not ok:
        logger.warning("Demand dataset missing required columns: %s — will attempt fallback columns.", missing)
    return df


def load_stations_dataset() -> Optional[pd.DataFrame]:
    """Load and schema-map the EV stations dataset (Dataset B)."""
    cfg = load_config()
    data_dir = PROJECT_ROOT / cfg["datasets"]["stations"]["local_dir"]
    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.warning("Stations dataset not found at %s.", data_dir)
        return None
    raw = _load_directory(data_dir)
    if raw is None:
        return None
    df = map_schema(raw, "stations")
    return df


def load_adoption_dataset() -> Optional[pd.DataFrame]:
    """Load and schema-map the Indian EV adoption dataset (Dataset C)."""
    cfg = load_config()
    data_dir = PROJECT_ROOT / cfg["datasets"]["adoption"]["local_dir"]
    if not data_dir.exists() or not any(data_dir.iterdir()):
        logger.warning("Adoption dataset not found at %s.", data_dir)
        return None
    raw = _load_directory(data_dir)
    if raw is None:
        return None
    df = map_schema(raw, "adoption")
    return df


def load_all() -> dict[str, Optional[pd.DataFrame]]:
    """Load all datasets and return a dict."""
    return {
        "demand": load_demand_dataset(),
        "stations": load_stations_dataset(),
        "adoption": load_adoption_dataset(),
    }
