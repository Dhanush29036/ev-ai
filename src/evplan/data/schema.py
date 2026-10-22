"""Schema mapper — maps raw dataset columns to standard pipeline names."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import pandas as pd

from evplan.utils.config import load_schema
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def build_rename_map(df: pd.DataFrame, standard_columns: dict) -> dict[str, str]:
    """
    Build {raw_col -> standard_col} by matching raw column names
    against the alias lists in schema.yaml.
    """
    raw_cols_lower = {c.lower(): c for c in df.columns}
    rename: dict[str, str] = {}
    for std_name, aliases in standard_columns.items():
        for alias in aliases:
            if alias.lower() in raw_cols_lower:
                raw_col = raw_cols_lower[alias.lower()]
                if raw_col not in rename:
                    rename[raw_col] = std_name
                break
    return rename


def map_schema(df: pd.DataFrame, dataset_role: str = "demand") -> pd.DataFrame:
    """
    Apply schema mapping to a DataFrame.

    Parameters
    ----------
    df : raw DataFrame
    dataset_role : 'demand' | 'stations' | 'adoption'
    """
    schema = load_schema()
    std_cols = schema.get("standard_columns", {})
    rename_map = build_rename_map(df, std_cols)

    if rename_map:
        logger.info("Schema rename map: %s", rename_map)
        df = df.rename(columns=rename_map)
    else:
        logger.warning("No column renames found for dataset role '%s'", dataset_role)

    # Log unmapped columns
    mapped = set(rename_map.values())
    raw_remaining = [c for c in df.columns if c not in mapped]
    if raw_remaining:
        logger.debug("Unmapped columns (kept as-is): %s", raw_remaining)

    return df


def audit_required_columns(
    df: pd.DataFrame, dataset_role: str = "demand"
) -> tuple[bool, list[str]]:
    """Check that required columns for a role are present after mapping."""
    schema = load_schema()
    role_key = f"{dataset_role}_dataset"
    required = schema.get(role_key, {}).get("required", [])
    missing = [c for c in required if c not in df.columns]
    ok = len(missing) == 0
    if not ok:
        logger.warning("Missing required columns for '%s': %s", dataset_role, missing)
    return ok, missing
