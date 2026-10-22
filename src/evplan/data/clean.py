"""Data cleaning — standardization, outlier treatment, missing value handling."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evplan.utils.config import load_config
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


# ─── Datetime parsing ───────────────────────────────────────────────────────

def parse_datetime(df: pd.DataFrame) -> pd.DataFrame:
    """
    Parse the datetime column. Handles:
    - Combined timestamp string
    - Separate year/month/day/hour columns
    Falls back gracefully if parsing fails.
    """
    cfg = load_config()
    dt_col = cfg["data"]["datetime_column"]  # 'timestamp'

    if dt_col in df.columns:
        try:
            df[dt_col] = pd.to_datetime(df[dt_col], utc=False)
            df[dt_col] = df[dt_col].dt.tz_localize(None)  # strip tz
            logger.info("Parsed '%s' as datetime.", dt_col)
            return df
        except Exception as exc:
            logger.warning("Failed to parse '%s' as datetime: %s", dt_col, exc)

    # Try building from components
    parts = {"year": "year", "month": "month", "day": "day", "hour": "hour"}
    found = {k: v for k, v in parts.items() if v in df.columns}
    if len(found) >= 3:
        try:
            df["timestamp"] = pd.to_datetime(
                {
                    "year": df.get("year", pd.Series([2023] * len(df))),
                    "month": df.get("month", pd.Series([1] * len(df))),
                    "day": df.get("day", pd.Series([1] * len(df))),
                    "hour": df.get("hour", pd.Series([0] * len(df))),
                }
            )
            logger.info("Built 'timestamp' from year/month/day/hour columns.")
        except Exception as exc:
            logger.warning("Could not build timestamp from components: %s", exc)
    return df


# ─── Standardization ────────────────────────────────────────────────────────

def standardize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Strip whitespace from string columns, normalize dtypes."""
    for col in df.select_dtypes(include="object"):
        df[col] = df[col].astype(str).str.strip()
    return df


def remove_duplicates(df: pd.DataFrame) -> pd.DataFrame:
    """Remove exact duplicate rows."""
    before = len(df)
    df = df.drop_duplicates()
    removed = before - len(df)
    if removed:
        logger.info("Removed %d duplicate rows.", removed)
    return df


# ─── Missing values ─────────────────────────────────────────────────────────

def handle_missing(df: pd.DataFrame, target_col: str = "energy_consumed_kwh") -> pd.DataFrame:
    """
    Per-column missing value strategy:
    - Target column: drop rows with missing target
    - Numeric lag/time features: forward fill then median fill
    - Categoricals: fill with mode or 'Unknown'
    - All others: median for numeric, mode for categorical
    """
    # Drop rows where target is missing
    if target_col in df.columns:
        before = len(df)
        df = df.dropna(subset=[target_col])
        if len(df) < before:
            logger.info("Dropped %d rows with missing target '%s'.", before - len(df), target_col)

    num_cols = df.select_dtypes(include="number").columns.tolist()
    cat_cols = df.select_dtypes(include="object").columns.tolist()

    for col in num_cols:
        if df[col].isna().any():
            fill = df[col].median()
            df[col] = df[col].fillna(fill)
            logger.debug("Filled numeric '%s' with median=%.4f", col, fill)

    for col in cat_cols:
        if df[col].isna().any():
            mode_vals = df[col].mode()
            fill = mode_vals.iloc[0] if len(mode_vals) else "Unknown"
            df[col] = df[col].fillna(fill)
            logger.debug("Filled categorical '%s' with mode='%s'", col, fill)

    return df


# ─── Outlier detection ───────────────────────────────────────────────────────

def detect_outliers_iqr(series: pd.Series, threshold: float = 3.0) -> pd.Series:
    """Return boolean mask of outliers using IQR method."""
    q1 = series.quantile(0.25)
    q3 = series.quantile(0.75)
    iqr = q3 - q1
    lower = q1 - threshold * iqr
    upper = q3 + threshold * iqr
    return (series < lower) | (series > upper)


def treat_outliers(df: pd.DataFrame, target_col: str = "energy_consumed_kwh") -> pd.DataFrame:
    """
    Flag outliers in numeric columns with a boolean column (no blind deletion).
    Clip extreme values to 1st/99th percentile for model stability.
    """
    cfg = load_config()
    threshold = cfg["data"]["outlier_threshold"]
    num_cols = df.select_dtypes(include="number").columns.tolist()

    for col in num_cols:
        mask = detect_outliers_iqr(df[col], threshold)
        flag_col = f"{col}_outlier_flag"
        df[flag_col] = mask.astype(int)
        n = mask.sum()
        if n > 0:
            # Clip rather than delete
            p1 = df[col].quantile(0.01)
            p99 = df[col].quantile(0.99)
            df[col] = df[col].clip(lower=p1, upper=p99)
            logger.info("Flagged & clipped %d outliers in '%s'.", n, col)

    # Physical sanity for demand
    if target_col in df.columns:
        min_val = cfg["data"]["min_demand_kwh"]
        max_val = cfg["data"]["max_demand_kwh"]
        bad = (df[target_col] < min_val) | (df[target_col] > max_val)
        if bad.any():
            df = df[~bad]
            logger.info("Removed %d rows failing physical sanity check on '%s'.", bad.sum(), target_col)

    return df


# ─── Validation ──────────────────────────────────────────────────────────────

def validate_dataframe(df: pd.DataFrame, target_col: str) -> dict:
    """Return data quality statistics."""
    report: dict = {
        "n_rows": int(len(df)),
        "n_cols": int(len(df.columns)),
        "missing_per_col": df.isnull().sum().to_dict(),
        "missing_pct": (df.isnull().mean() * 100).round(2).to_dict(),
        "duplicates": int(df.duplicated().sum()),
    }
    if target_col in df.columns:
        report["target_stats"] = df[target_col].describe().to_dict()
    return report


# ─── Main clean entrypoint ───────────────────────────────────────────────────

def clean_demand(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """Full cleaning pipeline for demand dataset. Returns (cleaned_df, quality_report)."""
    cfg = load_config()
    target = cfg["data"]["target_column"]

    df = standardize_columns(df)
    df = remove_duplicates(df)
    df = parse_datetime(df)
    df = handle_missing(df, target_col=target)
    df = treat_outliers(df, target_col=target)

    report = validate_dataframe(df, target)
    logger.info("Cleaning complete: %d rows, %d cols", len(df), len(df.columns))
    return df, report


def clean_stations(df: pd.DataFrame) -> pd.DataFrame:
    """Clean stations dataset: validate coords, fill missing."""
    df = standardize_columns(df)
    df = remove_duplicates(df)

    # Validate coordinate ranges (India bounds)
    if "latitude" in df.columns and "longitude" in df.columns:
        df["latitude"] = pd.to_numeric(df["latitude"], errors="coerce")
        df["longitude"] = pd.to_numeric(df["longitude"], errors="coerce")
        before = len(df)
        df = df.dropna(subset=["latitude", "longitude"])
        # India bounds approximate
        df = df[
            (df["latitude"].between(6.4, 37.6)) &
            (df["longitude"].between(68.1, 97.4))
        ]
        logger.info("Stations: kept %d/%d within India bounds.", len(df), before)

    df = handle_missing(df)
    return df


def clean_adoption(df: pd.DataFrame) -> pd.DataFrame:
    """Clean EV adoption dataset."""
    df = standardize_columns(df)
    df = remove_duplicates(df)
    if "ev_sales" in df.columns:
        df["ev_sales"] = pd.to_numeric(df["ev_sales"], errors="coerce")
        df = df.dropna(subset=["ev_sales"])
        df = df[df["ev_sales"] >= 0]
    df = handle_missing(df)
    return df
