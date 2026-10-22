"""Feature engineering — temporal, lag, spatial, categorical features."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from evplan.utils.config import load_config
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


# ─── Holiday helper ──────────────────────────────────────────────────────────

def _get_india_holidays(years: list[int]) -> set:
    """Return a set of India holiday dates."""
    try:
        import holidays as hol
        india = hol.India(years=years)
        return set(india.keys())
    except ImportError:
        logger.warning("'holidays' package not found — holiday feature will be zero.")
        return set()


# ─── Time features ───────────────────────────────────────────────────────────

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add hour, day of week, month, season, weekend, cyclic encodings, holiday."""
    if "timestamp" not in df.columns:
        logger.warning("No 'timestamp' column — skipping time features.")
        return df

    ts = df["timestamp"]
    df["hour"] = ts.dt.hour
    df["day_of_week"] = ts.dt.dayofweek          # 0=Monday
    df["day_of_month"] = ts.dt.day
    df["month"] = ts.dt.month
    df["quarter"] = ts.dt.quarter
    df["week_of_year"] = ts.dt.isocalendar().week.astype(int)
    df["is_weekend"] = (ts.dt.dayofweek >= 5).astype(int)

    # Season (India-centric)
    month = ts.dt.month
    df["season"] = pd.cut(
        month,
        bins=[0, 2, 5, 8, 11, 12],
        labels=["Winter", "Spring", "Summer", "Monsoon", "Winter"],
        ordered=False,
    ).astype(str)

    # Cyclic encodings
    df["hour_sin"] = np.sin(2 * np.pi * df["hour"] / 24)
    df["hour_cos"] = np.cos(2 * np.pi * df["hour"] / 24)
    df["dow_sin"] = np.sin(2 * np.pi * df["day_of_week"] / 7)
    df["dow_cos"] = np.cos(2 * np.pi * df["day_of_week"] / 7)
    df["month_sin"] = np.sin(2 * np.pi * (df["month"] - 1) / 12)
    df["month_cos"] = np.cos(2 * np.pi * (df["month"] - 1) / 12)

    # Holiday flag
    cfg = load_config()
    if cfg["features"]["include_holidays"]:
        years = sorted(ts.dt.year.unique().tolist())
        holidays = _get_india_holidays(years)
        df["is_holiday"] = ts.dt.date.apply(lambda d: int(d in holidays))
    else:
        df["is_holiday"] = 0

    logger.info("Time features added: hour, dow, month, season, cyclic, holiday.")
    return df


# ─── Lag features ────────────────────────────────────────────────────────────

def add_lag_features(
    df: pd.DataFrame,
    target_col: str = "energy_consumed_kwh",
    group_col: Optional[str] = None,
) -> pd.DataFrame:
    """
    Add lag and rolling features strictly from past data (no leakage).
    Requires df to be sorted by timestamp.
    """
    cfg = load_config()
    lags = cfg["features"]["lag_hours"]
    windows = cfg["features"]["rolling_windows"]

    if target_col not in df.columns:
        logger.warning("Target '%s' not found — skipping lag features.", target_col)
        return df

    if "timestamp" in df.columns:
        df = df.sort_values("timestamp").reset_index(drop=True)

    def _add_lags(series: pd.Series) -> pd.DataFrame:
        parts = {}
        for lag in lags:
            parts[f"{target_col}_lag_{lag}"] = series.shift(lag)
        for w in windows:
            shifted = series.shift(1)
            parts[f"{target_col}_roll_mean_{w}"] = shifted.rolling(w, min_periods=1).mean()
            parts[f"{target_col}_roll_std_{w}"] = shifted.rolling(w, min_periods=1).std().fillna(0)
        return pd.DataFrame(parts)

    if group_col and group_col in df.columns:
        lag_dfs = df.groupby(group_col)[target_col].transform
        # Per-group approach
        all_lag_frames = []
        for gid, gdf in df.groupby(group_col):
            lag_frame = _add_lags(gdf[target_col])
            lag_frame.index = gdf.index
            all_lag_frames.append(lag_frame)
        lag_full = pd.concat(all_lag_frames).sort_index()
    else:
        lag_full = _add_lags(df[target_col])

    df = pd.concat([df, lag_full], axis=1)
    logger.info("Lag features added: lags=%s, rolling=%s", lags, windows)
    return df


# ─── Spatial features ────────────────────────────────────────────────────────

def add_spatial_features(
    df: pd.DataFrame,
    zone_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Add zone-level spatial features:
    - station_count, station_density, total_capacity_kw, ev_adoption_score
    from the integrated zone table.
    """
    if zone_stats is None or zone_stats.empty:
        logger.warning("No zone stats available — spatial features will be zeros.")
        for col in ["station_count", "ev_adoption_score", "total_capacity_kw"]:
            df[col] = 0.0
        return df

    if "zone_id" not in df.columns:
        from evplan.data.integrate import add_zone_id
        df = add_zone_id(df)

    zone_cols = [c for c in ["zone_id", "station_count", "ev_adoption_score",
                              "total_capacity_kw", "zone_lat", "zone_lon",
                              "total_chargers", "avg_capacity_kw"] if c in zone_stats.columns]

    df = df.merge(zone_stats[zone_cols], on="zone_id", how="left")

    # Fill missing
    for col in ["station_count", "ev_adoption_score", "total_capacity_kw"]:
        if col in df.columns:
            df[col] = df[col].fillna(0.0)

    logger.info("Spatial features joined from %d zones.", len(zone_stats))
    return df


# ─── Categorical encoding ────────────────────────────────────────────────────

def encode_categoricals(df: pd.DataFrame, target_col: str = "energy_consumed_kwh") -> pd.DataFrame:
    """
    Encode categoricals:
    - Low-cardinality (<= 10 unique): one-hot
    - High-cardinality (> 10, includes zone_id, city, state): target encode or label encode
    - Season: ordinal
    """
    season_map = {"Winter": 0, "Spring": 1, "Summer": 2, "Monsoon": 3}
    if "season" in df.columns:
        df["season"] = df["season"].map(season_map).fillna(0).astype(int)

    cat_cols = [c for c in df.select_dtypes(include="object").columns
                if c not in ("timestamp", "_source_file", "is_synthetic_demand")]

    low_card = [c for c in cat_cols if df[c].nunique() <= 10]
    high_card = [c for c in cat_cols if df[c].nunique() > 10]

    # One-hot for low-cardinality
    if low_card:
        df = pd.get_dummies(df, columns=low_card, drop_first=False, dtype=int)
        logger.info("One-hot encoded: %s", low_card)

    # Label encode high-cardinality
    for col in high_card:
        df[col] = pd.factorize(df[col])[0]
        logger.info("Label encoded (high-card): %s", col)

    return df


# ─── Master feature engineering ──────────────────────────────────────────────

def engineer_features(
    df: pd.DataFrame,
    zone_stats: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """Run the full feature engineering pipeline."""
    cfg = load_config()
    target = cfg["data"]["target_column"]

    df = add_time_features(df)
    df = add_lag_features(df, target_col=target)
    df = add_spatial_features(df, zone_stats=zone_stats)
    df = encode_categoricals(df, target_col=target)

    # Drop raw datetime (keep components)
    if "timestamp" in df.columns:
        df = df.drop(columns=["timestamp"])

    # Drop helper columns
    for col in ["_source_file", "is_synthetic_demand"]:
        if col in df.columns:
            df = df.drop(columns=[col])

    logger.info("Feature engineering complete: %d rows, %d cols", len(df), len(df.columns))
    return df


def get_feature_columns(df: pd.DataFrame, target_col: str = "energy_consumed_kwh") -> list[str]:
    """Return sorted list of feature columns (excludes target and flag columns)."""
    exclude = {target_col, "timestamp"}
    return sorted([c for c in df.columns if c not in exclude and not c.endswith("_outlier_flag")])
