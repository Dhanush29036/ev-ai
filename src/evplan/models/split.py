"""Chronological train/test split and time-series cross-validation. No random splits."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit

from evplan.utils.config import load_config
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def chronological_split(
    df: pd.DataFrame,
    target_col: str = "energy_consumed_kwh",
    test_size: float = 0.2,
    val_size: float = 0.1,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """
    Split DataFrame into train / val / test using chronological order.
    NEVER uses random shuffling.

    Returns: (train_df, val_df, test_df)
    """
    n = len(df)
    test_n = max(1, int(n * test_size))
    val_n = max(1, int(n * val_size))
    train_n = n - val_n - test_n

    if train_n <= 0:
        raise ValueError(f"Not enough data for split: {n} rows, test={test_n}, val={val_n}")

    train = df.iloc[:train_n].copy()
    val = df.iloc[train_n: train_n + val_n].copy()
    test = df.iloc[train_n + val_n:].copy()

    logger.info(
        "Chronological split: train=%d | val=%d | test=%d",
        len(train), len(val), len(test),
    )
    return train, val, test


def get_X_y(
    df: pd.DataFrame,
    target_col: str = "energy_consumed_kwh",
    feature_cols: list[str] | None = None,
) -> tuple[pd.DataFrame, pd.Series]:
    """Extract feature matrix X and target y. Drop non-numeric junk."""
    if feature_cols is None:
        exclude = {target_col}
        feature_cols = [c for c in df.columns if c not in exclude]

    X = df[feature_cols].copy()
    # Drop any non-numeric columns that might have slipped through
    X = X.select_dtypes(include="number")
    y = df[target_col].copy()
    return X, y


def check_leakage(
    X: pd.DataFrame,
    y: pd.Series,
    threshold: float = 0.98,
) -> list[str]:
    """
    Check for potential data leakage: features with Pearson |r| > threshold with target.
    Returns list of suspicious feature names.
    """
    suspicious = []
    for col in X.columns:
        try:
            corr = X[col].corr(y)
            if abs(corr) > threshold:
                suspicious.append((col, corr))
                logger.warning(
                    "POTENTIAL LEAKAGE: '%s' has corr=%.4f with target!", col, corr
                )
        except Exception:
            pass
    if suspicious:
        logger.warning("Leakage check flagged %d features: %s", len(suspicious), suspicious)
    else:
        logger.info("Leakage check passed — no features above |r|=%.2f threshold.", threshold)
    return [col for col, _ in suspicious]


def make_tscv(n_splits: int = 5) -> TimeSeriesSplit:
    """Create a TimeSeriesSplit for cross-validation."""
    return TimeSeriesSplit(n_splits=n_splits)
