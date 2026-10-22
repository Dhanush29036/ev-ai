"""Baseline models — naive, seasonal-naive, linear regression."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from evplan.models.evaluate import compute_metrics
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


class NaiveBaseline:
    """Last-observation naive forecast."""

    name = "Naive"

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "NaiveBaseline":
        self._last = float(y.iloc[-1])
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        return np.full(len(X), self._last)


class SeasonalNaiveBaseline:
    """Seasonal naive: repeat last week (168 lags for hourly data)."""

    name = "Seasonal Naive"

    def __init__(self, period: int = 168):
        self.period = period
        self._history: np.ndarray = np.array([])

    def fit(self, X: pd.DataFrame, y: pd.Series) -> "SeasonalNaiveBaseline":
        self._history = y.values.copy()
        return self

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        n = len(X)
        preds = []
        hist = self._history.tolist()
        for i in range(n):
            idx = len(hist) - self.period + (i % self.period)
            if idx < 0 or idx >= len(hist):
                preds.append(np.mean(hist[-self.period:]) if len(hist) >= self.period else np.mean(hist))
            else:
                preds.append(hist[idx])
        return np.array(preds)


def build_linear_baseline() -> Pipeline:
    """Return sklearn Pipeline with SimpleImputer + StandardScaler + Ridge regression."""
    return Pipeline([
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
        ("ridge", Ridge(alpha=1.0)),
    ])


def evaluate_baselines(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    X_test: pd.DataFrame,
    y_test: pd.Series,
) -> dict[str, dict]:
    """Fit and evaluate all baselines. Return dict of metrics."""
    results: dict[str, dict] = {}

    for baseline in [NaiveBaseline(), SeasonalNaiveBaseline()]:
        baseline.fit(X_train, y_train)
        y_pred = baseline.predict(X_test)
        m = compute_metrics(y_test, y_pred, prefix=baseline.name.lower().replace(" ", "_"))
        results[baseline.name] = {"metrics": m, "y_pred": y_pred.tolist()}
        logger.info("Baseline '%s': RMSE=%.4f", baseline.name, list(m.values())[0])

    # Linear regression
    try:
        lr = build_linear_baseline()
        lr.fit(X_train, y_train)
        y_pred_lr = lr.predict(X_test)
        m_lr = compute_metrics(y_test, y_pred_lr, prefix="linear")
        results["Linear Regression"] = {"metrics": m_lr, "y_pred": y_pred_lr.tolist(), "model": lr}
        logger.info("Baseline 'Linear Regression': RMSE=%.4f", list(m_lr.values())[0])
    except Exception as exc:
        logger.warning("Linear baseline failed: %s", exc)

    return results
