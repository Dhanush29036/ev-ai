"""Model evaluation — RMSE, MAE, R², MAPE, sMAPE, residual analysis."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def _safe_mape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """MAPE, guarded against zero-division."""
    return float(np.mean(np.abs((y_true - y_pred) / (np.abs(y_true) + eps))) * 100)


def _smape(y_true: np.ndarray, y_pred: np.ndarray, eps: float = 1e-8) -> float:
    """Symmetric MAPE."""
    return float(
        np.mean(2 * np.abs(y_pred - y_true) / (np.abs(y_true) + np.abs(y_pred) + eps)) * 100
    )


def compute_metrics(
    y_true: np.ndarray | pd.Series,
    y_pred: np.ndarray | pd.Series,
    prefix: str = "",
) -> dict[str, float]:
    """Compute regression metrics."""
    yt = np.array(y_true, dtype=float)
    yp = np.array(y_pred, dtype=float)

    rmse = float(np.sqrt(mean_squared_error(yt, yp)))
    mae = float(mean_absolute_error(yt, yp))
    r2 = float(r2_score(yt, yp))
    mape = _safe_mape(yt, yp)
    smape = _smape(yt, yp)

    metrics = {
        "rmse": rmse, "mae": mae, "r2": r2, "mape": mape, "smape": smape,
    }
    if prefix:
        metrics = {f"{prefix}_{k}": v for k, v in metrics.items()}

    for k, v in metrics.items():
        logger.info("  %s = %.4f", k, v)

    return metrics


def evaluate_model(
    model: Any,
    X_test: pd.DataFrame,
    y_test: pd.Series,
    model_name: str = "model",
) -> dict[str, Any]:
    """Full evaluation: metrics + residuals + error by hour/zone."""
    y_pred = model.predict(X_test)
    metrics = compute_metrics(y_test, y_pred, prefix=model_name)

    residuals = pd.Series(y_test.values - y_pred, index=y_test.index, name="residual")

    result = {
        "metrics": metrics,
        "y_pred": y_pred.tolist(),
        "residuals": residuals.tolist(),
    }

    # Error by hour
    if "hour" in X_test.columns:
        hourly_rmse = (
            pd.DataFrame({"hour": X_test["hour"], "err": (y_test.values - y_pred) ** 2})
            .groupby("hour")["err"]
            .mean()
            .apply(np.sqrt)
            .to_dict()
        )
        result["hourly_rmse"] = {int(k): float(v) for k, v in hourly_rmse.items()}

    logger.info("Evaluation complete for %s", model_name)
    return result


def compare_models(results: dict[str, dict]) -> pd.DataFrame:
    """Build a comparison table from multiple model result dicts."""
    rows = []
    canonical_metrics = ["rmse", "mae", "r2", "mape", "smape"]
    for model_name, res in results.items():
        row = {"model": model_name}
        mets = res.get("metrics", {})
        for cm in canonical_metrics:
            val = None
            for k, v in mets.items():
                if k == cm or k.endswith(f"_{cm}"):
                    val = round(float(v), 4)
                    break
            if val is not None:
                row[cm] = val
        rows.append(row)
    df = pd.DataFrame(rows)
    if "rmse" in df.columns:
        df = df.sort_values("rmse")
    return df


def select_best_model(results: dict[str, dict], metric: str = "rmse") -> str:
    """Return the model name with the best (lowest) CV RMSE."""
    scores = {}
    for name, res in results.items():
        mets = res.get("metrics", {})
        val = None
        for k, v in mets.items():
            if k == metric or k.endswith(f"_{metric}"):
                val = float(v)
                break
        if val is not None:
            scores[name] = val
    if not scores:
        return list(results.keys())[0]
    best = min(scores, key=scores.get)
    logger.info("Best model by %s: %s (%.4f)", metric, best, scores[best])
    return best
