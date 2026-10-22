"""Model training — RF, XGBoost, optional LightGBM, with model registry."""

from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Optional

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from xgboost import XGBRegressor

from evplan.models.baselines import evaluate_baselines
from evplan.models.evaluate import compute_metrics, compare_models, evaluate_model, select_best_model
from evplan.models.split import chronological_split, get_X_y, check_leakage, make_tscv
from evplan.models.tune import tune_model
from evplan.utils.config import load_config, get_path
from evplan.utils.logging import get_logger
from evplan.utils.seed import set_global_seed

logger = get_logger(__name__)


def _data_hash(df: pd.DataFrame) -> str:
    """Stable hash of a DataFrame for reproducibility tracking."""
    return hashlib.md5(pd.util.hash_pandas_object(df, index=True).values).hexdigest()[:10]


def _build_rf(params: dict) -> RandomForestRegressor:
    cfg = load_config()
    defaults = cfg["models"]["rf"]
    p = {**defaults, **params, "random_state": cfg["project"]["seed"]}
    p.pop("enabled", None)
    return RandomForestRegressor(**p)


def _build_xgb(params: dict) -> XGBRegressor:
    cfg = load_config()
    defaults = cfg["models"]["xgboost"]
    p = {**defaults, **params, "random_state": cfg["project"]["seed"], "verbosity": 0}
    p.pop("enabled", None)
    return XGBRegressor(**p)


def _build_lgbm(params: dict) -> Any:
    try:
        from lightgbm import LGBMRegressor  # type: ignore
        cfg = load_config()
        defaults = cfg["models"]["lightgbm"]
        p = {**defaults, **params, "random_state": cfg["project"]["seed"], "verbose": -1}
        p.pop("enabled", None)
        return LGBMRegressor(**p)
    except ImportError:
        logger.warning("LightGBM not installed — skipping.")
        return None


def _save_model(
    model: Any,
    name: str,
    feature_cols: list[str],
    metrics: dict,
    data_hash: str,
) -> Path:
    """Serialize model + metadata to outputs/models/."""
    models_dir = get_path("models")
    models_dir.mkdir(parents=True, exist_ok=True)

    model_path = models_dir / f"{name}.joblib"
    joblib.dump(model, model_path)

    cfg = load_config()
    registry_entry = {
        "name": name,
        "version": "1.0",
        "timestamp": datetime.utcnow().isoformat(),
        "seed": cfg["project"]["seed"],
        "data_hash": data_hash,
        "feature_cols": feature_cols,
        "metrics": metrics,
        "path": str(model_path),
    }

    registry_path = models_dir / "model_registry.json"
    registry: list = []
    if registry_path.exists():
        registry = json.loads(registry_path.read_text())
    registry.append(registry_entry)
    registry_path.write_text(json.dumps(registry, indent=2))

    logger.info("Saved model '%s' → %s", name, model_path)
    return model_path


def train_pipeline(
    df: pd.DataFrame,
    tune: bool = True,
    n_trials: int = 30,
) -> dict[str, Any]:
    """
    Full training pipeline:
    1. Chronological split
    2. Leakage check
    3. Baseline evaluation
    4. Optuna tuning (optional)
    5. RF + XGBoost (+ optional LightGBM) training
    6. Evaluation on held-out test set
    7. Save models and registry
    Returns dict with all results.
    """
    cfg = load_config()
    set_global_seed(cfg["project"]["seed"])
    target = cfg["data"]["target_column"]

    logger.info("=== Training Pipeline Start ===")

    # Split
    train_df, val_df, test_df = chronological_split(
        df, target_col=target,
        test_size=cfg["data"]["test_size"],
        val_size=cfg["data"]["val_size"],
    )

    # Combine train+val for final fit (test stays held-out)
    trainval_df = pd.concat([train_df, val_df], ignore_index=True)

    # Feature columns
    feat_cols = [c for c in df.columns if c != target and not c.endswith("_outlier_flag")]
    feat_cols_num = [c for c in feat_cols if pd.api.types.is_numeric_dtype(df[c])]

    X_train, y_train = get_X_y(train_df, target, feat_cols_num)
    X_val, y_val = get_X_y(val_df, target, feat_cols_num)
    X_test, y_test = get_X_y(test_df, target, feat_cols_num)
    X_trainval, y_trainval = get_X_y(trainval_df, target, feat_cols_num)

    logger.info("Feature matrix: %d train, %d val, %d test rows | %d features",
                len(X_train), len(X_val), len(X_test), len(feat_cols_num))

    # Leakage check
    suspicious = check_leakage(X_trainval, y_trainval)
    if suspicious:
        logger.warning("Dropping suspicious leaky features: %s", suspicious)
        feat_cols_num = [c for c in feat_cols_num if c not in suspicious]
        X_train = X_train[[c for c in feat_cols_num if c in X_train.columns]]
        X_trainval = X_trainval[[c for c in feat_cols_num if c in X_trainval.columns]]
        X_test = X_test[[c for c in feat_cols_num if c in X_test.columns]]

    data_hash = _data_hash(trainval_df)
    all_results: dict[str, Any] = {}

    # Baselines
    logger.info("--- Baselines ---")
    baseline_results = evaluate_baselines(X_train, y_train, X_test, y_test)
    all_results.update(baseline_results)

    # Random Forest
    if cfg["models"]["rf"]["enabled"]:
        logger.info("--- Random Forest ---")
        rf_params = {}
        if tune:
            rf_params, _ = tune_model("rf", X_trainval, y_trainval, n_trials=n_trials)
        rf = _build_rf(rf_params)
        rf.fit(X_trainval, y_trainval)
        rf_eval = evaluate_model(rf, X_test, y_test, model_name="rf")
        all_results["Random Forest"] = {**rf_eval, "model": rf, "feature_cols": list(X_trainval.columns)}
        _save_model(rf, "random_forest", list(X_trainval.columns), rf_eval["metrics"], data_hash)

    # XGBoost
    if cfg["models"]["xgboost"]["enabled"]:
        logger.info("--- XGBoost ---")
        xgb_params = {}
        if tune:
            xgb_params, _ = tune_model("xgboost", X_trainval, y_trainval, n_trials=n_trials)
        xgb = _build_xgb(xgb_params)
        xgb.fit(X_trainval, y_trainval)
        xgb_eval = evaluate_model(xgb, X_test, y_test, model_name="xgboost")
        all_results["XGBoost"] = {**xgb_eval, "model": xgb, "feature_cols": list(X_trainval.columns)}
        _save_model(xgb, "xgboost", list(X_trainval.columns), xgb_eval["metrics"], data_hash)

    # LightGBM (optional)
    if cfg["models"]["lightgbm"]["enabled"]:
        logger.info("--- LightGBM (optional) ---")
        lgbm = _build_lgbm({})
        if lgbm is not None:
            lgbm.fit(X_trainval, y_trainval)
            lgbm_eval = evaluate_model(lgbm, X_test, y_test, model_name="lightgbm")
            all_results["LightGBM"] = {**lgbm_eval, "model": lgbm, "feature_cols": list(X_trainval.columns)}
            _save_model(lgbm, "lightgbm", list(X_trainval.columns), lgbm_eval["metrics"], data_hash)

    # Comparison table
    comparison = compare_models(all_results)
    table_path = get_path("tables") / "model_comparison.csv"
    table_path.parent.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(table_path, index=False)
    logger.info("Model comparison table saved.")

    best_name = select_best_model(
        {k: v for k, v in all_results.items() if "model" in v}
    )
    logger.info("=== Best model: %s ===", best_name)

    return {
        "results": all_results,
        "comparison": comparison,
        "best_model_name": best_name,
        "best_model": all_results[best_name]["model"],
        "feature_cols": all_results[best_name].get("feature_cols", feat_cols_num),
        "X_test": X_test,
        "y_test": y_test,
    }


def load_model(name: str) -> Optional[Any]:
    """Load a saved model by name from the model registry."""
    models_dir = get_path("models")
    path = models_dir / f"{name}.joblib"
    if not path.exists():
        logger.warning("Model file not found: %s", path)
        return None
    model = joblib.load(path)
    logger.info("Loaded model: %s", path)
    return model
