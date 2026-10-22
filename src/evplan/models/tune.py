"""Hyperparameter tuning with Optuna for RF and XGBoost."""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Any, Callable, Optional

import numpy as np
import optuna
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import cross_val_score
from xgboost import XGBRegressor

from evplan.models.split import make_tscv
from evplan.utils.config import load_config, get_path
from evplan.utils.logging import get_logger
from evplan.utils.seed import set_global_seed

logger = get_logger(__name__)
optuna.logging.set_verbosity(optuna.logging.WARNING)


def _rf_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: Any) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 25),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
        "max_features": trial.suggest_float("max_features", 0.3, 1.0),
        "random_state": 42,
        "n_jobs": -1,
    }
    model = RandomForestRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    return float(-scores.mean())


def _xgb_objective(trial: optuna.Trial, X: pd.DataFrame, y: pd.Series, cv: Any) -> float:
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 400),
        "max_depth": trial.suggest_int("max_depth", 3, 10),
        "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
        "subsample": trial.suggest_float("subsample", 0.5, 1.0),
        "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
        "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
        "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        "reg_alpha": trial.suggest_float("reg_alpha", 0.0, 1.0),
        "reg_lambda": trial.suggest_float("reg_lambda", 0.5, 5.0),
        "random_state": 42,
        "n_jobs": -1,
        "verbosity": 0,
    }
    model = XGBRegressor(**params)
    scores = cross_val_score(model, X, y, cv=cv, scoring="neg_root_mean_squared_error", n_jobs=1)
    return float(-scores.mean())


def tune_model(
    model_type: str,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_trials: int = 30,
    timeout: int = 600,
) -> tuple[dict[str, Any], float]:
    """
    Run Optuna study for model_type in {'rf', 'xgboost'}.
    Returns (best_params, best_cv_rmse).
    """
    cfg = load_config()
    set_global_seed(cfg["project"]["seed"])
    cv = make_tscv(cfg["data"]["cv_folds"])

    objectives: dict[str, Callable] = {
        "rf": _rf_objective,
        "xgboost": _xgb_objective,
    }
    if model_type not in objectives:
        raise ValueError(f"Unknown model_type '{model_type}'. Choose from: {list(objectives)}")

    obj = objectives[model_type]
    study = optuna.create_study(direction="minimize", sampler=optuna.samplers.TPESampler(seed=42))

    history_path = get_path("tables") / f"optuna_{model_type}_history.csv"
    if history_path.exists():
        try:
            df_hist = pd.read_csv(history_path)
            if not df_hist.empty and "value" in df_hist.columns:
                best_row = df_hist.sort_values("value").iloc[0]
                best = {}
                for col in df_hist.columns:
                    if col.startswith("params_"):
                        k = col.replace("params_", "")
                        v = best_row[col]
                        if k in ("n_estimators", "max_depth", "min_samples_leaf", "min_child_weight"):
                            v = int(v)
                        best[k] = v
                best_rmse = float(best_row["value"])
                logger.info("Loaded previous Optuna tuning for '%s': best RMSE=%.4f", model_type, best_rmse)
                logger.info("Best params: %s", best)
                return best, best_rmse
        except Exception as exc:
            logger.warning("Could not read previous Optuna history: %s", exc)

    logger.info("Starting Optuna tuning for '%s' (%d trials, timeout=%ds)…", model_type, n_trials, timeout)
    t0 = time.time()
    study.optimize(
        lambda trial: obj(trial, X_train, y_train, cv),
        n_trials=n_trials,
        timeout=timeout,
        show_progress_bar=False,
    )
    elapsed = time.time() - t0

    best = study.best_params
    best_rmse = study.best_value
    logger.info("Tuning complete for '%s': best RMSE=%.4f in %.1fs", model_type, best_rmse, elapsed)
    logger.info("Best params: %s", best)

    # Save search history
    history_path = get_path("tables") / f"optuna_{model_type}_history.csv"
    history_path.parent.mkdir(parents=True, exist_ok=True)
    study.trials_dataframe().to_csv(history_path, index=False)

    return best, best_rmse
