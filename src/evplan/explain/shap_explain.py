"""SHAP-based explainability — global and local explanations, plain-English narratives."""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Any, Optional

import numpy as np
import pandas as pd
import shap

from evplan.utils.config import load_config, get_path
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def _cache_path(model_name: str) -> Path:
    p = get_path("outputs") / f"shap_values_{model_name}.pkl"
    p.parent.mkdir(parents=True, exist_ok=True)
    return p


def compute_shap_values(
    model: Any,
    X: pd.DataFrame,
    model_name: str = "model",
    max_samples: int = 1000,
    force_recompute: bool = False,
) -> tuple[shap.Explanation, Any]:
    """
    Compute SHAP values using TreeExplainer.
    Caches to disk for dashboard speed.
    Returns (shap_values, explainer).
    """
    cache = _cache_path(model_name)

    if not force_recompute and cache.exists():
        logger.info("Loading cached SHAP values from %s", cache)
        with open(cache, "rb") as f:
            return pickle.load(f)

    # Sample for speed
    n = min(max_samples, len(X))
    X_sample = X.sample(n=n, random_state=42) if len(X) > n else X.copy()

    logger.info("Computing SHAP values for '%s' on %d samples…", model_name, n)
    explainer = shap.TreeExplainer(model)

    try:
        shap_values = explainer(X_sample)
    except Exception:
        # Fallback for some model types
        raw = explainer.shap_values(X_sample)
        shap_values = shap.Explanation(
            values=raw,
            base_values=explainer.expected_value,
            data=X_sample.values,
            feature_names=list(X_sample.columns),
        )

    with open(cache, "wb") as f:
        pickle.dump((shap_values, explainer), f)

    logger.info("SHAP computation complete. Cached to %s", cache)
    return shap_values, explainer


def get_global_importance(shap_values: shap.Explanation) -> pd.DataFrame:
    """Return DataFrame of mean |SHAP| per feature, sorted descending."""
    if hasattr(shap_values, "values"):
        vals = np.abs(shap_values.values)
        names = shap_values.feature_names
    else:
        vals = np.abs(shap_values)
        names = [f"f{i}" for i in range(vals.shape[1])]

    mean_abs = vals.mean(axis=0)
    df = pd.DataFrame({"feature": names, "mean_abs_shap": mean_abs})
    df = df.sort_values("mean_abs_shap", ascending=False).reset_index(drop=True)
    total = df["mean_abs_shap"].sum()
    df["importance_pct"] = (df["mean_abs_shap"] / (total if total > 0 else 1.0) * 100).round(2)
    return df


def generate_global_plots(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    model_name: str = "model",
    top_n: int = 15,
) -> dict[str, Path]:
    """Generate and save global SHAP plots (bar, beeswarm). Return {plot_type: path}."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = get_path("figures")
    figs_dir.mkdir(parents=True, exist_ok=True)
    saved: dict[str, Path] = {}

    # Bar plot
    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.bar(shap_values, max_display=top_n, show=False, ax=ax)
        ax.set_title(f"SHAP Global Feature Importance — {model_name}")
        path = figs_dir / f"shap_bar_{model_name}.png"
        fig.savefig(path, bbox_inches="tight", dpi=150)
        plt.close(fig)
        saved["bar"] = path
        logger.info("Saved SHAP bar plot: %s", path)
    except Exception as exc:
        logger.warning("SHAP bar plot failed: %s", exc)

    # Beeswarm
    try:
        fig, ax = plt.subplots(figsize=(10, 8))
        shap.plots.beeswarm(shap_values, max_display=top_n, show=False)
        path = figs_dir / f"shap_beeswarm_{model_name}.png"
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close("all")
        saved["beeswarm"] = path
        logger.info("Saved SHAP beeswarm plot: %s", path)
    except Exception as exc:
        logger.warning("SHAP beeswarm plot failed: %s", exc)

    return saved


def generate_dependence_plots(
    shap_values: shap.Explanation,
    X_sample: pd.DataFrame,
    model_name: str = "model",
    n_features: int = 5,
) -> list[Path]:
    """Generate SHAP dependence plots for top-N features."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = get_path("figures")
    figs_dir.mkdir(parents=True, exist_ok=True)
    paths = []

    importance = get_global_importance(shap_values)
    top_features = importance["feature"].head(n_features).tolist()

    for feat in top_features:
        if feat not in X_sample.columns:
            continue
        try:
            fig, ax = plt.subplots(figsize=(8, 5))
            shap.plots.scatter(shap_values[:, feat], show=False, ax=ax)
            ax.set_title(f"SHAP Dependence: {feat} — {model_name}")
            path = figs_dir / f"shap_dep_{model_name}_{feat.replace('/', '_')}.png"
            fig.savefig(path, bbox_inches="tight", dpi=150)
            plt.close(fig)
            paths.append(path)
        except Exception as exc:
            logger.warning("Dependence plot for '%s' failed: %s", feat, exc)

    return paths


def local_explanation(
    shap_values: shap.Explanation,
    idx: int,
    model_name: str = "model",
) -> Path:
    """Generate and save a waterfall plot for a single prediction."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    figs_dir = get_path("figures")
    figs_dir.mkdir(parents=True, exist_ok=True)

    try:
        fig, ax = plt.subplots(figsize=(10, 6))
        shap.plots.waterfall(shap_values[idx], show=False)
        path = figs_dir / f"shap_waterfall_{model_name}_{idx}.png"
        plt.savefig(path, bbox_inches="tight", dpi=150)
        plt.close("all")
        return path
    except Exception as exc:
        logger.warning("Waterfall plot failed for idx=%d: %s", idx, exc)
        return None


def narrative_explanation(
    shap_values: shap.Explanation,
    idx: int,
    y_pred: float,
    top_n: int = 5,
) -> str:
    """
    Generate a plain-English explanation for a single prediction.
    E.g.: "Predicted 34.2 kWh, driven mainly by evening hour (+6.1), weekday (+2.3)…"
    """
    if hasattr(shap_values, "values") and hasattr(shap_values, "feature_names"):
        if idx < 0 or idx >= len(shap_values.values):
            return f"Predicted {y_pred:.2f} kWh."
        vals = shap_values.values[idx]
        names = shap_values.feature_names
        base = float(shap_values.base_values[idx]) if hasattr(shap_values.base_values, "__len__") else float(shap_values.base_values)
    else:
        return f"Predicted {y_pred:.2f} kWh."

    # Sort by absolute SHAP
    pairs = sorted(zip(names, vals), key=lambda x: abs(x[1]), reverse=True)[:top_n]

    parts = []
    for feat, sv in pairs:
        sign = "+" if sv >= 0 else ""
        parts.append(f"{feat} ({sign}{sv:.2f})")

    drivers = ", ".join(parts)
    return (
        f"Predicted {y_pred:.2f} kWh (baseline: {base:.2f} kWh). "
        f"Top drivers: {drivers}."
    )


def consistency_check(
    shap_vals_rf: shap.Explanation,
    shap_vals_xgb: shap.Explanation,
) -> float:
    """
    Rank correlation (Spearman) of mean |SHAP| between RF and XGBoost.
    Returns correlation coefficient.
    """
    from scipy.stats import spearmanr

    imp_rf = get_global_importance(shap_vals_rf).set_index("feature")["mean_abs_shap"]
    imp_xgb = get_global_importance(shap_vals_xgb).set_index("feature")["mean_abs_shap"]

    common = imp_rf.index.intersection(imp_xgb.index)
    if len(common) < 2:
        return 0.0

    corr, _ = spearmanr(imp_rf[common], imp_xgb[common])
    logger.info("RF vs XGBoost SHAP rank correlation: %.4f", corr)
    return float(corr)
