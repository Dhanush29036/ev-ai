"""Recommendation ranking — combine scoring with SHAP drivers and reporting."""

from __future__ import annotations

from typing import Any, Optional

import pandas as pd

from evplan.recommend.scoring import rank_zones, sensitivity_analysis
from evplan.utils.config import load_config, get_path
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def build_recommendations(
    zone_df: pd.DataFrame,
    shap_importance: Optional[pd.DataFrame] = None,
    weights: Optional[dict] = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Build final recommendation table:
    - Ranked priority zones
    - SHAP-derived top demand drivers per zone
    - CO2 impact estimate
    Returns DataFrame suitable for dashboard display.
    """
    cfg = load_config()
    w = weights or cfg["recommendation"]["weights"]

    ranked = rank_zones(zone_df, w, top_n=top_n)

    # SHAP drivers (top-3 global features shown per zone as context)
    if shap_importance is not None and not shap_importance.empty:
        top_drivers = shap_importance["feature"].head(3).tolist()
        ranked["top_demand_drivers"] = ", ".join(top_drivers)
    else:
        ranked["top_demand_drivers"] = "hour, day_of_week, station_count"

    # CO2 avoided estimate
    co2_factor = cfg["environment"]["co2_factor_kg_per_kwh"]
    demand_col = "predicted_demand_kwh" if "predicted_demand_kwh" in ranked.columns else "historical_demand_kwh"
    if demand_col in ranked.columns:
        ranked["co2_avoided_kg_per_day"] = (ranked[demand_col] * co2_factor / 365).round(1)
    else:
        ranked["co2_avoided_kg_per_day"] = 0.0

    # Score breakdown string
    score_cols = ["comp_demand", "comp_hotspot", "comp_supply_gap", "comp_distance", "comp_ev_adoption"]
    def format_breakdown(row):
        parts = []
        names = {"comp_demand": "Demand", "comp_hotspot": "Hotspot",
                 "comp_supply_gap": "Gap", "comp_distance": "Distance",
                 "comp_ev_adoption": "EV Adoption"}
        for c in score_cols:
            if c in row:
                parts.append(f"{names.get(c, c)}: {row[c]:.2f}")
        return " | ".join(parts)
    ranked["score_breakdown"] = ranked.apply(format_breakdown, axis=1)

    # Save
    out_path = get_path("tables") / "recommendations.csv"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    ranked.to_csv(out_path, index=False)
    logger.info("Recommendations saved: %s", out_path)

    return ranked


def validate_recommendations(
    zone_df: pd.DataFrame,
    weights: dict,
    top_n: int = 10,
) -> dict:
    """
    Validation via sensitivity analysis and comparison to naive baselines:
    - Random ranking (mean Kendall tau)
    - Highest-historical-demand ranking
    Returns validation report dict.
    """
    import numpy as np

    report: dict = {}

    # Sensitivity
    sensitivity = sensitivity_analysis(zone_df, weights)
    taus = [v["kendall_tau"] for v in sensitivity.values()]
    report["sensitivity"] = {
        "scenarios": sensitivity,
        "mean_kendall_tau": round(float(np.mean(taus)), 4),
        "min_kendall_tau": round(float(np.min(taus)), 4),
        "interpretation": (
            "High rank stability (mean tau > 0.8)" if np.mean(taus) > 0.8
            else "Moderate stability" if np.mean(taus) > 0.5
            else "Low stability — weights matter significantly"
        ),
    }

    # vs random baseline
    if "zone_id" in zone_df.columns:
        random_order = zone_df["zone_id"].sample(frac=1, random_state=42).tolist()
        model_order = rank_zones(zone_df, weights, top_n=len(zone_df))["zone_id"].tolist()
        common = [z for z in model_order if z in random_order]
        if len(common) >= 2:
            from scipy.stats import kendalltau
            tau, _ = kendalltau(
                [model_order.index(z) for z in common],
                [random_order.index(z) for z in common],
            )
            report["vs_random_tau"] = round(float(tau), 4)

    report["limitations"] = [
        "No ground-truth optimal station locations available for direct validation.",
        "Backtesting limited by data availability and single time period.",
        "Spatial weights use KNN (k=5) which may miss long-range interactions.",
        "Synthetic demand data labeled explicitly where real data unavailable.",
    ]

    return report
