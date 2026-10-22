"""Station recommendation scoring — weighted priority score per zone."""

from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import kendalltau

from evplan.utils.config import load_config
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def normalize_col(s: pd.Series) -> pd.Series:
    """Min-max normalize a Series to [0, 1]."""
    mn, mx = s.min(), s.max()
    if mx == mn:
        return pd.Series([0.5] * len(s), index=s.index)
    return (s - mn) / (mx - mn)


def compute_priority_scores(
    zone_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
) -> pd.DataFrame:
    """
    Compute a weighted priority score for each zone:
    Score = w1*demand + w2*hotspot_z + w3*supply_gap + w4*distance + w5*ev_adoption

    All components normalized to [0,1] before weighting.
    Returns zone_df with added score columns.
    """
    cfg = load_config()
    w = weights or cfg["recommendation"]["weights"]

    df = zone_df.copy()

    # Component 1: Normalized predicted demand
    demand_col = "predicted_demand_kwh" if "predicted_demand_kwh" in df.columns else "historical_demand_kwh"
    df["comp_demand"] = normalize_col(df[demand_col].fillna(0)) if demand_col in df.columns else 0.0

    # Component 2: Hotspot Gi* z-score (clamp and normalize)
    if "gi_z_score" in df.columns:
        z = df["gi_z_score"].fillna(0).clip(-3, 3)
        df["comp_hotspot"] = normalize_col(z)
    else:
        df["comp_hotspot"] = 0.0

    # Component 3: Supply gap (demand / stations ratio)
    if "demand_supply_ratio" in df.columns:
        df["comp_supply_gap"] = normalize_col(df["demand_supply_ratio"].fillna(0))
    else:
        stations = df.get("station_count", pd.Series([1]*len(df))).clip(lower=1)
        gap = df[demand_col].fillna(0) / stations if demand_col in df.columns else pd.Series([0]*len(df))
        df["comp_supply_gap"] = normalize_col(gap)

    # Component 4: Distance proxy (inverse of station density — more distance = higher score)
    if "station_count" in df.columns:
        inv_stations = 1.0 / (df["station_count"].clip(lower=1))
        df["comp_distance"] = normalize_col(inv_stations)
    else:
        df["comp_distance"] = 0.5

    # Component 5: EV adoption context
    df["comp_ev_adoption"] = normalize_col(df.get("ev_adoption_score", pd.Series([0]*len(df))).fillna(0))

    # Weighted sum
    df["priority_score"] = (
        w.get("demand", 0.30) * df["comp_demand"] +
        w.get("hotspot", 0.25) * df["comp_hotspot"] +
        w.get("supply_gap", 0.20) * df["comp_supply_gap"] +
        w.get("distance", 0.15) * df["comp_distance"] +
        w.get("ev_adoption", 0.10) * df["comp_ev_adoption"]
    )

    # Recommended chargers
    cfg_rec = cfg["recommendation"]
    per_kwh = cfg_rec["chargers_per_100_kwh_demand"] / 100.0
    demand_val = df[demand_col] if demand_col in df.columns else pd.Series([0]*len(df))
    raw_chargers = (demand_val * per_kwh).fillna(0).round().astype(int)
    df["recommended_chargers"] = raw_chargers.clip(
        cfg_rec["min_chargers"], cfg_rec["max_chargers"]
    )

    return df


def rank_zones(
    zone_df: pd.DataFrame,
    weights: dict[str, float] | None = None,
    top_n: int = 10,
) -> pd.DataFrame:
    """
    Rank zones by priority_score.
    Returns top-N rows with rank column.
    """
    df = compute_priority_scores(zone_df, weights)
    df = df.sort_values("priority_score", ascending=False).reset_index(drop=True)
    df.insert(0, "rank", range(1, len(df) + 1))

    cfg = load_config()
    n = top_n or cfg["recommendation"]["top_n"]
    ranked = df.head(n).copy()

    logger.info("Top-%d zones ranked. Best: %s (score=%.4f)",
                n, ranked.iloc[0].get("zone_id", "?"), ranked.iloc[0]["priority_score"])
    return ranked


def sensitivity_analysis(
    zone_df: pd.DataFrame,
    base_weights: dict[str, float],
    perturbation: float = 0.20,
) -> dict:
    """
    Analyze rank stability under ±20% weight perturbations.
    Returns: {perturbation_scenario: Kendall tau vs base ranking}
    """
    cfg = load_config()
    base_ranked = rank_zones(zone_df, base_weights)
    base_order = base_ranked["zone_id"].tolist() if "zone_id" in base_ranked.columns else list(range(len(base_ranked)))

    results = {}
    for key in base_weights:
        for sign, label in [(1 + perturbation, f"+{int(perturbation*100)}%"),
                            (1 - perturbation, f"-{int(perturbation*100)}%")]:
            new_w = {k: v * (sign if k == key else 1.0) for k, v in base_weights.items()}
            # Renormalize
            total = sum(new_w.values())
            new_w = {k: v / total for k, v in new_w.items()}
            perturbed = rank_zones(zone_df, new_w)
            p_order = perturbed["zone_id"].tolist() if "zone_id" in perturbed.columns else list(range(len(perturbed)))

            # Kendall tau on the intersection
            common = [z for z in base_order if z in p_order]
            if len(common) < 2:
                tau = 0.0
            else:
                base_ranks = [base_order.index(z) for z in common]
                perturbed_ranks = [p_order.index(z) for z in common]
                tau, _ = kendalltau(base_ranks, perturbed_ranks)

            scenario = f"{key} {label}"
            results[scenario] = {"kendall_tau": round(float(tau), 4), "weights": new_w}

    return results
