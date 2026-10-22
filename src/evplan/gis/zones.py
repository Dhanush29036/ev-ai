"""Zone builder — city-level zone aggregation from available data."""

from __future__ import annotations

from typing import Optional
import pandas as pd

from evplan.utils.logging import get_logger

logger = get_logger(__name__)


def build_zone_table(
    demand_df: Optional[pd.DataFrame],
    stations_df: Optional[pd.DataFrame],
    adoption_df: Optional[pd.DataFrame],
    predicted_demand: Optional[pd.Series] = None,
    feature_df: Optional[pd.DataFrame] = None,
) -> pd.DataFrame:
    """
    Build a zone-level summary table combining all available data sources.
    This is the core input to GIS analysis and recommendation scoring.
    """
    from evplan.data.integrate import (
        aggregate_stations_by_zone,
        aggregate_adoption_by_state,
        integrate_datasets,
        add_zone_id,
        normalize_state,
    )

    stations_zone = aggregate_stations_by_zone(stations_df) if stations_df is not None else pd.DataFrame()
    adoption_state = aggregate_adoption_by_state(adoption_df) if adoption_df is not None else pd.DataFrame()

    zone_df = integrate_datasets(demand_df, stations_zone, adoption_state)

    if zone_df.empty and demand_df is not None:
        # Fallback: aggregate demand by zone
        logger.warning("Integration failed — building zone table from demand data alone.")
        ddf = demand_df.copy()
        ddf = add_zone_id(ddf)
        if "energy_consumed_kwh" in ddf.columns:
            zone_df = (
                ddf.groupby("zone_id")["energy_consumed_kwh"]
                .agg(["sum", "mean", "count"])
                .reset_index()
                .rename(columns={"sum": "historical_demand_kwh", "mean": "avg_demand_kwh", "count": "n_sessions"})
            )
        else:
            zone_df = pd.DataFrame()

    if zone_df.empty:
        logger.warning("Zone table is empty — using dummy placeholder.")
        zone_df = pd.DataFrame({
            "zone_id": ["Delhi", "Mumbai", "Bangalore", "Chennai", "Hyderabad",
                        "Kolkata", "Pune", "Ahmedabad", "Jaipur", "Lucknow"],
            "zone_lat": [28.61, 19.08, 12.97, 13.08, 17.38, 22.57, 18.52, 23.02, 26.91, 26.85],
            "zone_lon": [77.21, 72.88, 77.59, 80.27, 78.49, 88.36, 73.86, 72.57, 75.79, 80.95],
            "historical_demand_kwh": [1200, 950, 800, 700, 650, 500, 450, 400, 350, 300],
            "station_count": [45, 38, 32, 25, 22, 18, 15, 12, 9, 8],
            "ev_adoption_score": [0.9, 0.85, 0.82, 0.65, 0.70, 0.55, 0.60, 0.50, 0.40, 0.38],
            "is_synthetic_demand": [True]*10,
        })
        logger.warning("Using SYNTHETIC PLACEHOLDER zone data — clearly labeled is_synthetic_demand=True")

    # Attach model-predicted demand if available
    if predicted_demand is not None and "zone_id" in feature_df.columns if feature_df is not None else False:
        pred_by_zone = pd.Series(
            predicted_demand.values,
            index=feature_df["zone_id"] if feature_df is not None else pd.Series([]),
        ).groupby(level=0).sum()
        zone_df["predicted_demand_kwh"] = zone_df["zone_id"].map(pred_by_zone).fillna(
            zone_df.get("historical_demand_kwh", 0)
        )
    else:
        zone_df["predicted_demand_kwh"] = zone_df.get(
            "historical_demand_kwh", pd.Series([0.0] * len(zone_df))
        )

    return zone_df
