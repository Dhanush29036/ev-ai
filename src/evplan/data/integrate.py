"""Data integration — spatial/administrative join across datasets."""

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd

from evplan.utils.logging import get_logger

logger = get_logger(__name__)


# ─── State name normalizer ────────────────────────────────────────────────────

STATE_ALIASES: dict[str, str] = {
    "delhi": "Delhi", "new delhi": "Delhi", "ncr": "Delhi",
    "maharashtra": "Maharashtra", "mh": "Maharashtra",
    "karnataka": "Karnataka", "ka": "Karnataka",
    "tamil nadu": "Tamil Nadu", "tn": "Tamil Nadu", "tamilnadu": "Tamil Nadu",
    "gujarat": "Gujarat", "gj": "Gujarat",
    "rajasthan": "Rajasthan", "rj": "Rajasthan",
    "uttar pradesh": "Uttar Pradesh", "up": "Uttar Pradesh",
    "andhra pradesh": "Andhra Pradesh", "ap": "Andhra Pradesh",
    "telangana": "Telangana", "ts": "Telangana",
    "kerala": "Kerala", "kl": "Kerala",
    "west bengal": "West Bengal", "wb": "West Bengal",
    "punjab": "Punjab", "pb": "Punjab",
    "haryana": "Haryana", "hr": "Haryana",
    "madhya pradesh": "Madhya Pradesh", "mp": "Madhya Pradesh",
    "odisha": "Odisha", "or": "Odisha",
    "bihar": "Bihar", "br": "Bihar",
    "assam": "Assam", "as": "Assam",
    "jharkhand": "Jharkhand", "jh": "Jharkhand",
    "uttarakhand": "Uttarakhand", "uk": "Uttarakhand",
    "himachal pradesh": "Himachal Pradesh", "hp": "Himachal Pradesh",
    "goa": "Goa", "ga": "Goa",
    "chhattisgarh": "Chhattisgarh", "cg": "Chhattisgarh",
}


def normalize_state(s: str) -> str:
    """Normalize state name to canonical form."""
    return STATE_ALIASES.get(str(s).strip().lower(), str(s).strip().title())


def normalize_city(s: str) -> str:
    """Normalize city name."""
    return str(s).strip().title()


def add_zone_id(df: pd.DataFrame) -> pd.DataFrame:
    """Create a canonical zone_id from city/state columns."""
    if "city" in df.columns:
        df["zone_id"] = df["city"].apply(normalize_city)
    elif "state" in df.columns:
        df["zone_id"] = df["state"].apply(normalize_state)
    else:
        df["zone_id"] = "Unknown"
    return df


# ─── Station aggregation ──────────────────────────────────────────────────────

def aggregate_stations_by_zone(stations_df: pd.DataFrame) -> pd.DataFrame:
    """
    Aggregate station data to zone level.
    Returns: zone_id, station_count, total_capacity_kw, avg_chargers, lat, lon
    """
    if stations_df is None or stations_df.empty:
        return pd.DataFrame()

    df = stations_df.copy()
    df = add_zone_id(df)

    agg = {"station_count": ("zone_id", "count")}
    if "charger_capacity_kw" in df.columns:
        df["charger_capacity_kw"] = pd.to_numeric(df["charger_capacity_kw"], errors="coerce")
        agg["total_capacity_kw"] = ("charger_capacity_kw", "sum")
        agg["avg_capacity_kw"] = ("charger_capacity_kw", "mean")
    if "num_chargers" in df.columns:
        df["num_chargers"] = pd.to_numeric(df["num_chargers"], errors="coerce")
        agg["total_chargers"] = ("num_chargers", "sum")

    zone_stats = df.groupby("zone_id").agg(**{k: v for k, v in agg.items()}).reset_index()

    # Add representative centroid
    if "latitude" in df.columns and "longitude" in df.columns:
        coords = df.groupby("zone_id")[["latitude", "longitude"]].mean().reset_index()
        coords.columns = ["zone_id", "zone_lat", "zone_lon"]
        zone_stats = zone_stats.merge(coords, on="zone_id", how="left")

    logger.info("Aggregated stations: %d zones", len(zone_stats))
    return zone_stats


# ─── Adoption aggregation ─────────────────────────────────────────────────────

def aggregate_adoption_by_state(adoption_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate EV adoption data to state level (latest year)."""
    if adoption_df is None or adoption_df.empty:
        return pd.DataFrame()

    df = adoption_df.copy()
    if "state" in df.columns:
        df["state_norm"] = df["state"].apply(normalize_state)
    else:
        return pd.DataFrame()

    if "year" in df.columns:
        latest_year = df["year"].max()
        df = df[df["year"] == latest_year]

    if "ev_sales" in df.columns:
        state_stats = (
            df.groupby("state_norm")["ev_sales"]
            .sum()
            .reset_index()
            .rename(columns={"state_norm": "state", "ev_sales": "total_ev_sales"})
        )
        # Normalize to 0-1 for scoring
        max_sales = state_stats["total_ev_sales"].max()
        if max_sales > 0:
            state_stats["ev_adoption_score"] = state_stats["total_ev_sales"] / max_sales
        else:
            state_stats["ev_adoption_score"] = 0.0
        logger.info("Adoption: %d states aggregated.", len(state_stats))
        return state_stats

    return pd.DataFrame()


# ─── Main integration ─────────────────────────────────────────────────────────

INTEGRATION_NOTES = """
## Data Integration Strategy

### Primary Link
- Dataset A (demand) → zone-level demand via city/state column
- Dataset B (stations) → aggregated to zone level by city
- Dataset C (adoption) → aggregated to state level, joined via state name

### Assumption
If demand dataset has no lat/lon: the trained model is applied to zone-level
feature profiles built from B and C to generate zone-level predicted demand.
This is clearly labeled as 'model-generated demand' (is_synthetic=True).

### Join Key
city/state text match after normalization (title-case, alias mapping).
No direct spatial key exists between A, B, C — the join is administrative.
"""


def integrate_datasets(
    demand_df: Optional[pd.DataFrame],
    stations_zone_df: pd.DataFrame,
    adoption_state_df: pd.DataFrame,
) -> pd.DataFrame:
    """
    Build a zone-level feature table by joining station and adoption data.
    Used for GIS hotspot analysis and recommendation scoring.
    """
    if stations_zone_df.empty:
        logger.warning("No station zone data available for integration.")
        return pd.DataFrame()

    zone_df = stations_zone_df.copy()

    # Add state column for joining with adoption
    # Approximate: use zone_id as state if no separate state column
    if "state" not in zone_df.columns:
        zone_df["state"] = zone_df["zone_id"].apply(normalize_state)

    # Join adoption scores
    if not adoption_state_df.empty:
        zone_df = zone_df.merge(
            adoption_state_df[["state", "ev_adoption_score", "total_ev_sales"]],
            on="state", how="left"
        )
        zone_df["ev_adoption_score"] = zone_df["ev_adoption_score"].fillna(0.0)
        logger.info("Joined adoption scores.")
    else:
        zone_df["ev_adoption_score"] = 0.0
        zone_df["total_ev_sales"] = 0.0

    # Add demand from demand_df if available (zone-level actual demand)
    if demand_df is not None and not demand_df.empty and "energy_consumed_kwh" in demand_df.columns:
        if "zone_id" not in demand_df.columns:
            demand_df = add_zone_id(demand_df)
        zone_demand = (
            demand_df.groupby("zone_id")["energy_consumed_kwh"]
            .sum()
            .reset_index()
            .rename(columns={"energy_consumed_kwh": "historical_demand_kwh"})
        )
        zone_df = zone_df.merge(zone_demand, on="zone_id", how="left")
        zone_df["historical_demand_kwh"] = zone_df["historical_demand_kwh"].fillna(0.0)
        zone_df["is_synthetic_demand"] = False
    else:
        zone_df["historical_demand_kwh"] = 0.0
        zone_df["is_synthetic_demand"] = True

    logger.info("Integrated zone table: %d zones, %d cols", len(zone_df), len(zone_df.columns))
    return zone_df
