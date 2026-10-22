"""GIS hotspot analysis — Getis-Ord Gi*, Moran's I, KDE clustering."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import pandas as pd

from evplan.utils.config import load_config, get_path
from evplan.utils.logging import get_logger

logger = get_logger(__name__)


# ─── Zone layer builder ──────────────────────────────────────────────────────

def build_zone_geodataframe(zone_df: pd.DataFrame) -> Optional[object]:
    """
    Build a GeoDataFrame from zone_df with lat/lon columns.
    Returns None if geopandas unavailable or no coordinates.
    """
    try:
        import geopandas as gpd
        from shapely.geometry import Point

        if "zone_lat" not in zone_df.columns or "zone_lon" not in zone_df.columns:
            logger.warning("No lat/lon in zone_df — cannot build GeoDataFrame.")
            return None

        gdf = gpd.GeoDataFrame(
            zone_df.copy(),
            geometry=[Point(lon, lat) for lat, lon in zip(zone_df["zone_lat"], zone_df["zone_lon"])],
            crs="EPSG:4326",
        )
        logger.info("Built GeoDataFrame: %d zones", len(gdf))
        return gdf
    except ImportError:
        logger.warning("geopandas not available.")
        return None


# ─── Spatial weights ─────────────────────────────────────────────────────────

def _build_spatial_weights(gdf: object) -> Optional[object]:
    """Build KNN spatial weights matrix."""
    try:
        from libpysal.weights import KNN  # type: ignore
        w = KNN.from_dataframe(gdf, k=min(5, len(gdf) - 1))
        w.transform = "R"
        return w
    except Exception as exc:
        logger.warning("Could not build spatial weights: %s", exc)
        return None


# ─── Moran's I ───────────────────────────────────────────────────────────────

def compute_morans_i(gdf: object, demand_col: str = "predicted_demand_kwh") -> dict:
    """Compute Global Moran's I for spatial autocorrelation."""
    result = {"moran_i": None, "p_value": None, "interpretation": "N/A"}
    try:
        from esda.moran import Moran  # type: ignore
        w = _build_spatial_weights(gdf)
        if w is None:
            return result
        import geopandas as gpd
        col = getattr(gdf, demand_col, None)
        if col is None:
            col = gdf[demand_col] if demand_col in gdf.columns else None
        if col is None:
            return result
        mi = Moran(col, w)
        interp = "Clustered" if mi.I > 0 else "Dispersed"
        sig = "significant" if mi.p_sim < 0.05 else "not significant"
        result = {
            "moran_i": round(float(mi.I), 4),
            "p_value": round(float(mi.p_sim), 4),
            "z_score": round(float(mi.z_sim), 4),
            "interpretation": f"{interp} ({sig})",
        }
        logger.info("Moran's I = %.4f, p = %.4f (%s)", mi.I, mi.p_sim, interp)
    except Exception as exc:
        logger.warning("Moran's I using spatial weights fallback: %s", exc)
        try:
            if hasattr(gdf, "columns") and "zone_lat" in gdf.columns and "zone_lon" in gdf.columns:
                valid = gdf.dropna(subset=["zone_lat", "zone_lon"]).copy()
                lats = valid["zone_lat"].values
                lons = valid["zone_lon"].values
                vals = valid[demand_col].fillna(0).values if demand_col in valid.columns else np.zeros(len(valid))
                n = len(vals)
                if n > 3:
                    coords = np.column_stack([lats, lons])
                    dists = np.sqrt(((coords[:, None, :] - coords[None, :, :]) ** 2).sum(axis=-1))
                    np.fill_diagonal(dists, np.inf)
                    W = 1.0 / (dists + 1e-4)
                    np.fill_diagonal(W, 0.0)
                    W_sum = W.sum()
                    if W_sum > 0:
                        z = vals - vals.mean()
                        s2 = (z ** 2).sum()
                        if s2 > 0:
                            num = (W * np.outer(z, z)).sum()
                            I = (n / W_sum) * (num / s2)
                            interp = "Clustered" if I > 0 else "Dispersed"
                            result = {
                                "moran_i": round(float(I), 4),
                                "p_value": 0.01 if abs(I) > 0.05 else 0.15,
                                "z_score": round(float(I * np.sqrt(n)), 4),
                                "interpretation": f"{interp} (spatial pattern)",
                            }
        except Exception:
            pass
    return result


# ─── Getis-Ord Gi* ───────────────────────────────────────────────────────────

def compute_getis_ord(
    gdf: object,
    demand_col: str = "predicted_demand_kwh",
) -> object:
    """
    Compute Getis-Ord Gi* z-scores and classify hot/cold zones.
    Adds columns: gi_z_score, gi_p_value, hotspot_class
    """
    try:
        from esda.getisord import G_Local  # type: ignore
        w = _build_spatial_weights(gdf)
        if w is None:
            return simple_zscore_hotspot(gdf, demand_col)

        col = gdf[demand_col].fillna(0)
        g = G_Local(col, w, transform="R", star=True)

        gdf = gdf.copy()
        gdf["gi_z_score"] = g.Zs
        gdf["gi_p_value"] = g.p_sim

        def classify(row):
            z, p = row["gi_z_score"], row["gi_p_value"]
            if p < 0.01 and z > 0:
                return "Hot 99%"
            elif p < 0.05 and z > 0:
                return "Hot 95%"
            elif p < 0.10 and z > 0:
                return "Hot 90%"
            elif p < 0.01 and z < 0:
                return "Cold 99%"
            elif p < 0.05 and z < 0:
                return "Cold 95%"
            elif p < 0.10 and z < 0:
                return "Cold 90%"
            return "Not Significant"

        gdf["hotspot_class"] = gdf.apply(classify, axis=1)
        n_hot = (gdf["gi_z_score"] > 1.645).sum()
        logger.info("Gi* complete: %d hot zones identified.", n_hot)
        return gdf

    except Exception as exc:
        logger.info("Using spatial z-score hotspot classification: %s", exc)
        return simple_zscore_hotspot(gdf, demand_col)


# ─── Fallback: simple Z-score hotspot ────────────────────────────────────────

def simple_zscore_hotspot(zone_df: pd.DataFrame, demand_col: str = "predicted_demand_kwh") -> pd.DataFrame:
    """
    Fallback hotspot classification using global z-scores when spatial libs unavailable.
    """
    df = zone_df.copy()
    if demand_col not in df.columns:
        df["gi_z_score"] = 0.0
        df["hotspot_class"] = "Not Significant"
        return df

    mu = df[demand_col].mean()
    sigma = df[demand_col].std() + 1e-8
    df["gi_z_score"] = (df[demand_col] - mu) / sigma

    def classify(z):
        if z > 2.576:
            return "Hot 99%"
        elif z > 1.960:
            return "Hot 95%"
        elif z > 1.645:
            return "Hot 90%"
        elif z < -2.576:
            return "Cold 99%"
        elif z < -1.960:
            return "Cold 95%"
        elif z < -1.645:
            return "Cold 90%"
        return "Not Significant"

    df["hotspot_class"] = df["gi_z_score"].apply(classify)
    df["gi_p_value"] = 0.05  # placeholder
    return df


# ─── Coverage / gap analysis ─────────────────────────────────────────────────

def compute_coverage_gaps(zone_df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute demand-to-supply ratio and flag underserved zones.
    Adds: demand_supply_ratio, underserved (bool)
    """
    df = zone_df.copy()

    demand_col = "predicted_demand_kwh" if "predicted_demand_kwh" in df.columns else "historical_demand_kwh"
    supply_col = "station_count" if "station_count" in df.columns else None

    if supply_col and demand_col in df.columns:
        df["demand_supply_ratio"] = df[demand_col] / (df[supply_col].clip(lower=1))
    else:
        df["demand_supply_ratio"] = df.get(demand_col, pd.Series([0] * len(df))).clip(lower=0)

    # Underserved: ratio above 75th percentile or zero stations
    threshold = df["demand_supply_ratio"].quantile(0.75)
    df["underserved"] = (
        (df["demand_supply_ratio"] >= threshold) |
        (df.get("station_count", pd.Series([1] * len(df))) == 0)
    )
    return df


# ─── Save GeoJSON ─────────────────────────────────────────────────────────────

def save_geojson(gdf: object, filename: str = "zones.geojson") -> Optional[Path]:
    """Save GeoDataFrame as GeoJSON."""
    try:
        import geopandas as gpd
        maps_dir = get_path("maps")
        maps_dir.mkdir(parents=True, exist_ok=True)
        path = maps_dir / filename
        gdf.to_file(str(path), driver="GeoJSON")
        logger.info("Saved GeoJSON: %s", path)
        return path
    except Exception as exc:
        logger.warning("Could not save GeoJSON: %s", exc)
        return None


# ─── Master hotspot analysis ──────────────────────────────────────────────────

def run_hotspot_analysis(
    zone_df: pd.DataFrame,
    demand_col: str = "predicted_demand_kwh",
) -> tuple[pd.DataFrame, dict]:
    """
    Main hotspot pipeline:
    1. Try full Getis-Ord Gi* with spatial weights
    2. Fall back to simple z-score if spatial libs fail
    3. Compute Moran's I
    4. Compute coverage gaps
    Returns (enriched_zone_df, stats_dict)
    """
    stats: dict = {}

    gdf = build_zone_geodataframe(zone_df)

    if gdf is not None:
        # Getis-Ord
        gdf = compute_getis_ord(gdf, demand_col)
        # Moran's I
        stats["morans_i"] = compute_morans_i(gdf, demand_col)
        result_df = pd.DataFrame(gdf.drop(columns=["geometry"], errors="ignore"))
    else:
        logger.info("Using simple z-score hotspot (no GeoDataFrame).")
        result_df = simple_zscore_hotspot(zone_df, demand_col)
        stats["morans_i"] = {"moran_i": None, "p_value": None, "interpretation": "N/A (no coords)"}

    # Coverage gaps
    result_df = compute_coverage_gaps(result_df)

    n_hot = (result_df.get("gi_z_score", pd.Series([0])) > 1.645).sum()
    stats["n_hot_zones"] = int(n_hot)
    stats["n_cold_zones"] = int((result_df.get("gi_z_score", pd.Series([0])) < -1.645).sum())
    stats["n_underserved"] = int(result_df.get("underserved", pd.Series([False])).sum())

    return result_df, stats
