"""Interactive Folium map builder for hotspots, stations, and recommendations."""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import pandas as pd

from evplan.utils.config import load_config, get_path
from evplan.utils.logging import get_logger

logger = get_logger(__name__)

HOTSPOT_COLORS = {
    "Hot 99%": "#d73027",
    "Hot 95%": "#fc8d59",
    "Hot 90%": "#fee090",
    "Not Significant": "#91bfdb",
    "Cold 90%": "#74add1",
    "Cold 95%": "#4575b4",
    "Cold 99%": "#313695",
}


def build_map(
    zone_df: pd.DataFrame,
    stations_df: Optional[pd.DataFrame] = None,
    recommendations_df: Optional[pd.DataFrame] = None,
) -> object:
    """
    Build a multi-layer Folium map with:
    - Hotspot layer (colored circles by Gi* class)
    - Station layer (blue markers)
    - Recommendation layer (green stars)
    Returns a folium.Map object.
    """
    try:
        import folium
        from folium.plugins import MarkerCluster
    except ImportError:
        logger.error("folium not installed.")
        return None

    cfg = load_config()
    center = cfg["dashboard"]["map_center"]
    zoom = cfg["dashboard"]["map_zoom"]

    m = folium.Map(location=center, zoom_start=zoom, tiles="OpenStreetMap")

    # ── Hotspot layer ─────────────────────────────────────────
    hotspot_group = folium.FeatureGroup(name="🔥 Demand Hotspots", show=True)
    for _, row in zone_df.iterrows():
        if "zone_lat" not in row or pd.isna(row.get("zone_lat")):
            continue
        color = HOTSPOT_COLORS.get(str(row.get("hotspot_class", "Not Significant")), "#91bfdb")
        demand = row.get("predicted_demand_kwh", row.get("historical_demand_kwh", 0))
        zone = row.get("zone_id", "Unknown")
        z_score = row.get("gi_z_score", 0)
        ratio = row.get("demand_supply_ratio", 0)
        popup_html = f"""
        <b>{zone}</b><br>
        Demand: {demand:.1f} kWh<br>
        Hotspot: {row.get('hotspot_class','N/A')}<br>
        Gi* z-score: {z_score:.2f}<br>
        Demand/Supply Ratio: {ratio:.2f}<br>
        Stations: {int(row.get('station_count',0))}
        """
        folium.CircleMarker(
            location=[row["zone_lat"], row["zone_lon"]],
            radius=max(6, min(20, abs(z_score) * 3 + 5)),
            color=color, fill=True, fill_color=color, fill_opacity=0.7,
            popup=folium.Popup(popup_html, max_width=250),
            tooltip=f"{zone}: {row.get('hotspot_class','N/A')}",
        ).add_to(hotspot_group)
    hotspot_group.add_to(m)

    # ── Station layer ─────────────────────────────────────────
    if stations_df is not None and not stations_df.empty:
        station_group = folium.FeatureGroup(name="⚡ Existing Stations", show=True)
        cluster = MarkerCluster().add_to(station_group)
        for _, row in stations_df.iterrows():
            if "latitude" not in row or pd.isna(row.get("latitude")):
                continue
            city = row.get("city", row.get("zone_id", "Station"))
            ctype = row.get("charger_type", "N/A")
            cap = row.get("charger_capacity_kw", "N/A")
            popup_html = f"<b>{city}</b><br>Type: {ctype}<br>Capacity: {cap} kW"
            folium.Marker(
                location=[row["latitude"], row["longitude"]],
                popup=folium.Popup(popup_html, max_width=200),
                tooltip=str(city),
                icon=folium.Icon(color="blue", icon="plug", prefix="fa"),
            ).add_to(cluster)
        station_group.add_to(m)

    # ── Recommendation layer ─────────────────────────────────
    if recommendations_df is not None and not recommendations_df.empty:
        rec_group = folium.FeatureGroup(name="🏆 Recommended Locations", show=True)
        for rank, (_, row) in enumerate(recommendations_df.iterrows(), start=1):
            if "zone_lat" not in row or pd.isna(row.get("zone_lat")):
                continue
            zone = row.get("zone_id", "Zone")
            score = row.get("priority_score", 0)
            chargers = row.get("recommended_chargers", "N/A")
            popup_html = f"""
            <b>#{rank} {zone}</b><br>
            Priority Score: {score:.3f}<br>
            Recommended Chargers: {chargers}<br>
            Demand: {row.get('predicted_demand_kwh', 0):.1f} kWh
            """
            folium.Marker(
                location=[row["zone_lat"], row["zone_lon"]],
                popup=folium.Popup(popup_html, max_width=250),
                tooltip=f"#{rank} {zone}",
                icon=folium.Icon(color="green", icon="star", prefix="fa"),
            ).add_to(rec_group)
        rec_group.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


def save_map(m: object, filename: str = "ev_map.html") -> Optional[Path]:
    """Save folium map to outputs/maps/."""
    if m is None:
        return None
    maps_dir = get_path("maps")
    maps_dir.mkdir(parents=True, exist_ok=True)
    path = maps_dir / filename
    m.save(str(path))
    logger.info("Saved map: %s", path)
    return path
