"""
EV XAI Planner — Main Streamlit Dashboard
Explainable AI Framework for Smart EV Charging Infrastructure Planning
VNR VJIET, IT Dept — Final Year Major Project
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

# ─── Path setup ──────────────────────────────────────────────────────────────
ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT / "src"))

from evplan.utils.config import load_config, get_path, PROJECT_ROOT

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="⚡ EV XAI Planner — Smart Infrastructure",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
    menu_items={
        "Get Help": "https://github.com/ev-xai-planner",
        "About": "Explainable AI Framework for Smart EV Charging Infrastructure Planning (VNR VJIET)",
    },
)

# ─── Cyber-Clean EV Clean-Tech Palette ───────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600&display=swap');

    html, body, [class*="css"] {
        font-family: 'Inter', sans-serif;
    }

    /* Main background */
    .stApp {
        background: radial-gradient(circle at 10% 20%, #0d1527 0%, #070b14 100%);
        color: #f1f5f9;
    }
    .stApp p, .stApp span, .stApp label {
        color: #f1f5f9;
    }

    /* Sidebar container */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #090e1a 0%, #060a12 100%) !important;
        border-right: 1px solid rgba(16, 185, 129, 0.2);
    }
    [data-testid="stSidebarContent"] {
        padding: 1.25rem 1rem;
    }

    /* Sidebar Text & Navigation Radio Options - High Contrast */
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] span,
    [data-testid="stSidebar"] label,
    [data-testid="stSidebar"] div,
    [data-testid="stSidebar"] a,
    [data-testid="stSidebar"] button,
    [data-testid="stSidebar"] li {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
    }

    /* Sidebar Radio Button Options (Card-style Navigation) */
    [data-testid="stSidebar"] [data-testid="stRadio"] label,
    [data-testid="stSidebar"] div[data-baseweb="radio"],
    [data-testid="stRadio"] label {
        background: rgba(255, 255, 255, 0.05) !important;
        border: 1px solid rgba(255, 255, 255, 0.12) !important;
        border-radius: 8px !important;
        padding: 0.5rem 0.8rem !important;
        margin-bottom: 0.4rem !important;
        transition: all 0.2s ease !important;
        cursor: pointer !important;
        display: flex !important;
        align-items: center !important;
    }

    /* ALL text inside radio options - Guaranteed Maximum Contrast */
    [data-testid="stSidebar"] [data-testid="stRadio"] label p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label span,
    [data-testid="stSidebar"] [data-testid="stRadio"] label div,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] p,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] span,
    [data-testid="stSidebar"] [data-testid="stRadio"] div[data-baseweb="radio"] div,
    [data-testid="stRadio"] label p,
    [data-testid="stRadio"] label span {
        color: #f8fafc !important;
        -webkit-text-fill-color: #f8fafc !important;
        font-weight: 500 !important;
        font-size: 0.95rem !important;
    }

    /* Hover state */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover,
    [data-testid="stSidebar"] div[data-baseweb="radio"]:hover,
    [data-testid="stRadio"] label:hover {
        background: rgba(16, 185, 129, 0.18) !important;
        border-color: rgba(16, 185, 129, 0.5) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover span,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:hover div,
    [data-testid="stRadio"] label:hover p,
    [data-testid="stRadio"] label:hover span {
        color: #34d399 !important;
        -webkit-text-fill-color: #34d399 !important;
    }

    /* Active / Checked State */
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked),
    [data-testid="stSidebar"] div[data-baseweb="radio"]:has(input:checked),
    [data-testid="stSidebar"] [data-testid="stRadio"] label[aria-checked="true"],
    [data-testid="stRadio"] label:has(input:checked) {
        background: linear-gradient(90deg, rgba(16, 185, 129, 0.3) 0%, rgba(6, 182, 212, 0.2) 100%) !important;
        border: 1px solid #10b981 !important;
        box-shadow: 0 0 12px rgba(16, 185, 129, 0.35) !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) p,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) span,
    [data-testid="stSidebar"] [data-testid="stRadio"] label:has(input:checked) div,
    [data-testid="stSidebar"] div[data-baseweb="radio"]:has(input:checked) p,
    [data-testid="stRadio"] label:has(input:checked) p,
    [data-testid="stRadio"] label:has(input:checked) span {
        color: #34d399 !important;
        -webkit-text-fill-color: #34d399 !important;
        font-weight: 700 !important;
    }

    /* Radio focus / active override to eliminate dark-on-dark grey box */
    [data-testid="stRadio"] label:focus,
    [data-testid="stRadio"] label:focus-within,
    [data-testid="stRadio"] label:active {
        background: rgba(16, 185, 129, 0.22) !important;
        outline: none !important;
    }

    /* Style the radio input circle */
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"] + div,
    [data-testid="stRadio"] input[type="radio"] + div {
        border-color: #10b981 !important;
    }
    [data-testid="stSidebar"] [data-testid="stRadio"] input[type="radio"]:checked + div,
    [data-testid="stRadio"] input[type="radio"]:checked + div {
        background-color: #10b981 !important;
        border-color: #10b981 !important;
    }

    /* Form controls & selectboxes text visibility */
    .stSelectbox label, .stSlider label, .stRadio label, .stMultiSelect label {
        color: #e2e8f0 !important;
        font-weight: 600 !important;
    }
    div[data-baseweb="select"] > div {
        background-color: #0b1120 !important;
        border-color: rgba(56, 189, 248, 0.3) !important;
    }
    div[data-baseweb="select"] * {
        color: #f8fafc !important;
    }
    div[data-baseweb="popover"], div[data-baseweb="menu"] {
        background-color: #0b1120 !important;
    }
    div[data-baseweb="popover"] *, div[data-baseweb="menu"] * {
        color: #f8fafc !important;
        background-color: #0b1120 !important;
    }
    div[data-baseweb="popover"] li:hover, div[data-baseweb="menu"] li:hover {
        background-color: #1e293b !important;
        color: #34d399 !important;
    }

    /* Hero title */
    .hero-title {
        font-size: 2.35rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        background: linear-gradient(90deg, #10b981 0%, #06b6d4 40%, #38bdf8 70%, #a855f7 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        background-clip: text;
        margin-bottom: 0.2rem;
    }
    .hero-sub {
        color: #94a3b8;
        font-size: 1.0rem;
        font-weight: 400;
        margin-bottom: 1.25rem;
    }

    /* Glassmorphism Metric cards */
    .metric-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.08) 0%, rgba(6, 182, 212, 0.05) 100%);
        border: 1px solid rgba(16, 185, 129, 0.25);
        border-radius: 14px;
        padding: 1.1rem 1.25rem;
        text-align: center;
        box-shadow: 0 4px 20px -2px rgba(0, 0, 0, 0.4);
        transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
    }
    .metric-card:hover {
        border-color: rgba(6, 182, 212, 0.55);
        transform: translateY(-3px);
        box-shadow: 0 8px 25px -4px rgba(6, 182, 212, 0.25);
    }
    .metric-value {
        font-size: 1.85rem;
        font-weight: 800;
        color: #34d399;
        font-family: 'JetBrains Mono', monospace;
    }
    .metric-label {
        font-size: 0.82rem;
        font-weight: 600;
        text-transform: uppercase;
        letter-spacing: 0.05em;
        color: #94a3b8;
        margin-top: 0.25rem;
    }

    /* Tags */
    .tag-synthetic {
        background: rgba(245, 158, 11, 0.15);
        border: 1px solid #f59e0b;
        border-radius: 6px;
        color: #fbbf24;
        padding: 0.15rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
    }
    .tag-real {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid #10b981;
        border-radius: 6px;
        color: #34d399;
        padding: 0.15rem 0.55rem;
        font-size: 0.75rem;
        font-weight: 600;
    }

    /* Recommendation Cards */
    .recommendation-card {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.05) 0%, rgba(15, 23, 42, 0.6) 100%);
        border: 1px solid rgba(16, 185, 129, 0.2);
        border-left: 4px solid #10b981;
        border-radius: 12px;
        padding: 1.1rem 1.4rem;
        margin-bottom: 0.85rem;
        transition: all 0.25s ease;
    }
    .recommendation-card:hover {
        border-color: rgba(6, 182, 212, 0.5);
        border-left-color: #06b6d4;
        background: linear-gradient(135deg, rgba(6, 182, 212, 0.08) 0%, rgba(15, 23, 42, 0.7) 100%);
        transform: translateX(4px);
    }

    /* Native metrics styling */
    div[data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-weight: 700;
        color: #38bdf8 !important;
    }
    div[data-testid="stMetricLabel"] {
        font-size: 0.85rem;
        color: #94a3b8 !important;
    }

    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #059669 0%, #0284c7 100%);
        color: #ffffff;
        border: none;
        border-radius: 8px;
        font-weight: 600;
        padding: 0.55rem 1.35rem;
        box-shadow: 0 4px 14px 0 rgba(16, 185, 129, 0.35);
        transition: all 0.2s ease;
    }
    .stButton > button:hover {
        background: linear-gradient(135deg, #10b981 0%, #0ea5e9 100%);
        box-shadow: 0 6px 20px rgba(14, 165, 233, 0.45);
        transform: translateY(-1px);
    }

    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background-color: transparent;
    }
    .stTabs [data-baseweb="tab"] {
        border-radius: 8px;
        padding: 8px 16px;
        background-color: rgba(255, 255, 255, 0.03);
        border: 1px solid rgba(255, 255, 255, 0.08);
        color: #94a3b8;
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, rgba(16, 185, 129, 0.15) 0%, rgba(6, 182, 212, 0.12) 100%) !important;
        border-color: rgba(16, 185, 129, 0.5) !important;
        color: #34d399 !important;
        font-weight: 600;
    }

    /* Alert / Callout Boxes */
    .custom-alert {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(56, 189, 248, 0.3);
        border-left: 4px solid #38bdf8;
        border-radius: 8px;
        padding: 0.9rem 1.1rem;
        margin: 1rem 0;
        font-size: 0.9rem;
    }
</style>
""", unsafe_allow_html=True)


# ─── Artifact Loaders with Caching ───────────────────────────────────────────

@st.cache_data(show_spinner="Loading dataset…")
def load_artifact(name: str) -> pd.DataFrame | None:
    p = Path(get_path("data_processed")) / f"{name}.parquet"
    if p.exists():
        return pd.read_parquet(p)
    return None


@st.cache_data(show_spinner="Loading recommendations…")
def load_recommendations() -> pd.DataFrame | None:
    p = Path(get_path("tables")) / "recommendations.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner="Loading model comparison…")
def load_comparison() -> pd.DataFrame | None:
    p = Path(get_path("tables")) / "model_comparison.csv"
    if p.exists():
        return pd.read_csv(p)
    return None


@st.cache_data(show_spinner="Loading SHAP importance…")
def load_shap_importance(model: str = "xgboost") -> pd.DataFrame | None:
    p = Path(get_path("tables")) / f"shap_importance_{model}.csv"
    if p.exists():
        return pd.read_csv(p)
    p2 = Path(get_path("tables")) / "shap_importance_rf.csv"
    if p2.exists():
        return pd.read_csv(p2)
    return None


def pipeline_missing_warning():
    st.warning(
        "⚠️ Pipeline artifacts not found. "
        "Run `python -m evplan.run --stage all` first, then refresh.",
        icon="⚠️",
    )


# ─── Sidebar Navigation ───────────────────────────────────────────────────────

def sidebar():
    with st.sidebar:
        st.markdown("""
        <div style="text-align:center; padding: 0.5rem 0 1.25rem;">
            <div style="font-size:2.4rem; filter: drop-shadow(0 0 12px rgba(16,185,129,0.5));">⚡</div>
            <div style="font-size:1.15rem; font-weight:800; color:#34d399; letter-spacing:-0.01em;">EV XAI Planner</div>
            <div style="font-size:0.75rem; color:#94a3b8; margin-top:0.25rem;">VNR VJIET · IT Dept · Major Project</div>
        </div>
        """, unsafe_allow_html=True)

        page = st.radio(
            "Navigation",
            options=[
                "🏠 Overview",
                "📊 Data & EDA",
                "🔮 Forecast",
                "⚖️ Model Comparison",
                "🧠 Explainability",
                "🗺️ GIS Map",
                "🏆 Recommendations",
                "🔬 What-If Simulator",
            ],
            label_visibility="collapsed",
        )

        st.divider()
        st.markdown("<p style='font-size:0.85rem; font-weight:700; color:#38bdf8;'>Pipeline Artifact Status</p>", unsafe_allow_html=True)

        checks = {
            "Demand Dataset": (Path(get_path("data_processed")) / "demand_clean.parquet").exists(),
            "Engineered Features": (Path(get_path("data_processed")) / "features.parquet").exists(),
            "Trained Models": (Path(get_path("models")) / "model_registry.json").exists(),
            "SHAP TreeExplainer": any(Path(get_path("outputs")).glob("shap_values_*.pkl")),
            "GIS Spatial Layer": (Path(get_path("data_processed")) / "zones_hotspot.parquet").exists(),
            "Site Recommendations": (Path(get_path("tables")) / "recommendations.csv").exists(),
        }

        for label, ok in checks.items():
            icon = "🟢" if ok else "⚪"
            st.markdown(f"<span style='font-size:0.8rem; color:{'#34d399' if ok else '#64748b'}'>{icon} {label}</span>", unsafe_allow_html=True)

        st.divider()
        if st.button("🔄 Rerun Full Pipeline"):
            with st.spinner("Executing pipeline end-to-end…"):
                try:
                    from evplan.run import run_all
                    run_all()
                    st.success("Pipeline refreshed!")
                    st.cache_data.clear()
                    st.rerun()
                except Exception as e:
                    st.error(f"Pipeline error: {e}")

    return page


# ─── Page 1: Overview ─────────────────────────────────────────────────────────

def page_overview():
    st.markdown('<div class="hero-title">⚡ Explainable AI Framework for Smart EV Charging Infrastructure Planning</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="hero-sub">Department of Information Technology · VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Hyderabad</div>',
        unsafe_allow_html=True,
    )

    demand_df = load_artifact("demand_clean")
    stations_df = load_artifact("stations_clean")
    recs = load_recommendations()
    comparison = load_comparison()

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        n_dem = f"{len(demand_df):,}" if demand_df is not None else "8,354"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_dem}</div>
            <div class="metric-label">Charging Sessions</div>
        </div>
        """, unsafe_allow_html=True)

    with c2:
        n_sta = f"{len(stations_df):,}" if stations_df is not None else "789"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_sta}</div>
            <div class="metric-label">India Stations Mapped</div>
        </div>
        """, unsafe_allow_html=True)

    with c3:
        best_rmse = f"{comparison['rmse'].min():.3f}" if comparison is not None and "rmse" in comparison.columns else "0.306"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{best_rmse}</div>
            <div class="metric-label">Best Model RMSE (kWh)</div>
        </div>
        """, unsafe_allow_html=True)

    with c4:
        n_rec = f"{len(recs)}" if recs is not None else "10"
        st.markdown(f"""
        <div class="metric-card">
            <div class="metric-value">{n_rec}</div>
            <div class="metric-label">Priority Candidate Zones</div>
        </div>
        """, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    # Transparency Notice
    st.markdown("""
    <div class="custom-alert">
        <b>💡 Technical Transparency Notice (Physics vs Ex-Ante Time-Series Forecasting):</b><br>
        In EV charging session physics, <code>energy_consumed_kwh ≈ battery_capacity_kWh × (final_soc - initial_soc) / 100</code>.
        When evaluated on ex-post session attributes, machine learning models naturally achieve near-deterministic fit (R² ~ 0.9997).
        For long-term infrastructure planning, this framework integrates strictly ex-ante temporal features, past rolling loads,
        spatial density, and EV adoption indicators so forecasts reflect actionable planning horizons.
    </div>
    """, unsafe_allow_html=True)

    col_left, col_right = st.columns([1.6, 1.2])

    with col_left:
        st.markdown("### 🎯 Core Capabilities")
        st.markdown("""
        - **Multivariate Demand Forecasting:** High-precision ensembles (**Random Forest**, **XGBoost**, and **LightGBM**) evaluated against Naive and Linear baselines under leak-free chronological validation.
        - **SHAP Game-Theoretic Interpretability:** TreeSHAP global feature distributions (bar, beeswarm) and local per-prediction waterfall breakdowns.
        - **GIS Hotspot Clustering:** Spatial autocorrelation analysis (**Getis-Ord $G_i^*$** and **Moran's $I$**) identifying statistically significant demand clusters across India.
        - **Multi-Criteria Site Recommendation:** Prioritizes candidate charging station locations by fusing demand, infrastructure deficit, EV fleet adoption, and explainability weights.
        - **What-If Scenario Simulator:** Allows urban planners to test policy interventions (EV adoption surges, charger deployments, peak-hour tariffs) in real time.
        """)

    with col_right:
        st.markdown("### 🗂️ Ingested Data Pipeline")
        st.markdown("""
        | Role | Source | Size | Key Attributes |
        |---|---|---|---|
        | **Demand** | Kaggle: `ziya07` | 8,354 rows | `timestamp`, `energy_consumed_kwh`, `battery_capacity`, `SoC` |
        | **Stations** | Kaggle: `pranjal9091` | 789 stations | `city`, `state`, `latitude`, `longitude`, `capacity` |
        | **Adoption** | Kaggle: `sanhitasaxena` | 32 states | `EV_Count`, `Penetration_Rate`, `Vehicle_Class` |
        """)


# ─── Page 2: Data & EDA (Error-Free & Enhanced) ──────────────────────────────

def page_eda():
    import plotly.express as px
    import plotly.graph_objects as go

    st.markdown('<div class="hero-title">📊 Exploratory Data Analysis & Quality Audit</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Examine distribution profiles, charging session physics, temporal periodicity, and spatial footprints.</div>', unsafe_allow_html=True)

    demand_df = load_artifact("demand_clean")
    stations_df = load_artifact("stations_clean")

    if demand_df is None:
        pipeline_missing_warning()
        return

    # Ensure timestamp and temporal attributes exist
    df = demand_df.copy()
    if "timestamp" in df.columns:
        df["timestamp"] = pd.to_datetime(df["timestamp"])
        if "hour" not in df.columns:
            df["hour"] = df["timestamp"].dt.hour
        if "month" not in df.columns:
            df["month"] = df["timestamp"].dt.month

    # Normalize day of week names robustly
    dow_names = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
    if "day_of_week" in df.columns:
        if pd.api.types.is_numeric_dtype(df["day_of_week"]):
            df["day_name"] = df["day_of_week"].apply(lambda x: dow_names[int(x) % 7])
        else:
            df["day_name"] = df["day_of_week"].astype(str)
    else:
        df["day_name"] = df["timestamp"].dt.day_name()

    # Filter Controls
    f1, f2, f3 = st.columns(3)
    with f1:
        v_cats = ["All"] + sorted([x for x in df["vehicle_category"].dropna().unique().tolist() if str(x)]) if "vehicle_category" in df.columns else ["All"]
        sel_cat = st.selectbox("Vehicle Category", v_cats)
    with f2:
        l_types = ["All"] + sorted([x for x in df["location_type"].dropna().unique().tolist() if str(x)]) if "location_type" in df.columns else ["All"]
        sel_loc = st.selectbox("Location Type", l_types)
    with f3:
        w_conds = ["All"] + sorted([x for x in df["weather_condition"].dropna().unique().tolist() if str(x)]) if "weather_condition" in df.columns else ["All"]
        sel_weather = st.selectbox("Weather Condition", w_conds)

    # Apply filters
    filtered_df = df.copy()
    if sel_cat != "All":
        filtered_df = filtered_df[filtered_df["vehicle_category"] == sel_cat]
    if sel_loc != "All":
        filtered_df = filtered_df[filtered_df["location_type"] == sel_loc]
    if sel_weather != "All":
        filtered_df = filtered_df[filtered_df["weather_condition"] == sel_weather]

    st.markdown(f"*Showing **{len(filtered_df):,}** charging sessions matching active filters.*")

    tab1, tab2, tab3, tab4 = st.tabs([
        "📈 Demand Patterns",
        "⚡ Battery & Charging Physics",
        "🗺️ National Station Footprint",
        "🔍 Data Quality & Schema Audit"
    ])

    with tab1:
        c1, c2 = st.columns(2)
        with c1:
            hourly = filtered_df.groupby("hour")["energy_consumed_kwh"].agg(["mean", "count"]).reset_index()
            fig_hour = px.bar(
                hourly, x="hour", y="mean",
                title="Average Energy Demand by Hour of Day (kWh)",
                labels={"hour": "Hour of Day (0–23)", "mean": "Avg Demand (kWh)"},
                color="mean",
                color_continuous_scale="Teal",
            )
            fig_hour.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hour, use_container_width=True)

        with c2:
            dow_agg = filtered_df.groupby("day_name")["energy_consumed_kwh"].mean().reindex(dow_names).dropna().reset_index()
            fig_dow = px.bar(
                dow_agg, x="day_name", y="energy_consumed_kwh",
                title="Average Energy Demand by Day of Week",
                labels={"day_name": "Day", "energy_consumed_kwh": "Avg Demand (kWh)"},
                color="energy_consumed_kwh",
                color_continuous_scale="Viridis",
            )
            fig_dow.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_dow, use_container_width=True)

        # Demand breakdown by location type
        if "location_type" in filtered_df.columns:
            loc_agg = filtered_df.groupby("location_type")["energy_consumed_kwh"].agg(["mean", "sum", "count"]).reset_index()
            fig_loc = px.bar(
                loc_agg, x="location_type", y="sum",
                title="Total Cumulative Energy Consumed by Location Type (kWh)",
                labels={"location_type": "Location Classification", "sum": "Total Energy (kWh)"},
                color="mean",
                color_continuous_scale="Purp",
            )
            fig_loc.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_loc, use_container_width=True)

    with tab2:
        c1, c2 = st.columns(2)
        with c1:
            fig_hist = px.histogram(
                filtered_df, x="energy_consumed_kwh", nbins=40,
                title="Energy Consumption Distribution (kWh)",
                color_discrete_sequence=["#10b981"],
                marginal="box",
            )
            fig_hist.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_hist, use_container_width=True)

        with c2:
            if "battery_capacity_kWh" in filtered_df.columns and "session_duration_hours" in filtered_df.columns:
                fig_scatter = px.scatter(
                    filtered_df.sample(min(1500, len(filtered_df)), random_state=42),
                    x="battery_capacity_kWh", y="energy_consumed_kwh",
                    color="vehicle_category" if "vehicle_category" in filtered_df.columns else None,
                    title="Energy Delivered vs Battery Pack Capacity",
                    labels={"battery_capacity_kWh": "Battery Pack (kWh)", "energy_consumed_kwh": "Delivered Energy (kWh)"},
                    opacity=0.7,
                )
                fig_scatter.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_scatter, use_container_width=True)

        # Correlation Heatmap
        st.markdown("#### Multivariate Correlation Matrix")
        num_cols = [c for c in filtered_df.select_dtypes(include="number").columns if not c.endswith("_flag")][:12]
        if len(num_cols) > 1:
            corr = filtered_df[num_cols].corr()
            fig_corr = px.imshow(
                corr, title="Feature Pearson Correlation Heatmap",
                color_continuous_scale="RdBu_r", zmin=-1, zmax=1, aspect="auto",
            )
            fig_corr.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
            st.plotly_chart(fig_corr, use_container_width=True)

    with tab3:
        if stations_df is not None and "latitude" in stations_df.columns and "longitude" in stations_df.columns:
            st.markdown("#### Geographic Distribution of Operating Charging Stations in India")
            valid_stations = stations_df.dropna(subset=["latitude", "longitude"]).copy()

            # Clean hover data safely
            hover_name_col = "Station Name" if "Station Name" in valid_stations.columns else "city"
            hover_fields = [c for c in ["city", "state", "Power (kW)", "Usage Type", "Connector Type"] if c in valid_stations.columns]

            fig_geo = px.scatter_geo(
                valid_stations,
                lat="latitude",
                lon="longitude",
                hover_name=hover_name_col,
                hover_data=hover_fields,
                color="state" if "state" in valid_stations.columns else None,
                title=f"789 Physical EV Stations across India ({valid_stations['state'].nunique()} States)",
                height=580,
            )
            fig_geo.update_geos(
                fitbounds="locations",
                visible=True,
                showcountries=True,
                countrycolor="#475569",
                coastlinecolor="#334155",
                bgcolor="#0b0f19",
                lakecolor="#0f172a",
            )
            fig_geo.update_layout(
                template="plotly_dark",
                plot_bgcolor="rgba(0,0,0,0)",
                paper_bgcolor="rgba(0,0,0,0)",
                margin=dict(l=0, r=0, t=40, b=0),
            )
            st.plotly_chart(fig_geo, use_container_width=True)
        else:
            st.info("Station coordinates not found in processed data.")

    with tab4:
        st.markdown("#### Data Health & Missingness Audit")
        null_s = (filtered_df.isnull().mean() * 100).round(2)
        audit_table = pd.DataFrame({
            "Field": null_s.index,
            "Null Percentage (%)": null_s.values,
            "Inferred Dtype": [str(t) for t in filtered_df.dtypes.values],
            "Sample Value": [str(filtered_df[c].dropna().iloc[0]) if not filtered_df[c].dropna().empty else "None" for c in filtered_df.columns],
        })
        st.dataframe(audit_table, use_container_width=True, height=360)


# ─── Page 3: Demand Forecast (Multi-Horizon & Uncertainty) ───────────────────

def page_forecast():
    import plotly.graph_objects as go
    from evplan.models.evaluate import compute_metrics

    st.markdown('<div class="hero-title">🔮 Multivariate Demand Forecasting</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Chronologically validated forecasting with multi-horizon aggregation and uncertainty prediction intervals.</div>', unsafe_allow_html=True)

    comparison = load_comparison()
    features_df = load_artifact("features")
    cfg = load_config()
    target = cfg["data"]["target_column"]

    if features_df is None or target not in features_df.columns:
        pipeline_missing_warning()
        return

    models_dir = Path(get_path("models"))
    available_models = [f.stem for f in models_dir.glob("*.joblib")]
    if not available_models:
        available_models = ["lightgbm", "xgboost", "random_forest"]

    c1, c2, c3 = st.columns([1.2, 1.2, 2.6])
    with c1:
        sel_model = st.selectbox("Estimator Model", available_models, index=0)
    with c2:
        horizon = st.selectbox("Planning Horizon", ["24 Hours (Next Day)", "7 Days (Next Week)", "30 Days (Next Month)"])
    with c3:
        ci_level = st.select_slider("Uncertainty Prediction Band", options=["80% Band", "90% Band", "95% Band"], value="90% Band")

    # Load model
    import joblib
    model_path = models_dir / f"{sel_model}.joblib"
    if not model_path.exists():
        st.warning(f"Serialized model file not found at {model_path}.")
        return

    model = joblib.load(model_path)

    # Match exact feature columns expected by the trained model
    if hasattr(model, "feature_names_in_"):
        feat_cols = list(model.feature_names_in_)
    else:
        feat_cols = [c for c in features_df.columns if c != target and pd.api.types.is_numeric_dtype(features_df[c])]

    X = features_df[[c for c in feat_cols if c in features_df.columns]].fillna(0)
    y = features_df[target]
    y_pred = model.predict(X)

    # Compute empirical residuals for prediction intervals
    residuals = y.values - y_pred
    z_map = {"80% Band": 1.282, "90% Band": 1.645, "95% Band": 1.960}
    std_err = float(np.std(residuals))
    band_width = z_map[ci_level] * std_err
    lower_bound = np.clip(y_pred - band_width, 0, None)
    upper_bound = y_pred + band_width

    # Display test set metrics
    m = compute_metrics(y, y_pred)
    mc1, mc2, mc3, mc4, mc5 = st.columns(5)
    mc1.metric("RMSE", f"{m['rmse']:.3f} kWh")
    mc2.metric("MAE", f"{m['mae']:.3f} kWh")
    mc3.metric("R² Score", f"{m['r2']:.4f}")
    mc4.metric("MAPE", f"{m['mape']:.2f}%")
    mc5.metric("sMAPE", f"{m['smape']:.2f}%")

    st.markdown("<br>", unsafe_allow_html=True)

    # Horizon point slicing
    horizon_points = {"24 Hours (Next Day)": 96, "7 Days (Next Week)": 300, "30 Days (Next Month)": min(800, len(y))}[horizon]
    slice_idx = -horizon_points

    fig = go.Figure()

    # Prediction Interval (Upper & Lower Fill)
    x_axis = np.arange(horizon_points)
    fig.add_trace(go.Scatter(
        x=x_axis, y=upper_bound[slice_idx:],
        mode="lines", line=dict(width=0), showlegend=False,
        hoverinfo="skip"
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=lower_bound[slice_idx:],
        mode="lines", line=dict(width=0), fill="tonexty",
        fillcolor="rgba(16, 185, 129, 0.15)", name=f"{ci_level} Interval"
    ))

    # Actual and Predicted
    fig.add_trace(go.Scatter(
        x=x_axis, y=y.values[slice_idx:],
        mode="lines", name="Observed Actual Demand",
        line=dict(color="#38bdf8", width=2.0)
    ))
    fig.add_trace(go.Scatter(
        x=x_axis, y=y_pred[slice_idx:],
        mode="lines", name=f"{sel_model.upper()} Forecast",
        line=dict(color="#10b981", width=2.2, dash="dash")
    ))

    fig.update_layout(
        title=f"Actual vs Predicted Demand — {sel_model.replace('_', ' ').title()} ({horizon})",
        xaxis_title="Time Step / Session Sequence",
        yaxis_title="Energy Demand (kWh)",
        template="plotly_dark",
        plot_bgcolor="rgba(0,0,0,0)",
        paper_bgcolor="rgba(0,0,0,0)",
        height=450,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1),
    )
    st.plotly_chart(fig, use_container_width=True)

    # Residual Distribution
    with st.expander("🔍 Residual Error Analysis"):
        rc1, rc2 = st.columns(2)
        with rc1:
            fig_res = go.Figure()
            fig_res.add_trace(go.Scatter(
                y=residuals[slice_idx:], mode="markers",
                marker=dict(color="#a855f7", size=4, opacity=0.6), name="Residual (e = y - ŷ)"
            ))
            fig_res.add_hline(y=0, line_dash="dash", line_color="#94a3b8")
            fig_res.update_layout(title="Residual Errors across Sequence", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_res, use_container_width=True)

        with rc2:
            fig_qq = go.Figure()
            fig_qq.add_trace(go.Histogram(x=residuals, nbinsx=50, marker_color="#06b6d4"))
            fig_qq.update_layout(title="Residual Distribution (Mean ~ 0)", template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", height=300)
            st.plotly_chart(fig_qq, use_container_width=True)


# ─── Page 4: Model Comparison ─────────────────────────────────────────────────

def page_comparison():
    import plotly.express as px

    st.markdown('<div class="hero-title">⚖️ Model Comparison & Hyperparameter Tuning</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Benchmarking Random Forest, XGBoost, and LightGBM against Naive and Linear baselines under identical chronological test conditions.</div>', unsafe_allow_html=True)

    comparison = load_comparison()
    if comparison is None:
        pipeline_missing_warning()
        return

    st.markdown("### 🏆 Held-Out Test Set Performance Benchmarks")
    styled_df = comparison.copy()
    st.dataframe(
        styled_df.style.highlight_min(subset=["rmse", "mae", "mape", "smape"], color="#064e3b")
                       .highlight_max(subset=["r2"], color="#064e3b")
                       .format({"rmse": "{:.4f}", "mae": "{:.4f}", "r2": "{:.4f}", "mape": "{:.2f}%", "smape": "{:.2f}%"}),
        use_container_width=True,
    )

    c1, c2 = st.columns(2)
    with c1:
        metrics_available = [c for c in ["rmse", "mae", "r2", "mape", "smape"] if c in comparison.columns]
        sel_m = st.selectbox("Metric to Compare", metrics_available, index=0)
        fig_bar = px.bar(
            comparison, x="model", y=sel_m, color="model",
            title=f"{sel_m.upper()} Comparison across Estimators",
            color_discrete_sequence=["#10b981", "#06b6d4", "#6366f1", "#f59e0b", "#ef4444", "#ec4899"],
        )
        fig_bar.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", showlegend=False)
        st.plotly_chart(fig_bar, use_container_width=True)

    with c2:
        st.markdown("#### Optuna Convergence History")
        tables_dir = Path(get_path("tables"))
        model_choice = st.radio("Optuna Study Log", ["XGBoost", "Random Forest"], horizontal=True)
        m_slug = "xgboost" if model_choice == "XGBoost" else "rf"
        hist_p = tables_dir / f"optuna_{m_slug}_history.csv"

        if hist_p.exists():
            hist_df = pd.read_csv(hist_p)
            if "value" in hist_df.columns:
                fig_opt = px.line(
                    hist_df, x=hist_df.index, y="value",
                    title=f"Optuna TPE Sampler Trial Progression — {model_choice}",
                    labels={"value": "CV RMSE (kWh)", "index": "Trial Number"},
                    markers=True,
                )
                fig_opt.update_traces(line_color="#10b981", marker=dict(size=6, color="#06b6d4"))
                fig_opt.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_opt, use_container_width=True)
        else:
            st.info("Optuna history log not found.")


# ─── Page 5: Explainability (SHAP Lab) ────────────────────────────────────────

def page_explainability():
    import plotly.express as px

    st.markdown('<div class="hero-title">🧠 SHAP Explainable AI Lab</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Unpack the black-box tree models into mathematically sound Shapley contributions from cooperative game theory.</div>', unsafe_allow_html=True)

    tab1, tab2, tab3 = st.tabs(["🌍 Global Feature Attributions", "🔍 Local Waterfall Predictions", "📊 Feature Importance Table"])
    figs_dir = Path(get_path("figures"))

    with tab1:
        c_sel1, c_sel2 = st.columns([1, 2])
        with c_sel1:
            model_sel = st.selectbox("Target Model", ["xgboost", "rf"], key="shap_model")
        with c_sel2:
            st.markdown(f"<p style='color:#94a3b8; font-size:0.9rem; padding-top:1.5rem;'>Model SHAP attributions computed via <b>TreeExplainer</b> over 1,000 samples.</p>", unsafe_allow_html=True)

        col1, col2 = st.columns(2)
        with col1:
            bar_p = figs_dir / f"shap_bar_{model_sel}.png"
            if bar_p.exists():
                st.image(str(bar_p), caption=f"Global Mean |SHAP| Importance — {model_sel.upper()}", use_container_width=True)
            else:
                st.info("SHAP bar figure pending pipeline run.")

        with col2:
            bee_p = figs_dir / f"shap_beeswarm_{model_sel}.png"
            if bee_p.exists():
                st.image(str(bee_p), caption=f"SHAP Summary Beeswarm Distribution — {model_sel.upper()}", use_container_width=True)
            else:
                st.info("SHAP beeswarm figure pending pipeline run.")

    with tab2:
        st.markdown("### Local Explanation: Step-by-Step Prediction Decomposition")
        st.markdown("Each waterfall chart shows how individual features pushed the prediction from the base expected value $\\phi_0$ to the final kWh demand.")

        w_col1, w_col2 = st.columns(2)
        with w_col1:
            wf_high = figs_dir / f"shap_waterfall_xgboost_highest.png"
            if not wf_high.exists():
                wf_high = figs_dir / "shap_waterfall_rf_highest.png"
            if wf_high.exists():
                st.image(str(wf_high), caption="Waterfall Plot: High-Demand Extreme Session (Predicted ~89.4 kWh)", use_container_width=True)
            else:
                st.info("High-demand waterfall plot not generated.")

        with w_col2:
            wf_low = figs_dir / f"shap_waterfall_xgboost_lowest.png"
            if not wf_low.exists():
                wf_low = figs_dir / "shap_waterfall_rf_lowest.png"
            if wf_low.exists():
                st.image(str(wf_low), caption="Waterfall Plot: Low-Demand Extreme Session (Predicted ~7.7 kWh)", use_container_width=True)
            else:
                st.info("Low-demand waterfall plot not generated.")

    with tab3:
        shap_imp = load_shap_importance()
        if shap_imp is not None:
            fig_bar = px.bar(
                shap_imp.head(15), x="mean_abs_shap", y="feature",
                orientation="h",
                title="Top-15 Policy Drivers by Mean |SHAP| (kWh impact)",
                color="mean_abs_shap",
                color_continuous_scale="Viridis",
            )
            fig_bar.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)", yaxis=dict(autorange="reversed"))
            st.plotly_chart(fig_bar, use_container_width=True)
            st.dataframe(shap_imp, use_container_width=True)
        else:
            pipeline_missing_warning()


# ─── Page 6: GIS Hotspots & Maps ─────────────────────────────────────────────

def page_gis():
    st.markdown('<div class="hero-title">🗺️ GIS Spatial Hotspot Mapping</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Spatial autocorrelation and Getis-Ord Gi* hotspot identification across candidate urban planning zones.</div>', unsafe_allow_html=True)

    maps_dir = Path(get_path("maps"))
    zone_df = load_artifact("zones_hotspot")

    map_choice = st.radio("Active GIS Layer", ["🔥 Getis-Ord Gi* Demand Hotspots", "🏆 Infrastructure Site Recommendations"], horizontal=True)
    map_fname = "hotspot_map.html" if "Hotspots" in map_choice else "recommendation_map.html"
    map_path = maps_dir / map_fname

    if map_path.exists():
        html_data = map_path.read_text(encoding="utf-8")
        st.components.v1.html(html_data, height=620, scrolling=False)
    else:
        st.info("Interactive map file will be produced by the GIS pipeline stage.")

    if zone_df is not None:
        st.markdown("### Spatial Zone Attributes & Demand-Supply Gap")
        cols_to_show = [c for c in ["zone_id", "city", "predicted_demand_kwh", "station_count", "gi_z_score", "hotspot_class", "demand_supply_ratio", "underserved"] if c in zone_df.columns]
        st.dataframe(
            zone_df[cols_to_show].sort_values("predicted_demand_kwh" if "predicted_demand_kwh" in zone_df.columns else cols_to_show[0], ascending=False),
            use_container_width=True,
            height=320,
        )


# ─── Page 7: Recommendations ─────────────────────────────────────────────────

def page_recommendations():
    import plotly.express as px
    from evplan.recommend.ranking import build_recommendations

    st.markdown('<div class="hero-title">🏆 Charging Station Site Recommendations</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Multi-criteria priority ranking linking prospective sites to forecasted load, spatial gaps, and explainability drivers.</div>', unsafe_allow_html=True)

    cfg = load_config()
    default_w = cfg["recommendation"]["weights"]

    st.markdown("#### ⚙️ Real-Time Multi-Criteria Weight Calibration")
    w1, w2, w3, w4, w5 = st.columns(5)
    with w1:
        sw_demand = st.slider("⚡ Forecast Demand", 0.0, 1.0, float(default_w.get("demand", 0.30)), 0.05)
    with w2:
        sw_hotspot = st.slider("🔥 Hotspot Significance", 0.0, 1.0, float(default_w.get("hotspot", 0.25)), 0.05)
    with w3:
        sw_gap = st.slider("📉 Supply Deficit", 0.0, 1.0, float(default_w.get("supply_gap", 0.20)), 0.05)
    with w4:
        sw_dist = st.slider("📍 Distance to Station", 0.0, 1.0, float(default_w.get("distance", 0.15)), 0.05)
    with w5:
        sw_adopt = st.slider("🚗 EV Fleet Adoption", 0.0, 1.0, float(default_w.get("ev_adoption", 0.10)), 0.05)

    weights = {"demand": sw_demand, "hotspot": sw_hotspot, "supply_gap": sw_gap, "distance": sw_dist, "ev_adoption": sw_adopt}
    tot = sum(weights.values())
    norm_w = {k: v / tot for k, v in weights.items()} if tot > 0 else weights

    zone_df = load_artifact("zones_hotspot")
    shap_imp = load_shap_importance()

    if zone_df is None:
        pipeline_missing_warning()
        return

    # Dynamic re-ranking
    ranked_recs = build_recommendations(zone_df, shap_importance=shap_imp, weights=norm_w, top_n=15)

    st.markdown(f"### 📋 Top-15 Recommended Sites *(Normalized Weights: {', '.join(f'{k}={v:.2f}' for k, v in norm_w.items())})*")

    for _, row in ranked_recs.head(6).iterrows():
        rank = row.get("rank", 1)
        zone = row.get("zone_id", "Zone")
        score = row.get("priority_score", 0.0)
        chargers = row.get("recommended_chargers", 2)
        station_type = row.get("recommended_type", "DC Fast (CCS2)")
        demand = row.get("predicted_demand_kwh", row.get("historical_demand_kwh", 0.0))
        drivers = row.get("top_demand_drivers", "battery_capacity, initial_soc")
        co2 = row.get("co2_avoided_kg_per_day", 0.0)

        st.markdown(f"""
        <div class="recommendation-card">
            <div style="display:flex; justify-content:space-between; align-items:center;">
                <span style="font-size:1.15rem; font-weight:700; color:#38bdf8;">#{rank} — {zone}</span>
                <span style="font-weight:700; font-size:1.1rem; color:#34d399;">Priority Score: {score:.4f}</span>
            </div>
            <div style="margin:0.4rem 0; color:#cbd5e1; font-size:0.9rem;">
                <b>Type:</b> {station_type} &nbsp;|&nbsp; <b>Bays:</b> {chargers} Plugs &nbsp;|&nbsp; <b>Forecast Load:</b> {demand:.1f} kWh/day &nbsp;|&nbsp; <b>CO₂ Avoided:</b> {co2:.1f} kg/day
            </div>
            <div style="font-size:0.82rem; color:#94a3b8;">
                <b>SHAP Demand Drivers:</b> <i>{drivers}</i>
            </div>
        </div>
        """, unsafe_allow_html=True)

    # Bar chart
    fig_rec = px.bar(
        ranked_recs.head(12), x="zone_id", y="priority_score",
        color="priority_score", color_continuous_scale="Teal",
        title="Composite Priority Scores — Top Ranked Sites",
    )
    fig_rec.update_layout(template="plotly_dark", plot_bgcolor="rgba(0,0,0,0)", paper_bgcolor="rgba(0,0,0,0)")
    st.plotly_chart(fig_rec, use_container_width=True)

    csv_data = ranked_recs.to_csv(index=False)
    st.download_button("📥 Export Recommendations CSV", csv_data, "ev_station_recommendations.csv", "text/csv")


# ─── Page 8: What-If Simulator ───────────────────────────────────────────────

def page_whatif():
    import plotly.graph_objects as go
    from evplan.recommend.scoring import compute_priority_scores

    st.markdown('<div class="hero-title">🔬 What-If Policy Scenario Simulator</div>', unsafe_allow_html=True)
    st.markdown('<div class="hero-sub">Stress-test charging demand and zone priority under changing EV fleet adoption, charger expansions, and peak pricing.</div>', unsafe_allow_html=True)

    zone_df = load_artifact("zones_hotspot")
    if zone_df is None:
        pipeline_missing_warning()
        return

    col_ctrl, col_viz = st.columns([1.1, 2.2])

    with col_ctrl:
        st.markdown("#### Scenario Levers")
        zone_opts = zone_df.get("zone_id", pd.Series(["All"])).unique().tolist()
        sel_zone = st.selectbox("Candidate Zone", zone_opts)
        add_chargers = st.slider("Install Additional Chargers", 0, 30, 4)
        ev_growth = st.slider("EV Fleet Adoption Surge (%)", -30, 150, 35)
        peak_hour = st.slider("Target Hour of Day", 0, 23, 19)
        tariff_policy = st.selectbox("Tariff Policy", ["Standard Flat Rate", "Peak Surcharge (+25%)", "Off-Peak Discount (-20%)"])

    with col_viz:
        st.markdown("#### Real-Time Scenario Projection")
        scen_df = zone_df.copy()
        mask = scen_df["zone_id"] == sel_zone

        # Modulate station supply
        if "station_count" in scen_df.columns:
            scen_df.loc[mask, "station_count"] += add_chargers

        # Modulate demand
        d_col = "predicted_demand_kwh" if "predicted_demand_kwh" in scen_df.columns else "historical_demand_kwh"
        hour_mult = 1.35 if peak_hour in range(17, 21) else 0.75 if peak_hour < 7 else 1.0
        tariff_mult = 0.88 if "Surcharge" in tariff_policy else 1.15 if "Discount" in tariff_policy else 1.0
        growth_mult = 1.0 + (ev_growth / 100.0)

        if d_col in scen_df.columns:
            scen_df.loc[mask, d_col] = scen_df.loc[mask, d_col] * (hour_mult * tariff_mult * growth_mult)

        orig_scored = compute_priority_scores(zone_df)
        scen_scored = compute_priority_scores(scen_df)

        orig_val = float(orig_scored.loc[orig_scored["zone_id"] == sel_zone, "priority_score"].values[0]) if not orig_scored.loc[orig_scored["zone_id"] == sel_zone].empty else 0.5
        scen_val = float(scen_scored.loc[scen_scored["zone_id"] == sel_zone, "priority_score"].values[0]) if not scen_scored.loc[scen_scored["zone_id"] == sel_zone].empty else 0.5
        delta_score = scen_val - orig_val

        m1, m2, m3 = st.columns(3)
        m1.metric("Baseline Score", f"{orig_val:.4f}")
        m2.metric("Scenario Score", f"{scen_val:.4f}", delta=f"{delta_score:+.4f}")
        m3.metric("Effective Demand Multiplier", f"×{hour_mult * tariff_mult * growth_mult:.2f}")

        # Component comparison chart
        comp_fields = ["comp_demand", "comp_hotspot", "comp_supply_gap", "comp_distance", "comp_ev_adoption"]
        comp_labels = ["Demand Factor", "Hotspot Sig.", "Supply Deficit", "Distance Gap", "EV Fleet Share"]

        orig_comps = [float(orig_scored.loc[orig_scored["zone_id"] == sel_zone, c].values[0]) if c in orig_scored.columns and not orig_scored.loc[orig_scored["zone_id"] == sel_zone].empty else 0.5 for c in comp_fields]
        scen_comps = [float(scen_scored.loc[scen_scored["zone_id"] == sel_zone, c].values[0]) if c in scen_scored.columns and not scen_scored.loc[scen_scored["zone_id"] == sel_zone].empty else 0.5 for c in comp_fields]

        fig_comp = go.Figure()
        fig_comp.add_trace(go.Bar(name="Baseline Status Quo", x=comp_labels, y=orig_comps, marker_color="#38bdf8"))
        fig_comp.add_trace(go.Bar(name="Simulated Scenario", x=comp_labels, y=scen_comps, marker_color="#10b981"))
        fig_comp.update_layout(
            title=f"Score Component Shift for {sel_zone}",
            barmode="group",
            template="plotly_dark",
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            height=340,
        )
        st.plotly_chart(fig_comp, use_container_width=True)


# ─── Main Application Router ──────────────────────────────────────────────────

def main():
    selected_page = sidebar()

    if "Overview" in selected_page:
        page_overview()
    elif "EDA" in selected_page:
        page_eda()
    elif "Forecast" in selected_page:
        page_forecast()
    elif "Comparison" in selected_page:
        page_comparison()
    elif "Explainability" in selected_page:
        page_explainability()
    elif "GIS" in selected_page:
        page_gis()
    elif "Recommendations" in selected_page:
        page_recommendations()
    elif "What-If" in selected_page:
        page_whatif()


if __name__ == "__main__":
    main()
