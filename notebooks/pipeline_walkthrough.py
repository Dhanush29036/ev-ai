"""
EV XAI Planner — Interactive Pipeline Walkthrough
VNR VJIET IT Department — Final Year Major Project

Demonstrates:
1. Dataset inspection and summary statistics
2. Model performance comparison (RF, XGBoost, Baselines)
3. SHAP explainability insights (Global importance, Beeswarm, Local waterfall)
4. GIS Hotspot identification and recommended station locations
"""

import json
from pathlib import Path
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent

def main():
    print("=" * 70)
    print("  EXPLAINABLE AI FRAMEWORK FOR SMART EV CHARGING INFRASTRUCTURE PLANNING")
    print("  VNR VJIET - IT Department - Final Year Major Project")
    print("=" * 70)

    # 1. Inspect Processed Datasets
    print("\n[1] DATASET SUMMARY")
    demand_p = ROOT / "data" / "processed" / "demand_clean.parquet"
    if demand_p.exists():
        df_demand = pd.read_parquet(demand_p)
        print(f"  • EV Charging Sessions: {len(df_demand):,} rows, {len(df_demand.columns)} columns")
        print(f"  • Date range: {df_demand['timestamp'].min()} to {df_demand['timestamp'].max()}")
        print(f"  • Total energy delivered: {df_demand['energy_consumed_kwh'].sum():,.2f} kWh")
        print(f"  • Mean session duration: {df_demand['session_duration_hours'].mean():.2f} hours")
    
    stations_p = ROOT / "data" / "processed" / "stations_clean.parquet"
    if stations_p.exists():
        df_stations = pd.read_parquet(stations_p)
        print(f"  • EV Charging Stations (India): {len(df_stations):,} geocoded stations across {df_stations['state'].nunique()} states")

    # 2. Inspect Model Performance
    print("\n[2] MODEL PERFORMANCE COMPARISON")
    metrics_p = ROOT / "outputs" / "tables" / "model_comparison.csv"
    if metrics_p.exists():
        df_metrics = pd.read_csv(metrics_p)
        print(df_metrics.to_string(index=False))
    else:
        print("  • Model comparison table will be generated upon pipeline completion.")

    # 3. Inspect SHAP Explainability
    print("\n[3] SHAP EXPLAINABILITY INSIGHTS")
    shap_p = ROOT / "outputs" / "tables" / "shap_importance_xgboost.csv"
    if not shap_p.exists():
        shap_p = ROOT / "outputs" / "tables" / "shap_importance_rf.csv"
    
    if shap_p.exists():
        df_shap = pd.read_csv(shap_p)
        print("  • Top 5 Drivers of EV Charging Demand:")
        for idx, row in df_shap.head(5).iterrows():
            print(f"    {idx+1}. {row['feature']}: {row['mean_abs_shap']:.4f} mean |SHAP| ({row['importance_pct']:.1f}%)")
    else:
        print("  • SHAP importance table pending pipeline run.")

    # 4. Inspect Recommended Station Sites
    print("\n[4] TOP RECOMMENDED CHARGING STATIONS")
    rec_p = ROOT / "outputs" / "tables" / "recommendations.csv"
    if rec_p.exists():
        df_recs = pd.read_csv(rec_p)
        display_cols = ["rank", "city", "priority_score", "demand_score", "hotspot_score", "recommended_type", "recommended_chargers"]
        valid_cols = [c for c in display_cols if c in df_recs.columns]
        print(df_recs[valid_cols].head(5).to_string(index=False))
    else:
        print("  • Recommendation rankings pending pipeline completion.")

    print("\n" + "=" * 70)
    print("  To launch the interactive visual dashboard, run:")
    print("  python -m streamlit run app/streamlit_app.py")
    print("=" * 70)

if __name__ == "__main__":
    main()
