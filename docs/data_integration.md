# Data Integration Guide

**Project:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Institution:** VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Hyderabad  
**Department:** Information Technology  

---

## 1. Overview of Data Sources

The framework ingests and harmonizes three core multi-modal data streams to drive multivariate demand forecasting, spatial hotspot clustering, and explainable station site recommendation.

| Dataset Role | Primary Source | Kaggle Dataset Slug | File Format | Records (Cleaned) | Key Attributes |
| :--- | :--- | :--- | :--- | :--- | :--- |
| **EV Demand** | Real Charging Sessions | `ziya07/electric-vehicle-charging-demand-dataset` | CSV / Parquet | ~8,354 sessions | `timestamp`, `energy_consumed_kwh`, `session_duration_hours`, `station_id`, `vehicle_category`, `initial_soc`, `final_soc`, `charging_rate_kw` |
| **EV Stations** | Charging Infrastructure (India) | `pranjal9091/ev-charging-stations-in-india-simplified-2025` | CSV / Parquet | ~789 stations | `name`, `city`, `state`, `latitude`, `longitude`, `type`, `capacity` |
| **EV Adoption** | State/Vehicle Adoption | `sanhitasaxena/indian-electric-vehicle-dataset` | CSV / Parquet | ~32 records | `State`, `EV_Count`, `Vehicle_Class`, `Penetration_Rate` |

---

## 2. Automated Pipeline Ingestion

The pipeline automatically fetches these datasets using `kagglehub` without requiring manual credential configuration for public datasets:

```bash
# Automated download via pipeline runner
python -m evplan.run --stage data
```

### Download Directory Structure:
```
data/
├── raw/
│   ├── ev_charging_demand/
│   │   └── ev_charging_dataset.csv
│   ├── ev_stations_india/
│   │   └── Indian_EV_Stations_Simplified.csv
│   └── indian_ev_dataset/
│       └── final_dataset.csv
├── interim/
└── processed/
    ├── demand_clean.parquet
    ├── stations_clean.parquet
    ├── adoption_clean.parquet
    ├── zone_table.parquet
    ├── features.parquet
    └── zones_hotspot.parquet
```

---

## 3. Data Cleaning and Schema Mapping

### 3.1 Schema Harmonization (`configs/schema.yaml`)
Different datasets use varied naming conventions (`City` vs `city`, `Latitude` vs `latitude`, `energy_consumed_kWh` vs `energy_consumed_kwh`). The schema mapping layer standardizes them dynamically:
- Strip whitespace, normalize to lowercase.
- Map aliases to canonical schema keys.

### 3.2 Cleaning Rules
1. **Datetime Parsing:** Converts `timestamp` to standard UTC-naive `datetime64[ns]` and extracts temporal hierarchy (hour, day of week, month, weekend).
2. **Missing Value Imputation:**
   - Drops rows missing critical target variables (`energy_consumed_kwh`).
   - Imputes numeric features with median and categorical features with mode.
3. **Outlier Treatment:**
   - Detects extreme outliers using IQR with a 3.0 threshold.
   - Clips outliers to valid physical bounds (e.g., SoC bounded between 0% and 100%, duration > 0).
4. **Geographic Filtering:**
   - Bounds station coordinates to valid India geographic box (Lat: 6.0° to 38.0° N, Lon: 68.0° to 98.0° E).
   - Resolves spatial coordinates to geographic zones and city clusters.

---

## 4. Manual Download Fallback

If Kaggle API access is restricted in your network environment, download the CSV files manually from:
1. [Kaggle - EV Charging Demand Dataset](https://www.kaggle.com/datasets/ziya07/electric-vehicle-charging-demand-dataset)
2. [Kaggle - EV Charging Stations in India](https://www.kaggle.com/datasets/pranjal9091/ev-charging-stations-in-india-simplified-2025)
3. [Kaggle - Indian EV Dataset](https://www.kaggle.com/datasets/sanhitasaxena/indian-electric-vehicle-dataset)

Place the downloaded CSV files into their respective folders under `data/raw/` and rerun:
```bash
python -m evplan.run --stage data
```
