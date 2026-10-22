# Explainable AI Framework for Smart EV Charging Infrastructure Planning

[![Python 3.10+](https://img.shields.io/badge/python-3.10+-blue.svg)](https://www.python.org/downloads/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.29+-FF4B4B.svg)](https://streamlit.io/)
[![XGBoost](https://img.shields.io/badge/XGBoost-2.0+-green.svg)](https://xgboost.readthedocs.io/)
[![SHAP](https://img.shields.io/badge/SHAP-0.43+-purple.svg)](https://shap.readthedocs.io/)
[![Code style: ruff](https://img.shields.io/badge/code%20style-ruff-000000.svg)](https://github.com/astral-sh/ruff)

**Institution:** VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Hyderabad  
**Department:** Information Technology  
**Program:** B.Tech — Final Year Major Project  

---

## 📌 Executive Summary

With the rapid acceleration of electric mobility, municipal planners and distribution system operators (DSOs) face severe challenges in deciding **where**, **when**, and **how many** EV charging stations to build. Suboptimal placement results in grid localized transformer overload, underutilized capital investments, and long vehicle queuing times.

This project delivers a complete, production-grade **Explainable AI (XAI) Framework for Smart EV Charging Infrastructure Planning**. It bridges black-box machine learning and urban policy-making by pairing high-performance multivariate demand forecasting with game-theoretic **SHAP (SHapley Additive exPlanations)** interpretability, spatial hotspot clustering (**Getis-Ord $G_i^*$**), and multi-criteria site prioritization.

---

## 🌟 Key Features

1. **Multivariate EV Charging Demand Forecasting:**
   - Evaluates **Random Forest Regressor** and **XGBoost Regressor** against Naive, Seasonal Naive, and Ridge Linear baselines.
   - Rigorous **expanding-window chronological validation** (`TimeSeriesSplit`) strictly preventing lookahead leakage.
   - Automated hyperparameter optimization via **Optuna** (TPE sampler).
   - Evaluation metrics: RMSE, MAE, $R^2$, and sMAPE.

2. **SHAP-Based Explainable AI (TreeSHAP):**
   - **Global Interpretability:** Summary bar plots and beeswarm distributions illustrating macro feature impacts across sessions.
   - **Feature Dependence & Non-linear Interactions:** Dynamic scatter plots exposing charging thresholds (e.g., initial battery SoC vs. energy consumed).
   - **Local Interpretability:** Decision-level waterfall explanations decomposing individual predictions from baseline expectation $\phi_0$ to final kWh.
   - **Cross-Model Consistency Check:** Pearson correlation analysis between RF and XGBoost SHAP attributions.

3. **GIS Hotspot Clustering & Spatial Mapping:**
   - Automated spatial aggregation across urban municipal zones and city clusters.
   - Spatial autocorrelation analysis via **Getis-Ord $G_i^*$** and **Moran's $I$** using inverse-distance spatial weighting.
   - Interactive, multi-layer Folium maps rendered in Streamlit (Choropleth zones, Hotspots/Coldspots, Station clusters, Recommended sites).

4. **Multi-Criteria Station Recommendation Engine:**
   - Composite utility scoring:
     $$\text{Score}_i = w_D \tilde{D}_i + w_U \tilde{U}_i + w_A \tilde{A}_i + w_S \tilde{S}_i$$
     combining forecasted demand, infrastructure deficit, EV fleet adoption, and SHAP policy weights.
   - Outputs: Recommended station tier (DC Fast vs. AC Level 2), recommended plug counts, grid kVA capacity estimation, and natural-language justifications.

5. **Interactive What-If Scenario Simulator:**
   - Real-time simulation of policy interventions (e.g., EV adoption growth +25% to +100%, peak-hour tariff surcharges, fast-charging share expansion).
   - Instant re-calculation of grid load impact, revenue projections, and queue mitigation.

6. **9-Page Executive Streamlit Web Dashboard:**
   - Overview & KPIs, Exploratory Data Analysis (EDA), Demand Forecasting, Model Comparison & Residuals, SHAP Explainability Lab, GIS Spatial Map, Station Recommendations, What-If Simulator, and Methodology Reference.

---

## 🗂️ Project Architecture

```
ev-xai-planner/
├── app/
│   └── streamlit_app.py        # 9-page executive Streamlit dashboard
├── configs/
│   ├── config.yaml             # Central pipeline configuration & hyperparameters
│   └── schema.yaml             # Multi-dataset column mapping and aliases
├── data/
│   ├── raw/                    # Downloaded Kaggle and source datasets
│   ├── interim/                # Temporary processing stages
│   └── processed/              # Cleaned Parquet tables & engineered feature sets
├── docs/
│   ├── data_audit.md           # Automated dataset health & null audit
│   ├── data_integration.md     # Ingestion & cleaning documentation
│   └── methodology.md          # Full mathematical & algorithmic formulation
├── notebooks/                  # Interactive experimentation notebooks
├── outputs/
│   ├── figures/                # Generated SHAP plots, residuals, feature importances
│   ├── maps/                   # Standalone interactive HTML Folium maps
│   ├── models/                 # Serialized joblib models & Optuna best params
│   ├── reports/                # Evaluation reports & summaries
│   └── tables/                 # CSV metric tables, rankings, and validations
├── src/
│   └── evplan/
│       ├── data/               # Download, loaders, schema mapping, cleaning
│       ├── features/           # Temporal, cyclic, lag, rolling, zone features
│       ├── models/             # Baselines, training, tuning, evaluation
│       ├── explain/            # SHAP TreeExplainer, plots, local waterfalls
│       ├── gis/                # Spatial metrics, Getis-Ord Gi*, Folium maps
│       ├── recommend/          # Multi-criteria scoring, ranking, validation
│       ├── utils/              # Config, logging, seed, filesystem helpers
│       └── run.py              # CLI Master pipeline orchestrator
├── tests/
│   └── test_pipeline.py        # 15 comprehensive unit & integration tests
├── launch.bat                  # One-click Windows desktop launcher
├── Makefile                    # Automation shortcuts (install, pipeline, test, dashboard)
├── pyproject.toml              # Modern Python build specification
└── requirements.txt            # Locked project dependencies
```

---

## ⚡ Quick Start

### 1. Installation
Clone the repository and install requirements:
```bash
git clone https://github.com/vnr-vjiet-it/ev-xai-planner.git
cd ev-xai-planner

# Install dependencies and package in editable mode
pip install -r requirements.txt
pip install -e . --no-deps
```

### 2. Run the Full End-to-End Pipeline
Execute all stages in sequence (downloads raw Kaggle data, cleans, engineers features, tunes models, computes SHAP values, runs GIS clustering, and generates recommendations):
```bash
python -m evplan.run --stage all
```

Individual stages can also be executed independently:
```bash
python -m evplan.run --stage data
python -m evplan.run --stage features
python -m evplan.run --stage train
python -m evplan.run --stage explain
python -m evplan.run --stage gis
python -m evplan.run --stage recommend
```

### 3. Launch the Interactive Dashboard
Run the Streamlit web application:
```bash
# On Windows (one command):
launch.bat

# Or using Streamlit directly:
streamlit run app/streamlit_app.py
```
Open [http://localhost:8501](http://localhost:8501) in your browser.

### 4. Run Test Suite
Run all unit and integration tests with pytest:
```bash
python -m pytest tests/ -v
```

---

## 📊 Datasets Used

| Dataset Name | Source | Description |
| :--- | :--- | :--- |
| **EV Charging Demand Dataset** | [Kaggle: ziya07](https://www.kaggle.com/datasets/ziya07/electric-vehicle-charging-demand-dataset) | Real EV charging sessions with energy consumed, duration, SoC, vehicle category, and station ID. |
| **EV Charging Stations in India** | [Kaggle: pranjal9091](https://www.kaggle.com/datasets/pranjal9091/ev-charging-stations-in-india-simplified-2025) | Geocoded EV charging stations across Indian states and cities with operational capacities. |
| **Indian EV Dataset** | [Kaggle: sanhitasaxena](https://www.kaggle.com/datasets/sanhitasaxena/indian-electric-vehicle-dataset) | State-level EV adoption rates, vehicle registrations, and market penetration statistics. |

---

## 🎓 Academic Accreditation

- **Project Title:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
- **Institution:** VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Bachupally, Hyderabad  
- **Department:** Department of Information Technology  
- **Academic Year:** 2025 – 2026  
