# Product Requirements Document (PRD)

## Project Title: Explainable AI Framework for Smart EV Charging Infrastructure Planning
**Institution:** VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Hyderabad  
**Department:** Department of Information Technology  
**Program:** Final Year Major Project (B.Tech)  
**Academic Year:** 2025 – 2026  

---

## 1. Problem Statement & Motivation
The global transition toward electric mobility requires massive capital expenditure in charging infrastructure. Municipal planning bodies, distribution system operators (DSOs), and commercial charge point operators (CPOs) currently rely on heuristic or static demographic proxies when selecting new station sites. This causes:
- **Localized Grid Strain:** Transformer overloads during peak hours.
- **Stranded Capital:** Low charger utilization in suboptimal zones.
- **High Queuing Delay:** Severe congestion at bottleneck hubs.

Black-box machine learning approaches fail in public policy and infrastructure financing because municipal committees require **explainability, traceability, and justification** before approving permits or grid interconnections.

---

## 2. Target Users & Stakeholders
1. **Urban Planners & Municipal Authorities:** Need ranked priority zones with explainable socioeconomic and spatial justifications.
2. **Distribution System Operators (DSOs):** Require grid capacity impact estimates (kVA), peak load forecasts, and phase balance insights.
3. **Charge Point Operators (CPOs):** Need ROI, charger type recommendations (DC Fast vs AC Level 2), and bay sizing.
4. **Academic & Research Evaluators:** Require reproducible pipelines, leak-free validation, and game-theoretic interpretability.

---

## 3. Product Objectives & Success Metrics
- **Forecasting Accuracy:** Deliver multivariate demand predictions with competitive RMSE (< 1.5 kWh) and $R^2 > 0.90$ on held-out test data.
- **Explainability:** 100% of zone recommendations traceable to top SHAP feature drivers with auto-generated plain-English narratives.
- **Spatial Clustering:** Quantitative spatial autocorrelation via Getis-Ord $G_i^*$ and Moran's $I$.
- **Latency & Usability:** Sub-second interactive filter updates on the 9-page executive Streamlit dashboard.
- **Reproducibility:** Single-command execution via `python -m evplan.run --stage all` and `launch.bat`.

---

## 4. System Scope & Functional Requirements

### 4.1 Data Ingestion & Preprocessing
- Dynamic schema mapper (`configs/schema.yaml`) to normalize diverse dataset structures.
- Strict chronological validation (`TimeSeriesSplit`) without lookahead leakage.
- Missing value imputation and IQR-based outlier bounding with audit flags.

### 4.2 Multivariate Demand Forecasting
- Machine Learning models: **Random Forest Regressor**, **XGBoost Regressor**, and **LightGBM Regressor**.
- Baselines: Naive, Seasonal Naive (168-hour weekly lag), and Ridge Linear Regression.
- Automated hyperparameter tuning with **Optuna** (TPE sampler, 30 trials).

### 4.3 Explainable AI (XAI)
- Global TreeSHAP bar importance and beeswarm distributions.
- Feature dependence plots with automatic interaction detection.
- Local waterfall plots for individual prediction decomposition from base expectation $\phi_0$.
- Spearman rank correlation consistency check between RF and XGBoost attributions.

### 4.4 Spatial Analysis & GIS Hotspot Clustering
- Multi-layer interactive maps (Demand Hotspots, Existing Stations, Recommended Sites).
- Statistical clustering: Getis-Ord $G_i^*$ ($p < 0.01, 0.05, 0.10$) and Global Moran's $I$.

### 4.5 Recommendation Engine
- Multi-criteria priority score:
  $$\text{Priority Score}_i = w_D \tilde{D}_i + w_H \tilde{H}_i + w_G \tilde{G}_i + w_d \tilde{d}_i + w_A \tilde{A}_i$$
- Live weight adjustment sliders with real-time re-ranking.
- Sensitivity analysis under $\pm 20\%$ weight perturbations.
- Export to CSV format for municipal reporting.

### 4.6 What-If Scenario Simulator
- Parametric adjustment of EV fleet adoption growth (-50% to +200%).
- Real-time simulation of charger additions, peak-hour shifts, and seasonal variations.

---

## 5. Non-Functional Requirements
- **Robustness:** Zero unhandled crashes on missing columns or empty filters; graceful fallbacks throughout.
- **Documentation:** Full algorithmic, mathematical, and data audit guides in `docs/`.
- **Code Quality:** Type hints, Google docstrings, Ruff linter compliance, and modular structure in `src/evplan/`.
- **Test Coverage:** Pytest suite testing feature leakage, schema translation, scoring monotonicity, and ranking determinism.
