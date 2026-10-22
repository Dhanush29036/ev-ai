# Model Card: EV Charging Demand Forecasters & XAI Engine

**Project:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Institution:** VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Hyderabad  
**Department:** Information Technology  
**Version:** 1.0.0  

---

## 1. Model Details
- **Architecture:** 
  - Primary Ensemble: **Random Forest Regressor** (tuned: 325 estimators, max_depth 20)
  - Primary Gradient Boosting: **XGBoost Regressor** (tuned: 397 estimators, max_depth 5, lr 0.022)
  - Comparison Model: **LightGBM Regressor** (tuned: 200 estimators, max_depth 6, lr 0.050)
  - Comparison Baselines: Ridge Linear Regression, Last-Observation Naive, Seasonal Naive (168-hr lag).
- **Frameworks:** scikit-learn 1.9, xgboost 3.4, lightgbm 4.0, shap 0.54, optuna 5.0.
- **Explainability:** TreeSHAP (interventional / path-dependent feature attributions).

---

## 2. Intended Use
- **Primary Use Case:** Forecasting municipal-scale EV charging energy demand (kWh) across spatial zones to prioritize new charging station deployments.
- **Secondary Use Case:** Interpreting policy levers (battery capacity trends, time-of-day tariffs, fleet electrification ratios) through Shapley attribution values.
- **Out-of-Scope Uses:** Real-time millisecond-level inverter control or billing transaction processing.

---

## 3. Training & Validation Data
- **Dataset A:** Real EV Charging Demand Dataset (`ziya07/electric-vehicle-charging-demand-dataset`), ~8,354 sessions across commercial, residential, workplace, and highway locations.
- **Dataset B:** Geocoded EV Charging Stations in India (`pranjal9091/ev-charging-stations-in-india-simplified-2025`), ~789 physical stations.
- **Dataset C:** Indian Electric Vehicle State Adoption (`sanhitasaxena/indian-electric-vehicle-dataset`).
- **Validation Protocol:** Chronological train (70%) / validation (10%) / held-out test (20%) split using TimeSeriesSplit cross-validation. Zero random shuffling.

---

## 4. Evaluation Metrics & Benchmarks
Evaluated on strictly held-out chronological test set:
- **LightGBM:** RMSE = 0.3060, MAE = 0.1954, $R^2 = 0.9997$, MAPE = 0.65%, sMAPE = 0.65%
- **XGBoost:** RMSE = 0.3182, MAE = 0.2147, $R^2 = 0.9997$, MAPE = 0.70%, sMAPE = 0.70%
- **Random Forest:** RMSE = 0.3646, MAE = 0.2120, $R^2 = 0.9996$, MAPE = 0.67%, sMAPE = 0.66%
- **Ridge Linear Regression:** RMSE = 3.6209, MAE = 2.7262, $R^2 = 0.9632$, MAPE = 9.51%
- **Naive Baseline:** RMSE = 21.9946, MAE = 18.8918, $R^2 = -0.3560$, MAPE = 76.88%

---

## 5. Critical Technical Transparency: Physics vs. Ex-Ante Forecasting
### Deterministic Relationship (Ex-Post):
In physics-based charging records where initial battery SoC, final battery SoC, and battery pack capacity are all known after a session ends:
$$\text{Energy Consumed (kWh)} \approx \text{Battery Capacity (kWh)} \times \frac{\text{Final SoC} - \text{Initial SoC}}{100}$$
When models are evaluated on this ex-post feature vector, $R^2$ approaches 1.0 (mean |SHAP| correctly attributes >90% of importance to battery capacity and state of charge).

### Ex-Ante Forecasting Mode:
For long-term infrastructure planning where future session SoC is unknown at deployment time, the framework provides an **ex-ante time-series mode** utilizing historical rolling demand, temporal periodicity (hour, day, month, seasonality), station density, and regional EV adoption context.

---

## 6. Ethical Considerations & Limitations
- **Data Representativeness:** Real-world charging datasets reflect early adopter behavior in urban corridors and may underestimate rural/semi-urban commercial demand.
- **Spatial Granularity:** Indian station locations represent point coordinates; polygon boundary approximations are used for zone aggregations.
- **Environmental Impact:** CO₂ mitigation estimates use an average grid emission factor of $0.82\text{ kg CO}_2/\text{kWh}$, which varies regionally depending on thermal vs renewable generation mix.
