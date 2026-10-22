# Methodology and Algorithmic Framework

**Project:** Explainable AI Framework for Smart EV Charging Infrastructure Planning  
**Institution:** VNR Vignana Jyothi Institute of Engineering & Technology (VNR VJIET), Hyderabad  
**Department:** Information Technology  

---

## 1. Mathematical Formulation of EV Charging Demand Forecasting

Let $\mathcal{D} = \{(\mathbf{x}_t, y_t)\}_{t=1}^N$ denote a multivariate time series of EV charging sessions, where:
- $\mathbf{x}_t \in \mathbb{R}^D$ is the feature vector at time $t$, capturing temporal indicators, cyclical encodings, historical lag consumption, rolling window statistics, session dynamics, and spatial zone aggregations.
- $y_t \in \mathbb{R}^+$ represents the target variable: aggregate energy demand ($E_t$ in kWh) or charging load ($P_t$ in kW).

The regression objective is to learn an estimator $f^*(\mathbf{x}): \mathbb{R}^D \to \mathbb{R}$ minimizing expected loss over the empirical distribution:
$$\min_{\theta} \frac{1}{N} \sum_{t=1}^N \mathcal{L}(y_t, f(\mathbf{x}_t; \theta)) + \Omega(\theta)$$
where $\mathcal{L}(\cdot)$ is squared error $(y - \hat{y})^2$ and $\Omega(\theta)$ represents model regularization (tree depth, L1/L2 leaf penalties).

### 1.1 Non-Leakage Chronological Validation
To prevent temporal data leakage, we enforce strict chronological splitting:
$$\mathcal{D}_{\text{train}} = \{(\mathbf{x}_t, y_t) \mid t \le T_{\text{split}}\}, \quad \mathcal{D}_{\text{test}} = \{(\mathbf{x}_t, y_t) \mid t > T_{\text{split}}\}$$
Cross-validation uses expanding-window Time Series Split (TimeSeriesSplit) where fold $k$ is trained exclusively on past horizons $1 \dots t_k$ and evaluated on $[t_k + 1, t_k + \Delta]$.

### 1.2 Evaluation Metrics
- **Root Mean Squared Error (RMSE):** $\sqrt{\frac{1}{N}\sum_{i=1}^N (y_i - \hat{y}_i)^2}$
- **Mean Absolute Error (MAE):** $\frac{1}{N}\sum_{i=1}^N |y_i - \hat{y}_i|$
- **Coefficient of Determination ($R^2$):** $1 - \frac{\sum (y_i - \hat{y}_i)^2}{\sum (y_i - \bar{y})^2}$
- **Symmetric MAPE (sMAPE):** $\frac{100\%}{N} \sum_{i=1}^N \frac{2|y_i - \hat{y}_i|}{|y_i| + |\hat{y}_i| + \epsilon}$

---

## 2. Explainable AI: SHAP (SHapley Additive exPlanations)

To render the black-box tree ensembles interpretable to municipal planners and electrical grid engineers, we apply cooperative game-theoretic **TreeSHAP**.

For an individual prediction $f(\mathbf{x})$, the model output decomposes additively:
$$f(\mathbf{x}) = \phi_0 + \sum_{j=1}^D \phi_j(\mathbf{x})$$
where:
- $\phi_0 = \mathbb{E}[f(\mathbf{X})]$ is the base value (dataset expected demand).
- $\phi_j(\mathbf{x})$ is the Shapley value of feature $j$, defined as:
$$\phi_j(\mathbf{x}) = \sum_{S \subseteq \mathcal{F} \setminus \{j\}} \frac{|S|!(|\mathcal{F}| - |S| - 1)!}{|\mathcal{F}|!} \Big[ f_{\mathbf{x}}(S \cup \{j\}) - f_{\mathbf{x}}(S) \Big]$$

### 2.1 Explanation Modes
1. **Global Summary & Beeswarm:** Highlights macro-level feature importance across the entire cohort and shows directional effect (e.g., high initial SoC reduces energy consumed).
2. **Feature Dependence & Interaction:** Plots feature value vs. its SHAP value, coloring by an automatically selected interaction feature to reveal non-linear thresholds and saturation effects.
3. **Local Waterfall & Force Plots:** Decomposes a single charging session or spatial zone prediction from base expectation $\phi_0$ to final predicted demand $f(\mathbf{x})$.

---

## 3. Spatial Hotspot Analysis: Getis-Ord $G_i^*$ & Moran's $I$

To identify statistically significant geographic clusters of high charging demand (hotspots) and low demand (coldspots), we formulate spatial weight matrices $\mathbf{W}$ based on inverse distance:
$$w_{ij} = \frac{1}{d(z_i, z_j)^\alpha}, \quad \forall i \neq j, \quad w_{ii} = 0$$

### 3.1 Getis-Ord Local Statistic ($G_i^*$)
For each spatial zone $i$:
$$G_i^* = \frac{\sum_{j=1}^n w_{ij} x_j - \bar{X}\sum_{j=1}^n w_{ij}}{S \sqrt{\frac{n\sum_{j=1}^n w_{ij}^2 - (\sum_{j=1}^n w_{ij})^2}{n - 1}}}$$
where $\bar{X} = \frac{1}{n}\sum x_j$ and $S = \sqrt{\frac{1}{n}\sum x_j^2 - (\bar{X})^2}$.

- **$Z(G_i^*) > +1.96$ ($p < 0.05$):** Statistically significant EV Charging Hotspot.
- **$Z(G_i^*) < -1.96$ ($p < 0.05$):** Statistically significant Coldspot.
- **$-1.96 \le Z(G_i^*) \le +1.96$:** Not statistically significant spatial concentration.

---

## 4. Multi-Criteria Recommendation & Optimization

Candidate zone suitability for new EV infrastructure deployment is ranked through a multi-attribute utility function:

$$\text{Priority Score}_i = w_{\text{demand}} \cdot \tilde{D}_i + w_{\text{deficit}} \cdot \tilde{U}_i + w_{\text{adoption}} \cdot \tilde{A}_i + w_{\text{shap}} \cdot \tilde{S}_i$$

where each component is normalized via Min-Max scaling:
$$\tilde{V}_i = \frac{V_i - \min(V)}{\max(V) - \min(V)}$$

- $\tilde{D}_i$: Forecasted Charging Energy Demand in zone $i$.
- $\tilde{U}_i$: Infrastructure Deficit Score (Demand-to-Capacity ratio / distance to nearest station).
- $\tilde{A}_i$: EV Fleet Adoption & Registered Vehicles density in the surrounding region.
- $\tilde{S}_i$: Model-derived Explainability Weight (composite impact of top policy-actionable features).
- Normalized weights: $\sum_{k} w_k = 1.0$.

### 4.1 Station Recommendation Output
Each recommended candidate site outputs:
1. Recommended Station Type: DC Fast Charging Hub vs. AC Level-2 Destination.
2. Suggested Number of Charging Bays / Plugs.
3. Estimated Grid Capacity Impact (kVA requirement).
4. Investment Urgency Tier (Critical, High, Moderate).
5. Comprehensive Natural Language Justification citing local demand factors and SHAP feature drivers.
