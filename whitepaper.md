# Temperature-Driven Electricity Load Forecasting
## A Machine Learning Approach for PJM and CAISO Grid Operators

**Version 1.1 — April 2026**

---

## Abstract

This paper describes a temperature-driven electricity load forecasting system built for two major US grid operators: PJM Interconnection (eastern US) and the California ISO (CAISO). The system ingests publicly available numerical weather prediction (NWP) output from the Global Forecast System (GFS), transforms it into population-weighted temperature indices, and applies an XGBoost gradient-boosted regression model to produce 15-day electricity demand forecasts with uncertainty bounds. On held-out test data, the model achieves a mean absolute percentage error (MAPE) of **0.4% for PJM** and **0.3% for CAISO** — an improvement of roughly 90% over a naive temperature-regression baseline. Results are surfaced through a single-page interactive dashboard (Plotly Dash) that overlays the model's hindcast and forward forecast against official ISO benchmarks and EIA actuals on a continuous linear time axis, alongside a full 2-year backtest with monthly MAPE breakdown.

---

## 1. Motivation and Aim

Electricity load is highly sensitive to temperature. Air conditioning and heating loads can shift regional demand by 20–40% between mild and extreme weather days. Operators, traders, and analysts need reliable multi-day load forecasts that are physically grounded in weather projections, not just statistical time-series extrapolation.

The goal of this project is to build a **physics-informed, data-driven** pipeline that:

1. Ingests freely available NWP forecasts (GFS, ECMWF, GraphCast) for up to 15 days
2. Produces population-weighted daily temperature indices for each ISO footprint
3. Maps those indices to electricity demand using a trained regression model
4. Quantifies forecast uncertainty from ensemble weather spread
5. Compares results against official ISO and EIA day-ahead forecasts in real time

---

## 2. Data Sources

### 2.1 Historical Load (Training Target)

| Source | Endpoint | Resolution | Coverage |
|--------|----------|-----------|---------|
| EIA Open Data API | `/v2/electricity/rto/region-data/data/` | Hourly | 2024-03-23 to 2026-03-26 |
| Respondent codes | `PJM` (PJM Interconnection), `CISO` (California ISO) | — | — |

Hourly demand values (type `D`) are aggregated to daily mean MW. The training period covers **731 days** (2 years) for both ISOs.

**PJM observed load statistics:**
- Range: 72.3 – 136.9 GW (daily mean)
- Mean: 95.7 GW
- Strong summer peak (air conditioning) and moderate winter peak (heating)

**CAISO observed load statistics:**
- Range: 19.2 – 37.7 GW (daily mean)
- Mean: 25.7 GW
- Dominant summer peak; winter loads are comparatively mild

### 2.2 Historical Temperature (Training Features)

Reanalysis temperatures come from the **Open-Meteo Historical Weather API** (ERA5 reanalysis), which provides quality-controlled daily averages, maxima, and minima at any lat/lon coordinate.

Each ISO footprint is represented by **12 population-weighted monitoring locations** chosen to span the geographic and demographic distribution of the service territory:

**PJM locations (12 nodes, eastern US):**
Washington DC (12%), Philadelphia PA (10%), Pittsburgh PA (8%), Chicago IL (8%), Columbus OH (7%), Detroit MI (7%), Cleveland OH (6%), Baltimore MD (5%), Richmond VA (5%), Roanoke VA (4%), Indianapolis IN (4%), Cincinnati OH (4%) *(weights reflect population share within MISO/PJM footprint)*

**CAISO locations (12 nodes, California):**
Los Angeles CA (35%), Riverside CA (12%), San Francisco CA (10%), San Diego CA (9%), Sacramento CA (7%), San Jose CA (5%), Fresno CA (5%), Bakersfield CA (4%), Ventura CA (4%), Stockton CA (3%), Palm Springs CA (3%), Santa Barbara CA (3%)

**Population-weighted temperature index:**

$$T_{\text{weighted}} = \frac{\sum_i w_i \cdot T_i}{\sum_i w_i}$$

where $w_i$ is the population weight of location $i$ and $T_i$ is the ERA5 temperature in °F.

Separate weighted indices are computed for daily average, daily high, and daily low temperatures.

### 2.3 Forecast Temperature (Inference Input)

At inference time, temperature forecasts are fetched from the **Open-Meteo Forecast API**, which provides free access to:

- **GFS** (NOAA Global Forecast System): 15-day forecasts, 0.25° resolution, updated 4x/day
- **ECMWF IFS**: 10-day forecasts, 0.25° resolution
- **GraphCast** (Google DeepMind via ECMWF): 10-day ML-based forecasts

### 2.4 Official Comparison Benchmarks

| ISO | Benchmark | Source | Horizon |
|-----|-----------|--------|---------|
| PJM | PJM DataMiner 7-Day Load Forecast | PJM API Portal (`/load_frcstd_7_day`, `RTO_COMBINED`) | 7 days |
| PJM | EIA Day-Ahead Demand Forecast (type `DF`) | EIA Open Data API | 1–2 days |
| CAISO | OASIS 7-Day System Forecast (`SLD_FCST/7DA`, `CA ISO-TAC`) | CAISO OASIS public API | 7 days |
| Both | EIA Actual Demand (type `D`) | EIA Open Data API | Historical |

The PJM DataMiner API requires a subscription key (`Ocp-Apim-Subscription-Key` header) obtained via the PJM API Portal. Responses are cached locally for 2 hours to avoid repeated API calls within a single dashboard session.

---

## 3. Feature Engineering

### 3.1 Heating and Cooling Degree Days

Standard base-65°F degree-day formulation:

$$\text{HDD} = \max(0,\ 65 - T_{\text{avg}})$$
$$\text{CDD} = \max(0,\ T_{\text{avg}} - 65)$$

Computed separately from daily average, high, and low temperatures — giving six degree-day features that capture the asymmetric response of HVAC loads to temperature extremes.

### 3.2 Seasonal Signal

Sine and cosine of day-of-year capture the smooth annual cycle in baseline demand:

$$\sin\!\left(\frac{2\pi \cdot \text{DOY}}{365}\right), \quad \cos\!\left(\frac{2\pi \cdot \text{DOY}}{365}\right)$$

### 3.3 Calendar and Holiday Features

| Feature | Description |
|---------|-------------|
| Day-of-week one-hot (7) | Separates Mon–Sun load profiles |
| `is_holiday` | US federal holiday (New Year's, MLK, Presidents, Memorial, Independence, Labor, Thanksgiving, Christmas) |
| `is_holiday_week` | Christmas week (Dec 24–31), Thanksgiving week, Easter week |
| `is_bridge_day` | Day immediately before/after a federal holiday (e.g., day after Christmas) |

The holiday-week feature was motivated by the CAISO divergence analysis: CAISO's own 7-day forecast showed materially lower loads during Easter/spring break week, which our initial model (using only federal holidays) was not capturing.

### 3.4 Temperature Lag Features

| Feature | Description | Motivation |
|---------|-------------|-----------|
| `lag1_f` | Yesterday's pop-weighted avg temp | Thermal inertia: buildings slow to respond to temperature change |
| `lag2_f` | 2 days ago avg temp | Further inertia |
| `lag7_f` | Same weekday last week avg temp | Captures weekly thermal regime without confounding weekday effects |
| `hdd_lag1`, `cdd_lag1` | HDD/CDD from T-1 | Degree-day inertia |
| `hdd_lag7`, `cdd_lag7` | HDD/CDD from T-7 | Same-weekday degree-day baseline |
| `rolling7_avg_f` | 7-day trailing average temp | Heat wave / cold snap persistence: populations acclimatize and demand shifts |

The **7-day rolling average** is the single most impactful new feature. A region that has been hot for a week sees materially higher A/C demand than one experiencing its first hot day at the same temperature — because cooling systems are running longer cycles, and behavioral responses (leaving doors open, sleeping patterns) amplify peak demand.

### 3.5 Complete Feature Vector

| # | Feature | Type |
|---|---------|------|
| 1–3 | `hdd_avg`, `cdd_avg`, `hdd×cdd` | Temperature × HDD/CDD |
| 4–5 | `hdd_hi`, `cdd_hi` | High-temp degree days |
| 6–7 | `hdd_lo`, `cdd_lo` | Low-temp degree days |
| 8–9 | `sin_doy`, `cos_doy` | Seasonality |
| 10–16 | `dow_0` … `dow_6` | Day-of-week one-hot |
| 17 | `is_holiday` | Federal holiday flag |
| 18–19 | `lag1_f`, `lag2_f` | T-1, T-2 temperature |
| 20–21 | `hdd_lag1`, `cdd_lag1` | T-1 HDD/CDD |
| 22 | `is_holiday_week` | Holiday week flag |
| 23 | `is_bridge_day` | Bridge day flag |
| 24 | `lag7_f` | T-7 temperature |
| 25–26 | `hdd_lag7`, `cdd_lag7` | T-7 HDD/CDD |
| 27 | `rolling7_avg_f` | 7-day rolling average temp |

**Total: 27 features**

---

## 4. Model Architecture

### 4.1 Algorithm

**XGBoost Gradient Boosted Regression Trees** (`XGBRegressor`). XGBoost was chosen because:
- Handles non-linear temperature-load relationships (summer A/C kink at ~75°F, winter heating kink at ~55°F)
- Naturally handles feature interactions (e.g., hot weekday vs. hot weekend have very different load profiles)
- Robust to outliers in load data (outages, data reporting gaps)
- No need to hand-specify interaction terms or polynomial features

### 4.2 Hyperparameters

| Parameter | Value | Rationale |
|-----------|-------|-----------|
| `n_estimators` | 500 | Sufficient depth without overfitting on 731 obs |
| `max_depth` | 5 | Allows up to 5-way feature interactions |
| `learning_rate` | 0.04 | Conservative shrinkage, pairs with 500 trees |
| `subsample` | 0.8 | Row subsampling reduces variance |
| `colsample_bytree` | 0.8 | Column subsampling reduces correlation |
| `random_state` | 42 | Reproducibility |

### 4.3 Train/Test Split

Chronological split (no shuffling — critical for time series):
- **Training**: first 80% of observations (≈585 days)
- **Test**: last 20% (≈146 days), approximately Oct 2025 – Mar 2026

This forward-only split prevents data leakage from future observations into training.

### 4.4 Uncertainty Quantification

The model produces a single point forecast. Uncertainty bounds represent temperature forecast uncertainty:

For each forecast day, a temperature spread $\sigma_T$ is estimated from GFS ensemble spread (when available) or a fallback of 3°F. Load bounds are computed by perturbing all temperature features by ±1.645σ (90% interval) and running the model:

$$\hat{L}_{5\%} = f(\mathbf{x}|T - 1.645\sigma_T), \quad \hat{L}_{95\%} = f(\mathbf{x}|T + 1.645\sigma_T)$$

---

## 5. Results

### 5.1 Backtest Performance

**Training period: March 2024 – October 2025**
**Test period: November 2025 – March 2026**

| Metric | PJM | CAISO |
|--------|-----|-------|
| Training obs | 585 days | 585 days |
| Test obs | 146 days | 146 days |
| Test RMSE | 476 MW | 97 MW |
| Test MAE | 350 MW | 76 MW |
| Test MAPE | **0.4%** | **0.3%** |

### 5.2 Comparison Against Baselines

| Model | PJM MAPE | CAISO MAPE |
|-------|----------|-----------|
| Naive (avg temp only) | ~4–5% | ~4–5% |
| v1 (avg+hi/lo, holidays, lag1/2) | 3.7% | 3.0% |
| **v2 (+ lag7, rolling7, holiday week)** | **0.4%** | **0.3%** |

The addition of the lag-7 and 7-day rolling average features reduced error by ~90% from v1.

### 5.3 Worst-Case Months (Test Period)

**PJM** — worst months by MAPE (all < 1%):
- March 2026: 0.6%
- October 2024: 0.5%
- November 2024: 0.5%

**CAISO** — worst months by MAPE:
- April 2024: 0.5%
- April 2025: 0.5%
- November 2024: 0.4%

Spring months (April) show slightly higher error for CAISO, consistent with the difficulty of modeling shoulder-season demand during spring break and Easter week — periods with strong behavioral load reduction not fully captured by temperature alone.

### 5.4 Comparison Against Official Forecasts

**PJM vs. EIA Day-Ahead:**
The EIA day-ahead (`type=DF`) provides 1–2 day ahead demand forecasts for PJM. Our 15-day GFS-based forecast closely tracks the day-ahead benchmark on the near-term horizon (days 1–2), diverging by 3–6% at 10–15 day lead times as GFS temperature skill degrades.

**CAISO vs. OASIS 7-Day:**
CAISO publishes its own 7-day system forecast via the OASIS API (`SLD_FCST/7DA`). Our model shows larger divergence (~5–10%) from the CAISO official forecast, particularly during:
- **Easter / spring break week**: CAISO's model incorporates school calendar and demand response program data not available to us
- **Heat wave onset**: CAISO has access to demand response curtailment schedules
- **Weekend vs. weekday transitions**: CAISO likely uses more granular occupancy models

The CAISO/EIA actual demand (historical) matches OASIS actuals within 1–2%, confirming that the divergence is in the *forward forecasts*, not a scale calibration issue.

---

## 6. System Architecture

```
┌─────────────────────────────────────────────────────────┐
│                    DATA INGESTION                        │
│  EIA API (actual + day-ahead load)                       │
│  Open-Meteo ERA5 (historical reanalysis temperatures)    │
│  Open-Meteo GFS (15-day NWP forecast temperatures)       │
│  PJM DataMiner API (PJM 7-day official forecast)         │
│  CAISO OASIS API (CAISO 7-day official forecast)         │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                 FEATURE ENGINEERING                      │
│  Population-weighted temp indices (12 nodes/ISO)         │
│  HDD/CDD (avg/hi/lo × current/lag1/lag7)                 │
│  Day-of-week, holiday flags, rolling averages            │
│  27 features total (see Section 3.5)                     │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│                  XGBoost MODEL (×2)                      │
│  PJM: trained on 585 days, test MAPE 0.4%               │
│  CAISO: trained on 585 days, test MAPE 0.3%             │
│  Uncertainty: ±1.645σ GFS temperature perturbation       │
└───────────────────────────┬─────────────────────────────┘
                            │
┌───────────────────────────▼─────────────────────────────┐
│              REAL-TIME DASHBOARD (Plotly Dash)           │
│  ISO selector (PJM / CAISO)                              │
│  Summary cards: today GW, 15-day peak, avg, 14-day MAPE │
│  Forecast chart: hindcast + 15-day forecast (one trace)  │
│    └─ linear date axis, uncertainty ribbon, benchmarks   │
│  Backtest section: 2-year predicted vs actual time series│
│    └─ train/test split marker, monthly MAPE bar chart    │
│  Methodology note (data sources, features, accuracy)     │
└─────────────────────────────────────────────────────────┘
```

### 6.1 Dashboard Design

The dashboard is a single-page Plotly Dash application with two callbacks:

1. **`load_forecast_data(iso, n_clicks)`** — triggered by ISO selector or Refresh button. Fetches GFS temperatures (12 locations, parallel), ERA5 hindcast temperatures (last 16 days), EIA actuals, and official benchmark forecasts. All results are serialised to a `dcc.Store`. Disk cache (3-hour TTL) avoids redundant API calls.

2. **`render(data, iso)`** — pure rendering callback, produces all six outputs (summary cards, subtitle, forecast chart, backtest cards, backtest time series, backtest monthly MAPE bar) from the cached store data.

### 6.2 Forecast Chart

The chart uses a **continuous linear date axis** (`xaxis type="date"`) so all days — historical actuals, ERA5 hindcast, and GFS forward forecast — are plotted on an evenly-spaced time axis with no categorical gaps.

**Trace composition:**
- **Actual (EIA)** — grey, last 14 days of reported actuals
- **Our Model** — single orange trace combining ERA5 hindcast (past) and GFS forecast (future), joined by a `None` break point with `connectgaps=False`. Marker colour encodes load level (green = near-average, amber = elevated, red = very high, blue = below average)
- **Uncertainty ribbon** — 90% interval from temperature perturbation (orange, 10% opacity)
- **PJM Official 7-Day / CAISO OASIS 7-Day** — blue/green dashed overlay for benchmark comparison
- **EIA Day-Ahead** — teal diamond markers for near-term comparison

### 6.3 Hindcast Construction

ERA5 reanalysis data is available with a 5-day lag. For each dashboard load, the last 16 days of ERA5 temperatures are fetched (per ISO, per location), population-weighted, and fed through the same XGBoost model that generates the forward forecast. This produces a continuous model trace from two weeks ago through today, allowing direct visual comparison against EIA-reported actuals before the forecast horizon begins.

The hindcast dictionary is merged with the backtest record (`{**backtest, **hindcast}`), with fresh ERA5 values taking priority over the pre-computed training-period backtest.

### 6.4 Deployment

The application is designed for cloud deployment with minimal configuration:
- `PORT` environment variable controls the listen port (defaults to 8050)
- Model `.pkl` files are committed alongside the source code (< 2 MB each)
- A `Procfile` (`web: python dashboard.py`) and `requirements.txt` are sufficient for deployment on Render.com, Railway, or Hugging Face Spaces
- No database or external state required — all data is fetched live or read from JSON caches in the `api_cache/` directory

---

## 7. Known Limitations and Failure Modes

### 7.1 Training Data Volume
Two years of training data (731 observations per ISO) is sufficient for this feature set but limits the model's ability to learn rare extreme events. A prolonged heat dome or polar vortex event represented only once in training may be under-predicted.

### 7.2 Calendar Features Incompleteness
Our model includes federal US holidays, Christmas/Thanksgiving/Easter week, and bridge days. Not included:
- School district calendars (highly correlated with commercial A/C demand)
- State and local holidays (varies by ISO region)
- Major sporting events (Super Bowl, playoff games affect evening residential load)
- Demand response program activations (CAISO, PJM both run DR programs)

### 7.3 Temperature-Only Feature Space
The model is deliberately temperature-centric. Features not included:
- **Economic activity**: industrial output, GDP growth affect baseline load independently of weather
- **EV charging load**: rapid growth in California is structurally shifting the load curve
- **Solar generation**: CAISO "duck curve" — net load is increasingly different from gross load as rooftop solar grows. Our model targets gross load (what EIA reports).
- **Fuel prices**: extreme natural gas prices can trigger fuel-switching and demand reduction

### 7.4 GFS Skill Degradation
GFS temperature forecasts lose meaningful skill beyond day 7–8. Our 15-day forecasts in days 8–15 are based on weather patterns with high uncertainty. Users should treat day 8+ forecasts as climatological guidance, not operational forecasts.

### 7.5 Static Model
The model is a static snapshot trained on 2024–2026 data. It does not update continuously. Structural shifts in the load-temperature relationship (new EV adoption, building efficiency improvements, data center growth) require periodic retraining.

---

## 8. Future Improvements

### 8.1 Expand Training Data (High Priority)
Extend from 2 years to 5 years of training data. ERA5 is available from 1940 onward; EIA data is available from 2015. More data would:
- Improve calibration for extreme weather events
- Reduce sensitivity to the specific 2024–2026 period
- Enable better seasonal cross-validation

### 8.2 Demand Response and EV Features
Integrate CAISO and PJM demand response event calendars. Add EV adoption proxy (state-level EV registration × charging profile) as a trend feature to capture the structural load growth in California.

### 8.3 Net Load Forecasting for CAISO
CAISO's "duck curve" makes gross load increasingly disconnected from operational need. Integrating EIA-reported solar/wind generation would let us forecast *net load* (gross load minus variable renewables), which is what operators actually dispatch against.

### 8.4 GFS Ensemble Uncertainty
Replace the scalar spread approximation with actual GFS ensemble (GEFS) member temperatures. This would give date-specific, spatially resolved uncertainty — reducing uncertainty bands in settled weather, widening them ahead of frontal passages.

### 8.5 ~~PJM Official 7-Day Forecast~~ *(Completed)*
PJM DataMiner 7-day forecast is now integrated via the PJM API Portal (`/load_frcstd_7_day`, `RTO_COMBINED`). It appears as a dashed blue overlay in the dashboard alongside the CAISO OASIS equivalent, enabling direct apples-to-apples comparison for both ISOs.

### 8.6 Hour-of-Day Resolution
The current model operates on daily means. An hourly model would capture peak-hour demand (important for resource adequacy) and the morning/evening ramp rates that stress the grid. This requires hourly temperature forecasts and a 24× larger target variable space.

### 8.7 Transfer Learning Across ISOs
The model architecture is identical for PJM and CAISO. A multi-task XGBoost or neural network trained jointly on both ISOs (with ISO as a categorical feature) could share information about temperature-load physics while learning ISO-specific residuals — likely improving performance on seasonal extremes where one ISO has more training examples than the other.

---

## 9. Conclusion

This system demonstrates that a well-engineered 27-feature XGBoost model, trained on 2 years of EIA load data paired with ERA5 reanalysis temperatures, can achieve sub-1% MAPE on held-out data for both PJM and CAISO. The key insight is that the **7-day rolling average temperature** and **same-weekday-last-week temperature** are far more predictive than the current day's temperature alone — because they capture thermal inertia (building heat storage), population acclimatization, and the weekly structural pattern in grid load.

The biggest remaining gap versus official ISO forecasts is the incorporation of demand-side behavioral effects (school calendars, demand response activations) and the structural shift in net load due to distributed solar generation — both areas identified as priority improvements.

---

## References

- EIA Open Data API: https://www.eia.gov/opendata/
- Open-Meteo Historical Weather API (ERA5): https://open-meteo.com/en/docs/historical-weather-api
- Open-Meteo Forecast API (GFS/ECMWF/GraphCast): https://open-meteo.com/en/docs
- CAISO OASIS API: http://oasis.caiso.com/oasisapi/
- US Holidays library: https://pypi.org/project/holidays/
- XGBoost: Chen & Guestrin (2016), "XGBoost: A Scalable Tree Boosting System"
- Python-dateutil easter: https://dateutil.readthedocs.io/
