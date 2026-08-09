---
title: Grid Load & Price Dashboard
emoji: ⚡
colorFrom: indigo
colorTo: purple
sdk: docker
pinned: false
---

# US Electricity Market Forecasting Tool

A personal research project covering all 7 major US electricity markets. Combines short-term load forecasting with a 12-month forward price curve — the kind of view a power trader needs in one place.

**Live demo:** https://huggingface.co/spaces/JollyAkshay/grid-dashboard

---

## What it does

**Load forecasting (15-day horizon)**
- Separate XGBoost models trained per ISO on 2 years of hourly EIA demand data
- Features: HDD/CDD, apparent temperature, dewpoint, wind speed, day-of-week, holiday flags, load autocorrelation lags
- Uncertainty bands via Conformalized Quantile Regression (guaranteed ≥90% empirical coverage)
- Benchmarked against ISO-published day-ahead forecasts (PJM DataMiner, CAISO OASIS, ERCOT public reports)

**12-month forward price curve**
- OLS log-linear regression trained on 730 days of day-ahead LMP history per ISO
- Three weather scenarios per delivery month (cold / base / hot) using NOAA 30-year climate normals
- Peak / off-peak splits per month (NERC definition: Mon–Fri HE07–22)
- Gas price assumptions from EIA Short-Term Energy Outlook (Henry Hub, monthly)
- Outputs: monthly strip price, on-peak, off-peak, spark spreads (CCGT 7,000 BTU and CT 10,000 BTU), implied heat rates
- Winsorisation at 3× p95 before fitting to prevent scarcity spikes biasing seasonal coefficients

**ISO coverage**

| ISO | Load model | Price model | Price data source |
|-----|-----------|-------------|-------------------|
| NYISO | XGBoost | OLS | Monthly ZIP archives (DA LMP) |
| CAISO | XGBoost | OLS | OASIS API (rate-limited, 28-day chunks) |
| MISO | XGBoost | OLS | EIA API |
| ERCOT | XGBoost | OLS | NP4-190-CD via OAuth2 (HB_BUSAVG) |
| SPP | XGBoost | OLS | SPP Marketplace DA-LMP CSVs |
| PJM | XGBoost | Heuristic | Requires PJM DataMiner key |
| ISO-NE | XGBoost | Heuristic | Requires ISO-NE credentials |

---

## Structure

```
src/temperature_modeling/
    price_forecast.py     # ISO price fetchers, OLS model, winsorisation
    forward_curve.py      # 12-month curve builder, spark spreads, archive
    carbon_intensity.py   # Grid carbon intensity and fuel mix
    demand_response.py    # DR window detection
    *_load.py             # Per-ISO XGBoost load models
dashboard.py              # Plotly Dash front-end
api.py                    # FastAPI REST layer (separate from dashboard)
api_cache/                # Model weights (Git LFS) and JSON caches
```

---

## Running locally

```bash
pip install -e ".[dev]"

# Set environment variables
export EIA_API_KEY=...
export ERCOT_API_KEY=...
export ERCOT_USERNAME=...
export ERCOT_PASSWORD=...

python dashboard.py        # Dashboard on http://localhost:8050
uvicorn api:app --port 8001  # REST API on http://localhost:8001
```

API documentation available at `http://localhost:8001/docs`.

---

## Key endpoints

```
GET  /v1/forecast/{iso}           # 15-day load forecast
GET  /v1/prices/{iso}             # Day-ahead price forecast
GET  /v1/forward-curve/{iso}      # 12-month forward curve
GET  /v1/carbon/{iso}             # Live grid carbon intensity
GET  /v1/accuracy/{iso}           # Live forecast verification vs EIA actuals
POST /v1/refresh/{iso}            # Trigger background cache refresh
```

---

## Tests

```bash
pytest
```
