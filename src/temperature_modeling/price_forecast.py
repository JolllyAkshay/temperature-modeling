"""
Day-ahead electricity price fetching and simple regression-based forecast.

Data sources:
  - EIA v2 /electricity/rto/region-data  → historical load (MW)
  - EIA v2 /electricity/wholesale-markets/prices → historical day-ahead LMP ($/MWh)

Forecast approach:
  A simple load-price regression (log-linear) trained on the last 90 days of
  EIA actuals. Applies learned relationship to the 15-day load forecast to
  produce an indicative price forecast. NOT a substitute for a proper LMP model
  but gives directionally correct signals for high/low price days.

Public API
----------
fetch_price_history(iso, days=90) -> list[dict]
    [{date, load_mw, price_usd_mwh}]

forecast_prices(iso, load_forecast) -> list[dict]
    [{date, mean_load_mw, forecast_price, low_price, high_price}]
"""

import logging
import math
import os
from datetime import date, timedelta

import requests

log = logging.getLogger(__name__)

_EIA_PRICE_URL = "https://api.eia.gov/v2/electricity/wholesale-markets/prices/data/"
_EIA_LOAD_URL  = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

# EIA respondent codes for load and price series
_ISO_EIA = {
    "pjm":   {"load_region": "PJM",  "price_region": "PJM",  "price_type": "DA"},
    "caiso": {"load_region": "CAL",  "price_region": "MIDA", "price_type": "DA"},
    "ercot": {"load_region": "TEX",  "price_region": "TEX",  "price_type": "DA"},
    "miso":  {"load_region": "MISO", "price_region": "MIDW", "price_type": "DA"},
}


# ---------------------------------------------------------------------------
# EIA data fetching
# ---------------------------------------------------------------------------

def _eia_get(url: str, params: dict) -> list:
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        return []
    params["api_key"] = api_key
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("response", {}).get("data", [])
    except Exception:
        log.warning("EIA request failed: %s", url)
        return []


def fetch_price_history(iso: str, days: int = 90) -> list:
    """
    Return [{date, load_mw, price_usd_mwh}] for the last `days` days.
    Both load and price are daily averages from EIA hourly data.
    """
    cfg = _ISO_EIA.get(iso)
    if not cfg:
        return []

    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)
    start, end = start_dt.isoformat(), end_dt.isoformat()

    # Fetch hourly load
    load_rows = _eia_get(_EIA_LOAD_URL, {
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": cfg["load_region"],
        "facets[type][]":       "D",
        "start": start, "end": end, "length": 5000,
    })

    # Fetch hourly DA price
    price_rows = _eia_get(_EIA_PRICE_URL, {
        "frequency":             "hourly",
        "data[0]":               "price",
        "facets[respondent][]":  cfg["price_region"],
        "facets[type][]":        cfg["price_type"],
        "start": start, "end": end, "length": 5000,
    })

    # Aggregate to daily
    daily_load:  dict = {}
    daily_price: dict = {}

    for row in load_rows:
        d = row.get("period", "")[:10]
        v = row.get("value")
        if d and v is not None:
            daily_load.setdefault(d, []).append(float(v))

    for row in price_rows:
        d = row.get("period", "")[:10]
        v = row.get("price")
        if d and v is not None:
            try:
                daily_price.setdefault(d, []).append(float(v))
            except (TypeError, ValueError):
                pass

    result = []
    for d in sorted(set(daily_load) & set(daily_price)):
        load_vals  = daily_load[d]
        price_vals = daily_price[d]
        result.append({
            "date":          d,
            "load_mw":       round(sum(load_vals)  / len(load_vals)),
            "price_usd_mwh": round(sum(price_vals) / len(price_vals), 2),
        })

    log.info("%s: fetched %d days of load+price history from EIA", iso.upper(), len(result))
    return result


# ---------------------------------------------------------------------------
# Regression model (log-linear load → price)
# ---------------------------------------------------------------------------

def _fit_price_model(history: list) -> dict | None:
    """
    Fit log(price) = a * log(load) + b via OLS on the last 90 days.
    Returns {"a": float, "b": float, "rmse": float} or None if insufficient data.
    """
    pairs = [
        (math.log(max(r["load_mw"], 1)), math.log(max(r["price_usd_mwh"], 1)))
        for r in history
        if r["price_usd_mwh"] > 0 and r["load_mw"] > 0
    ]
    if len(pairs) < 14:
        log.warning("Insufficient price history for regression (%d pairs)", len(pairs))
        return None

    n    = len(pairs)
    sx   = sum(x for x, _ in pairs)
    sy   = sum(y for _, y in pairs)
    sxx  = sum(x * x for x, _ in pairs)
    sxy  = sum(x * y for x, y in pairs)
    denom = n * sxx - sx * sx
    if abs(denom) < 1e-10:
        return None

    a = (n * sxy - sx * sy) / denom
    b = (sy - a * sx) / n

    # RMSE in price space
    resid = [math.exp(a * x + b) - math.exp(y) for x, y in pairs]
    rmse  = math.sqrt(sum(r * r for r in resid) / len(resid))

    return {"a": a, "b": b, "rmse": round(rmse, 2)}


def forecast_prices(iso: str, load_forecast: list) -> list:
    """
    Forecast day-ahead prices from the load forecast.

    Parameters
    ----------
    iso           : ISO code
    load_forecast : list of {date, mean_load_gw, ...} from the load model store

    Returns
    -------
    list of {date, forecast_price, low_price, high_price, model_rmse}
    Empty list if EIA data is unavailable or regression fails.
    """
    history = fetch_price_history(iso, days=90)
    if not history:
        return []

    model = _fit_price_model(history)
    if not model:
        return []

    a, b, rmse = model["a"], model["b"], model["rmse"]

    results = []
    for day in load_forecast:
        load_mw = day.get("mean_load_gw", 0) * 1000
        if load_mw <= 0:
            continue
        log_load = math.log(max(load_mw, 1))
        price    = math.exp(a * log_load + b)
        results.append({
            "date":           day["date"],
            "forecast_price": round(price, 2),
            "low_price":      round(max(price - rmse, 0), 2),
            "high_price":     round(price + rmse, 2),
            "model_rmse":     rmse,
        })

    log.info("%s: price forecast complete — %d days, model RMSE $%.2f/MWh",
             iso.upper(), len(results), rmse)
    return results
