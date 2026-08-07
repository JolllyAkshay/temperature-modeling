"""
Forecast verification: persist daily forecasts and compare against EIA actuals.

Usage
-----
At forecast time:
    from temperature_modeling.verification import record_forecast
    record_forecast(iso, load_list)   # load_list = dcc.Store "load" payload

On dashboard load:
    from temperature_modeling.verification import load_verification_stats
    stats = load_verification_stats(iso)
    # stats["mape_7d"], stats["mape_30d"], stats["bias_mw"], stats["records"]

The verification log lives in api_cache/<iso>_verification.jsonl — one JSON line
per forecast issue date. Actuals are fetched from EIA v2 /electricity/rto/region-data.
"""

import json
import logging
import os
from datetime import date, timedelta
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "api_cache"

_EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"

_ISO_EIA_REGION = {
    "pjm":   "PJM",
    "caiso": "CAL",
    "ercot": "TEX",
    "miso":  "MISO",
    "nyiso": "NY",
    "isone": "NE",
    "spp":   "SW",
}


# ---------------------------------------------------------------------------
# Write side: record a forecast when it is generated
# ---------------------------------------------------------------------------

def record_forecast(iso: str, load_list: list) -> None:
    """
    Persist today's load forecast to the verification log.
    Skips silently if the ISO has already been recorded today.
    """
    if not load_list:
        return

    log_path = _CACHE_DIR / f"{iso}_verification.jsonl"
    today = date.today().isoformat()

    # Avoid duplicating today's entry
    if log_path.exists():
        try:
            last = log_path.read_text().strip().splitlines()[-1]
            if json.loads(last).get("issue_date") == today:
                return
        except Exception:
            pass

    record = {
        "issue_date": today,
        "forecasts": [
            {"date": d["date"], "mean_mw": d["mean_load_gw"] * 1000}
            for d in load_list
        ],
    }
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        with log_path.open("a") as f:
            f.write(json.dumps(record) + "\n")
        log.debug("Recorded %s forecast for %s (%d days)", iso.upper(), today, len(load_list))
    except OSError:
        log.warning("Could not write verification log for %s", iso.upper())


# ---------------------------------------------------------------------------
# EIA actuals fetch
# ---------------------------------------------------------------------------

def _fetch_eia_actuals(iso: str, start: str, end: str) -> dict:
    """
    Return {date_str: actual_load_mw} from EIA for the given date range.
    Uses hourly data averaged to daily.
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        return {}

    region = _ISO_EIA_REGION.get(iso)
    if not region:
        return {}

    params = {
        "api_key":              api_key,
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": region,
        "facets[type][]":       "D",
        "start":                start,
        "end":                  end,
        "length":               2000,
        "offset":               0,
    }
    try:
        r = requests.get(_EIA_URL, params=params, timeout=30)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
    except Exception:
        log.warning("EIA actuals fetch failed for %s", iso.upper())
        return {}

    # Aggregate hourly MW to daily average
    daily: dict = {}
    for row in rows:
        dt = row.get("period", "")[:10]
        val = row.get("value")
        if dt and val is not None:
            daily.setdefault(dt, []).append(float(val))

    return {dt: sum(vals) / len(vals) for dt, vals in daily.items()}


# ---------------------------------------------------------------------------
# Read side: compute verification statistics
# ---------------------------------------------------------------------------

def load_verification_stats(iso: str) -> dict:
    """
    Compare stored day-ahead forecasts against EIA actuals and return summary stats.

    Returns dict with keys:
        records    list of {date, forecast_mw, actual_mw, error_pct}
        mape_7d    float | None
        mape_30d   float | None
        bias_mw    float | None  (positive = over-forecast)
        n_verified int
    """
    empty = {"records": [], "mape_7d": None, "mape_30d": None,
             "bias_mw": None, "n_verified": 0}

    log_path = _CACHE_DIR / f"{iso}_verification.jsonl"
    if not log_path.exists():
        return empty

    stored = []
    try:
        for line in log_path.read_text().splitlines():
            if line.strip():
                stored.append(json.loads(line))
    except Exception:
        return empty

    if not stored:
        return empty

    all_dates = sorted(
        f["date"]
        for rec in stored
        for f in rec["forecasts"]
        if f["date"] < date.today().isoformat()
    )
    if not all_dates:
        return empty

    actuals = _fetch_eia_actuals(iso, all_dates[0], all_dates[-1])
    if not actuals:
        return empty

    records = []
    for rec in stored:
        issue = rec["issue_date"]
        for f in rec["forecasts"]:
            fdate = f["date"]
            if fdate <= issue or fdate not in actuals:
                continue
            lead = (date.fromisoformat(fdate) - date.fromisoformat(issue)).days
            if lead != 1:
                continue
            actual = actuals[fdate]
            if not actual:
                continue
            error_pct = (f["mean_mw"] - actual) / actual * 100
            records.append({
                "date":        fdate,
                "forecast_mw": round(f["mean_mw"]),
                "actual_mw":   round(actual),
                "error_pct":   round(error_pct, 2),
            })

    if not records:
        return {**empty, "records": records}

    records.sort(key=lambda r: r["date"])
    biases = [r["forecast_mw"] - r["actual_mw"] for r in records]

    cutoff_7  = (date.today() - timedelta(days=7)).isoformat()
    cutoff_30 = (date.today() - timedelta(days=30)).isoformat()

    def _mape(subset):
        if not subset:
            return None
        return round(sum(abs(r["error_pct"]) for r in subset) / len(subset), 2)

    return {
        "records":    records,
        "mape_7d":    _mape([r for r in records if r["date"] >= cutoff_7]),
        "mape_30d":   _mape([r for r in records if r["date"] >= cutoff_30]),
        "bias_mw":    round(sum(biases) / len(biases)) if biases else None,
        "n_verified": len(records),
    }
