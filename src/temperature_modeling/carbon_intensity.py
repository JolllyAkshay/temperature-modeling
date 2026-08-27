"""
Real-time carbon intensity per ISO from EIA v2 fuel-type data.

Returns lbs CO2/MWh for the most recent hour plus a fuel-mix breakdown.
"""

import os
import time
import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, Optional

import requests

log = logging.getLogger(__name__)

_EIA_KEY = os.environ.get("EIA_API_KEY", "")
_EIA_FUEL_URL = "https://api.eia.gov/v2/electricity/rto/fuel-type-data/data/"

# lbs CO2 / MWh — EPA eGRID 2022 national averages
_CO2_LBS_PER_MWH: Dict[str, float] = {
    "NG":  910.0,   # natural gas
    "COL": 2230.0,  # coal
    "OIL": 1670.0,  # petroleum
    "OTH": 1100.0,  # other (biomass, geothermal, misc)
    "NUC": 0.0,     # nuclear
    "WND": 0.0,     # wind
    "SUN": 0.0,     # solar
    "WAT": 0.0,     # hydro
    "PS":  0.0,     # pumped storage
    "GEO": 0.0,     # geothermal (some ISOs split this out)
}

_FUEL_LABELS: Dict[str, str] = {
    "NG":  "Natural Gas",
    "COL": "Coal",
    "OIL": "Oil",
    "NUC": "Nuclear",
    "WND": "Wind",
    "SUN": "Solar",
    "WAT": "Hydro",
    "PS":  "Pumped Storage",
    "OTH": "Other",
    "GEO": "Geothermal",
}

# EIA respondent codes
_ISO_RESPONDENT: Dict[str, str] = {
    "pjm":   "PJM",
    "caiso": "CISO",
    "ercot": "ERCO",
    "miso":  "MISO",
    "nyiso": "NYIS",
    "isone": "ISNE",
    "spp":   "SWPP",
}

# Simple in-process cache: {iso: (timestamp, result_dict)}
_CACHE: Dict[str, tuple] = {}
_CACHE_TTL = 3600  # 1 hour — fuel mix changes slowly enough

# Separate cache for the (larger) history payload: {"iso:hours": (timestamp, result_dict)}
_HISTORY_CACHE: Dict[str, tuple] = {}
_HISTORY_CACHE_TTL = 1800  # 30 min


def fetch_carbon_intensity(iso: str, session: Optional[requests.Session] = None) -> dict:
    """
    Fetch the most recent hour's fuel mix for `iso` and return:
    {
        "lbs_co2_per_mwh": float,
        "fuel_mix": {"Natural Gas": float_mw, "Solar": float_mw, ...},
        "total_mw": float,
        "period": "2026-07-27T15",   # UTC hour string
        "clean_pct": float,          # % from zero-carbon sources
    }
    Returns {} on failure.
    """
    if not _EIA_KEY:
        log.warning("EIA_API_KEY not set — carbon intensity unavailable")
        return {}

    cached_ts, cached_val = _CACHE.get(iso, (0, {}))
    if time.time() - cached_ts < _CACHE_TTL and cached_val:
        return cached_val

    respondent = _ISO_RESPONDENT.get(iso)
    if not respondent:
        return {}

    # EIA fuel-type data has ~12-24h lag — look back 48h to always find data
    now_utc   = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=48)
    start_str = start_utc.strftime("%Y-%m-%dT%H")
    end_str   = now_utc.strftime("%Y-%m-%dT%H")

    params = {
        "api_key":              _EIA_KEY,
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": respondent,
        "start":                start_str,
        "end":                  end_str,
        "length":               200,
        "sort[0][column]":      "period",
        "sort[0][direction]":   "desc",
    }

    sess = session or requests.Session()
    try:
        r = sess.get(_EIA_FUEL_URL, params=params, timeout=20)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
    except Exception as exc:
        log.warning("Carbon intensity fetch failed for %s: %s", iso.upper(), exc)
        return {}

    if not rows:
        log.warning("No fuel-type data returned for %s", iso.upper())
        return {}

    # Use the most recent period that has data
    latest_period = rows[0]["period"]  # already sorted desc
    period_rows   = [row for row in rows if row["period"] == latest_period]

    fuel_mw: Dict[str, float] = {}
    for row in period_rows:
        ftype = row.get("fueltype", "OTH")
        val   = row.get("value")
        if val is None:
            continue
        fuel_mw[ftype] = float(val)

    total_mw = sum(fuel_mw.values())
    if total_mw <= 0:
        return {}

    # Weighted lbs CO2/MWh
    co2_total = sum(
        mw * _CO2_LBS_PER_MWH.get(ftype, _CO2_LBS_PER_MWH["OTH"])
        for ftype, mw in fuel_mw.items()
    )
    lbs_per_mwh = co2_total / total_mw

    clean_mw = sum(
        mw for ftype, mw in fuel_mw.items()
        if _CO2_LBS_PER_MWH.get(ftype, 1) == 0.0
    )
    clean_pct = 100.0 * clean_mw / total_mw if total_mw else 0.0

    # Build labelled fuel mix (human-readable names, sorted descending by MW)
    labelled: Dict[str, float] = {}
    for ftype, mw in sorted(fuel_mw.items(), key=lambda x: -x[1]):
        label = _FUEL_LABELS.get(ftype, ftype)
        labelled[label] = round(mw, 0)

    result = {
        "lbs_co2_per_mwh": round(lbs_per_mwh, 1),
        "fuel_mix":         labelled,
        "total_mw":         round(total_mw, 0),
        "period":           latest_period,
        "clean_pct":        round(clean_pct, 1),
    }
    _CACHE[iso] = (time.time(), result)
    return result


def fetch_carbon_intensity_history(iso: str, hours: int = 24, session: Optional[requests.Session] = None) -> dict:
    """
    Fetch a trailing fuel-mix / clean-energy-% trend for `iso`, for charting
    how the mix has moved over the last `hours` hours (distinct from
    fetch_carbon_intensity's single most-recent-hour snapshot).

    Returns:
        {
            "available": True,
            "periods": ["2026-08-25T00", ...],   # ascending, UTC hour strings
            "clean_pct": [42.1, 45.0, ...],       # % zero-carbon, aligned to periods
            "lbs_co2_per_mwh": [810.2, ...],
            "fuel_mix_mw": {"Natural Gas": [1200.0, 1300.0, ...], ...},  # aligned to periods
        }
        or {"available": False, "reason": str}
    """
    if not _EIA_KEY:
        return {"available": False, "reason": "EIA_API_KEY not set."}

    cache_key = f"{iso}:{hours}"
    cached_ts, cached_val = _HISTORY_CACHE.get(cache_key, (0, None))
    if cached_val and time.time() - cached_ts < _HISTORY_CACHE_TTL:
        return cached_val

    respondent = _ISO_RESPONDENT.get(iso)
    if not respondent:
        return {"available": False, "reason": f"Unsupported ISO: {iso}"}

    # EIA fuel-type data lags ~12-24h; look back hours + 36h buffer so
    # `hours` worth of actual data is always covered.
    now_utc = datetime.now(timezone.utc)
    start_utc = now_utc - timedelta(hours=hours + 36)
    start_str = start_utc.strftime("%Y-%m-%dT%H")
    end_str = now_utc.strftime("%Y-%m-%dT%H")

    params = {
        "api_key": _EIA_KEY,
        "frequency": "hourly",
        "data[0]": "value",
        "facets[respondent][]": respondent,
        "start": start_str,
        "end": end_str,
        "length": 5000,
        "sort[0][column]": "period",
        "sort[0][direction]": "desc",
    }

    sess = session or requests.Session()
    try:
        r = sess.get(_EIA_FUEL_URL, params=params, timeout=25)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
    except Exception as exc:
        log.warning("Carbon intensity history fetch failed for %s: %s", iso.upper(), exc)
        return {"available": False, "reason": "Fuel mix history temporarily unavailable."}

    if not rows:
        return {"available": False, "reason": "No fuel mix history returned."}

    by_period: Dict[str, Dict[str, float]] = {}
    for row in rows:
        period = row["period"]
        ftype = row.get("fueltype", "OTH")
        val = row.get("value")
        if val is None:
            continue
        by_period.setdefault(period, {})[ftype] = float(val)

    periods_sorted = sorted(by_period.keys())[-hours:]
    if not periods_sorted:
        return {"available": False, "reason": "No fuel mix history returned."}

    all_fuel_types = sorted({ft for p in periods_sorted for ft in by_period[p].keys()})
    fuel_mix_mw: Dict[str, list] = {_FUEL_LABELS.get(ft, ft): [] for ft in all_fuel_types}
    clean_pct_list = []
    lbs_co2_list = []

    for p in periods_sorted:
        fuel_mw = by_period[p]
        total_mw = sum(fuel_mw.values())
        for ft in all_fuel_types:
            fuel_mix_mw[_FUEL_LABELS.get(ft, ft)].append(round(fuel_mw.get(ft, 0.0), 0))
        if total_mw > 0:
            co2_total = sum(
                mw * _CO2_LBS_PER_MWH.get(ft, _CO2_LBS_PER_MWH["OTH"])
                for ft, mw in fuel_mw.items()
            )
            clean_mw = sum(mw for ft, mw in fuel_mw.items() if _CO2_LBS_PER_MWH.get(ft, 1) == 0.0)
            clean_pct_list.append(round(100.0 * clean_mw / total_mw, 1))
            lbs_co2_list.append(round(co2_total / total_mw, 1))
        else:
            clean_pct_list.append(None)
            lbs_co2_list.append(None)

    result = {
        "available": True,
        "periods": periods_sorted,
        "clean_pct": clean_pct_list,
        "lbs_co2_per_mwh": lbs_co2_list,
        "fuel_mix_mw": fuel_mix_mw,
    }
    _HISTORY_CACHE[cache_key] = (time.time(), result)
    return result
