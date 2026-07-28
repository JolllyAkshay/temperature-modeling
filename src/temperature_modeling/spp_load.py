"""
SPP electricity load forecasting from temperature projections.
Mirrors caiso_load.py but targets the SWPP EIA respondent (Southwest Power Pool).
"""

import os
import time
from datetime import date, timedelta
from typing import Dict, List, Tuple

import requests

from ._era5 import _CACHE_DIR, _cache_load, _cache_save
from .spp import SPP_LOAD_LOCATIONS
from .models import LoadForecast, LoadObservation
from .pjm_load import (
    _build_features,
    _is_us_holiday,
    compute_hdd_cdd,
    evaluate_load_model,
    fetch_era5_daily_hi_lo,
    load_load_model,
    run_load_backtest,
    save_load_model,
    LoadCorrectionModel,
    _EIA_KEY,
    _EIA_URL as _EIA_BASE,
)

_SPP_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "spp_load_model.pkl"
)

_EIA_RESPONDENT = "SWPP"


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def fetch_spp_load_daily(start: date, end: date, session: requests.Session) -> Dict[date, float]:
    """Fetch SPP hourly demand from EIA and aggregate to daily mean (MW)."""
    all_rows: list = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key":              _EIA_KEY,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": _EIA_RESPONDENT,
            "facets[type][]":       "D",
            "start":                start.strftime("%Y-%m-%dT00"),
            "end":                  end.strftime("%Y-%m-%dT23"),
            "length":               page_size,
            "offset":               offset,
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
        }
        cached = _cache_load(_EIA_BASE, params)
        if cached is not None:
            data = cached
        else:
            for attempt in range(5):
                try:
                    r = session.get(_EIA_BASE, params=params, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(f"EIA SPP fetch failed: {exc}") from exc
                    time.sleep(10 * (attempt + 1))
            _cache_save(_EIA_BASE, params, data)

        rows = data.get("response", {}).get("data", [])
        all_rows.extend(rows)
        total = int(data.get("response", {}).get("total", 0))
        offset += page_size
        if offset >= total:
            break

    daily: Dict[date, list] = {}
    for row in all_rows:
        val = row.get("value")
        if val is None:
            continue
        try:
            d = date.fromisoformat(row.get("period", "")[:10])
            daily.setdefault(d, []).append(float(val))
        except (ValueError, TypeError):
            continue

    return {d: sum(v) / len(v) for d, v in daily.items() if v}


def weighted_avg_temp_f_spp(temps_c_by_label: Dict[str, float]) -> float:
    total_wt = total_w = 0.0
    for loc in SPP_LOAD_LOCATIONS:
        label = loc["label"]
        if label not in temps_c_by_label:
            continue
        w = loc["weight"]
        total_wt += w * _c_to_f(temps_c_by_label[label])
        total_w  += w
    if total_w == 0:
        raise ValueError("No SPP location temperatures available")
    return total_wt / total_w


def build_spp_training_data(
    load_daily: Dict[date, float],
    era5_avg_by_label: Dict[str, Dict[date, float]],
    era5_hi_by_label:  Dict[str, Dict[date, Tuple[float, float]]],
) -> List[LoadObservation]:
    avg_f_by_date:      Dict[date, float] = {}
    hi_f_by_date:       Dict[date, float] = {}
    lo_f_by_date:       Dict[date, float] = {}
    apparent_hi_by_date: Dict[date, float] = {}

    for d in sorted(load_daily):
        avg_c = {
            loc["label"]: era5_avg_by_label.get(loc["label"], {}).get(d)
            for loc in SPP_LOAD_LOCATIONS
            if era5_avg_by_label.get(loc["label"], {}).get(d) is not None
        }
        if not avg_c:
            continue
        avg_f_by_date[d] = weighted_avg_temp_f_spp(avg_c)

        hi_c, lo_c, ap_c = {}, {}, {}
        for loc in SPP_LOAD_LOCATIONS:
            pair = era5_hi_by_label.get(loc["label"], {}).get(d)
            if pair is not None:
                hi_c[loc["label"]] = pair[0]
                lo_c[loc["label"]] = pair[1]
                if len(pair) > 2 and pair[2] is not None:
                    ap_c[loc["label"]] = pair[2]
        if hi_c:
            hi_f_by_date[d] = weighted_avg_temp_f_spp(hi_c)
        if lo_c:
            lo_f_by_date[d] = weighted_avg_temp_f_spp(lo_c)
        if ap_c:
            apparent_hi_by_date[d] = weighted_avg_temp_f_spp(ap_c)

    obs: List[LoadObservation] = []
    for d in sorted(avg_f_by_date):
        if d not in load_daily:
            continue
        avg_f = avg_f_by_date[d]
        hdd, cdd = compute_hdd_cdd(avg_f)
        roll_vals = [avg_f_by_date.get(d - timedelta(days=k)) for k in range(7)]
        obs.append(LoadObservation(
            date=d,
            hdd=hdd, cdd=cdd,
            avg_temp_f=avg_f,
            hi_temp_f=hi_f_by_date.get(d, avg_f),
            lo_temp_f=lo_f_by_date.get(d, avg_f),
            actual_load_mw=load_daily[d],
            is_weekend=(d.weekday() >= 5),
            day_of_week=d.weekday(),
            is_holiday=_is_us_holiday(d),
            day_of_year=d.timetuple().tm_yday,
            temp_lag1_f=avg_f_by_date.get(d - timedelta(days=1)),
            temp_lag2_f=avg_f_by_date.get(d - timedelta(days=2)),
            temp_lag7_f=avg_f_by_date.get(d - timedelta(days=7)),
            rolling7_avg_f=(
                sum(v for v in roll_vals if v is not None) /
                len([v for v in roll_vals if v is not None])
                if any(v is not None for v in roll_vals) else None
            ),
            apparent_hi_f=_c_to_f(apparent_hi_by_date[d]) if d in apparent_hi_by_date else None,
        ))
    return obs


# ---------------------------------------------------------------------------
# Official comparison (actual demand + day-ahead forecast from EIA)
# ---------------------------------------------------------------------------

_SPP_COMPARISON_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "spp_comparison_cache.json"
)


def fetch_spp_official_comparison(
    session: requests.Session,
    lookback_days: int = 14,
) -> Dict[str, Dict]:
    """
    Fetch EIA actual demand (D) and day-ahead forecast (DF) for SPP (SWPP).
    Returns {"actual": {date_str: gw}, "da_fcst": {date_str: gw}}.
    """
    import json as _json

    today = date.today()
    if os.path.exists(_SPP_COMPARISON_CACHE):
        try:
            cached = _json.loads(open(_SPP_COMPARISON_CACHE).read())
            age_h = (time.time() - os.path.getmtime(_SPP_COMPARISON_CACHE)) / 3600
            if age_h < 6 and cached.get("actual") and today.isoformat() in str(cached):
                return cached
        except Exception:
            pass

    hist_start = today - timedelta(days=lookback_days)
    fwd_end    = today + timedelta(days=2)

    def _fetch_type(type_code: str, start: date, end: date) -> Dict[date, float]:
        all_rows: list = []
        offset = 0
        while True:
            params = {
                "api_key":              _EIA_KEY,
                "frequency":            "hourly",
                "data[0]":              "value",
                "facets[respondent][]": _EIA_RESPONDENT,
                "facets[type][]":       type_code,
                "start":                start.strftime("%Y-%m-%dT00"),
                "end":                  end.strftime("%Y-%m-%dT23"),
                "length":               5000,
                "offset":               offset,
                "sort[0][column]":      "period",
                "sort[0][direction]":   "asc",
            }
            try:
                r = session.get(_EIA_BASE, params=params, timeout=30)
                r.raise_for_status()
                data = r.json()
            except Exception:
                break
            rows  = data.get("response", {}).get("data", [])
            total = int(data.get("response", {}).get("total", 0))
            all_rows.extend(rows)
            offset += 5000
            if offset >= total:
                break
        daily: Dict[date, list] = {}
        for row in all_rows:
            val = row.get("value")
            if val is None:
                continue
            try:
                d = date.fromisoformat(row["period"][:10])
                daily.setdefault(d, []).append(float(val))
            except (ValueError, TypeError, KeyError):
                continue
        return {d: sum(v) / len(v) for d, v in daily.items() if v}

    actual  = _fetch_type("D",  hist_start, today)
    da_fcst = _fetch_type("DF", today,      fwd_end)

    result = {
        "actual":  {d.isoformat(): round(mw / 1000, 2) for d, mw in actual.items()},
        "da_fcst": {d.isoformat(): round(mw / 1000, 2) for d, mw in da_fcst.items()},
    }
    try:
        os.makedirs(os.path.dirname(_SPP_COMPARISON_CACHE), exist_ok=True)
        open(_SPP_COMPARISON_CACHE, "w").write(_json.dumps(result))
    except Exception:
        pass
    return result
