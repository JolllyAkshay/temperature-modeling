"""
MISO electricity load forecasting from temperature projections.
Mirrors caiso_load.py but targets the MISO EIA respondent.

Key differences:
  - MISO is historically coal-heavy; EPA eGRID 2023 MROW marginal: ~1,580 lbs CO2/MWh.
  - MISO Planning Resource Auction capacity prices ~$35/MW-day (zone-dependent).
  - Official 7-day forecast via MISO Market Information System (login required);
    falls back to EIA day-ahead comparison.
"""

import os
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import requests

from ._era5 import _CACHE_DIR, _cache_load, _cache_save
from .miso import MISO_LOAD_LOCATIONS
from .models import LoadForecast, LoadObservation
from .pjm_load import (
    LoadCorrectionModel,
    _build_features,
    _is_us_holiday,
    _obs_to_features,
    compute_hdd_cdd,
    evaluate_load_model,
    fetch_era5_daily_hi_lo,
    load_load_model,
    run_load_backtest,
    save_load_model,
    _EIA_KEY,
    _EIA_URL as _EIA_BASE,
)

_MISO_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "miso_load_model.pkl"
)

_MISO_COMPARISON_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "miso_comparison_cache.json"
)

_MISO_7DAY_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "miso_7day_cache.json"
)


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


# ---------------------------------------------------------------------------
# Population-weighted temperature (MISO locations)
# ---------------------------------------------------------------------------

def weighted_avg_temp_f_miso(temps_c_by_label: Dict[str, float]) -> float:
    total_wt = total_w = 0.0
    for loc in MISO_LOAD_LOCATIONS:
        label = loc["label"]
        if label not in temps_c_by_label:
            continue
        w = loc["weight"]
        total_wt += w * _c_to_f(temps_c_by_label[label])
        total_w  += w
    if total_w == 0:
        raise ValueError("No MISO location temperatures available")
    return total_wt / total_w


# ---------------------------------------------------------------------------
# EIA load fetch (MISO respondent)
# ---------------------------------------------------------------------------

def fetch_miso_load_daily(
    start: date,
    end: date,
    session: requests.Session,
) -> Dict[date, float]:
    """Fetch MISO hourly demand from EIA and aggregate to daily mean (MW)."""
    all_rows: list = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key":              _EIA_KEY,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "MISO",
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
                        raise RuntimeError(f"EIA MISO fetch failed: {exc}") from exc
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
        period = row.get("period", "")
        val    = row.get("value")
        if val is None:
            continue
        try:
            d = date.fromisoformat(period[:10])
            daily.setdefault(d, []).append(float(val))
        except (ValueError, TypeError):
            continue

    return {d: sum(vals) / len(vals) for d, vals in daily.items() if vals}


# ---------------------------------------------------------------------------
# Training data assembly
# ---------------------------------------------------------------------------

def build_miso_training_data(
    load_daily: Dict[date, float],
    era5_avg_by_label: Dict[str, Dict[date, float]],
    era5_hi_by_label:  Dict[str, Dict[date, Tuple[float, float]]],
) -> List[LoadObservation]:
    sorted_dates = sorted(load_daily.keys())
    avg_f_by_date:      Dict[date, float] = {}
    hi_f_by_date:       Dict[date, float] = {}
    lo_f_by_date:       Dict[date, float] = {}
    apparent_hi_by_date: Dict[date, float] = {}

    for d in sorted_dates:
        avg_c = {
            loc["label"]: era5_avg_by_label.get(loc["label"], {}).get(d)
            for loc in MISO_LOAD_LOCATIONS
            if era5_avg_by_label.get(loc["label"], {}).get(d) is not None
        }
        if not avg_c:
            continue
        avg_f_by_date[d] = weighted_avg_temp_f_miso(avg_c)

        hi_c = {}
        lo_c = {}
        ap_c = {}
        for loc in MISO_LOAD_LOCATIONS:
            label = loc["label"]
            pair  = era5_hi_by_label.get(label, {}).get(d)
            if pair is not None:
                hi_c[label] = pair[0]
                lo_c[label] = pair[1]
                if len(pair) > 2 and pair[2] is not None:
                    ap_c[label] = pair[2]
        if hi_c:
            hi_f_by_date[d] = weighted_avg_temp_f_miso(hi_c)
        if lo_c:
            lo_f_by_date[d] = weighted_avg_temp_f_miso(lo_c)
        if ap_c:
            apparent_hi_by_date[d] = weighted_avg_temp_f_miso(ap_c)

    obs: List[LoadObservation] = []
    for d in sorted(avg_f_by_date.keys()):
        if d not in load_daily:
            continue
        avg_f = avg_f_by_date[d]
        hi_f  = hi_f_by_date.get(d, avg_f)
        lo_f  = lo_f_by_date.get(d, avg_f)
        hdd, cdd = compute_hdd_cdd(avg_f)

        lag1_f = avg_f_by_date.get(d - timedelta(days=1))
        lag2_f = avg_f_by_date.get(d - timedelta(days=2))
        lag7_f = avg_f_by_date.get(d - timedelta(days=7))

        roll7_vals = [avg_f_by_date.get(d - timedelta(days=k)) for k in range(7)]
        roll7_vals = [v for v in roll7_vals if v is not None]
        rolling7_f = sum(roll7_vals) / len(roll7_vals) if roll7_vals else None

        obs.append(LoadObservation(
            date=d,
            hdd=hdd,
            cdd=cdd,
            avg_temp_f=avg_f,
            hi_temp_f=hi_f,
            lo_temp_f=lo_f,
            actual_load_mw=load_daily[d],
            is_weekend=(d.weekday() >= 5),
            day_of_week=d.weekday(),
            is_holiday=_is_us_holiday(d),
            day_of_year=d.timetuple().tm_yday,
            temp_lag1_f=lag1_f,
            temp_lag2_f=lag2_f,
            temp_lag7_f=lag7_f,
            rolling7_avg_f=rolling7_f,
            apparent_hi_f=_c_to_f(apparent_hi_by_date[d]) if d in apparent_hi_by_date else None,
        ))
    return obs


# ---------------------------------------------------------------------------
# MISO load model
# ---------------------------------------------------------------------------

class MISOLoadModel(LoadCorrectionModel):
    """XGBoost model: daily MISO load (MW) from temperature features."""

    def predict_with_uncertainty(
        self,
        forecast_temps_c:       Dict[str, List[float]],
        forecast_hi_c:          Dict[str, List[float]],
        forecast_lo_c:          Dict[str, List[float]],
        gefs_spread_c:          Dict[str, List[float]],
        forecast_dates:         List[date],
        recent_avg_temps_f:     List[float] = None,
        locations=None,
        weighted_avg_fn=None,
        forecast_apparent_hi_c: Dict[str, List[float]] = None,
    ) -> List[LoadForecast]:
        _locs   = locations    if locations    is not None else MISO_LOAD_LOCATIONS
        _wt_avg = weighted_avg_fn if weighted_avg_fn is not None else weighted_avg_temp_f_miso

        forecast_avg_f: List[Optional[float]] = []
        for i, _ in enumerate(forecast_dates):
            temps_c = {
                loc["label"]: (forecast_temps_c.get(loc["label"]) or [None] * 20)[i]
                for loc in _locs
                if (forecast_temps_c.get(loc["label"]) or [None] * 20)[i] is not None
            }
            if temps_c:
                forecast_avg_f.append(_wt_avg(temps_c))
            else:
                forecast_avg_f.append(None)

        def _lag(i, lag):
            j = i - lag
            if j >= 0:
                return forecast_avg_f[j]
            if recent_avg_temps_f is not None:
                k = len(recent_avg_temps_f) + j
                if 0 <= k < len(recent_avg_temps_f):
                    return recent_avg_temps_f[k]
            return None

        def _rolling7(i):
            vals = [_lag(i, k) for k in range(7)]
            vals = [v for v in vals if v is not None]
            return sum(vals) / len(vals) if vals else None

        results: List[LoadForecast] = []
        for i, d in enumerate(forecast_dates):
            hi_c_map = {
                loc["label"]: (forecast_hi_c.get(loc["label"]) or [None] * 20)[i]
                for loc in _locs
                if (forecast_hi_c.get(loc["label"]) or [None] * 20)[i] is not None
            }
            lo_c_map = {
                loc["label"]: (forecast_lo_c.get(loc["label"]) or [None] * 20)[i]
                for loc in _locs
                if (forecast_lo_c.get(loc["label"]) or [None] * 20)[i] is not None
            }
            avg_f = forecast_avg_f[i]
            if avg_f is None:
                continue
            hi_f = _wt_avg(hi_c_map) if hi_c_map else avg_f
            lo_f = _wt_avg(lo_c_map) if lo_c_map else avg_f

            app_hi_c_map = {}
            if forecast_apparent_hi_c:
                app_hi_c_map = {
                    loc["label"]: (forecast_apparent_hi_c.get(loc["label"]) or [None]*20)[i]
                    for loc in _locs
                    if (forecast_apparent_hi_c.get(loc["label"]) or [None]*20)[i] is not None
                }
            app_hi_f = _wt_avg(app_hi_c_map) if app_hi_c_map else None

            feats, hdd, cdd = _build_features(
                avg_f, hi_f, lo_f, d,
                _lag(i, 1), _lag(i, 2), _lag(i, 7), _rolling7(i),
                apparent_hi_f=app_hi_f,
            )
            mean_mw = self.predict([feats])[0]

            spread_c_vals = [
                (gefs_spread_c.get(loc["label"]) or [1.0] * 20)[i]
                for loc in MISO_LOAD_LOCATIONS
                if gefs_spread_c.get(loc["label"])
            ]
            spread_f = (sum(spread_c_vals) / len(spread_c_vals) * 9 / 5) if spread_c_vals else 1.8
            delta_load = abs(mean_mw * 0.015 * spread_f)

            results.append(LoadForecast(
                valid_date=d,
                lead_days=(d - date.today()).days,
                mean_load_mw=round(mean_mw, 1),
                low_load_mw=round(mean_mw - 1.645 * delta_load, 1),
                high_load_mw=round(mean_mw + 1.645 * delta_load, 1),
                hdd=hdd,
                cdd=cdd,
                avg_temp_f=avg_f,
            ))
        return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_miso_model(model: MISOLoadModel, path: str = _MISO_MODEL_PATH) -> None:
    save_load_model(model, path)


def load_miso_model(path: str = _MISO_MODEL_PATH) -> MISOLoadModel:
    return load_load_model(path)


# ---------------------------------------------------------------------------
# EIA official comparison (MISO actual + day-ahead forecast)
# ---------------------------------------------------------------------------

def fetch_miso_official_comparison(
    session: requests.Session,
    lookback_days: int = 14,
) -> Dict[str, Dict]:
    import json as _json, time as _time

    today = date.today()
    _cache_path = _MISO_COMPARISON_CACHE
    if os.path.exists(_cache_path):
        try:
            cached = _json.loads(open(_cache_path).read())
            age_h = (_time.time() - os.path.getmtime(_cache_path)) / 3600
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
                "facets[respondent][]": "MISO",
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
            period = row.get("period", "")
            val    = row.get("value")
            if val is None:
                continue
            try:
                d = date.fromisoformat(period[:10])
                daily.setdefault(d, []).append(float(val))
            except (ValueError, TypeError):
                continue
        return {d: sum(v) / len(v) for d, v in daily.items() if v}

    actual  = _fetch_type("D",  hist_start, today)
    da_fcst = _fetch_type("DF", today,      fwd_end)

    result = {
        "actual":  {d.isoformat(): round(mw / 1000, 2) for d, mw in actual.items()},
        "da_fcst": {d.isoformat(): round(mw / 1000, 2) for d, mw in da_fcst.items()},
    }

    if result["actual"]:
        try:
            os.makedirs(os.path.dirname(_cache_path), exist_ok=True)
            open(_cache_path, "w").write(_json.dumps(result))
        except Exception:
            pass

    if not result["actual"] and os.path.exists(_cache_path):
        try:
            return _json.loads(open(_cache_path).read())
        except Exception:
            pass

    return result


# ---------------------------------------------------------------------------
# MISO 7-day load forecast proxy
# MISO's detailed forecast requires a full Data Exchange account.
# We use EIA day-ahead forecast (type=DF, respondent=MISO) as a proxy.
# ---------------------------------------------------------------------------

def fetch_miso_7day(session: requests.Session) -> Dict[str, float]:
    import json as _json, time as _time
    from collections import defaultdict

    cache_path = _MISO_7DAY_CACHE
    if os.path.exists(cache_path):
        try:
            age_h = (_time.time() - os.path.getmtime(cache_path)) / 3600
            if age_h < 2:
                cached = _json.loads(open(cache_path).read())
                if cached:
                    return cached
        except Exception:
            pass

    today   = date.today()
    fwd_end = today + timedelta(days=7)

    all_rows: list = []
    offset = 0
    while True:
        params = {
            "api_key":              _EIA_KEY,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "MISO",
            "facets[type][]":       "DF",
            "start":                today.strftime("%Y-%m-%dT00"),
            "end":                  fwd_end.strftime("%Y-%m-%dT23"),
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

    daily_lists: dict = defaultdict(list)
    for row in all_rows:
        period = row.get("period", "")
        val    = row.get("value")
        if val is None:
            continue
        try:
            daily_lists[period[:10]].append(float(val))
        except (ValueError, TypeError):
            continue

    result = {
        d: round(sum(v) / len(v) / 1000, 2)
        for d, v in daily_lists.items()
        if v
    }

    if result:
        try:
            os.makedirs(os.path.dirname(cache_path), exist_ok=True)
            open(cache_path, "w").write(_json.dumps(result))
        except Exception:
            pass

    return result
