"""
PJM electricity load forecasting from temperature projections.

Improvements over v1:
  - Federal holidays flag (us holidays library)
  - Day-of-week one-hot (7 levels, not just weekend/weekday)
  - Lagged temperature features (T-1, T-2)
  - Separate daily high / low in addition to avg (better HDD/CDD split)
"""

import math
import os
import pickle
import time
from datetime import date, timedelta
from typing import Dict, List, Optional, Tuple

import requests

try:
    import holidays as _holidays_lib
    _US_HOLIDAYS = _holidays_lib.US()
except ImportError:
    _US_HOLIDAYS = None

from ._era5 import ERA5_ARCHIVE_URL, _CACHE_DIR, _cache_load, _cache_save, get_json
from .models import Coordinates, LoadForecast, LoadObservation
from .pjm import PJM_LOAD_LOCATIONS

# ---------------------------------------------------------------------------
# EIA API
# ---------------------------------------------------------------------------

_EIA_URL = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
_EIA_KEY  = os.environ.get("EIA_API_KEY", "DEMO_KEY")

_MODEL_PATH = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "pjm_load_model.pkl"
)


# ---------------------------------------------------------------------------
# EIA load fetch (unchanged from v1)
# ---------------------------------------------------------------------------

def fetch_pjm_load_daily(
    start: date,
    end: date,
    session: requests.Session,
) -> Dict[date, float]:
    """Fetch PJM RTO hourly demand from EIA and aggregate to daily mean (MW)."""
    all_rows: list = []
    offset = 0
    page_size = 5000

    while True:
        params = {
            "api_key":              _EIA_KEY,
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": "PJM",
            "facets[type][]":       "D",
            "start":                start.strftime("%Y-%m-%dT00"),
            "end":                  end.strftime("%Y-%m-%dT23"),
            "length":               page_size,
            "offset":               offset,
            "sort[0][column]":      "period",
            "sort[0][direction]":   "asc",
        }
        cached = _cache_load(_EIA_URL, params)
        if cached is not None:
            data = cached
        else:
            for attempt in range(5):
                try:
                    r = session.get(_EIA_URL, params=params, timeout=30)
                    r.raise_for_status()
                    data = r.json()
                    break
                except Exception as exc:
                    if attempt == 4:
                        raise RuntimeError(f"EIA fetch failed: {exc}") from exc
                    time.sleep(10 * (attempt + 1))
            _cache_save(_EIA_URL, params, data)

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
# ERA5 daily max / min fetch
# ---------------------------------------------------------------------------

def fetch_era5_daily_hi_lo(
    coords: Coordinates,
    start: date,
    end: date,
    session: requests.Session,
) -> Dict[date, Tuple[float, float]]:
    """
    Fetch daily max and min 2m temperature from the Open-Meteo archive.
    Returns {date: (max_c, min_c)}.
    """
    params = {
        "latitude":   coords.lat,
        "longitude":  coords.lon,
        "daily":      "temperature_2m_max,temperature_2m_min",
        "start_date": start.isoformat(),
        "end_date":   end.isoformat(),
        "timezone":   "UTC",
    }
    data = get_json(ERA5_ARCHIVE_URL, params, session, label="ERA5 daily hi/lo")
    daily = data.get("daily", {})
    dates = daily.get("time", [])
    maxes = daily.get("temperature_2m_max", [])
    mins  = daily.get("temperature_2m_min", [])
    result = {}
    for d_str, mx, mn in zip(dates, maxes, mins):
        if mx is not None and mn is not None:
            result[date.fromisoformat(d_str)] = (float(mx), float(mn))
    return result


# ---------------------------------------------------------------------------
# HDD / CDD helpers
# ---------------------------------------------------------------------------

def compute_hdd_cdd(avg_temp_f: float, base_f: float = 65.0) -> Tuple[float, float]:
    return max(0.0, base_f - avg_temp_f), max(0.0, avg_temp_f - base_f)


def _c_to_f(c: float) -> float:
    return c * 9 / 5 + 32


def weighted_avg_temp_f(temps_c_by_label: Dict[str, float]) -> float:
    total_w = total_wt = 0.0
    for loc in PJM_LOAD_LOCATIONS:
        label = loc["label"]
        if label not in temps_c_by_label:
            continue
        w = loc["weight"]
        total_wt += w * _c_to_f(temps_c_by_label[label])
        total_w  += w
    if total_w == 0:
        raise ValueError("No PJM location temperatures available")
    return total_wt / total_w


def _is_us_holiday(d: date) -> bool:
    if _US_HOLIDAYS is None:
        return False
    return d in _US_HOLIDAYS


# ---------------------------------------------------------------------------
# Training data assembly
# ---------------------------------------------------------------------------

def build_load_training_data(
    load_daily: Dict[date, float],
    era5_avg_by_label:  Dict[str, Dict[date, float]],
    era5_hi_by_label:   Dict[str, Dict[date, Tuple[float, float]]],  # {label: {date: (max_c, min_c)}}
) -> List[LoadObservation]:
    """
    Join PJM daily load with population-weighted temperature features from ERA5.
    Includes hi/lo split, holidays, day-of-week, and T-1/T-2 lag features.
    """
    sorted_dates = sorted(load_daily.keys())
    avg_f_by_date: Dict[date, float] = {}
    hi_f_by_date:  Dict[date, float] = {}
    lo_f_by_date:  Dict[date, float] = {}

    for d in sorted_dates:
        # Weighted avg
        avg_c = {
            loc["label"]: era5_avg_by_label.get(loc["label"], {}).get(d)
            for loc in PJM_LOAD_LOCATIONS
            if era5_avg_by_label.get(loc["label"], {}).get(d) is not None
        }
        if not avg_c:
            continue
        avg_f_by_date[d] = weighted_avg_temp_f(avg_c)

        # Weighted hi
        hi_c = {}
        lo_c = {}
        for loc in PJM_LOAD_LOCATIONS:
            label = loc["label"]
            pair  = era5_hi_by_label.get(label, {}).get(d)
            if pair is not None:
                hi_c[label] = pair[0]
                lo_c[label] = pair[1]
        if hi_c:
            hi_f_by_date[d] = weighted_avg_temp_f(hi_c)
        if lo_c:
            lo_f_by_date[d] = weighted_avg_temp_f(lo_c)

    obs = []
    dates_with_avg = sorted(avg_f_by_date.keys())
    for i, d in enumerate(dates_with_avg):
        load_mw = load_daily.get(d)
        if load_mw is None:
            continue
        avg_f = avg_f_by_date[d]
        hi_f  = hi_f_by_date.get(d, avg_f + 5)   # fallback: avg + typical diurnal swing
        lo_f  = lo_f_by_date.get(d, avg_f - 5)
        hdd, cdd = compute_hdd_cdd(avg_f)

        lag1_f = avg_f_by_date.get(d - timedelta(days=1))
        lag2_f = avg_f_by_date.get(d - timedelta(days=2))

        obs.append(LoadObservation(
            date=d,
            hdd=hdd, cdd=cdd,
            avg_temp_f=avg_f,
            hi_temp_f=hi_f,
            lo_temp_f=lo_f,
            actual_load_mw=load_mw,
            is_weekend=d.weekday() >= 5,
            day_of_week=d.weekday(),
            is_holiday=_is_us_holiday(d),
            day_of_year=d.timetuple().tm_yday,
            temp_lag1_f=lag1_f,
            temp_lag2_f=lag2_f,
        ))
    return obs


# ---------------------------------------------------------------------------
# Feature vector
# ---------------------------------------------------------------------------

def _build_features(
    avg_f:     float,
    hi_f:      float,
    lo_f:      float,
    d:         date,
    lag1_f:    Optional[float],
    lag2_f:    Optional[float],
) -> Tuple[list, float, float]:
    hdd, cdd    = compute_hdd_cdd(avg_f)
    hdd_hi, cdd_hi = compute_hdd_cdd(hi_f)
    hdd_lo, cdd_lo = compute_hdd_cdd(lo_f)

    doy_rad = 2 * math.pi * d.timetuple().tm_yday / 365.0
    dow = [1.0 if d.weekday() == i else 0.0 for i in range(7)]

    lag1 = lag1_f if lag1_f is not None else avg_f
    lag2 = lag2_f if lag2_f is not None else avg_f
    hdd_lag1, cdd_lag1 = compute_hdd_cdd(lag1)

    features = [
        hdd, cdd, hdd * cdd,          # avg-based HDD/CDD + interaction (3)
        hdd_hi, cdd_hi,                # high-temp HDD/CDD (2)
        hdd_lo, cdd_lo,                # low-temp HDD/CDD (2)
        math.sin(doy_rad), math.cos(doy_rad),  # seasonality (2)
        *dow,                          # day of week one-hot (7)
        float(_is_us_holiday(d)),      # holiday flag (1)
        lag1, lag2,                    # lagged avg temp (2)
        hdd_lag1, cdd_lag1,            # lagged HDD/CDD (2)
    ]
    return features, hdd, cdd  # 21 features total


def _obs_to_features(o: LoadObservation) -> list:
    feats, _, _ = _build_features(
        avg_f=o.avg_temp_f,
        hi_f=o.hi_temp_f,
        lo_f=o.lo_temp_f,
        d=o.date,
        lag1_f=o.temp_lag1_f,
        lag2_f=o.temp_lag2_f,
    )
    return feats


# ---------------------------------------------------------------------------
# Load correction model
# ---------------------------------------------------------------------------

class LoadCorrectionModel:
    """XGBoost model: daily PJM RTO load (MW) from temperature features."""

    def __init__(self):
        self._model = None

    def fit(self, observations: List[LoadObservation]) -> "LoadCorrectionModel":
        try:
            from xgboost import XGBRegressor
        except ImportError:
            from sklearn.ensemble import GradientBoostingRegressor as XGBRegressor

        X = [_obs_to_features(o) for o in observations]
        y = [o.actual_load_mw for o in observations]

        self._model = XGBRegressor(
            n_estimators=500,
            max_depth=5,
            learning_rate=0.04,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
        )
        self._model.fit(X, y)
        return self

    def predict(self, feature_rows: List[list]) -> List[float]:
        if self._model is None:
            raise RuntimeError("Model not fitted.")
        return [float(p) for p in self._model.predict(feature_rows)]

    def predict_with_uncertainty(
        self,
        forecast_temps_c:   Dict[str, List[float]],  # avg °C per location
        forecast_hi_c:      Dict[str, List[float]],  # daily high °C per location
        forecast_lo_c:      Dict[str, List[float]],  # daily low °C per location
        gefs_spread_c:      Dict[str, List[float]],  # GEFS spread °C (optional)
        forecast_dates:     List[date],
        recent_avg_temps_f: List[float] = None,      # [T-2, T-1] actual °F for lag init
    ) -> List[LoadForecast]:
        """
        Produce a LoadForecast for each date, propagating temperature
        uncertainty from GEFS spread into load uncertainty.
        """
        today = date.today()
        # Build a running list of avg temp forecasts (index 0 = today) for lags
        forecast_avg_f: List[Optional[float]] = []
        for i, d in enumerate(forecast_dates):
            temps_c = {
                loc["label"]: (forecast_temps_c.get(loc["label"]) or [None]*20)[i]
                for loc in PJM_LOAD_LOCATIONS
                if (forecast_temps_c.get(loc["label"]) or [None]*20)[i] is not None
            }
            if temps_c:
                forecast_avg_f.append(weighted_avg_temp_f(temps_c))
            else:
                forecast_avg_f.append(None)

        def _lag(i, lag):
            # lag=1 → day i-1, lag=2 → day i-2
            j = i - lag
            if j >= 0:
                return forecast_avg_f[j]
            # Before the forecast window — use recent actual temps if provided
            if recent_avg_temps_f is not None:
                idx = len(recent_avg_temps_f) + j  # j is negative
                if 0 <= idx < len(recent_avg_temps_f):
                    return recent_avg_temps_f[idx]
            return None

        results = []
        for i, d in enumerate(forecast_dates):
            avg_f = forecast_avg_f[i]
            if avg_f is None:
                continue

            # Daily hi/lo
            hi_c_vals = {
                loc["label"]: (forecast_hi_c.get(loc["label"]) or [None]*20)[i]
                for loc in PJM_LOAD_LOCATIONS
                if (forecast_hi_c.get(loc["label"]) or [None]*20)[i] is not None
            }
            lo_c_vals = {
                loc["label"]: (forecast_lo_c.get(loc["label"]) or [None]*20)[i]
                for loc in PJM_LOAD_LOCATIONS
                if (forecast_lo_c.get(loc["label"]) or [None]*20)[i] is not None
            }
            hi_f = weighted_avg_temp_f(hi_c_vals) if hi_c_vals else avg_f + 5
            lo_f = weighted_avg_temp_f(lo_c_vals) if lo_c_vals else avg_f - 5

            lag1 = _lag(i, 1)
            lag2 = _lag(i, 2)

            feats, hdd, cdd = _build_features(avg_f, hi_f, lo_f, d, lag1, lag2)
            mean_load = self.predict([feats])[0]

            # Uncertainty from GEFS spread
            total_w = total_ws = 0.0
            for loc in PJM_LOAD_LOCATIONS:
                spreads = gefs_spread_c.get(loc["label"])
                if spreads and i < len(spreads) and spreads[i] is not None:
                    w = loc["weight"]
                    total_ws += w * spreads[i] * 9 / 5
                    total_w  += w
            spread_f = (total_ws / total_w) if total_w > 0 else 3.0  # fallback 3°F

            z = 1.645
            low_feats,  _, _ = _build_features(avg_f - z * spread_f, hi_f - z * spread_f,
                                                lo_f - z * spread_f, d, lag1, lag2)
            high_feats, _, _ = _build_features(avg_f + z * spread_f, hi_f + z * spread_f,
                                                lo_f + z * spread_f, d, lag1, lag2)
            p_low  = self.predict([low_feats])[0]
            p_high = self.predict([high_feats])[0]

            results.append(LoadForecast(
                valid_date=d,
                lead_days=(d - today).days,
                mean_load_mw=mean_load,
                low_load_mw=min(p_low, p_high),
                high_load_mw=max(p_low, p_high),
                hdd=hdd, cdd=cdd,
                avg_temp_f=avg_f,
            ))

        return results


# ---------------------------------------------------------------------------
# Persistence
# ---------------------------------------------------------------------------

def save_load_model(model: LoadCorrectionModel, path: str = _MODEL_PATH) -> None:
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "wb") as f:
        pickle.dump(model, f)
    print(f"  Saved load model: {path}")


def load_load_model(path: str = _MODEL_PATH) -> LoadCorrectionModel:
    with open(path, "rb") as f:
        return pickle.load(f)


# ---------------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------------

def evaluate_load_model(
    model: LoadCorrectionModel,
    observations: List[LoadObservation],
    test_fraction: float = 0.2,
) -> dict:
    n_test = max(1, int(len(observations) * test_fraction))
    train  = observations[:-n_test]
    test   = observations[-n_test:]

    X_test = [_obs_to_features(o) for o in test]
    y_test = [o.actual_load_mw for o in test]
    y_pred = model.predict(X_test)

    rmse = math.sqrt(sum((p - a) ** 2 for p, a in zip(y_pred, y_test)) / len(y_test))
    mae  = sum(abs(p - a) for p, a in zip(y_pred, y_test)) / len(y_test)
    mape = sum(abs(p - a) / a for p, a in zip(y_pred, y_test) if a) / len(y_test) * 100

    return {
        "n_train":       len(train),
        "n_test":        len(test),
        "test_rmse_mw":  round(rmse, 1),
        "test_mae_mw":   round(mae, 1),
        "test_mape_pct": round(mape, 2),
    }
