"""
AutoTheta time-series ensemble for load forecasting.

AutoTheta (Assimakopoulos & Nikolopoulos 2000, winner of M3 competition)
decomposes the series into trend + seasonality without weather features,
providing an orthogonal signal to the weather-driven XGBoost model.

Blend: 75% XGBoost + 25% AutoTheta in the final forecast.
"""

import json
import logging
import time
from datetime import date
from pathlib import Path
from typing import Dict, List, Optional

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "api_cache"
_CACHE_TTL  = 6 * 3600   # 6 hours
THETA_WEIGHT = 0.25       # 25% Theta, 75% XGBoost


def _cache_path(iso: str) -> Path:
    return _CACHE_DIR / f"{iso}_theta_cache.json"


def _load_cache(iso: str) -> Optional[Dict]:
    p = _cache_path(iso)
    if not p.exists():
        return None
    if time.time() - p.stat().st_mtime > _CACHE_TTL:
        return None
    try:
        data = json.loads(p.read_text())
        if data.get("date") == date.today().isoformat():
            return data.get("forecasts")
    except Exception:
        pass
    return None


def _save_cache(iso: str, forecasts: Dict) -> None:
    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _cache_path(iso).write_text(
            json.dumps({"date": date.today().isoformat(), "forecasts": forecasts})
        )
    except Exception:
        pass


def predict_theta(
    iso: str,
    historical_loads_mw: List[float],
    horizon: int = 15,
) -> Optional[Dict[str, float]]:
    """
    Fit AutoTheta on historical daily load and return {date_str: mean_gw} forecasts.

    Parameters
    ----------
    iso : str
        ISO code — used only for caching.
    historical_loads_mw : list
        Daily load history in MW, ordered oldest to most recent.
        Needs ≥ 60 observations; ≥ 365 recommended.
    horizon : int
        Number of days ahead to forecast.

    Returns
    -------
    dict or None
        {date_str: mean_gw} for each forecast day, or None on failure.
    """
    cached = _load_cache(iso)
    if cached:
        return cached

    if len(historical_loads_mw) < 60:
        log.debug("Theta: insufficient history (%d obs) for %s", len(historical_loads_mw), iso)
        return None

    try:
        import pandas as pd
        from statsforecast import StatsForecast
        from statsforecast.models import AutoTheta

        n = len(historical_loads_mw)
        today = date.today()
        start_date = today - __import__("datetime").timedelta(days=n)

        df = pd.DataFrame({
            "unique_id": iso,
            "ds": pd.date_range(start=start_date, periods=n, freq="D"),
            "y": [float(v) for v in historical_loads_mw],
        })

        sf = StatsForecast(
            models=[AutoTheta(season_length=7)],
            freq="D",
            n_jobs=1,
        )
        sf.fit(df)
        preds = sf.predict(h=horizon)

        result: Dict[str, float] = {}
        for _, row in preds.iterrows():
            d = row["ds"]
            if hasattr(d, "date"):
                d = d.date()
            result[str(d)[:10]] = round(float(row["AutoTheta"]) / 1000, 2)

        _save_cache(iso, result)
        log.info("AutoTheta forecast generated for %s (%d days)", iso.upper(), len(result))
        return result

    except ImportError:
        log.debug("statsforecast not installed — AutoTheta ensemble unavailable")
        return None
    except Exception as exc:
        log.warning("AutoTheta forecast failed for %s: %s", iso.upper(), exc)
        return None


def blend_with_xgboost(
    load_data: List[Dict],
    theta_forecasts: Dict[str, float],
) -> List[Dict]:
    """
    Blend load_data (XGBoost) with AutoTheta forecasts (75/25 mix).
    Widens the CI by half the model-disagreement distance.
    Modifies load_data in place and returns it.
    """
    for item in load_data:
        d_str = item["date"]
        th_gw = theta_forecasts.get(d_str)
        if th_gw is None:
            continue
        xgb_gw = item["mean_load_gw"]
        blended = round((1 - THETA_WEIGHT) * xgb_gw + THETA_WEIGHT * th_gw, 2)
        disagree = abs(xgb_gw - th_gw) * 0.5  # half-disagreement widens CI
        item["mean_load_gw"]  = blended
        item["low_load_gw"]   = round(item["low_load_gw"]  - disagree, 2)
        item["high_load_gw"]  = round(item["high_load_gw"] + disagree, 2)
    return load_data
