"""
Pre-generate forecast caches for NYISO, ISO-NE, and SPP so HuggingFace
returns results instantly without recomputing on every page load.

Usage:
    python precache_forecasts.py
"""
import json
import logging
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv

load_dotenv()
logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
log = logging.getLogger(__name__)

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from temperature_modeling._era5 import fetch_era5_daily
from temperature_modeling.models import Coordinates
from temperature_modeling.pjm import PJM_LOAD_LOCATIONS
from temperature_modeling.pjm_load import (
    LoadCorrectionModel,
    _build_features,
    load_load_model,
    fetch_era5_daily_hi_lo,
    fetch_gefs_spread,
    run_load_backtest,
    fetch_pjm_official_comparison,
    fetch_pjm_dataminer_7day,
    weighted_avg_temp_f as weighted_avg_temp_f_pjm,
    _MODEL_PATH as _PJM_MODEL_PATH,
)
from temperature_modeling.net_load import fetch_net_load_forecast
from temperature_modeling.price_forecast import forecast_prices

from temperature_modeling.caiso import CAISO_LOAD_LOCATIONS
from temperature_modeling.caiso_load import (
    weighted_avg_temp_f_caiso, _CAISO_MODEL_PATH,
    fetch_caiso_official_comparison, fetch_caiso_oasis_7day,
)
from temperature_modeling.ercot import ERCOT_LOAD_LOCATIONS
from temperature_modeling.ercot_load import (
    weighted_avg_temp_f_ercot, _ERCOT_MODEL_PATH,
    fetch_ercot_official_comparison, fetch_ercot_7day,
)
from temperature_modeling.miso import MISO_LOAD_LOCATIONS
from temperature_modeling.miso_load import (
    weighted_avg_temp_f_miso, _MISO_MODEL_PATH,
    fetch_miso_official_comparison, fetch_miso_7day,
)
from temperature_modeling.nyiso import NYISO_LOAD_LOCATIONS
from temperature_modeling.nyiso_load import (
    weighted_avg_temp_f_nyiso, _NYISO_MODEL_PATH, fetch_nyiso_official_comparison,
)
from temperature_modeling.isone import ISONE_LOAD_LOCATIONS
from temperature_modeling.isone_load import (
    weighted_avg_temp_f_isone, _ISONE_MODEL_PATH, fetch_isone_official_comparison,
)
from temperature_modeling.spp import SPP_LOAD_LOCATIONS
from temperature_modeling.spp_load import (
    weighted_avg_temp_f_spp, _SPP_MODEL_PATH, fetch_spp_official_comparison,
)

_API_CACHE = _HERE / "api_cache"

_ISOS = {
    "pjm": dict(
        locations=PJM_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_pjm,
        model_path=_PJM_MODEL_PATH,
        cache_file=_API_CACHE / "pjm_forecast_cache.json",
        comparison_fn=fetch_pjm_official_comparison,
        bench_fn=fetch_pjm_dataminer_7day,
        bench_key="pjm_7day",
        training_path=_API_CACHE / "pjm_load_training.json",
    ),
    "caiso": dict(
        locations=CAISO_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_caiso,
        model_path=_CAISO_MODEL_PATH,
        cache_file=_API_CACHE / "caiso_forecast_cache.json",
        comparison_fn=fetch_caiso_official_comparison,
        bench_fn=fetch_caiso_oasis_7day,
        bench_key="oasis_7day",
        training_path=_API_CACHE / "caiso_load_training.json",
    ),
    "ercot": dict(
        locations=ERCOT_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_ercot,
        model_path=_ERCOT_MODEL_PATH,
        cache_file=_API_CACHE / "ercot_forecast_cache.json",
        comparison_fn=fetch_ercot_official_comparison,
        bench_fn=fetch_ercot_7day,
        bench_key="ercot_7day",
        training_path=_API_CACHE / "ercot_load_training.json",
    ),
    "miso": dict(
        locations=MISO_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_miso,
        model_path=_MISO_MODEL_PATH,
        cache_file=_API_CACHE / "miso_forecast_cache.json",
        comparison_fn=fetch_miso_official_comparison,
        bench_fn=fetch_miso_7day,
        bench_key="miso_7day",
        training_path=_API_CACHE / "miso_load_training.json",
    ),
    "nyiso": dict(
        locations=NYISO_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_nyiso,
        model_path=_NYISO_MODEL_PATH,
        cache_file=_API_CACHE / "nyiso_forecast_cache.json",
        comparison_fn=fetch_nyiso_official_comparison,
        bench_fn=lambda s: {},
        bench_key="nyiso_7day",
        training_path=_API_CACHE / "nyiso_load_training.json",
    ),
    "isone": dict(
        locations=ISONE_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_isone,
        model_path=_ISONE_MODEL_PATH,
        cache_file=_API_CACHE / "isone_forecast_cache.json",
        comparison_fn=fetch_isone_official_comparison,
        bench_fn=lambda s: {},
        bench_key="isone_7day",
        training_path=_API_CACHE / "isone_load_training.json",
    ),
    "spp": dict(
        locations=SPP_LOAD_LOCATIONS,
        weighted_avg_fn=weighted_avg_temp_f_spp,
        model_path=_SPP_MODEL_PATH,
        cache_file=_API_CACHE / "spp_forecast_cache.json",
        comparison_fn=fetch_spp_official_comparison,
        bench_fn=lambda s: {},
        bench_key="spp_7day",
        training_path=_API_CACHE / "spp_load_training.json",
    ),
}


def _fetch_one(label, lat, lon, session, forecast_days=15):
    try:
        r = session.get(
            "https://api.open-meteo.com/v1/forecast",
            params={
                "latitude": lat, "longitude": lon,
                "daily": ("temperature_2m_max,temperature_2m_min,"
                          "apparent_temperature_max,"
                          "dew_point_2m_max,"
                          "wind_speed_10m_max"),
                "forecast_days": forecast_days,
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
            },
            timeout=30,
        )
        r.raise_for_status()
        d = r.json()["daily"]
        return {
            "label": label, "dates": d["time"],
            "hi": d["temperature_2m_max"], "lo": d["temperature_2m_min"],
            "apparent_hi":  d.get("apparent_temperature_max"),
            "dewpoint_hi":  d.get("dew_point_2m_max"),   # °F (temperature_unit=fahrenheit)
            "wind_kph":     d.get("wind_speed_10m_max"),  # km/h (independent of temp unit)
        }
    except Exception as exc:
        log.warning("Open-Meteo error for %s: %s", label, exc)
        return None


def generate_cache(iso, cfg):
    log.info("=== %s: starting ===", iso.upper())

    model = load_load_model(cfg["model_path"])
    if model is None:
        log.error("%s: model not found at %s", iso.upper(), cfg["model_path"])
        return

    locations      = cfg["locations"]
    weighted_avg_fn = cfg["weighted_avg_fn"]

    session = requests.Session()
    session.headers["User-Agent"] = "precache-forecasts/1.0"

    def _f_to_c(lst):
        return [(v - 32) * 5 / 9 if v is not None else None for v in lst]

    # GFS temperature fetch
    avg_c, hi_c, lo_c, apparent_hi_c, dewpoint_c, wind_kph = {}, {}, {}, {}, {}, {}
    forecast_dates_strs = None

    log.info("%s: fetching GFS for %d locations ...", iso.upper(), len(locations))
    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch_one, loc["label"], loc["lat"], loc["lon"], session): loc
                for loc in locations}
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            label   = res["label"]
            loc_hi  = _f_to_c(res["hi"])
            loc_lo  = _f_to_c(res["lo"])
            loc_avg = [(h + l) / 2 if h and l else None for h, l in zip(loc_hi, loc_lo)]
            avg_c[label] = loc_avg
            hi_c[label]  = loc_hi
            lo_c[label]  = loc_lo
            if res.get("apparent_hi"):
                apparent_hi_c[label] = _f_to_c(res["apparent_hi"])
            if res.get("dewpoint_hi"):
                dewpoint_c[label] = _f_to_c(res["dewpoint_hi"])  # °F → °C
            if res.get("wind_kph"):
                wind_kph[label] = res["wind_kph"]  # already km/h
            if forecast_dates_strs is None:
                forecast_dates_strs = res["dates"][:15]

    if not avg_c or not forecast_dates_strs:
        log.error("%s: no GFS data — aborting", iso.upper())
        return

    forecast_dates_list = [date.fromisoformat(d) for d in forecast_dates_strs]
    log.info("%s: GFS done — %d dates", iso.upper(), len(forecast_dates_list))

    # GEFS ensemble spread
    gefs_spread_c: dict = {}
    try:
        gefs_spread_c = fetch_gefs_spread(locations, session, forecast_days=len(forecast_dates_list))
        log.info("%s: GEFS spread fetched for %d/%d locations",
                 iso.upper(), len(gefs_spread_c), len(locations))
    except Exception:
        log.exception("%s: GEFS spread fetch failed — using fallback 3°F spread", iso.upper())

    # ERA5 hindcast
    era5_session = requests.Session()
    era5_session.headers["User-Agent"] = "precache-forecasts/1.0"
    today = date.today()
    era5_avg_hist, recent_avg_f = {}, []

    log.info("%s: fetching ERA5 for %d locations ...", iso.upper(), len(locations))
    try:
        per_label = {}
        for i, loc in enumerate(locations, 1):
            log.info("  [%d/%d] %s", i, len(locations), loc["label"])
            per_label[loc["label"]] = fetch_era5_daily(
                Coordinates(loc["lat"], loc["lon"]),
                today - timedelta(days=16), today - timedelta(days=1),
                era5_session,
            )
        era5_avg_hist = per_label
        for lag_d in sorted(today - timedelta(days=k) for k in range(8, 0, -1)):
            c_map = {loc["label"]: per_label[loc["label"]][lag_d]
                     for loc in locations
                     if lag_d in per_label.get(loc["label"], {})}
            if c_map:
                recent_avg_f.append(weighted_avg_fn(c_map))
    except Exception:
        log.exception("%s: ERA5 failed — lags will use GFS fallback", iso.upper())
        recent_avg_f = []

    log.info("%s: ERA5 done — %d recent temps", iso.upper(), len(recent_avg_f))

    # Hindcast series
    hindcast = {}
    if era5_avg_hist:
        try:
            avg_f_hist = {}
            for off in range(16):
                d2 = today - timedelta(days=off + 1)
                c_map = {loc["label"]: era5_avg_hist[loc["label"]][d2]
                         for loc in locations
                         if d2 in era5_avg_hist.get(loc["label"], {})}
                if c_map:
                    avg_f_hist[d2] = weighted_avg_fn(c_map)
            for d2, avg_f in avg_f_hist.items():
                lag1  = avg_f_hist.get(d2 - timedelta(days=1))
                lag2  = avg_f_hist.get(d2 - timedelta(days=2))
                lag7  = avg_f_hist.get(d2 - timedelta(days=7))
                rv    = [avg_f_hist.get(d2 - timedelta(days=k)) for k in range(7)]
                roll7 = sum(v for v in rv if v) / max(sum(1 for v in rv if v), 1)
                feats, _, _ = _build_features(avg_f, avg_f + 5, avg_f - 5, d2, lag1, lag2, lag7, roll7)
                hindcast[d2.isoformat()] = round(model.predict([feats])[0] / 1000, 2)
        except Exception:
            log.exception("%s: hindcast failed", iso.upper())

    # 15-day forecast
    log.info("%s: running predict_with_uncertainty ...", iso.upper())
    load_forecasts = model.predict_with_uncertainty(
        forecast_temps_c=avg_c, forecast_hi_c=hi_c, forecast_lo_c=lo_c,
        gefs_spread_c=gefs_spread_c, forecast_dates=forecast_dates_list,
        recent_avg_temps_f=recent_avg_f if len(recent_avg_f) >= 2 else None,
        locations=locations, weighted_avg_fn=weighted_avg_fn,
        forecast_apparent_hi_c=apparent_hi_c if apparent_hi_c else None,
        forecast_dewpoint_c=dewpoint_c if dewpoint_c else None,
        forecast_wind_kph=wind_kph if wind_kph else None,
    )
    load_data = [{"date": lf.valid_date.isoformat(),
                  "mean_load_gw": round(lf.mean_load_mw / 1000, 2),
                  "low_load_gw":  round(lf.low_load_mw  / 1000, 2),
                  "high_load_gw": round(lf.high_load_mw / 1000, 2),
                  "avg_temp_f":   round(lf.avg_temp_f, 1) if lf.avg_temp_f else None,
                  "hdd":          round(lf.hdd, 1),
                  "cdd":          round(lf.cdd, 1)}
                 for lf in load_forecasts]
    log.info("%s: forecast done — %d days", iso.upper(), len(load_data))

    # Net load + price
    net_load = fetch_net_load_forecast(iso, forecast_dates_list, session)
    prices   = forecast_prices(iso, load_data)

    # Official ISO comparison (actual + day-ahead forecast)
    comparison: dict = {"actual": {}, "da_fcst": {}}
    try:
        comparison = cfg["comparison_fn"](session)
        log.info("%s: comparison fetched — %d actual, %d da_fcst days",
                 iso.upper(), len(comparison.get("actual", {})), len(comparison.get("da_fcst", {})))
    except Exception:
        log.exception("%s: comparison fetch failed", iso.upper())

    # 7-day benchmark from ISO
    bench_data: dict = {}
    try:
        bench_data = cfg["bench_fn"](session)
    except Exception:
        log.exception("%s: bench fetch failed", iso.upper())

    # Model accuracy backtest (reads training JSON; returns {} if not present)
    backtest: dict = {}
    try:
        tp = cfg.get("training_path")
        if tp and Path(tp).exists():
            backtest = run_load_backtest(model, str(tp))
            log.info("%s: backtest done — test MAPE %.1f%%",
                     iso.upper(), backtest.get("mape_test") or 0)
    except Exception:
        log.exception("%s: backtest failed", iso.upper())

    result = {
        "load":              load_data,
        "dates":             forecast_dates_strs,
        "comparison":        comparison,
        cfg["bench_key"]:    bench_data,
        "backtest":          backtest,
        "hindcast":          hindcast,
        "net_load":          net_load,
        "price_forecast":    prices,
    }

    cfg["cache_file"].parent.mkdir(parents=True, exist_ok=True)
    cfg["cache_file"].write_text(json.dumps(result))
    log.info("%s: cache written to %s", iso.upper(), cfg["cache_file"])


if __name__ == "__main__":
    isos_to_run = sys.argv[1:] or list(_ISOS.keys())
    for iso in isos_to_run:
        if iso not in _ISOS:
            log.error("Unknown ISO: %s", iso)
            continue
        generate_cache(iso, _ISOS[iso])
    log.info("Done.")
