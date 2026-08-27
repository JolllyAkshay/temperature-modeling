"""
US Grid Load Forecast Dashboard — PJM & CAISO
Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""

import json
import logging
import logging.config
import os
import sys
import time

from dotenv import load_dotenv
load_dotenv()
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import dash
from dash import dcc, html, Input, Output, State, ALL
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Logging — configure before any module imports so child loggers inherit
# ---------------------------------------------------------------------------
logging.config.dictConfig({
    "version": 1,
    "disable_existing_loggers": False,
    "formatters": {
        "standard": {"format": "%(asctime)s [%(levelname)s] %(name)s: %(message)s",
                     "datefmt": "%Y-%m-%d %H:%M:%S"},
    },
    "handlers": {
        "console": {"class": "logging.StreamHandler", "formatter": "standard"},
    },
    "root": {"level": "INFO", "handlers": ["console"]},
})
log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Bootstrap
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

_env_file = _HERE / ".env"
if _env_file.exists():
    for _line in _env_file.read_text().splitlines():
        _line = _line.strip()
        if _line and not _line.startswith("#") and "=" in _line:
            _k, _v = _line.split("=", 1)
            os.environ.setdefault(_k.strip(), _v.strip())

from temperature_modeling.pjm import PJM_LOAD_LOCATIONS
from temperature_modeling.pjm_load import (
    LoadCorrectionModel, load_load_model,
    fetch_pjm_official_comparison, fetch_pjm_dataminer_7day, run_load_backtest,
    fetch_gefs_spread,
)
from temperature_modeling.caiso import CAISO_LOAD_LOCATIONS
from temperature_modeling.caiso_load import (
    CAISOLoadModel, load_caiso_model,
    fetch_caiso_official_comparison, fetch_caiso_oasis_7day,
    weighted_avg_temp_f_caiso, _CAISO_MODEL_PATH,
)
from temperature_modeling.ercot import ERCOT_LOAD_LOCATIONS
from temperature_modeling.ercot_load import (
    ERCOTLoadModel, load_ercot_model,
    fetch_ercot_official_comparison, fetch_ercot_7day,
    weighted_avg_temp_f_ercot,
)
from temperature_modeling.miso import MISO_LOAD_LOCATIONS
from temperature_modeling.miso_load import (
    MISOLoadModel, load_miso_model,
    fetch_miso_official_comparison, fetch_miso_7day,
    weighted_avg_temp_f_miso,
)
from temperature_modeling.nyiso import NYISO_LOAD_LOCATIONS
from temperature_modeling.nyiso_load import (
    weighted_avg_temp_f_nyiso, _NYISO_MODEL_PATH,
    fetch_nyiso_official_comparison,
)
from temperature_modeling.isone import ISONE_LOAD_LOCATIONS
from temperature_modeling.isone_load import (
    weighted_avg_temp_f_isone, _ISONE_MODEL_PATH,
    fetch_isone_official_comparison,
)
from temperature_modeling.spp import SPP_LOAD_LOCATIONS
from temperature_modeling.spp_load import (
    weighted_avg_temp_f_spp, _SPP_MODEL_PATH,
    fetch_spp_official_comparison,
)
from temperature_modeling import _llm
from temperature_modeling.ai_brief import generate_forecast_brief, generate_chat_response
from temperature_modeling.verification import record_forecast, load_verification_stats
from temperature_modeling.net_load import fetch_net_load_forecast
from temperature_modeling.price_forecast import forecast_prices, price_unavailable_reason
from temperature_modeling.capacity_market import get_capacity_market_data, get_reserve_margin_color
from temperature_modeling.ensemble import get_ensemble_forecast
from temperature_modeling.carbon_intensity import fetch_carbon_intensity
from temperature_modeling.demand_response import compute_dr_windows
from temperature_modeling.theta_ensemble import predict_theta, blend_with_xgboost

FORECAST_CACHE_TTL_HOURS = 3

_LOAD_FORECAST_CACHE_FILE   = _HERE / "api_cache" / "pjm_forecast_cache.json"
_LOAD_TRAINING_DATA_PATH    = _HERE / "api_cache" / "pjm_load_training.json"
_CAISO_FORECAST_CACHE_FILE  = _HERE / "api_cache" / "caiso_forecast_cache.json"
_CAISO_TRAINING_DATA_PATH   = _HERE / "api_cache" / "caiso_load_training.json"
_ERCOT_FORECAST_CACHE_FILE  = _HERE / "api_cache" / "ercot_forecast_cache.json"
_ERCOT_TRAINING_DATA_PATH   = _HERE / "api_cache" / "ercot_load_training.json"
_MISO_FORECAST_CACHE_FILE   = _HERE / "api_cache" / "miso_forecast_cache.json"
_MISO_TRAINING_DATA_PATH    = _HERE / "api_cache" / "miso_load_training.json"
_NYISO_FORECAST_CACHE_FILE  = _HERE / "api_cache" / "nyiso_forecast_cache.json"
_NYISO_TRAINING_DATA_PATH   = _HERE / "api_cache" / "nyiso_load_training.json"
_ISONE_FORECAST_CACHE_FILE  = _HERE / "api_cache" / "isone_forecast_cache.json"
_ISONE_TRAINING_DATA_PATH   = _HERE / "api_cache" / "isone_load_training.json"
_SPP_FORECAST_CACHE_FILE    = _HERE / "api_cache" / "spp_forecast_cache.json"
_SPP_TRAINING_DATA_PATH     = _HERE / "api_cache" / "spp_load_training.json"

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
_LOAD_MODEL: LoadCorrectionModel | None = None
try:
    _LOAD_MODEL = load_load_model()
    log.info("PJM load model loaded OK")
except FileNotFoundError:
    log.error("PJM load model not found — run training script first")
except Exception:
    log.exception("Failed to load PJM load model")

_CAISO_MODEL: CAISOLoadModel | None = None
try:
    _CAISO_MODEL = load_caiso_model()
    log.info("CAISO load model loaded OK")
except FileNotFoundError:
    log.error("CAISO load model not found — run training script first")
except Exception:
    log.exception("Failed to load CAISO load model")

_ERCOT_MODEL: ERCOTLoadModel | None = None
try:
    _ERCOT_MODEL = load_ercot_model()
    log.info("ERCOT load model loaded OK")
except FileNotFoundError:
    log.error("ERCOT load model not found — run training script first")
except Exception:
    log.exception("Failed to load ERCOT load model")

_MISO_MODEL: MISOLoadModel | None = None
try:
    _MISO_MODEL = load_miso_model()
    log.info("MISO load model loaded OK")
except FileNotFoundError:
    log.error("MISO load model not found — run training script first")
except Exception:
    log.exception("Failed to load MISO load model")


_NYISO_MODEL: LoadCorrectionModel | None = None
try:
    _NYISO_MODEL = load_load_model(_NYISO_MODEL_PATH)
    log.info("NYISO load model loaded OK")
except FileNotFoundError:
    log.warning("NYISO model not found — run collect_nyiso_load.py first")
except Exception:
    log.exception("Failed to load NYISO model")

_ISONE_MODEL: LoadCorrectionModel | None = None
try:
    _ISONE_MODEL = load_load_model(_ISONE_MODEL_PATH)
    log.info("ISO-NE load model loaded OK")
except FileNotFoundError:
    log.warning("ISO-NE model not found — run collect_isone_load.py first")
except Exception:
    log.exception("Failed to load ISO-NE model")

_SPP_MODEL: LoadCorrectionModel | None = None
try:
    _SPP_MODEL = load_load_model(_SPP_MODEL_PATH)
    log.info("SPP load model loaded OK")
except FileNotFoundError:
    log.warning("SPP model not found — run collect_spp_load.py first")
except Exception:
    log.exception("Failed to load SPP model")



# ---------------------------------------------------------------------------
# Temperature fetch helper (Open-Meteo GFS)
# ---------------------------------------------------------------------------
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
            timeout=20,
        )
        r.raise_for_status()
        d = r.json()["daily"]
        return {
            "label": label, "dates": d["time"],
            "hi": d["temperature_2m_max"], "lo": d["temperature_2m_min"],
            "apparent_hi":  d.get("apparent_temperature_max"),
            "dewpoint_hi":  d.get("dew_point_2m_max"),   # °F (temperature_unit=fahrenheit)
            "wind_kph":     d.get("wind_speed_10m_max"),  # km/h
        }
    except requests.HTTPError as exc:
        log.warning("Open-Meteo HTTP error for %s: %s", label, exc)
    except requests.RequestException as exc:
        log.warning("Open-Meteo network error for %s: %s", label, exc)
    except (KeyError, ValueError) as exc:
        log.warning("Open-Meteo unexpected response for %s: %s", label, exc)
    return None


# ---------------------------------------------------------------------------
# Shared ISO forecast helper — replaces 4 nearly-identical blocks
# ---------------------------------------------------------------------------

# Per-ISO config: (locations_list, weighted_avg_fn, model, cache_file,
#                  training_data_path, comparison_fn, bench_7day_fn, bench_key)
_ISO_CONFIGS: dict = {}  # populated after imports above


def _build_iso_configs():
    from temperature_modeling.pjm_load import weighted_avg_temp_f
    _ISO_CONFIGS.update({
        "pjm": dict(
            locations=PJM_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f,
            model_ref=lambda: _LOAD_MODEL,
            cache_file=_LOAD_FORECAST_CACHE_FILE,
            training_path=_LOAD_TRAINING_DATA_PATH,
            comparison_fn=fetch_pjm_official_comparison,
            bench_fn=fetch_pjm_dataminer_7day,
            bench_key="pjm_7day",
        ),
        "caiso": dict(
            locations=CAISO_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f_caiso,
            model_ref=lambda: _CAISO_MODEL,
            cache_file=_CAISO_FORECAST_CACHE_FILE,
            training_path=_CAISO_TRAINING_DATA_PATH,
            comparison_fn=fetch_caiso_official_comparison,
            bench_fn=fetch_caiso_oasis_7day,
            bench_key="oasis_7day",
        ),
        "ercot": dict(
            locations=ERCOT_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f_ercot,
            model_ref=lambda: _ERCOT_MODEL,
            cache_file=_ERCOT_FORECAST_CACHE_FILE,
            training_path=_ERCOT_TRAINING_DATA_PATH,
            comparison_fn=fetch_ercot_official_comparison,
            bench_fn=fetch_ercot_7day,
            bench_key="ercot_7day",
        ),
        "miso": dict(
            locations=MISO_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f_miso,
            model_ref=lambda: _MISO_MODEL,
            cache_file=_MISO_FORECAST_CACHE_FILE,
            training_path=_MISO_TRAINING_DATA_PATH,
            comparison_fn=fetch_miso_official_comparison,
            bench_fn=fetch_miso_7day,
            bench_key="miso_7day",
        ),
        "nyiso": dict(
            locations=NYISO_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f_nyiso,
            model_ref=lambda: _NYISO_MODEL,
            cache_file=_NYISO_FORECAST_CACHE_FILE,
            training_path=_NYISO_TRAINING_DATA_PATH,
            comparison_fn=fetch_nyiso_official_comparison,
            bench_fn=lambda s: [],
            bench_key="nyiso_7day",
        ),
        "isone": dict(
            locations=ISONE_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f_isone,
            model_ref=lambda: _ISONE_MODEL,
            cache_file=_ISONE_FORECAST_CACHE_FILE,
            training_path=_ISONE_TRAINING_DATA_PATH,
            comparison_fn=fetch_isone_official_comparison,
            bench_fn=lambda s: [],
            bench_key="isone_7day",
        ),
        "spp": dict(
            locations=SPP_LOAD_LOCATIONS,
            weighted_avg_fn=weighted_avg_temp_f_spp,
            model_ref=lambda: _SPP_MODEL,
            cache_file=_SPP_FORECAST_CACHE_FILE,
            training_path=_SPP_TRAINING_DATA_PATH,
            comparison_fn=fetch_spp_official_comparison,
            bench_fn=lambda s: [],
            bench_key="spp_7day",
        ),
    })


def _is_cache_valid(cache_file: Path) -> bool:
    """Return True only if the cache file is fresh, from today, and has comparison data."""
    if not cache_file.exists():
        return False
    age_h = (time.time() - cache_file.stat().st_mtime) / 3600
    if age_h >= FORECAST_CACHE_TTL_HOURS:
        return False
    try:
        cached = json.loads(cache_file.read_text())
        load_list = cached.get("load") or []
        if not load_list:
            return False
        if load_list[0].get("date") != date.today().isoformat():
            return False
        if not cached.get("comparison", {}).get("actual"):
            return False
        return True
    except (json.JSONDecodeError, KeyError, IndexError):
        log.debug("Cache %s is corrupt or incomplete — will refresh", cache_file.name)
        return False


def _fetch_iso_forecast(iso: str, force: bool = False) -> dict:
    """
    Shared forecast pipeline for all ISOs.
    Fetches GFS temperatures, builds ERA5 hindcast, runs the load model,
    and collects official benchmark data.
    """
    from temperature_modeling._era5 import fetch_era5_daily
    from temperature_modeling.pjm_load import _build_features as _bf
    from temperature_modeling.models import Coordinates as _C

    cfg   = _ISO_CONFIGS[iso]
    model = cfg["model_ref"]()
    if model is None:
        log.warning("%s model not loaded — skipping forecast", iso.upper())
        return {}

    cache_file = cfg["cache_file"]
    if not force and _is_cache_valid(cache_file):
        try:
            return json.loads(cache_file.read_text())
        except (json.JSONDecodeError, OSError):
            log.warning("Could not read %s cache — regenerating", iso.upper())

    locations      = cfg["locations"]
    weighted_avg_fn = cfg["weighted_avg_fn"]

    session = requests.Session()
    session.headers["User-Agent"] = "load-forecast-dashboard/1.0"

    avg_c:         dict = {}
    hi_c:          dict = {}
    lo_c:          dict = {}
    apparent_hi_c: dict = {}
    dewpoint_c:    dict = {}
    wind_kph:      dict = {}
    forecast_dates_strs = None

    def _f_to_c(lst):
        return [(v - 32) * 5 / 9 if v is not None else None for v in lst]

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch_one, loc["label"], loc["lat"], loc["lon"], session): loc
                for loc in locations}
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            label    = res["label"]
            loc_hi   = _f_to_c(res["hi"])
            loc_lo   = _f_to_c(res["lo"])
            loc_avg  = [(h + l) / 2 if h and l else None for h, l in zip(loc_hi, loc_lo)]
            avg_c[label] = loc_avg
            hi_c[label]  = loc_hi
            lo_c[label]  = loc_lo
            if res.get("apparent_hi"):
                apparent_hi_c[label] = _f_to_c(res["apparent_hi"])
            if res.get("dewpoint_hi"):
                dewpoint_c[label] = _f_to_c(res["dewpoint_hi"])  # °F → °C
            if res.get("wind_kph"):
                wind_kph[label] = res["wind_kph"]
            if forecast_dates_strs is None:
                forecast_dates_strs = res["dates"][:15]

    if not avg_c or not forecast_dates_strs:
        log.error("%s: no GFS temperature data returned — aborting forecast", iso.upper())
        return {}

    forecast_dates_list = [date.fromisoformat(d) for d in forecast_dates_strs]

    # GEFS ensemble spread — real 30-member σ for uncertainty bands
    gefs_spread_c: dict = {}
    try:
        gefs_spread_c = fetch_gefs_spread(locations, session, forecast_days=len(forecast_dates_list))
        log.info("%s: GEFS spread fetched for %d/%d locations",
                 iso.upper(), len(gefs_spread_c), len(locations))
    except Exception:
        log.exception("%s: GEFS spread fetch failed — using fallback 3°F spread", iso.upper())

    # ── Comparison data (EIA actuals) — fetched early to supply load autocorrelation lags ──
    comparison: dict = {"actual": {}, "da_fcst": {}}
    try:
        comparison = cfg["comparison_fn"](session)
    except Exception:
        log.exception("%s: official comparison fetch failed — chart will show no actuals", iso.upper())

    # Extract last 7 actual daily loads (MW) for load autocorrelation lag features
    recent_actual_loads_mw = None
    try:
        actual_gw = comparison.get("actual", {})
        if actual_gw:
            sorted_act_dates = sorted(actual_gw.keys())[-7:]
            loads_mw = [actual_gw[d] * 1000 for d in sorted_act_dates]
            if loads_mw:
                recent_actual_loads_mw = loads_mw
    except Exception:
        pass

    # ERA5 hindcast (last 16 days)
    era5_session = requests.Session()
    era5_session.headers["User-Agent"] = "load-forecast-dashboard/1.0"
    today = date.today()
    era5_avg_hist: dict = {}
    recent_avg_f: list = []
    try:
        per_label = {}
        for loc in locations:
            per_label[loc["label"]] = fetch_era5_daily(
                _C(loc["lat"], loc["lon"]),
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
        log.exception("%s: ERA5 hindcast fetch failed — lags will use GFS-only fallback", iso.upper())
        recent_avg_f = []

    # Build hindcast GW series
    hindcast: dict = {}
    if era5_avg_hist:
        try:
            avg_f_hist: dict = {}
            for off in range(16):
                d2 = today - timedelta(days=off + 1)
                c_map = {loc["label"]: era5_avg_hist[loc["label"]][d2]
                         for loc in locations
                         if d2 in era5_avg_hist.get(loc["label"], {})}
                if c_map:
                    avg_f_hist[d2] = weighted_avg_fn(c_map)
            # Build load lookup from EIA actuals for hindcast lag features
            load_actual_mw_by_date = {
                date.fromisoformat(ds): gw * 1000
                for ds, gw in comparison.get("actual", {}).items()
            }
            hindcast_load_mw: dict = {}  # rolling predictions for forward-lag propagation
            for d2, avg_f in sorted(avg_f_hist.items()):
                lag1  = avg_f_hist.get(d2 - timedelta(days=1))
                lag2  = avg_f_hist.get(d2 - timedelta(days=2))
                lag7  = avg_f_hist.get(d2 - timedelta(days=7))
                rv    = [avg_f_hist.get(d2 - timedelta(days=k)) for k in range(7)]
                roll7 = sum(v for v in rv if v) / max(sum(1 for v in rv if v), 1)
                # Load lags: prefer EIA actuals, fall back to prior hindcast predictions
                def _hl(dt):
                    return load_actual_mw_by_date.get(dt) or hindcast_load_mw.get(dt)
                ll1  = _hl(d2 - timedelta(days=1))
                ll7  = _hl(d2 - timedelta(days=7))
                llv  = [_hl(d2 - timedelta(days=k)) for k in range(1, 8)]
                llv  = [v for v in llv if v is not None]
                ll_r = sum(llv) / len(llv) if llv else None
                feats, _, _ = _bf(avg_f, avg_f + 5, avg_f - 5, d2, lag1, lag2, lag7, roll7,
                                   load_lag1_mw=ll1, load_lag7_mw=ll7, rolling7_load_mw=ll_r)
                pred_mw = model.predict([feats])[0]
                hindcast_load_mw[d2] = pred_mw
                hindcast[d2.isoformat()] = round(pred_mw / 1000, 2)
        except Exception:
            log.exception("%s: hindcast computation failed", iso.upper())
            hindcast = {}

    # 15-day GFS forecast with uncertainty
    load_forecasts = model.predict_with_uncertainty(
        forecast_temps_c=avg_c, forecast_hi_c=hi_c, forecast_lo_c=lo_c,
        gefs_spread_c=gefs_spread_c, forecast_dates=forecast_dates_list,
        recent_avg_temps_f=recent_avg_f if len(recent_avg_f) >= 2 else None,
        locations=locations, weighted_avg_fn=weighted_avg_fn,
        forecast_apparent_hi_c=apparent_hi_c if apparent_hi_c else None,
        forecast_dewpoint_c=dewpoint_c if dewpoint_c else None,
        forecast_wind_kph=wind_kph if wind_kph else None,
        recent_actual_loads_mw=recent_actual_loads_mw,
    )
    load_data = [{"date": lf.valid_date.isoformat(),
                  "mean_load_gw": round(lf.mean_load_mw / 1000, 2),
                  "low_load_gw":  round(lf.low_load_mw  / 1000, 2),
                  "high_load_gw": round(lf.high_load_mw / 1000, 2),
                  "avg_temp_f":   round(lf.avg_temp_f, 1) if lf.avg_temp_f else None,
                  "hdd":          round(lf.hdd, 1),
                  "cdd":          round(lf.cdd, 1)}
                 for lf in load_forecasts]

    bench_data: dict = {}
    try:
        bench_data = cfg["bench_fn"](session)
    except Exception:
        log.exception("%s: 7-day benchmark fetch failed", iso.upper())

    backtest: dict = {}
    try:
        backtest = run_load_backtest(model, str(cfg["training_path"]))
    except Exception:
        log.exception("%s: backtest computation failed", iso.upper())

    # ── AutoTheta ensemble blend (75% XGBoost + 25% Theta) ──────────────────
    try:
        hist_mw = [gw * 1000 for gw in backtest.get("actual_gw", [])]
        if len(hist_mw) >= 60:
            theta_fcst = predict_theta(iso, hist_mw, horizon=len(load_data))
            if theta_fcst:
                load_data = blend_with_xgboost(load_data, theta_fcst)
                log.info("%s: AutoTheta blend applied", iso.upper())
    except Exception:
        log.exception("%s: AutoTheta blend failed — using XGBoost-only forecast", iso.upper())

    result = {
        "load":            load_data,
        "dates":           forecast_dates_strs,
        "comparison":      comparison,
        cfg["bench_key"]:  bench_data,
        "backtest":        backtest,
        "hindcast":        hindcast,
    }
    # Enrich result with net load and price forecast before caching
    try:
        result["net_load"] = fetch_net_load_forecast(iso, forecast_dates_list, session)
    except Exception:
        log.exception("%s: net load computation failed", iso.upper())
        result["net_load"] = []

    try:
        result["price_forecast"] = forecast_prices(iso, load_data)
    except Exception:
        log.exception("%s: price forecast failed", iso.upper())
        result["price_forecast"] = []

    # ── SHAP explanation for day-0 forecast ──────────────────────────────────
    _SHAP_GROUPS = {
        "Temperature today": [0, 1, 2, 3, 4, 5, 6],
        "Seasonal pattern":  [7, 8],
        "Day / Calendar":    [9, 10, 11, 12, 13, 14, 15, 16, 21, 22],
        "Recent temps":      [17, 18, 19, 20, 23, 24, 25, 26],
        "Humidity & Wind":   [27, 28, 29, 30],
        "Load momentum":     [31, 32, 33],
    }
    try:
        d0 = forecast_dates_list[0]
        def _c_map_d0(src):
            return {loc["label"]: (src.get(loc["label"]) or [None]*15)[0]
                    for loc in locations
                    if (src.get(loc["label"]) or [None]*15)[0] is not None}
        avg_f0 = weighted_avg_fn(_c_map_d0(avg_c)) if _c_map_d0(avg_c) else 65.0
        hi_f0  = weighted_avg_fn(_c_map_d0(hi_c))  if _c_map_d0(hi_c)  else avg_f0
        lo_f0  = weighted_avg_fn(_c_map_d0(lo_c))  if _c_map_d0(lo_c)  else avg_f0
        app_f0 = weighted_avg_fn(_c_map_d0(apparent_hi_c)) if _c_map_d0(apparent_hi_c) else None
        dew_f0 = weighted_avg_fn(_c_map_d0(dewpoint_c))    if _c_map_d0(dewpoint_c)    else None
        wmap   = _c_map_d0(wind_kph)
        wind_mph0 = None
        if wmap:
            tw = sum(loc["weight"] for loc in locations if loc["label"] in wmap)
            wind_mph0 = sum(loc["weight"] * wmap[loc["label"]]
                            for loc in locations if loc["label"] in wmap) / tw * 0.621371
        lag1_f  = recent_avg_f[-1] if len(recent_avg_f) >= 1 else avg_f0
        lag2_f  = recent_avg_f[-2] if len(recent_avg_f) >= 2 else avg_f0
        lag7_f  = recent_avg_f[-7] if len(recent_avg_f) >= 7 else avg_f0
        rv      = [recent_avg_f[-k] for k in range(1, 8) if k <= len(recent_avg_f)]
        roll7_f = sum(rv) / len(rv) if rv else avg_f0
        # Load lags for SHAP day-0
        ll1_shap = recent_actual_loads_mw[-1] if recent_actual_loads_mw else None
        ll7_shap = recent_actual_loads_mw[-7] if recent_actual_loads_mw and len(recent_actual_loads_mw) >= 7 else None
        ll_r7_shap = (sum(recent_actual_loads_mw[-7:]) / min(7, len(recent_actual_loads_mw))
                      if recent_actual_loads_mw else None)
        feats_d0, _, _ = _bf(avg_f0, hi_f0, lo_f0, d0, lag1_f, lag2_f, lag7_f, roll7_f,
                              apparent_hi_f=app_f0, dewpoint_hi_f=dew_f0, wind_speed_mph=wind_mph0,
                              load_lag1_mw=ll1_shap, load_lag7_mw=ll7_shap,
                              rolling7_load_mw=ll_r7_shap)
        contribs = model.explain_prediction(feats_d0)
        if contribs:
            result["shap"] = {
                "groups": {g: round(sum(contribs[i] for i in idx), 1)
                           for g, idx in _SHAP_GROUPS.items()},
                "base_mw": round(contribs[-1], 1),
            }
    except Exception:
        log.exception("%s: SHAP computation failed — skipping explainability panel", iso.upper())

    # ── Carbon + demand-response windows ─────────────────────────────────────
    try:
        daily_gw   = {d["date"]: d["mean_load_gw"] for d in load_data[:2]}
        current_ci = fetch_carbon_intensity(iso, session)
        if current_ci:
            result["carbon"] = current_ci
        dr = compute_dr_windows(iso, daily_gw, current_ci, session)
        if dr:
            result["demand_response"] = dr
    except Exception:
        log.exception("%s: carbon / DR computation failed", iso.upper())

    # ── Demand percentile (uses backtest actual history) ──────────────────────
    try:
        today_gw = load_data[0]["mean_load_gw"] if load_data else None
        if today_gw and backtest.get("dates"):
            bt_dates   = backtest["dates"]
            bt_actuals = backtest.get("actual_gw", [])
            today_month = date.today().month
            same_month  = [(d, gw) for d, gw in zip(bt_dates, bt_actuals)
                           if int(d[5:7]) == today_month]
            if len(same_month) >= 15:
                vals = [gw for _, gw in same_month]
                pct  = round(sum(1 for v in vals if v <= today_gw) / len(vals) * 100)
                m_avg = round(sum(vals) / len(vals), 1)
                m_max = max(vals)
                # Last date in history when load exceeded today's forecast
                higher = [(d, gw) for d, gw in same_month if gw > today_gw]
                last_higher = higher[-1][0] if higher else None
                result["percentile"] = {
                    "percentile":      pct,
                    "month_avg_gw":    m_avg,
                    "month_max_gw":    round(m_max, 1),
                    "last_higher_date": last_higher,
                    "n_days":          len(same_month),
                }
    except Exception:
        log.exception("%s: percentile computation failed", iso.upper())

    try:
        cache_file.parent.mkdir(parents=True, exist_ok=True)
        cache_file.write_text(json.dumps(result))
    except OSError:
        log.warning("%s: could not write forecast cache to %s", iso.upper(), cache_file)

    record_forecast(iso, load_data)

    log.info("%s: forecast complete — %d days, hindcast %d days, backtest MAPE test=%.1f%%",
             iso.upper(), len(load_data), len(hindcast),
             backtest.get("mape_test") or 0)
    return result


def fetch_pjm_load_forecast(force=False):
    return _fetch_iso_forecast("pjm", force=force)


def fetch_caiso_load_forecast(force=False):
    return _fetch_iso_forecast("caiso", force=force)


def fetch_ercot_load_forecast(force=False):
    return _fetch_iso_forecast("ercot", force=force)


def fetch_miso_load_forecast(force=False):
    return _fetch_iso_forecast("miso", force=force)


# ---------------------------------------------------------------------------
# Startup — pre-load all ISO comparison data so EIA quota isn't exhausted
# by the time the user clicks ERCOT or MISO tabs.
# ---------------------------------------------------------------------------
_build_iso_configs()

log.info("Loading PJM forecast at startup...")
_startup_data = fetch_pjm_load_forecast()


def _prefetch_remaining_isos():
    """
    Warm the full forecast pipeline (load, price, net load, comparison,
    backtest) for every ISO besides PJM, which is already warmed
    synchronously above. Without this, the first dashboard visitor to click
    an un-warmed ISO tab triggers a live fetch mid-request — up to ~90s for
    CAISO/SPP cold — instead of hitting the cache _fetch_iso_forecast
    already checks for. Runs in the background so it never blocks app
    startup; each ISO's own try/except means one slow/failing ISO can't
    hold up the rest.
    """
    import time as _t
    for iso in ("caiso", "ercot", "miso", "nyiso", "isone", "spp"):
        try:
            result = _fetch_iso_forecast(iso)
            n = len(result.get("load") or [])
            log.info("%s: startup prefetch done — %d days cached", iso.upper(), n)
        except Exception:
            log.exception("%s: startup prefetch failed", iso.upper())
        _t.sleep(1)


import threading as _threading
_threading.Thread(target=_prefetch_remaining_isos, daemon=True).start()

log.info("Dashboard ready at http://127.0.0.1:8050")


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------
def card(title, value, sub="", color="#0f172a"):
    return html.Div(
        style={"background": "#ffffff", "border": "1px solid #e2e8f0",
               "borderRadius": "8px", "padding": "10px 16px",
               "boxShadow": "0 1px 3px rgba(0,0,0,0.05)", "minWidth": "120px"},
        children=[
            html.Div(title, style={"fontSize": "10px", "color": "#94a3b8",
                                   "textTransform": "uppercase", "letterSpacing": "0.06em"}),
            html.Div(value, style={"fontSize": "22px", "fontWeight": 700,
                                   "color": color, "margin": "2px 0"}),
            html.Div(sub, style={"fontSize": "11px", "color": "#94a3b8"}),
        ],
    )


_TAB_STYLE = {"padding": "10px 22px", "fontSize": "13px", "color": "#64748b",
              "backgroundColor": "#ffffff"}
_TAB_SEL   = {"padding": "10px 22px", "fontSize": "13px", "fontWeight": 600,
              "color": "#0f172a", "backgroundColor": "#ffffff",
              "borderTop": "2px solid #2563eb"}



# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
_SITE_URL = "https://jollyakshay-grid-dashboard.hf.space"
_SITE_DESCRIPTION = (
    "Live load forecasts, wholesale forward curves, and capacity-market analysis for all "
    "7 major US ISOs (PJM, CAISO, ERCOT, MISO, NYISO, ISO-NE, SPP) — plus a free public tool "
    "explaining what powers any US zip code and why electricity costs are rising, including "
    "the effect of data center demand growth on capacity prices."
)

app = dash.Dash(
    __name__,
    title="Grid Load Forecast — US Power Grid",
    meta_tags=[
        {"name": "viewport", "content": "width=device-width, initial-scale=1"},
        {"name": "description", "content": _SITE_DESCRIPTION},
        {"property": "og:type", "content": "website"},
        {"property": "og:url", "content": _SITE_URL},
        {"property": "og:title", "content": "US Grid Intelligence Platform"},
        {"property": "og:description", "content": _SITE_DESCRIPTION},
        {"name": "twitter:card", "content": "summary"},
        {"name": "twitter:title", "content": "US Grid Intelligence Platform"},
        {"name": "twitter:description", "content": _SITE_DESCRIPTION},
    ],
    suppress_callback_exceptions=True,
)

_main_dashboard_layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif",
           "backgroundColor": "#f8fafc", "minHeight": "100vh", "color": "#1e293b"},
    children=[

        # ── Header ────────────────────────────────────────────────────────────
        html.Div(
            style={"padding": "14px 28px", "borderBottom": "1px solid #e2e8f0",
                   "backgroundColor": "#ffffff",
                   "display": "flex", "alignItems": "center",
                   "justifyContent": "space-between",
                   "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"},
            children=[
                html.Div([
                    html.H1("Grid Load Forecast",
                            style={"margin": 0, "fontSize": "18px", "fontWeight": 700,
                                   "color": "#0f172a"}),
                    html.Span(f"PJM · CAISO · ERCOT · MISO · NYISO · ISO-NE · SPP  ·  Updated {datetime.now().strftime('%d %b %Y')}",
                              style={"color": "#94a3b8", "fontSize": "12px"}),
                ]),
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "14px"},
                    children=[
                        dcc.Link(
                            "What Powers My Zip Code →", href="/my-electricity",
                            style={"color": "#2563eb", "fontSize": "13px", "fontWeight": 500,
                                   "textDecoration": "none"},
                        ),
                        dcc.Link(
                            "Futures Pricer →", href="/futures-pricer",
                            style={"color": "#2563eb", "fontSize": "13px", "fontWeight": 500,
                                   "textDecoration": "none"},
                        ),
                        html.Button(
                            "⟳ Refresh", id="refresh-btn",
                            style={"background": "#f1f5f9", "border": "1px solid #e2e8f0",
                                   "color": "#475569", "padding": "6px 16px",
                                   "borderRadius": "6px", "cursor": "pointer", "fontSize": "13px"},
                        ),
                    ],
                ),
            ],
        ),

        # ── Body ──────────────────────────────────────────────────────────────
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px 24px"},
            children=[

                # Intro / navigation cards — first-time visitors (not just traders)
                # land here, so this exists to say what the platform is before
                # dropping into the ISO-selector/forecast tool below.
                html.Div(
                    style={"marginBottom": "24px"},
                    children=[
                        html.P(
                            "A free, live intelligence platform for the US electricity grid — built to track how "
                            "data center and AI-driven demand growth is reshaping wholesale prices, capacity "
                            "auctions, and consumer electricity costs across all 7 major US ISOs.",
                            style={"fontSize": "14px", "color": "#475569", "maxWidth": "820px",
                                   "lineHeight": "1.5", "marginBottom": "16px"},
                        ),
                        html.Div(
                            style={"display": "flex", "gap": "14px", "flexWrap": "wrap"},
                            children=[
                                dcc.Link(html.Div([
                                    html.Div("Load Forecast & Market Data", style={"fontWeight": 700, "fontSize": "14px", "color": "#0f172a", "marginBottom": "4px"}),
                                    html.Div("15-day demand forecasts, real-time fuel mix, and capacity-market data for any of the 7 ISOs.",
                                              style={"fontSize": "12px", "color": "#64748b"}),
                                ], style={"padding": "14px 16px", "background": "#ffffff", "border": "1px solid #e2e8f0",
                                          "borderRadius": "8px", "width": "230px"}), href="/", style={"textDecoration": "none"}),
                                dcc.Link(html.Div([
                                    html.Div("Futures Pricer", style={"fontWeight": 700, "fontSize": "14px", "color": "#0f172a", "marginBottom": "4px"}),
                                    html.Div("Fair-value check for PJM Western Hub monthly forward power quotes against a modeled forward curve.",
                                              style={"fontSize": "12px", "color": "#64748b"}),
                                ], style={"padding": "14px 16px", "background": "#ffffff", "border": "1px solid #e2e8f0",
                                          "borderRadius": "8px", "width": "230px"}), href="/futures-pricer", style={"textDecoration": "none"}),
                                dcc.Link(html.Div([
                                    html.Div("What Powers My Zip Code", style={"fontWeight": 700, "fontSize": "14px", "color": "#0f172a", "marginBottom": "4px"}),
                                    html.Div("Enter any US zip code: see your fuel mix, provider, rate trend, and why capacity costs are rising.",
                                              style={"fontSize": "12px", "color": "#64748b"}),
                                ], style={"padding": "14px 16px", "background": "#ffffff", "border": "1px solid #e2e8f0",
                                          "borderRadius": "8px", "width": "230px"}), href="/my-electricity", style={"textDecoration": "none"}),
                            ],
                        ),
                    ],
                ),

                # ISO selector
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "12px",
                           "marginBottom": "18px"},
                    children=[
                        html.Span("Grid / ISO:",
                                  style={"fontSize": "13px", "color": "#64748b",
                                         "fontWeight": 500}),
                        dcc.RadioItems(
                            id="iso-selector",
                            options=[
                                {"label": "PJM (Eastern US)",   "value": "pjm"},
                                {"label": "CAISO (California)", "value": "caiso"},
                                {"label": "ERCOT (Texas)",      "value": "ercot"},
                                {"label": "MISO (Midwest)",     "value": "miso"},
                                {"label": "NYISO (New York)",   "value": "nyiso"},
                                {"label": "ISO-NE (New England)", "value": "isone"},
                                {"label": "SPP (Southwest)",    "value": "spp"},
                            ],
                            value="pjm", inline=True,
                            inputStyle={"marginRight": "5px"},
                            labelStyle={"marginRight": "20px", "fontSize": "14px",
                                        "cursor": "pointer", "fontWeight": 500},
                        ),
                    ],
                ),

                # Extreme event alert (hidden when not needed)
                html.Div(id="extreme-alert", style={"marginBottom": "12px"}),

                # Summary cards
                html.Div(id="load-cards",
                         style={"display": "flex", "gap": "12px",
                                "flexWrap": "wrap", "marginBottom": "16px"}),

                # ── AI Forecast Brief ──────────────────────────────────────────
                html.Div(id="ai-brief-card",
                         style={"marginBottom": "16px"}),

                # ── Ensemble toggle ────────────────────────────────────────────
                html.Div(
                    style={"display": "flex", "alignItems": "center", "gap": "10px",
                           "marginBottom": "12px"},
                    children=[
                        dcc.Checklist(
                            id="ensemble-toggle",
                            options=[{"label": " Apply teleconnection ensemble (NAO / AO / PNA / MJO)",
                                      "value": "on"}],
                            value=[],
                            inputStyle={"cursor": "pointer"},
                            labelStyle={"fontSize": "12px", "color": "#475569",
                                        "cursor": "pointer"},
                        ),
                        html.Span(id="ensemble-badge", style={"fontSize": "11px"}),
                    ],
                ),

                # ── Forecast chart ─────────────────────────────────────────────
                html.Div(
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "baseline", "marginBottom": "4px"},
                            children=[
                                html.Div("15-Day Load Forecast",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Div(id="forecast-subtitle",
                                         style={"fontSize": "11px", "color": "#94a3b8"}),
                            ],
                        ),
                        dcc.Graph(id="load-forecast-chart", style={"height": "320px"},
                                  config={"displayModeBar": False}),
                    ],
                ),

                # ── SHAP Explainability ────────────────────────────────────────
                html.Div(
                    id="shap-panel",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "4px"},
                            children=[
                                html.Div("What's driving today's forecast?",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Span("Tree SHAP — XGBoost attribution",
                                          style={"fontSize": "10px", "color": "#94a3b8",
                                                 "background": "#f1f5f9",
                                                 "borderRadius": "4px",
                                                 "padding": "2px 7px"}),
                            ],
                        ),
                        html.Div("Each bar shows how much that factor adds or subtracts from "
                                 "the expected baseline load. Positive = pushes load higher.",
                                 style={"fontSize": "11px", "color": "#94a3b8",
                                        "marginBottom": "8px"}),
                        dcc.Graph(id="shap-chart", style={"height": "200px"},
                                  config={"displayModeBar": False}),
                    ],
                ),

                # ── Carbon intensity ──────────────────────────────────────────
                html.Div(
                    id="carbon-panel",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "4px"},
                            children=[
                                html.Div("Grid Carbon Intensity",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Span("EIA v2 fuel-type · ~12-24h lag",
                                          style={"fontSize": "10px", "color": "#94a3b8",
                                                 "background": "#f1f5f9",
                                                 "borderRadius": "4px",
                                                 "padding": "2px 7px"}),
                            ],
                        ),
                        html.Div(id="carbon-intensity-number",
                                 style={"marginBottom": "10px"}),
                        dcc.Graph(id="carbon-fuel-mix-chart", style={"height": "200px"},
                                  config={"displayModeBar": False}),
                    ],
                ),

                # ── Demand-response windows ────────────────────────────────────
                html.Div(
                    id="dr-panel",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "4px"},
                            children=[
                                html.Div("Flexible Load Opportunity Windows",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Span("Best 4-hour blocks · next 48 h",
                                          style={"fontSize": "10px", "color": "#94a3b8",
                                                 "background": "#f1f5f9",
                                                 "borderRadius": "4px",
                                                 "padding": "2px 7px"}),
                            ],
                        ),
                        html.Div("When to shift EV charging, HVAC pre-conditioning, or "
                                 "industrial demand for lowest carbon and cost.",
                                 style={"fontSize": "11px", "color": "#94a3b8",
                                        "marginBottom": "10px"}),
                        html.Div(id="dr-recommendation",
                                 style={"marginBottom": "12px"}),
                        dcc.Graph(id="dr-chart", style={"height": "220px"},
                                  config={"displayModeBar": False}),
                    ],
                ),

                # ── Day-ahead price forecast ───────────────────────────────────
                html.Div(
                    id="price-forecast-panel",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "12px"},
                            children=[
                                html.Div("Day-Ahead Price Forecast ($/MWh)",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Span("EIA regression model",
                                          style={"fontSize": "10px", "color": "#94a3b8",
                                                 "background": "#f1f5f9",
                                                 "borderRadius": "4px",
                                                 "padding": "2px 7px"}),
                            ],
                        ),
                        dcc.Graph(id="price-forecast-chart", style={"height": "220px"},
                                  config={"displayModeBar": False}),
                    ],
                ),

                # ── 12-Month Forward Curve ────────────────────────────────────
                html.Div(
                    id="forward-curve-panel",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "center", "marginBottom": "12px"},
                            children=[
                                html.Div("12-Month Forward Price Curve",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Span("OLS + EIA STEO gas curve",
                                          style={"fontSize": "10px", "color": "#94a3b8",
                                                 "background": "#f1f5f9",
                                                 "borderRadius": "4px",
                                                 "padding": "2px 7px"}),
                            ],
                        ),
                        dcc.Graph(id="forward-curve-chart", style={"height": "280px"},
                                  config={"displayModeBar": False}),
                        dcc.Graph(id="spark-spread-chart", style={"height": "200px"},
                                  config={"displayModeBar": False}),
                        html.Div(id="forward-curve-stats",
                                 style={"marginTop": "10px", "fontSize": "11px",
                                        "color": "#94a3b8"}),
                    ],
                ),

                # ── Model accuracy backtest ───────────────────────────────────
                html.Div(
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "10px", "marginBottom": "12px"},
                            children=[
                                html.Div("Model Accuracy — Backtest",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Div(id="backtest-mape-badges",
                                         style={"display": "flex", "gap": "6px"}),
                            ],
                        ),
                        html.Div(
                            "Actual vs model-predicted load on the held-out test set (last 20% of "
                            "training data, shown in blue). Dotted line = model prediction.",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "8px"},
                        ),
                        dcc.Graph(id="backtest-chart", style={"height": "260px"},
                                  config={"displayModeBar": False}),
                    ],
                ),

                # ── Live forecast verification panel ──────────────────────────
                html.Div(id="verification-panel"),

                # ── Capacity market panel ─────────────────────────────────────
                html.Div(
                    id="capacity-market-panel",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div("Capacity Market Snapshot",
                                 style={"fontSize": "13px", "fontWeight": 600,
                                        "color": "#0f172a", "marginBottom": "14px"}),
                        html.Div(id="capacity-market-content"),
                    ],
                ),

                # ── Methodology note ───────────────────────────────────────────
                html.Div(
                    style={"backgroundColor": "#f8fafc", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "18px 22px",
                           "marginBottom": "32px"},
                    children=[
                        html.Div("Methodology",
                                 style={"fontSize": "12px", "fontWeight": 600,
                                        "color": "#475569", "textTransform": "uppercase",
                                        "letterSpacing": "0.06em", "marginBottom": "8px"}),
                        html.P(
                            "Separate XGBoost models trained for each of 7 ISOs on 2 years of hourly EIA demand "
                            "data aggregated to daily averages, paired with ERA5 reanalysis temperatures "
                            "population-weighted across 12 representative locations per ISO. "
                            "34 features: HDD/CDD from daily average, high, and low temperatures; apparent "
                            "temperature; dewpoint; wind speed; day-of-week encoding; US federal holiday flags; "
                            "T−1, T−2, T−7 temperature lags; 7-day rolling temperature average; load autocorrelation "
                            "lags (yesterday's actual load, 7 days ago, and 7-day rolling average). "
                            "Uncertainty bands use Conformalized Quantile Regression (CQR) — guaranteed ≥90% "
                            "empirical coverage. AutoTheta ensemble blended at 25% for trend-seasonality signal. "
                            "Forward forecasts use GFS NWP via Open-Meteo (15-day horizon); hindcast uses ERA5.",
                            style={"fontSize": "12px", "color": "#64748b",
                                   "lineHeight": "1.7", "margin": "0 0 8px 0"},
                        ),
                        html.P(
                            "ISO coverage — "
                            "PJM (Eastern US, ~65 GW peak): 12 locations from Chicago to Washington DC. "
                            "CAISO (California, ~45 GW peak): 12 locations from San Diego to Sacramento. "
                            "ERCOT (Texas, ~80 GW peak): 12 locations from Houston to Amarillo. "
                            "MISO (Midcontinent, ~120 GW peak): 12 locations from New Orleans to Fargo. "
                            "NYISO (New York, ~35 GW peak): 12 locations. "
                            "ISO-NE (New England, ~28 GW peak): 12 locations. "
                            "SPP (Southwest Power Pool, ~60 GW peak): 12 locations. "
                            "Benchmarks — PJM: PJM DataMiner 7-day forecast. CAISO: OASIS 7-day system forecast. "
                            "ERCOT: ERCOT public reports API or EIA day-ahead. MISO: EIA day-ahead demand. "
                            "Accuracy note: the hindcast MAPE (~0.4–0.5%) reflects in-sample model fit under ERA5 "
                            "observed temperatures — it does not represent live forecast skill. The 'Verified MAPE' "
                            "card shows actual day-ahead error verified against EIA reported actuals. "
                            "Live forward forecast error is consistent with industry norms of 1–3% day-ahead.",
                            style={"fontSize": "12px", "color": "#64748b",
                                   "lineHeight": "1.7", "margin": 0},
                        ),
                    ],
                ),

            ],
        ),

        # ── AI Chat panel ─────────────────────────────────────────────────────
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto",
                   "padding": "0 24px 40px 24px"},
            children=[
                html.Div(
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)"},
                    children=[
                        html.Div(
                            style={"display": "flex", "alignItems": "center",
                                   "gap": "8px", "marginBottom": "12px"},
                            children=[
                                html.Div("Ask the Forecast AI",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Span(f"Powered by {_llm.provider_label()}",
                                          style={"fontSize": "10px", "color": "#94a3b8",
                                                 "background": "#f1f5f9",
                                                 "borderRadius": "4px",
                                                 "padding": "2px 7px"}),
                            ],
                        ),
                        # Chat history display
                        html.Div(id="chat-history",
                                 style={"minHeight": "60px", "maxHeight": "320px",
                                        "overflowY": "auto", "marginBottom": "12px",
                                        "padding": "8px 0",
                                        "borderBottom": "1px solid #f1f5f9"}),
                        # Suggested question chips
                        html.Div(
                            style={"display": "flex", "gap": "6px", "flexWrap": "wrap",
                                   "marginBottom": "10px"},
                            children=[
                                html.Button(
                                    "What's driving the forecast peak?",
                                    id="chip-q1", n_clicks=0,
                                    style={"fontSize": "11px", "padding": "5px 10px",
                                           "borderRadius": "14px", "border": "1px solid #cbd5e1",
                                           "background": "#f8fafc", "color": "#475569",
                                           "cursor": "pointer", "whiteSpace": "nowrap"},
                                ),
                                html.Button(
                                    "How confident is this 15-day forecast?",
                                    id="chip-q2", n_clicks=0,
                                    style={"fontSize": "11px", "padding": "5px 10px",
                                           "borderRadius": "14px", "border": "1px solid #cbd5e1",
                                           "background": "#f8fafc", "color": "#475569",
                                           "cursor": "pointer", "whiteSpace": "nowrap"},
                                ),
                                html.Button(
                                    "How does our forecast compare to the ISO's day-ahead?",
                                    id="chip-q3", n_clicks=0,
                                    style={"fontSize": "11px", "padding": "5px 10px",
                                           "borderRadius": "14px", "border": "1px solid #cbd5e1",
                                           "background": "#f8fafc", "color": "#475569",
                                           "cursor": "pointer", "whiteSpace": "nowrap"},
                                ),
                            ],
                        ),
                        # Input row
                        html.Div(
                            style={"display": "flex", "gap": "8px", "alignItems": "flex-end"},
                            children=[
                                dcc.Textarea(
                                    id="chat-input",
                                    placeholder="Ask about the forecast… e.g. 'Why is load spiking Thursday?' or 'What does the ERCOT reserve margin mean?'",
                                    style={"flex": 1, "height": "60px", "resize": "vertical",
                                           "fontSize": "13px", "padding": "8px 12px",
                                           "border": "1px solid #e2e8f0",
                                           "borderRadius": "6px", "color": "#1e293b",
                                           "fontFamily": "Inter, system-ui, sans-serif"},
                                ),
                                html.Button(
                                    "Send", id="chat-submit",
                                    style={"background": "#2563eb", "color": "#ffffff",
                                           "border": "none", "borderRadius": "6px",
                                           "padding": "10px 20px", "fontSize": "13px",
                                           "fontWeight": 600, "cursor": "pointer",
                                           "height": "60px", "minWidth": "70px"},
                                ),
                                html.Button(
                                    "Clear", id="chat-clear",
                                    style={"background": "#f1f5f9", "color": "#475569",
                                           "border": "1px solid #e2e8f0",
                                           "borderRadius": "6px", "padding": "10px 16px",
                                           "fontSize": "13px", "cursor": "pointer",
                                           "height": "60px"},
                                ),
                            ],
                        ),
                    ],
                ),
            ],
        ),

        # Store
        dcc.Store(id="load-forecast-store"),
        dcc.Store(id="forward-curve-store"),
        dcc.Store(id="carbon-store"),
        dcc.Store(id="ensemble-store"),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Interval(id="daily-refresh", interval=24 * 60 * 60 * 1000, n_intervals=0),
        dcc.Interval(id="forecast-refresh", interval=15 * 60 * 1000, n_intervals=0),
    ],
)


# ---------------------------------------------------------------------------
# Futures Pricer — separate page (PJM Western Hub only for now)
# ---------------------------------------------------------------------------
def _next_n_months(n=24):
    """[(YYYY-MM, 'Mon YYYY'), ...] for the next n months, starting next month."""
    today = date.today()
    options = []
    y, m = today.year, today.month
    for _ in range(n):
        m += 1
        if m > 12:
            m = 1
            y += 1
        d = date(y, m, 1)
        options.append((d.strftime("%Y-%m"), d.strftime("%b %Y")))
    return options


def _futures_pricer_layout():
    return html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif",
               "backgroundColor": "#f8fafc", "minHeight": "100vh", "color": "#1e293b"},
        children=[
            html.Div(
                style={"padding": "14px 28px", "borderBottom": "1px solid #e2e8f0",
                       "backgroundColor": "#ffffff", "display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"},
                children=[
                    html.Div([
                        html.H1("Futures Contract Pricer", style={"margin": 0, "fontSize": "18px",
                                                                    "fontWeight": 700, "color": "#0f172a"}),
                        html.Span("PJM Western Hub — compare a real market quote against the model's forward curve",
                                  style={"color": "#94a3b8", "fontSize": "12px"}),
                    ]),
                    dcc.Link("← Back to Dashboard", href="/",
                             style={"color": "#2563eb", "fontSize": "13px", "fontWeight": 500,
                                    "textDecoration": "none"}),
                ],
            ),
            html.Div(
                style={"maxWidth": "760px", "margin": "0 auto", "padding": "28px 24px"},
                children=[
                    html.Div(
                        style={"background": "#ffffff", "border": "1px solid #e2e8f0", "borderRadius": "10px",
                               "padding": "22px 24px", "marginBottom": "20px"},
                        children=[
                            html.Div(
                                style={"display": "flex", "gap": "16px", "flexWrap": "wrap", "marginBottom": "16px"},
                                children=[
                                    html.Div([
                                        html.Label("Delivery month", style={"fontSize": "12px", "color": "#64748b",
                                                                             "display": "block", "marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="fp-delivery-month",
                                            options=[{"label": lbl, "value": ym} for ym, lbl in _next_n_months()],
                                            value=_next_n_months()[0][0],
                                            clearable=False, style={"width": "160px"},
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Type", style={"fontSize": "12px", "color": "#64748b",
                                                                   "display": "block", "marginBottom": "4px"}),
                                        dcc.RadioItems(
                                            id="fp-peak-type",
                                            options=[
                                                {"label": " Monthly avg", "value": "monthly_avg"},
                                                {"label": " On-peak",     "value": "on_peak"},
                                                {"label": " Off-peak",    "value": "off_peak"},
                                            ],
                                            value="monthly_avg", inline=True,
                                            style={"fontSize": "13px"}, inputStyle={"marginRight": "4px", "marginLeft": "10px"},
                                        ),
                                    ]),
                                    html.Div([
                                        html.Label("Weather scenario", style={"fontSize": "12px", "color": "#64748b",
                                                                               "display": "block", "marginBottom": "4px"}),
                                        dcc.Dropdown(
                                            id="fp-scenario",
                                            options=[{"label": "Cold", "value": "cold"},
                                                     {"label": "Base", "value": "base"},
                                                     {"label": "Hot",  "value": "hot"}],
                                            value="base", clearable=False, style={"width": "120px"},
                                        ),
                                    ]),
                                ],
                            ),
                            html.Div(
                                style={"display": "flex", "gap": "12px", "alignItems": "flex-end"},
                                children=[
                                    html.Div([
                                        html.Label("Quoted price ($/MWh)", style={"fontSize": "12px", "color": "#64748b",
                                                                                   "display": "block", "marginBottom": "4px"}),
                                        dcc.Input(id="fp-quoted-price", type="number", placeholder="e.g. 42.50",
                                                   style={"width": "140px", "padding": "7px 10px",
                                                          "border": "1px solid #cbd5e1", "borderRadius": "6px"}),
                                    ]),
                                    html.Button(
                                        "Price Contract", id="fp-submit", n_clicks=0,
                                        style={"background": "#2563eb", "color": "#ffffff", "border": "none",
                                               "padding": "9px 20px", "borderRadius": "6px", "cursor": "pointer",
                                               "fontSize": "13px", "fontWeight": 600},
                                    ),
                                ],
                            ),
                        ],
                    ),
                    html.Div(id="fp-result"),
                ],
            ),
        ],
    )


_FP_SIGNAL_COLOR = {
    "within_band":       "#16a34a",
    "above_band":        "#ef4444",
    "below_band":        "#ef4444",
    "no_band_available": "#94a3b8",
}
_FP_SIGNAL_LABEL = {
    "within_band":       "Within model's expected range",
    "above_band":        "Quoted above model's expected range",
    "below_band":        "Quoted below model's expected range",
    "no_band_available": "No statistical band available for comparison",
}


@app.callback(
    Output("fp-result", "children"),
    Input("fp-submit", "n_clicks"),
    State("fp-delivery-month", "value"),
    State("fp-peak-type", "value"),
    State("fp-scenario", "value"),
    State("fp-quoted-price", "value"),
)
def render_futures_pricer_result(n_clicks, delivery_month, peak_type, scenario, quoted_price):
    if not n_clicks:
        return []
    if quoted_price is None:
        return html.Div("Enter a quoted price.", style={"color": "#ef4444", "fontSize": "13px"})

    from temperature_modeling.futures_pricer import price_contract
    try:
        r = price_contract("pjm", delivery_month, peak_type, float(quoted_price), scenario=scenario)
    except ValueError as exc:
        return html.Div(str(exc), style={"color": "#ef4444", "fontSize": "13px"})
    except Exception:
        log.exception("Futures pricer failed")
        return html.Div("Pricing failed — try again in a moment.", style={"color": "#ef4444", "fontSize": "13px"})

    band_str = (f"${r['band_low']:.2f} – ${r['band_high']:.2f}"
                if r["band_source"] == "cqr" else "unavailable")
    spread_str = f"${r['spread_usd_mwh']:+.2f}/MWh"
    if r["spread_pct"] is not None:
        spread_str += f" ({r['spread_pct']:+.1f}%)"

    return html.Div(
        style={"background": "#ffffff", "border": "1px solid #e2e8f0", "borderRadius": "10px", "padding": "22px 24px"},
        children=[
            html.Div(
                style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "16px"},
                children=[
                    card("Model Price", f"${r['model_price']:.2f}", "per MWh", "#0f172a"),
                    card("90% Band", band_str, r["band_source"], "#6366f1"),
                    card("Quoted", f"${r['quoted_price']:.2f}", "per MWh", "#0f172a"),
                    card("Spread", spread_str, "quoted − model", "#f97316"),
                ],
            ),
            html.Div(
                _FP_SIGNAL_LABEL.get(r["signal"], r["signal"]),
                style={"fontSize": "14px", "fontWeight": 700,
                       "color": _FP_SIGNAL_COLOR.get(r["signal"], "#334155"),
                       "marginBottom": "10px"},
            ),
            html.Div(
                f"{r['lead_months']} month(s) out · model: {r['model_source']} · "
                f"peak split: {r.get('peak_split_method') or 'n/a'} · confidence: {r['confidence']}",
                style={"fontSize": "12px", "color": "#64748b", "marginBottom": "8px"},
            ),
            html.Ul(
                [html.Li(note, style={"fontSize": "12px", "color": "#94a3b8"}) for note in r["confidence_notes"]],
                style={"margin": 0, "paddingLeft": "18px"},
            ) if r["confidence_notes"] else None,
        ],
    )


# ---------------------------------------------------------------------------
# "What Powers My Zip Code" — consumer-facing page, public API too (no key)
# ---------------------------------------------------------------------------
def _electricity_explainer_layout():
    return html.Div(
        style={"fontFamily": "Inter, system-ui, sans-serif",
               "backgroundColor": "#f8fafc", "minHeight": "100vh", "color": "#1e293b"},
        children=[
            html.Div(
                style={"padding": "14px 28px", "borderBottom": "1px solid #e2e8f0",
                       "backgroundColor": "#ffffff", "display": "flex", "alignItems": "center",
                       "justifyContent": "space-between", "boxShadow": "0 1px 4px rgba(0,0,0,0.06)"},
                children=[
                    html.Div([
                        html.H1("What Powers My Zip Code", style={"margin": 0, "fontSize": "20px",
                                                                    "fontWeight": 700, "color": "#0f172a"}),
                        html.Span("Where your electricity comes from, how it's priced, and who provides it",
                                  style={"color": "#94a3b8", "fontSize": "13px"}),
                    ]),
                    dcc.Link("← Back to Dashboard", href="/",
                             style={"color": "#2563eb", "fontSize": "13px", "fontWeight": 500,
                                    "textDecoration": "none"}),
                ],
            ),
            html.Div(
                style={"maxWidth": "720px", "margin": "0 auto", "padding": "32px 24px"},
                children=[
                    html.Div(
                        style={"background": "#ffffff", "border": "1px solid #e2e8f0", "borderRadius": "10px",
                               "padding": "24px", "marginBottom": "24px", "display": "flex",
                               "gap": "12px", "alignItems": "flex-end"},
                        children=[
                            html.Div([
                                html.Label("Enter your zip code", style={"fontSize": "13px", "color": "#64748b",
                                                                          "display": "block", "marginBottom": "6px"}),
                                dcc.Input(id="elec-zip", type="text", placeholder="e.g. 19104",
                                          maxLength=5, style={"width": "160px", "padding": "9px 12px",
                                                               "border": "1px solid #cbd5e1", "borderRadius": "6px",
                                                               "fontSize": "15px"}),
                            ]),
                            html.Button(
                                "Look Up", id="elec-submit", n_clicks=0,
                                style={"background": "#2563eb", "color": "#ffffff", "border": "none",
                                       "padding": "10px 22px", "borderRadius": "6px", "cursor": "pointer",
                                       "fontSize": "14px", "fontWeight": 600},
                            ),
                        ],
                    ),
                    html.Div(id="elec-result"),
                ],
            ),
        ],
    )


def _elec_section(title, body):
    return html.Div(
        style={"background": "#ffffff", "border": "1px solid #e2e8f0", "borderRadius": "10px",
               "padding": "22px 24px", "marginBottom": "16px"},
        children=[
            html.H2(title, style={"fontSize": "16px", "fontWeight": 700, "color": "#0f172a",
                                   "margin": "0 0 12px"}),
            body,
        ],
    )


def _unavailable(reason):
    return html.Div(reason, style={"fontSize": "13px", "color": "#94a3b8", "fontStyle": "italic"})


@app.callback(
    Output("elec-result", "children"),
    Input("elec-submit", "n_clicks"),
    State("elec-zip", "value"),
)
def render_electricity_explainer_result(n_clicks, zip_code):
    if not n_clicks:
        return []
    if not zip_code:
        return html.Div("Enter a zip code.", style={"color": "#ef4444", "fontSize": "13px"})

    from temperature_modeling.electricity_explainer import explain_electricity
    try:
        r = explain_electricity(zip_code)
    except ValueError as exc:
        return html.Div(str(exc), style={"color": "#ef4444", "fontSize": "13px"})
    except Exception:
        log.exception("Electricity explainer failed")
        return html.Div("Lookup failed — try again in a moment.", style={"color": "#ef4444", "fontSize": "13px"})

    if not r["found"]:
        return html.Div(f"No data for zip code {zip_code}.", style={"color": "#ef4444", "fontSize": "13px"})

    sections = []

    # Map — where this zip code is, plus nearest plants if available
    loc = r.get("location")
    np_result = r.get("nearest_plants", {})
    plants = np_result["data"] if np_result.get("available") else []
    if loc:
        iso_label = {"pjm": "PJM", "caiso": "CAISO", "ercot": "ERCOT", "miso": "MISO",
                     "nyiso": "NYISO", "isone": "ISO-NE", "spp": "SPP"}.get(r["iso"])
        map_title = f"{iso_label} territory" if iso_label else "not in a wholesale market territory covered here"
        traces = [go.Scattergeo(
            lon=[loc["lon"]], lat=[loc["lat"]], mode="markers+text",
            text=[zip_code], textposition="top center",
            marker=dict(size=12, color="#2563eb", line=dict(width=1.5, color="#ffffff")),
            hovertemplate=f"{loc['display_name']}<extra></extra>", name="You",
        )]
        fig = go.Figure(traces)
        fig.update_layout(
            geo=dict(scope="usa", projection_type="albers usa",
                     showland=True, landcolor="#f1f5f9", showlakes=True, lakecolor="#ffffff",
                     subunitcolor="#cbd5e1", countrycolor="#cbd5e1",
                     center=dict(lat=loc["lat"], lon=loc["lon"]),
                     projection_scale=8 if r["iso"] else 4),
            margin=dict(l=0, r=0, t=0, b=0), height=220,
            paper_bgcolor="#ffffff", showlegend=False,
        )
        sections.append(_elec_section(
            f"Where you are — {map_title}",
            dcc.Graph(figure=fig, config={"displayModeBar": False}, style={"height": "220px"}),
        ))

    # Nearest generating plants
    if plants:
        plant_rows = [
            html.Div(f"{p['plant_name']} ({p['primary_fuel']}) — {p['distance_miles']:.1f} mi, "
                     f"{p['capacity_mw']:,.0f} MW — {p['owner']}",
                     style={"fontSize": "13px", "marginBottom": "5px"})
            for p in plants
        ]
        plant_rows.append(html.Div("From EIA-860, searched within your state only — a plant just over "
                                    "a state line won't show up here even if it's physically closer.",
                                    style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "6px"}))
        sections.append(_elec_section("Nearest generating plants", html.Div(plant_rows)))

    # Who provides your electricity
    util_rows = []
    for u in r["utilities"]:
        rate = f"${u['res_rate_usd_mwh']:.2f}/MWh avg residential" if u.get("res_rate_usd_mwh") else "rate n/a"
        note = f" — {u['note']}" if u.get("note") else ""
        util_rows.append(html.Div(f"{u['name']} ({u['state']}) — {rate}{note}",
                                   style={"fontSize": "14px", "marginBottom": "6px"}))
    state_rate_rows = [
        html.Div(f"Latest {state} statewide residential average ({sr['period']}): ${sr['res_rate_usd_mwh']:.2f}/MWh",
                  style={"fontSize": "13px", "color": "#2563eb", "marginTop": "4px"})
        for state, sr in r["latest_state_rates"].items()
    ]
    sections.append(_elec_section(
        "Who provides your electricity",
        html.Div(util_rows + state_rate_rows + [html.Div(
            f"Per-utility rate from {r['data_vintage_year']} (NREL/OpenEI, annual snapshot); "
            f"statewide average from EIA (monthly, most current available). Your actual bill depends on your specific plan.",
            style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "8px"})]),
    ))

    # Retail rate trend — trailing 24 months, state average
    rrh = r.get("retail_rate_history", {})
    if rrh.get("available"):
        hd = rrh["data"]
        periods = [row["period"] for row in hd["history"]]
        rates = [row["res_rate_usd_mwh"] for row in hd["history"]]
        rate_fig = go.Figure(go.Scatter(
            x=periods, y=rates, mode="lines+markers",
            line=dict(color="#2563eb", width=2), marker=dict(size=4),
            hovertemplate="%{x}<br>$%{y:.2f}/MWh<extra></extra>",
        ))
        rate_fig.update_layout(
            margin=dict(l=45, r=10, t=10, b=30), height=200,
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(title="$/MWh", gridcolor="#e2e8f0"),
            xaxis=dict(title=None, tickangle=-45, nticks=8, gridcolor="#e2e8f0"),
        )
        sections.append(_elec_section(
            f"{hd['state']} residential rate — last 24 months",
            dcc.Graph(figure=rate_fig, config={"displayModeBar": False}, style={"height": "200px"}),
        ))

    # Bill estimator — rate x adjustable usage. Prefer the fresher statewide
    # EIA rate over the per-utility annual snapshot when both exist.
    state_rates = list(r["latest_state_rates"].values())
    util_rates = [u["res_rate_usd_mwh"] for u in r["utilities"] if u.get("res_rate_usd_mwh")]
    if state_rates:
        bill_rate = state_rates[0]["res_rate_usd_mwh"]
    elif util_rates:
        bill_rate = sum(util_rates) / len(util_rates)
    else:
        bill_rate = None
    if bill_rate:
        sections.append(_elec_section(
            "Estimate your bill",
            html.Div([
                html.Label("Monthly usage (kWh) — US average household is about 900 kWh/month",
                           style={"fontSize": "12px", "color": "#64748b", "display": "block", "marginBottom": "6px"}),
                dcc.Input(id="elec-bill-usage", type="number", value=900, min=0, step=50,
                          style={"width": "140px", "padding": "8px", "fontSize": "14px",
                                 "border": "1px solid #cbd5e1", "borderRadius": "6px", "marginBottom": "10px"}),
                dcc.Store(id="elec-bill-rate", data=bill_rate),
                html.Div(id="elec-bill-estimate", style={"fontSize": "18px", "fontWeight": 700, "color": "#0f172a"}),
            ]),
        ))

    if r["non_rto"]:
        sections.append(_elec_section("Your area", _unavailable(r["fuel_mix"]["reason"])))
        return html.Div(sections)

    # What's generating your power right now
    fm = r["fuel_mix"]
    if fm["available"]:
        mix = fm["data"]["fuel_mix"]
        total = sum(mix.values()) or 1
        rows = [html.Div(f"{name}: {mw:,.0f} MW ({100*mw/total:.0f}%)", style={"fontSize": "14px", "marginBottom": "4px"})
                for name, mw in mix.items()]
        rows.append(html.Div(f"{fm['data']['clean_pct']:.0f}% from zero-carbon sources right now",
                              style={"fontSize": "13px", "color": "#16a34a", "marginTop": "8px", "fontWeight": 600}))
        sections.append(_elec_section("What's generating your power right now", html.Div(rows)))
    else:
        sections.append(_elec_section("What's generating your power right now", _unavailable(fm["reason"])))

    # Fuel mix trend — last 24h
    fmh = r.get("fuel_mix_history", {})
    if fmh.get("available"):
        periods = fmh["periods"]
        hist_fig = go.Figure(go.Scatter(
            x=periods, y=fmh["clean_pct"], mode="lines+markers",
            line=dict(color="#16a34a", width=2), marker=dict(size=4),
            hovertemplate="%{x}<br>%{y:.0f}%% clean<extra></extra>",
        ))
        hist_fig.update_layout(
            margin=dict(l=40, r=10, t=10, b=30), height=200,
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(title="% clean energy", range=[0, 100], gridcolor="#e2e8f0"),
            xaxis=dict(title=None, tickangle=-45, nticks=8, gridcolor="#e2e8f0"),
        )
        sections.append(_elec_section(
            "Clean energy % — last 24 hours",
            dcc.Graph(figure=hist_fig, config={"displayModeBar": False}, style={"height": "200px"}),
        ))

    # Best time to use electricity
    btu = r.get("best_time_to_use", {})
    if btu.get("available"):
        d = btu["data"]
        best, lc, lcost = d["best_window"], d["low_carbon_window"], d["low_cost_window"]
        rows = [
            html.Div(f"Best overall window today: {best['label']} — {best['reason']}",
                     style={"fontSize": "14px", "fontWeight": 600, "color": "#15803d", "marginBottom": "10px"}),
            html.Div(f"Lowest carbon: {lc['label']} (about {lc['carbon_reduction_pct']}% below the day's average carbon intensity)",
                     style={"fontSize": "13px", "marginBottom": "4px"}),
            html.Div(f"Lowest demand / cheapest to serve: {lcost['label']} (about {lcost['cost_reduction_pct']}% below the day's average grid load)",
                     style={"fontSize": "13px", "marginBottom": "8px"}),
            html.Div("Estimated from typical hourly demand shape, current fuel mix, and forecast sun/wind — "
                     "not a live price signal. Most useful if your plan has time-of-use pricing or you just "
                     "want to shift flexible use (EV charging, laundry, etc.) to a cleaner/cheaper window.",
                     style={"fontSize": "11px", "color": "#94a3b8"}),
        ]
        sections.append(_elec_section("Best time to use electricity today", html.Div(rows)))
    else:
        sections.append(_elec_section("Best time to use electricity today", _unavailable(btu.get("reason", "Temporarily unavailable."))))

    # How wholesale prices work
    wp = r["wholesale_price_context"]
    if wp["available"]:
        d = wp["data"]
        rows = [
            html.Div(f"Annual average forward price ({d['annual_avg_months']}-month strip): "
                     f"${d['annual_avg_usd_mwh']:.2f}/MWh — this is the figure comparable to your retail rate above.",
                     style={"fontSize": "14px", "marginBottom": "6px", "fontWeight": 600}),
            html.Div(f"Next month ({d['next_month']}) specifically: ${d['next_month_avg_usd_mwh']:.2f}/MWh "
                     f"(${d['next_month_on_peak_usd_mwh']:.2f} on-peak, ${d['next_month_off_peak_usd_mwh']:.2f} off-peak) "
                     f"— a single month, often not representative of the year.",
                     style={"fontSize": "13px", "color": "#64748b", "marginBottom": "10px"}),
        ]
        if wp.get("gap_explainer"):
            rows.append(html.Div(wp["gap_explainer"], style={"fontSize": "12px", "color": "#334155",
                                                                "background": "#f1f5f9", "borderRadius": "6px",
                                                                "padding": "10px 12px", "marginBottom": "10px"}))
        rows.append(html.Div(wp["disclaimer"], style={"fontSize": "12px", "color": "#b45309",
                                                         "background": "#fffbeb", "border": "1px solid #fde68a",
                                                         "borderRadius": "6px", "padding": "10px 12px"}))
        sections.append(_elec_section("How wholesale prices work", html.Div(rows)))
    else:
        sections.append(_elec_section("How wholesale prices work", _unavailable(wp["reason"])))

    # Wholesale price history — real settled daily prices, last 90 days
    wph = r.get("wholesale_price_history", {})
    if wph.get("available"):
        rows_data = wph["data"]
        dates = [row["date"] for row in rows_data]
        prices = [row["price_usd_mwh"] for row in rows_data]
        price_fig = go.Figure(go.Scatter(
            x=dates, y=prices, mode="lines",
            line=dict(color="#7c3aed", width=1.5),
            hovertemplate="%{x}<br>$%{y:.2f}/MWh<extra></extra>",
        ))
        price_fig.update_layout(
            margin=dict(l=45, r=10, t=10, b=30), height=200,
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(title="$/MWh", gridcolor="#e2e8f0"),
            xaxis=dict(title=None, tickangle=-45, nticks=8, gridcolor="#e2e8f0"),
        )
        sections.append(_elec_section(
            "Real settled wholesale prices — last 90 days",
            html.Div([
                dcc.Graph(figure=price_fig, config={"displayModeBar": False}, style={"height": "200px"}),
                html.Div("Actual settled daily prices, not the forward-looking model above.",
                         style={"fontSize": "11px", "color": "#94a3b8", "marginTop": "4px"}),
            ]),
        ))

    # Capacity auctions
    ca = r["capacity_auctions"]
    if ca["available"]:
        d = ca["data"]
        price = f"${d['clearing_price_mw_year']:,.0f}/MW-year" if d.get("clearing_price_mw_year") else "no centralized capacity price"
        rows = [
            html.Div(f"Mechanism: {d.get('mechanism', 'n/a')}", style={"fontSize": "14px", "marginBottom": "4px"}),
        ]
        if d.get("active_delivery_period"):
            rows.append(html.Div(f"Currently delivering under: {d['active_delivery_period']}",
                                  style={"fontSize": "14px", "marginBottom": "4px", "fontWeight": 600}))
        if d.get("auction_held_date"):
            rows.append(html.Div(f"That auction was held: {d['auction_held_date']}",
                                  style={"fontSize": "14px", "marginBottom": "4px", "color": "#2563eb"}))
        rows.append(html.Div(f"Clearing price: {price}", style={"fontSize": "14px", "marginBottom": "4px"}))
        rows.append(html.Div(d.get("notes", ""), style={"fontSize": "12px", "color": "#64748b", "marginTop": "8px"}))
        rows.append(html.Div(f"Source: {d.get('source', '')}", style={"fontSize": "10px", "color": "#cbd5e1", "marginTop": "6px"}))
        sections.append(_elec_section("Capacity auctions in your area", html.Div(rows)))
    else:
        sections.append(_elec_section("Capacity auctions in your area", _unavailable(ca["reason"])))

    # Capacity price trend — is it rising, and why?
    cpt = r.get("capacity_price_trend", {})
    if cpt.get("available"):
        hist = cpt["data"]["history"]
        periods = [h["delivery_period"] for h in hist]
        prices = [h["clearing_price_mw_year"] for h in hist]
        trend_fig = go.Figure(go.Bar(
            x=periods, y=prices, marker_color="#f97316",
            text=[f"{h['native_unit_price']}" for h in hist], textposition="outside",
            hovertemplate="%{x}<br>$%{y:,.0f}/MW-year<extra></extra>",
        ))
        trend_fig.update_layout(
            margin=dict(l=50, r=10, t=30, b=30), height=230,
            paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
            yaxis=dict(title="$/MW-year", gridcolor="#e2e8f0"),
            xaxis=dict(title=None, gridcolor="#e2e8f0"),
        )
        rows = [dcc.Graph(figure=trend_fig, config={"displayModeBar": False}, style={"height": "230px"})]
        attrib = cpt["data"].get("data_center_attribution")
        if attrib:
            rows.append(html.Div([
                html.Div(f"“{attrib['quote']}”", style={"fontStyle": "italic", "marginBottom": "4px"}),
                html.Div(f"— {attrib['attributed_to']}, {attrib['date']}", style={"fontWeight": 600, "marginBottom": "6px"}),
                html.Div(attrib["detail"], style={"marginBottom": "6px"}),
                html.Div(f"Source: {attrib['source']}", style={"fontSize": "10px", "color": "#94a3b8"}),
            ], style={"fontSize": "12px", "color": "#7c2d12", "background": "#fff7ed",
                       "border": "1px solid #fed7aa", "borderRadius": "6px", "padding": "10px 12px",
                       "marginTop": "10px"}))
        sections.append(_elec_section("Is your capacity price rising — and why?", html.Div(rows)))
    elif cpt.get("reason"):
        sections.append(_elec_section("Is your capacity price rising — and why?", _unavailable(cpt["reason"])))

    # Market competitiveness
    mc = r["market_competitiveness"]
    if mc["available"]:
        d = mc["data"]
        metric = f"{d['headline_metric']}: {d['headline_value']}" if d.get("headline_metric") else "See assessment below"
        sections.append(_elec_section(
            "How competitive is your market?",
            html.Div([
                html.Div(metric, style={"fontSize": "14px", "fontWeight": 600, "marginBottom": "6px"}),
                html.Div(d.get("assessment", ""), style={"fontSize": "13px", "color": "#334155", "marginBottom": "8px"}),
                html.Div(f"Source: {d.get('source', '')}", style={"fontSize": "10px", "color": "#cbd5e1"}),
            ]),
        ))
    else:
        sections.append(_elec_section("How competitive is your market?", _unavailable(mc["reason"])))

    return html.Div(sections)


app.clientside_callback(
    """
    function(usage_kwh, rate_usd_mwh) {
        if (!usage_kwh || usage_kwh <= 0 || !rate_usd_mwh) { return ""; }
        const rate_per_kwh = rate_usd_mwh / 1000;
        const bill = rate_per_kwh * usage_kwh;
        return "Estimated: $" + bill.toFixed(2) + "/month (at $" + rate_per_kwh.toFixed(3) + "/kWh × " + usage_kwh + " kWh)";
    }
    """,
    Output("elec-bill-estimate", "children"),
    Input("elec-bill-usage", "value"),
    Input("elec-bill-rate", "data"),
)


app.layout = html.Div([
    dcc.Location(id="url", refresh=False),
    html.Div(id="page-content"),
])


@app.callback(Output("page-content", "children"), Input("url", "pathname"))
def render_page(pathname):
    if pathname == "/futures-pricer":
        return _futures_pricer_layout()
    if pathname == "/my-electricity":
        return _electricity_explainer_layout()
    return _main_dashboard_layout


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("load-forecast-store", "data"),
    Input("iso-selector", "value"),
    Input("refresh-btn", "n_clicks"),
    Input("daily-refresh", "n_intervals"),
    Input("forecast-refresh", "n_intervals"),
)
def load_forecast_data(iso, n_clicks, _daily, _interval):
    force = n_clicks is not None and n_clicks > 0
    return _fetch_iso_forecast(iso, force=force)


@app.callback(
    Output("forward-curve-store", "data"),
    Input("iso-selector", "value"),
    Input("daily-refresh", "n_intervals"),
)
def load_forward_curve_data(iso, _daily):
    _cache_dir = _HERE / "api_cache"
    _slow_fetch = {"spp"}  # ISOs that do a multi-minute HTTP crawl without a cache
    if iso in _slow_fetch:
        has_cache = (_cache_dir / f"{iso}_price_history_archive.json").exists()
        if not has_cache:
            return None  # show "loading" state; warmup endpoint is building it
    try:
        from temperature_modeling.forward_curve import (
            build_forward_curve, backtest_forward_curve, _load_price_history,
        )
        # Both calls need the same price history — load it once here instead
        # of each function separately hitting _load_price_history's disk
        # cache (cheap once warm, but still two redundant reads+parses of a
        # file that can run to 730 rows per ISO on every render).
        try:
            history = _load_price_history(iso)
        except Exception:
            log.exception("%s: price history load failed", iso.upper())
            history = []
        result = build_forward_curve(iso, n_months=12, history=history)
        try:
            result["backtest"] = backtest_forward_curve(iso, history=history)
        except Exception:
            log.exception("%s: forward curve backtest failed", iso.upper())
            result["backtest"] = None
        return result
    except Exception as exc:
        log.warning("Forward curve failed for %s: %s", iso, exc)
        return None


@app.callback(
    Output("forward-curve-chart", "figure"),
    Output("spark-spread-chart", "figure"),
    Output("forward-curve-stats", "children"),
    Input("forward-curve-store", "data"),
    Input("iso-selector", "value"),
)
def render_forward_curve(data, iso):
    if not data or not data.get("curve"):
        _slow_fetch = {"spp"}
        msg = "Price history cache warming — please refresh in ~20 min" if iso in _slow_fetch else "Forward curve loading…"
        return _empty_fig(msg), _empty_fig(""), ""

    curve = data["curve"]
    months    = [m["month"] for m in curve]
    base_avg  = [m["scenarios"]["base"]["monthly_avg"] for m in curve]
    cold_avg  = [m["scenarios"]["cold"]["monthly_avg"] for m in curve]
    hot_avg   = [m["scenarios"]["hot"]["monthly_avg"]  for m in curve]
    base_on   = [m["scenarios"]["base"]["on_peak"]     for m in curve]
    base_off  = [m["scenarios"]["base"]["off_peak"]    for m in curve]
    base_low  = [m["scenarios"]["base"].get("low_usd_mwh")  for m in curve]
    base_high = [m["scenarios"]["base"].get("high_usd_mwh") for m in curve]

    _BLUE  = "#2563eb"
    _RED   = "#ef4444"
    _GREEN = "#16a34a"
    _GRAY  = "#94a3b8"

    fc_fig = go.Figure()
    # CQR band first (bottom layer) — this is the model's statistical
    # prediction interval (conformalized quantile regression), a distinct
    # and generally much wider uncertainty source than the cold/hot weather
    # scenario spread plotted on top of it. Only draw it if every month has
    # both bounds (older cached curves built before CQR shipped won't).
    if all(v is not None for v in base_low) and all(v is not None for v in base_high):
        fc_fig.add_trace(go.Scatter(
            x=months + months[::-1],
            y=base_high + base_low[::-1],
            fill="toself", fillcolor="rgba(148,163,184,0.18)",
            line=dict(width=0), showlegend=True, name="90% prediction interval (CQR)",
            hoverinfo="skip",
        ))
    fc_fig.add_trace(go.Scatter(
        x=months + months[::-1],
        y=hot_avg + cold_avg[::-1],
        fill="toself", fillcolor="rgba(37,99,235,0.08)",
        line=dict(width=0), showlegend=True, name="Cold–Hot range",
        hoverinfo="skip",
    ))
    fc_fig.add_trace(go.Scatter(
        x=months, y=cold_avg, mode="lines",
        line=dict(color=_GREEN, width=1.5, dash="dot"),
        name="Cold", hovertemplate="%{x}<br>Cold: $%{y:.1f}/MWh<extra></extra>",
    ))
    fc_fig.add_trace(go.Scatter(
        x=months, y=hot_avg, mode="lines",
        line=dict(color=_RED, width=1.5, dash="dot"),
        name="Hot", hovertemplate="%{x}<br>Hot: $%{y:.1f}/MWh<extra></extra>",
    ))
    fc_fig.add_trace(go.Scatter(
        x=months, y=base_avg, mode="lines+markers",
        line=dict(color=_BLUE, width=2.5),
        marker=dict(size=5, color=_BLUE),
        name="Base (strip)", hovertemplate="%{x}<br>Base: $%{y:.1f}/MWh<extra></extra>",
    ))
    fc_fig.add_trace(go.Scatter(
        x=months, y=base_on, mode="lines",
        line=dict(color=_BLUE, width=1, dash="dash"),
        name="Peak (base)", hovertemplate="%{x}<br>On-peak: $%{y:.1f}/MWh<extra></extra>",
    ))
    fc_fig.add_trace(go.Scatter(
        x=months, y=base_off, mode="lines",
        line=dict(color=_BLUE, width=1, dash="longdash"),
        name="Off-peak (base)", hovertemplate="%{x}<br>Off-peak: $%{y:.1f}/MWh<extra></extra>",
    ))
    fc_fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=0, r=8, t=8, b=0),
        legend=dict(orientation="h", y=-0.18, x=0, font=dict(size=10)),
        yaxis=dict(title="$/MWh", gridcolor="#f1f5f9", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10), tickangle=-30),
        hovermode="x unified",
    )

    # Spark spread chart (base CCGT)
    spark_ccgt = [m["scenarios"]["base"].get("spark_spread_ccgt") for m in curve]
    spark_ct   = [m["scenarios"]["base"].get("spark_spread_ct")   for m in curve]
    ss_fig = go.Figure()
    ss_fig.add_trace(go.Bar(
        x=months, y=spark_ccgt,
        marker_color=[_BLUE if v and v > 0 else _RED for v in spark_ccgt],
        name="Spark spread CCGT (7,000 BTU)",
        hovertemplate="%{x}<br>CCGT spark: $%{y:.2f}<extra></extra>",
    ))
    ss_fig.add_trace(go.Scatter(
        x=months, y=spark_ct, mode="lines+markers",
        line=dict(color=_GRAY, width=1.5, dash="dash"),
        marker=dict(size=4), name="CT spark (10,000 BTU)",
        hovertemplate="%{x}<br>CT spark: $%{y:.2f}<extra></extra>",
    ))
    ss_fig.add_hline(y=0, line_color="#cbd5e1", line_width=1)
    ss_fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        margin=dict(l=0, r=8, t=4, b=0),
        legend=dict(orientation="h", y=-0.25, x=0, font=dict(size=10)),
        yaxis=dict(title="$/MWh", gridcolor="#f1f5f9", tickfont=dict(size=10)),
        xaxis=dict(tickfont=dict(size=10), tickangle=-30),
        hovermode="x unified",
        title=dict(text="Spark Spreads — Base Scenario ($/MWh)", font=dict(size=11, color="#475569"), x=0),
    )

    model_src  = data.get("model_source", "ols")
    rmse       = data.get("model_rmse_usd_mwh")
    train_days = data.get("training_days")
    gas_months = len(data.get("gas_curve", []))
    rmse_str   = f"  |  RMSE ${rmse:.1f}/MWh" if rmse else ""
    train_str  = f"  |  Trained on {train_days}d" if train_days else ""

    bt = data.get("backtest") or {}
    n_comparisons = bt.get("n_comparisons") or 0
    if n_comparisons:
        bt_str = f"  |  Backtest: {n_comparisons} realized month{'s' if n_comparisons != 1 else ''}, MAPE {bt['mape']}%"
    elif bt.get("n_snapshots"):
        n_snap = bt["n_snapshots"]
        bt_str = f"  |  Backtest: accumulating ({n_snap} snapshot{'s' if n_snap != 1 else ''}, no completed months yet)"
    else:
        bt_str = ""

    stats_str = f"Model: {model_src}{rmse_str}{train_str}  |  Gas curve: {gas_months} EIA STEO months{bt_str}"

    return fc_fig, ss_fig, stats_str


def _empty_fig(msg="Loading…"):
    f = go.Figure()
    f.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(text=msg, showarrow=False,
                          font=dict(color="#94a3b8", size=13),
                          x=0.5, y=0.5, xref="paper", yref="paper")],
    )
    return f


@app.callback(
    Output("load-cards", "children"),
    Output("forecast-subtitle", "children"),
    Output("load-forecast-chart", "figure"),
    Output("extreme-alert", "children"),
    Output("verification-panel", "children"),
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
)
def render(data, iso):
    if not data or not data.get("load"):
        return ([], "", _empty_fig(), [], [])

    _iso_labels = {
        "pjm":   "PJM Interconnection",
        "caiso": "CAISO (California ISO)",
        "ercot": "ERCOT (Texas)",
        "miso":  "MISO (Midcontinent ISO)",
        "nyiso": "NYISO (New York)",
        "isone": "ISO-NE (New England)",
        "spp":   "SPP (Southwest Power Pool)",
    }
    iso_label = _iso_labels.get(iso, iso.upper())
    load_list  = data["load"]
    comparison = data.get("comparison", {})
    backtest   = data.get("backtest", {})
    hindcast   = data.get("hindcast", {})
    net_load_list = data.get("net_load", [])
    backtest_pre = data.get("backtest", {})
    bt_pred_by_date = dict(zip(backtest_pre.get("dates", []),
                               backtest_pre.get("predicted_gw", [])))
    combined_hindcast = {**bt_pred_by_date, **hindcast}

    dates  = [d["date"]         for d in load_list]
    means  = [d["mean_load_gw"] for d in load_list]
    lows   = [d["low_load_gw"]  for d in load_list]
    highs  = [d["high_load_gw"] for d in load_list]

    def dlabel(d):
        return (datetime.strptime(d, "%Y-%m-%d").strftime("%#d %b") if os.name == "nt"
                else datetime.strptime(d, "%Y-%m-%d").strftime("%-d %b"))

    today_gw = means[0] if means else 0
    peak_gw  = max(means) if means else 0
    peak_lbl = dlabel(dates[means.index(peak_gw)]) if means else "—"
    avg_gw   = sum(means) / len(means) if means else 0

    def load_color(v):
        r = v / avg_gw if avg_gw else 1
        if r > 1.15: return "#ef4444"
        if r > 1.06: return "#f97316"
        if r < 0.94: return "#3b82f6"
        return "#22c55e"

    actual_dict = comparison.get("actual", {})
    da_dict     = comparison.get("da_fcst", {})

    # Use held-out test-set MAPE from the 2-year backtest (last 20% ~ 150 days)
    test_mape = backtest.get("mape_test")
    mape_str  = f"{test_mape:.1f}%" if test_mape is not None else "—"
    mape_col  = "#22c55e" if test_mape is not None and test_mape < 3 else "#f97316"

    vstats = load_verification_stats(iso)
    v7  = f"{vstats['mape_7d']:.1f}%"  if vstats["mape_7d"]  is not None else "—"
    v30 = f"{vstats['mape_30d']:.1f}%" if vstats["mape_30d"] is not None else "—"
    vbias = (f"{vstats['bias_mw']:+,.0f} MW" if vstats["bias_mw"] is not None else "—")
    v_col = ("#22c55e" if vstats["mape_7d"] is not None and vstats["mape_7d"] < 3
             else "#f97316")

    # Percentile context
    pct_data = data.get("percentile", {})
    pct      = pct_data.get("percentile")
    m_avg    = pct_data.get("month_avg_gw")
    m_max    = pct_data.get("month_max_gw")
    last_hi  = pct_data.get("last_higher_date", "")
    pct_sub  = (f"avg {m_avg} GW · record {m_max} GW" if m_avg else "5-yr history")
    pct_col  = ("#ef4444" if pct and pct >= 90 else
                "#f97316" if pct and pct >= 75 else "#22c55e")
    pct_str  = f"{pct}th %ile" if pct is not None else "—"

    summary_cards = [
        card("Today (GW)", f"{today_gw:.1f}", "GFS-based", load_color(today_gw)),
        card("Monthly Rank", pct_str, pct_sub, pct_col),
        card("15-Day Peak", f"{peak_gw:.1f}", f"on {peak_lbl}", load_color(peak_gw)),
        card("15-Day Avg", f"{avg_gw:.1f}", "GW baseline", "#475569"),
        card("Hindcast MAPE", mape_str, "ERA5 obs. temps", mape_col),
        card("Verified MAPE 7d", v7, f"30d: {v30}  bias: {vbias}", v_col),
    ]

    # Extreme event alert banner
    alert_children = []
    if pct is not None and pct >= 85:
        last_hi_txt = f"  Last exceeded: {last_hi}." if last_hi else ""
        alert_color = "#ef4444" if pct >= 95 else "#f97316"
        alert_children = [html.Div(
            style={"background": alert_color + "10", "border": f"1px solid {alert_color}40",
                   "borderRadius": "8px", "padding": "10px 16px",
                   "display": "flex", "gap": "10px", "alignItems": "center"},
            children=[
                html.Span("⚠️" if pct >= 95 else "🔶",
                          style={"fontSize": "16px", "flexShrink": 0}),
                html.Span(
                    f"Extreme demand alert — today's {today_gw:.1f} GW forecast is the "
                    f"{pct}th percentile for this calendar month "
                    f"(5-yr avg {m_avg} GW, record {m_max} GW).{last_hi_txt}",
                    style={"fontSize": "13px", "color": alert_color, "fontWeight": 500},
                ),
            ],
        )]

    subtitle = f"{iso_label}  ·  GFS temperature input  ·  {len(dates)}-day horizon"

    # ── Forecast chart ────────────────────────────────────────────────────────
    hist_dates_raw = sorted(actual_dict.keys())
    hist_dates_raw = [d for d in hist_dates_raw if d < dates[0]][-14:]
    hist_vals      = [actual_dict[d] for d in hist_dates_raw]

    hc_dates = [d for d in hist_dates_raw if d in combined_hindcast]
    hc_vals  = [combined_hindcast[d] for d in hc_dates]

    combined_x      = hc_dates + [None] + dates
    combined_vals   = hc_vals  + [None] + means
    combined_colors = (["#f97316"] * len(hc_dates) +
                       ["rgba(0,0,0,0)"] +
                       [load_color(m) for m in means])

    fig = go.Figure()

    # Uncertainty ribbon
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=highs + lows[::-1],
        fill="toself", fillcolor="rgba(251,146,60,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))

    # Historical actual
    if hist_vals:
        fig.add_trace(go.Scatter(
            x=hist_dates_raw, y=hist_vals, mode="lines+markers",
            name="Actual (EIA)",
            line=dict(color="#94a3b8", width=2),
            marker=dict(size=5, color="#94a3b8"),
            hovertemplate="<b>%{x|%d %b}</b><br>Actual: %{y:.1f} GW<extra></extra>",
        ))

    # Our model (hindcast + forecast, one trace)
    fig.add_trace(go.Scatter(
        x=combined_x, y=combined_vals, mode="lines+markers",
        name="Our Model",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=6, color=combined_colors,
                    line=dict(color="#ffffff", width=1)),
        connectgaps=False,
        hovertemplate="<b>%{x|%d %b}</b><br>Model: %{y:.1f} GW<extra></extra>",
    ))

    # Official benchmark trace (ISO-specific)
    _bench_cfg = {
        "caiso": ("oasis_7day",  "CAISO Official 7-Day (OASIS)",        "#16a34a",
                  "CAISO OASIS"),
        "pjm":   ("pjm_7day",   "PJM Official 7-Day (DataMiner)",       "#2563eb",
                  "PJM official"),
        "ercot": ("ercot_7day",  "ERCOT 7-Day Forecast",                 "#7c3aed",
                  "ERCOT forecast"),
        "miso":  ("miso_7day",   "MISO EIA Day-Ahead Forecast",          "#0891b2",
                  "MISO EIA DA"),
    }
    bench_key, bench_name, bench_color, bench_hover = _bench_cfg.get(
        iso, ("pjm_7day", "Official 7-Day", "#2563eb", "Official")
    )
    bench_data  = data.get(bench_key, {})
    bench_dates = sorted(d for d in bench_data if d in dates)
    # Only show the benchmark trace when it's a real multi-day forecast (3+ days).
    # With only 1-2 days it's EIA day-ahead data, which is already shown below
    # as the "EIA Day-Ahead" markers — no point duplicating it.
    if len(bench_dates) >= 3:
        fig.add_trace(go.Scatter(
            x=bench_dates, y=[bench_data[d] for d in bench_dates],
            mode="lines+markers", name=bench_name,
            line=dict(color=bench_color, width=2, dash="dash"),
            marker=dict(size=7, color=bench_color, symbol="diamond",
                        line=dict(color="#ffffff", width=1.5)),
            hovertemplate=f"<b>%{{x|%d %b}}</b><br>{bench_hover}: %{{y:.1f}} GW<extra></extra>",
        ))
    da_in_fcst = sorted(d for d in da_dict if d in dates)
    if da_in_fcst:
        fig.add_trace(go.Scatter(
            x=da_in_fcst, y=[da_dict[d] for d in da_in_fcst],
            mode="markers", name="EIA Day-Ahead",
            marker=dict(size=9, color="#0ea5e9", symbol="diamond",
                        line=dict(color="#ffffff", width=1.5)),
            hovertemplate="<b>%{x|%d %b}</b><br>EIA day-ahead: %{y:.1f} GW<extra></extra>",
        ))

    # Net load trace (CAISO / ERCOT only — meaningful with high renewable penetration)
    if net_load_list and iso in ("caiso", "ercot"):
        nl_by_date = {r["date"]: r for r in net_load_list}
        nl_dates   = [d for d in dates if d in nl_by_date]
        net_vals   = [means[dates.index(d)] - nl_by_date[d]["renewable_gw"] for d in nl_dates]
        sol_vals   = [nl_by_date[d]["solar_gw"] for d in nl_dates]
        wnd_vals   = [nl_by_date[d]["wind_gw"]  for d in nl_dates]
        fig.add_trace(go.Scatter(
            x=nl_dates, y=net_vals, mode="lines",
            name="Net Load (ex-renewables)",
            line=dict(color="#8b5cf6", width=2, dash="dash"),
            customdata=list(zip(sol_vals, wnd_vals)),
            hovertemplate=(
                "<b>%{x|%d %b}</b><br>"
                "Net load: %{y:.1f} GW<br>"
                "Solar: %{customdata[0]:.1f} GW<br>"
                "Wind: %{customdata[1]:.1f} GW"
                "<extra></extra>"
            ),
        ))
        # Stacked renewable area
        fig.add_trace(go.Scatter(
            x=nl_dates, y=sol_vals, mode="none", name="Solar (est.)",
            fill="tozeroy", fillcolor="rgba(251,191,36,0.20)",
            hovertemplate="<b>%{x|%d %b}</b><br>Solar: %{y:.1f} GW<extra></extra>",
        ))
        fig.add_trace(go.Scatter(
            x=nl_dates, y=[s + w for s, w in zip(sol_vals, wnd_vals)],
            mode="none", name="Solar+Wind (est.)",
            fill="tonexty", fillcolor="rgba(52,211,153,0.20)",
            hovertemplate="<b>%{x|%d %b}</b><br>Solar+Wind: %{y:.1f} GW<extra></extra>",
        ))

    fig.add_hline(y=avg_gw, line_dash="dot", line_color="#cbd5e1", line_width=1.5,
                  annotation_text=f"15-day avg: {avg_gw:.1f} GW",
                  annotation_font_size=10, annotation_font_color="#94a3b8",
                  annotation_position="top left")
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.26, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", x=0),
        margin=dict(l=60, r=20, t=10, b=90),
        xaxis=dict(type="date", tickformat="%d %b", gridcolor="#f8fafc",
                   tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0",
                   dtick="D1"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".1f", ticksuffix=" GW",
                   title=dict(text="Load (GW)", font=dict(size=11, color="#64748b")),
                   linecolor="#e2e8f0"),
        height=320,
    )

    # ── Live verification panel ───────────────────────────────────────────────
    v_panel = []
    records = vstats.get("records", [])
    if records:
        _th_style = {"padding": "6px 10px", "fontWeight": 600, "color": "#64748b",
                     "fontSize": "11px", "borderBottom": "1px solid #e2e8f0",
                     "textTransform": "uppercase", "letterSpacing": "0.04em"}
        header = html.Thead(html.Tr([
            html.Th("Date",        style=_th_style),
            html.Th("Forecast MW", style={**_th_style, "textAlign": "right"}),
            html.Th("Actual MW",   style={**_th_style, "textAlign": "right"}),
            html.Th("Error %",     style={**_th_style, "textAlign": "right"}),
        ]))
        body_rows = []
        for r in records[-14:]:
            err = r["error_pct"]
            err_col = "#22c55e" if abs(err) < 2 else "#f97316" if abs(err) < 5 else "#ef4444"
            body_rows.append(html.Tr([
                html.Td(r["date"],               style={"padding": "5px 10px", "fontSize": "12px"}),
                html.Td(f"{r['forecast_mw']:,.0f}", style={"padding": "5px 10px", "fontSize": "12px",
                                                            "textAlign": "right"}),
                html.Td(f"{r['actual_mw']:,.0f}",   style={"padding": "5px 10px", "fontSize": "12px",
                                                            "textAlign": "right"}),
                html.Td(f"{err:+.1f}%",              style={"padding": "5px 10px", "fontSize": "12px",
                                                             "textAlign": "right", "color": err_col,
                                                             "fontWeight": 600}),
            ], style={"borderBottom": "1px solid #f8fafc"}))
        v_panel = [html.Div(
            style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                   "border": "1px solid #e2e8f0", "padding": "20px",
                   "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                   "marginBottom": "20px"},
            children=[
                html.Div(
                    style={"display": "flex", "alignItems": "baseline",
                           "gap": "12px", "marginBottom": "10px"},
                    children=[
                        html.Div("Live Forecast Verification",
                                 style={"fontSize": "13px", "fontWeight": 600,
                                        "color": "#0f172a"}),
                        html.Span(
                            f"7d MAPE: {v7}  ·  30d: {v30}  ·  bias: {vbias}",
                            style={"fontSize": "11px", "color": "#64748b"},
                        ),
                    ],
                ),
                html.Div(
                    "Day-ahead forecasts vs EIA-reported actuals (most recent 14 verified days).",
                    style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "10px"},
                ),
                html.Div(
                    style={"overflowX": "auto"},
                    children=[html.Table(
                        [header, html.Tbody(body_rows)],
                        style={"borderCollapse": "collapse", "width": "100%"},
                    )],
                ),
            ],
        )]

    return summary_cards, subtitle, fig, alert_children, v_panel


# ---------------------------------------------------------------------------
# SHAP explainability callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("shap-chart", "figure"),
    Input("load-forecast-store", "data"),
)
def render_shap(data):
    if not data or "shap" not in data:
        return _empty_fig("Forecast not loaded yet")

    shap = data["shap"]
    groups = shap.get("groups", {})
    base_mw = shap.get("base_mw", 0)

    if not groups:
        return _empty_fig("No SHAP data available")

    labels = list(groups.keys())
    values = [groups[g] for g in labels]
    colors = ["#ef4444" if v > 0 else "#3b82f6" for v in values]
    base_gw = round(base_mw / 1000, 2)

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:+,.0f} MW" for v in values],
        textposition="outside",
        cliponaxis=False,
    ))
    fig.add_vline(x=0, line_width=1, line_color="#94a3b8")
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=10, r=80, t=10, b=10),
        xaxis=dict(
            title=dict(text="Contribution vs. baseline (MW)", font=dict(size=10)),
            gridcolor="#f1f5f9", zeroline=False, tickfont=dict(size=10),
        ),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        showlegend=False,
        annotations=[dict(
            text=f"Baseline: {base_gw:.1f} GW",
            x=1, y=1.08, xref="paper", yref="paper",
            showarrow=False, font=dict(size=10, color="#94a3b8"),
            xanchor="right",
        )],
    )
    return fig


# ---------------------------------------------------------------------------
# Carbon intensity callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("carbon-store", "data"),
    Input("iso-selector", "value"),
    Input("refresh-btn", "n_clicks"),
    Input("forecast-refresh", "n_intervals"),
)
def load_carbon_data(iso, _clicks, _interval):
    session = requests.Session()
    session.headers["User-Agent"] = "grid-dashboard/1.0"
    try:
        return fetch_carbon_intensity(iso, session)
    except Exception:
        log.exception("Carbon intensity fetch failed for %s", iso.upper())
        return {}


@app.callback(
    Output("carbon-intensity-number", "children"),
    Output("carbon-fuel-mix-chart", "figure"),
    Input("carbon-store", "data"),
)
def render_carbon(data):
    empty_fig = _empty_fig("Carbon data unavailable (EIA_API_KEY required)")
    if not data or not data.get("lbs_co2_per_mwh"):
        return html.Div("Carbon data unavailable", style={"color": "#94a3b8", "fontSize": "13px"}), empty_fig

    lbs       = data["lbs_co2_per_mwh"]
    clean_pct = data.get("clean_pct", 0)
    period    = data.get("period", "")
    fuel_mix  = data.get("fuel_mix", {})
    total_mw  = data.get("total_mw", 0)

    # Color scale: green → yellow → orange → red
    if lbs < 300:
        ci_color = "#22c55e"
        ci_label = "Very Clean"
    elif lbs < 500:
        ci_color = "#84cc16"
        ci_label = "Clean"
    elif lbs < 750:
        ci_color = "#f97316"
        ci_label = "Mixed"
    else:
        ci_color = "#ef4444"
        ci_label = "Carbon Heavy"

    period_label = period[-5:].replace("T", " ") + ":00 UTC" if period else ""

    number_block = html.Div(
        style={"display": "flex", "alignItems": "baseline", "gap": "16px", "flexWrap": "wrap"},
        children=[
            html.Div([
                html.Span(f"{lbs:.0f}",
                          style={"fontSize": "36px", "fontWeight": 700, "color": ci_color}),
                html.Span(" lbs CO₂/MWh",
                          style={"fontSize": "13px", "color": "#64748b", "marginLeft": "4px"}),
            ]),
            html.Span(ci_label,
                      style={"fontSize": "11px", "fontWeight": 600,
                             "background": ci_color + "18", "color": ci_color,
                             "borderRadius": "4px", "padding": "3px 10px"}),
            html.Span(f"{clean_pct:.0f}% zero-carbon",
                      style={"fontSize": "12px", "color": "#64748b"}),
            html.Span(f"{total_mw/1000:.1f} GW total · {period_label}",
                      style={"fontSize": "11px", "color": "#94a3b8"}),
        ],
    )

    # Fuel mix bar chart
    _FUEL_COLORS: dict = {
        "Natural Gas": "#f97316",
        "Coal":        "#78716c",
        "Oil":         "#b45309",
        "Nuclear":     "#6366f1",
        "Wind":        "#22c55e",
        "Solar":       "#fbbf24",
        "Hydro":       "#0ea5e9",
        "Pumped Storage": "#94a3b8",
        "Geothermal":  "#10b981",
        "Other":       "#cbd5e1",
    }
    labels = list(fuel_mix.keys())
    values = [fuel_mix[k] / 1000 for k in labels]  # MW → GW
    colors = [_FUEL_COLORS.get(lbl, "#cbd5e1") for lbl in labels]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=values,
        y=labels,
        orientation="h",
        marker_color=colors,
        text=[f"{v:.1f} GW" for v in values],
        textposition="outside",
        cliponaxis=False,
        hovertemplate="<b>%{y}</b>: %{x:.1f} GW<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        margin=dict(l=10, r=70, t=10, b=10),
        xaxis=dict(
            title=dict(text="Generation (GW)", font=dict(size=10)),
            gridcolor="#f1f5f9", tickfont=dict(size=10),
        ),
        yaxis=dict(tickfont=dict(size=11), autorange="reversed"),
        showlegend=False,
        height=200,
    )
    return number_block, fig


# ---------------------------------------------------------------------------
# Demand-response callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("dr-recommendation", "children"),
    Output("dr-chart", "figure"),
    Input("load-forecast-store", "data"),
)
def render_dr(data):
    empty_fig = _empty_fig("Demand-response data unavailable")
    if not data or "demand_response" not in data:
        return html.Div("Loading…", style={"color": "#94a3b8", "fontSize": "13px"}), empty_fig

    dr        = data["demand_response"]
    hours     = dr.get("hours", [])
    best      = dr.get("best_window", {})
    low_ci    = dr.get("low_carbon_window", {})
    low_cost  = dr.get("low_cost_window", {})

    if not hours:
        return html.Div("No hourly data", style={"color": "#94a3b8"}), empty_fig

    # ── Recommendation chips ─────────────────────────────────────────────────
    def _chip(icon, label, sublabel, color):
        return html.Div(
            style={"background": color + "0f", "border": f"1px solid {color}30",
                   "borderRadius": "8px", "padding": "10px 14px",
                   "display": "flex", "gap": "10px", "alignItems": "flex-start",
                   "flex": "1", "minWidth": "200px"},
            children=[
                html.Span(icon, style={"fontSize": "20px", "flexShrink": 0}),
                html.Div([
                    html.Div(label,    style={"fontSize": "12px", "fontWeight": 700,
                                              "color": color}),
                    html.Div(sublabel, style={"fontSize": "11px", "color": "#475569",
                                              "lineHeight": "1.4", "marginTop": "2px"}),
                ]),
            ],
        )

    best_label    = best.get("label", "—")
    best_date     = best.get("date", "")
    best_reason   = best.get("reason", "")
    ci_label      = low_ci.get("label", "—")
    ci_pct        = low_ci.get("carbon_reduction_pct", 0)
    cost_label    = low_cost.get("label", "—")
    cost_pct      = low_cost.get("cost_reduction_pct", 0)
    ci_date       = low_ci.get("date", "")
    cost_date     = low_cost.get("date", "")

    def _fmt_date(d):
        if not d: return ""
        try:
            dt = date.fromisoformat(d)
            return "Today" if dt == date.today() else "Tomorrow"
        except Exception:
            return d

    recs = html.Div(
        style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "4px"},
        children=[
            _chip("⚡", f"Best Overall: {best_label} ({_fmt_date(best_date)})",
                  best_reason, "#2563eb"),
            _chip("🌿", f"Lowest Carbon: {ci_label} ({_fmt_date(ci_date)})",
                  f"{ci_pct}% less CO₂ than average hour", "#22c55e"),
            _chip("💰", f"Lowest Cost: {cost_label} ({_fmt_date(cost_date)})",
                  f"~{cost_pct}% below peak load pricing", "#f97316"),
        ],
    )

    # ── 48-hour stacked chart ────────────────────────────────────────────────
    ts_labels = [h["ts"].replace("T", " ") for h in hours]
    solars    = [h["solar_gw"]  for h in hours]
    winds     = [h["wind_gw"]   for h in hours]
    nets      = [h["net_load_gw"] for h in hours]

    fig = go.Figure()

    # Net load (gray)
    fig.add_trace(go.Bar(
        x=ts_labels, y=nets,
        name="Net Load (ex-renewables)",
        marker_color="#cbd5e1",
        hovertemplate="<b>%{x}</b><br>Net load: %{y:.1f} GW<extra></extra>",
    ))
    # Wind
    fig.add_trace(go.Bar(
        x=ts_labels, y=winds,
        name="Wind",
        marker_color="#34d399",
        hovertemplate="<b>%{x}</b><br>Wind: %{y:.1f} GW<extra></extra>",
    ))
    # Solar
    fig.add_trace(go.Bar(
        x=ts_labels, y=solars,
        name="Solar",
        marker_color="#fbbf24",
        hovertemplate="<b>%{x}</b><br>Solar: %{y:.1f} GW<extra></extra>",
    ))

    # Highlight best window with a rectangle
    if best.get("start_h") is not None and best.get("date"):
        hi_date = best["date"]
        hi_s    = best["start_h"]
        hi_e    = best["end_h"]
        hi_ts_s = f"{hi_date} {hi_s:02d}:00"
        hi_ts_e = f"{hi_date} {hi_e:02d}:00"
        fig.add_vrect(
            x0=hi_ts_s, x1=hi_ts_e,
            fillcolor="rgba(37,99,235,0.10)",
            line=dict(color="#2563eb", width=1.5, dash="dot"),
            annotation_text="Best window",
            annotation_font_size=9,
            annotation_font_color="#2563eb",
            annotation_position="top left",
        )

    fig.update_layout(
        barmode="stack",
        paper_bgcolor="#ffffff",
        plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.28, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", x=0),
        margin=dict(l=60, r=20, t=10, b=70),
        xaxis=dict(
            tickformat="%#d %b %H:%M" if os.name == "nt" else "%-d %b %H:%M",
            tickangle=-45, tickfont=dict(size=9), linecolor="#e2e8f0",
            gridcolor="#f8fafc",
        ),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".0f", ticksuffix=" GW",
                   linecolor="#e2e8f0"),
        height=220,
        hovermode="x unified",
    )
    return recs, fig


# ---------------------------------------------------------------------------
# Price forecast callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("price-forecast-chart", "figure"),
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
)
def render_price_chart(data, iso):
    if not data or not data.get("price_forecast"):
        msg = price_unavailable_reason(iso or "")
        fig = go.Figure()
        fig.add_annotation(text=msg,
                           xref="paper", yref="paper", x=0.5, y=0.5,
                           showarrow=False, font=dict(color="#94a3b8", size=12))
        fig.update_layout(paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
                          height=220, margin=dict(l=60, r=20, t=10, b=40))
        return fig

    prices = data["price_forecast"]
    dates  = [p["date"] for p in prices]
    means  = [p["forecast_price"] for p in prices]
    lows   = [p["low_price"]  for p in prices]
    highs  = [p["high_price"] for p in prices]
    rmse   = prices[0].get("model_rmse", 0) if prices else 0

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates + dates[::-1], y=highs + lows[::-1],
        fill="toself", fillcolor="rgba(99,102,241,0.10)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=means, mode="lines+markers",
        name="DA Price Forecast",
        line=dict(color="#6366f1", width=2),
        marker=dict(size=5, color="#6366f1"),
        hovertemplate="<b>%{x|%d %b}</b><br>Price: $%{y:.2f}/MWh<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        annotations=[dict(
            text=f"Model RMSE: ${rmse:.1f}/MWh · OLS regression (load + weekday + seasonal)",
            xref="paper", yref="paper", x=0, y=1.08, showarrow=False,
            font=dict(size=9, color="#94a3b8"), xanchor="left",
        )],
        legend=dict(orientation="h", y=-0.30, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", x=0),
        margin=dict(l=60, r=20, t=20, b=60),
        xaxis=dict(type="date", tickformat="%d %b", gridcolor="#f8fafc",
                   tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0",
                   dtick="D1"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".0f", tickprefix="$",
                   ticksuffix="/MWh", linecolor="#e2e8f0"),
        height=220,
    )
    return fig


# ---------------------------------------------------------------------------
# Backtest chart callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("backtest-chart", "figure"),
    Output("backtest-mape-badges", "children"),
    Input("load-forecast-store", "data"),
)
def render_backtest(data):
    empty = _empty_fig("Backtest data unavailable for this ISO")
    if not data:
        return empty, []

    bt = data.get("backtest", {})
    if not bt or not bt.get("dates"):
        return empty, []

    all_dates     = bt["dates"]
    all_actual    = bt["actual_gw"]
    all_predicted = bt["predicted_gw"]
    all_is_test   = bt["is_test"]
    split_date    = bt.get("split_date", "")
    mape_train    = bt.get("mape_train", 0)
    mape_test     = bt.get("mape_test", 0)

    # Show last 200 days for readability (enough to capture most of test period)
    SHOW = 200
    dates     = all_dates[-SHOW:]
    actual    = all_actual[-SHOW:]
    predicted = all_predicted[-SHOW:]
    is_test   = all_is_test[-SHOW:]

    train_dates   = [d for d, t in zip(dates, is_test) if not t]
    train_actual  = [a for a, t in zip(actual,    is_test) if not t]
    test_dates    = [d for d, t in zip(dates, is_test) if t]
    test_actual   = [a for a, t in zip(actual,    is_test) if t]
    test_pred     = [p for p, t in zip(predicted, is_test) if t]

    fig = go.Figure()
    if train_dates:
        fig.add_trace(go.Scatter(
            x=train_dates, y=train_actual,
            name="Actual (train)", mode="lines",
            line=dict(color="#cbd5e1", width=1),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.2f} GW<extra>Train</extra>",
        ))
    if test_dates:
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_actual,
            name="Actual (test)", mode="lines",
            line=dict(color="#1e293b", width=1.5),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Actual: %{y:.2f} GW<extra>Test</extra>",
        ))
        fig.add_trace(go.Scatter(
            x=test_dates, y=test_pred,
            name="Model prediction", mode="lines",
            line=dict(color="#2563eb", width=1.5, dash="dot"),
            hovertemplate="<b>%{x|%d %b %Y}</b><br>Predicted: %{y:.2f} GW<extra>Model</extra>",
        ))

    if split_date and split_date >= dates[0]:
        fig.add_vline(x=split_date, line_dash="dash", line_color="#94a3b8", line_width=1,
                      annotation_text="Train → Test", annotation_position="top right",
                      annotation_font_size=9, annotation_font_color="#94a3b8")

    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=1.1, x=0, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=30, b=50),
        xaxis=dict(type="date", tickformat="%d %b '%y", gridcolor="#f8fafc",
                   tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".1f", ticksuffix=" GW",
                   linecolor="#e2e8f0"),
        hovermode="x unified",
        height=260,
    )

    def _badge(label, val, color):
        return html.Span(
            f"{label}: {val:.1f}%",
            style={"fontSize": "11px", "background": color + "18", "color": color,
                   "borderRadius": "4px", "padding": "2px 8px"},
        )

    badges = [
        _badge("Train MAPE", mape_train, "#64748b"),
        _badge("Test MAPE",  mape_test,  "#2563eb"),
    ]
    return fig, badges


# ---------------------------------------------------------------------------
# Capacity market callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("capacity-market-content", "children"),
    Input("iso-selector", "value"),
)
def render_capacity_market(iso):
    cm = get_capacity_market_data(iso)
    if "error" in cm:
        return html.Div(cm["error"], style={"color": "#94a3b8", "fontSize": "13px"})

    rm_pct  = cm.get("reserve_margin_pct")
    rm_col  = get_reserve_margin_color(rm_pct)
    rm_str  = f"{rm_pct:.1f}%" if rm_pct else "—"
    price   = cm.get("clearing_price_mw_year")
    if price:
        price_s = f"${price:,.0f}/MW-year"
    elif cm.get("native_unit_price"):
        # Locational/no-single-figure markets (e.g. NYISO's per-zone ICAP
        # pricing) still have a real capacity mechanism — show the verified
        # native-unit price rather than falsely implying no market exists.
        price_s = cm["native_unit_price"]
    else:
        price_s = "Energy-only market"
    procured = cm.get("procured_mw")
    req      = cm.get("requirement_mw")

    def stat(label, value, color="#334155"):
        return html.Div(
            style={"background": "#f8fafc", "borderRadius": "8px", "padding": "12px 16px",
                   "minWidth": "140px", "flex": "1"},
            children=[
                html.Div(label, style={"fontSize": "10px", "color": "#94a3b8",
                                       "textTransform": "uppercase", "letterSpacing": "0.05em",
                                       "marginBottom": "4px"}),
                html.Div(value, style={"fontSize": "18px", "fontWeight": 700, "color": color}),
            ],
        )

    return html.Div([
        html.Div(
            style={"display": "flex", "gap": "10px", "flexWrap": "wrap", "marginBottom": "14px"},
            children=[
                stat("Mechanism", cm["mechanism"][:30], "#334155"),
                stat("Reserve Margin", rm_str, rm_col),
                stat("Capacity Price", price_s, "#6366f1"),
                stat("Procured", f"{procured/1000:.1f} GW" if procured else "—", "#0ea5e9"),
                stat("Requirement", f"{req/1000:.1f} GW" if req else "—", "#334155"),
                stat("2024 Peak", f"{cm['peak_mw_2024']/1000:.1f} GW", "#f97316"),
            ],
        ),
        html.Div(
            cm.get("notes", ""),
            style={"fontSize": "12px", "color": "#64748b", "lineHeight": "1.6",
                   "background": "#f8fafc", "borderRadius": "6px", "padding": "10px 14px",
                   "marginBottom": "6px"},
        ),
        html.Div(
            f"Source: {cm.get('source', '')}",
            style={"fontSize": "10px", "color": "#cbd5e1"},
        ),
    ])


# ---------------------------------------------------------------------------
# AI Brief callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("ai-brief-card", "children"),
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
)
def render_ai_brief(data, iso):
    if not data or not data.get("load"):
        return []
    brief = generate_forecast_brief(iso, data)
    if not brief:
        return []
    return html.Div(
        style={"backgroundColor": "#fffbeb", "border": "1px solid #fde68a",
               "borderRadius": "8px", "padding": "14px 18px",
               "display": "flex", "gap": "12px", "alignItems": "flex-start"},
        children=[
            html.Span("AI", style={"background": "#f59e0b", "color": "#ffffff",
                                   "borderRadius": "4px", "padding": "2px 6px",
                                   "fontSize": "10px", "fontWeight": 700,
                                   "flexShrink": 0, "marginTop": "1px"}),
            html.P(brief, style={"margin": 0, "fontSize": "13px",
                                  "color": "#78350f", "lineHeight": "1.65"}),
        ],
    )


# ---------------------------------------------------------------------------
# Ensemble callback — runs teleconnection adjustment when toggle is on
# ---------------------------------------------------------------------------
@app.callback(
    Output("ensemble-store", "data"),
    Output("ensemble-badge", "children"),
    Input("ensemble-toggle", "value"),
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
)
def run_ensemble(toggle_value, data, iso):
    if not toggle_value or not data or not data.get("load"):
        return {}, ""

    from temperature_modeling.models import LoadForecast as _LF
    from datetime import date as _date

    # Reconstruct LoadForecast objects from the store
    load_list = data.get("load", [])
    xgb_forecasts = [
        _LF(
            valid_date=_date.fromisoformat(d["date"]),
            lead_days=i,
            mean_load_mw=d["mean_load_gw"] * 1000,
            low_load_mw=d["low_load_gw"]   * 1000,
            high_load_mw=d["high_load_gw"] * 1000,
            hdd=0.0, cdd=0.0, avg_temp_f=0.0,
        )
        for i, d in enumerate(load_list)
    ]

    try:
        session = requests.Session()
        session.headers["User-Agent"] = "grid-dashboard-ensemble/1.0"
        adjusted, meta = get_ensemble_forecast(iso, xgb_forecasts, session)
    except Exception:
        log.exception("Ensemble callback failed")
        return {}, html.Span("Ensemble error", style={"color": "#ef4444", "fontSize": "11px"})

    if not meta.get("ensemble_available"):
        return {}, html.Span("Teleconnection data unavailable",
                              style={"color": "#94a3b8", "fontSize": "11px"})

    # Overwrite load data with adjusted values
    adj_data = [
        {"date": lf.valid_date.isoformat(),
         "mean_load_gw": round(lf.mean_load_mw / 1000, 2),
         "low_load_gw":  round(lf.low_load_mw  / 1000, 2),
         "high_load_gw": round(lf.high_load_mw / 1000, 2)}
        for lf in adjusted
    ]

    conf = meta.get("confidence", "low")
    conf_color = {"high": "#22c55e", "medium": "#f97316", "low": "#94a3b8"}.get(conf, "#94a3b8")
    badge = html.Span(
        [html.Span(f"Ensemble active — {conf} confidence: ", style={"color": "#475569"}),
         html.Span(meta.get("headline", ""), style={"color": "#0f172a"})],
        style={"fontSize": "11px"},
    )
    return {"load": adj_data, "reasoning": meta.get("reasoning", ""),
            "confidence": conf, "confidence_color": conf_color}, badge


# ---------------------------------------------------------------------------
# Chat callbacks
# ---------------------------------------------------------------------------
_CHIP_QUESTIONS = {
    "chip-q1": "What's driving the forecast peak?",
    "chip-q2": "How confident is this 15-day forecast?",
    "chip-q3": "How does our forecast compare to the ISO's day-ahead forecast?",
}

@app.callback(
    Output("chat-history-store", "data"),
    Output("chat-input", "value"),
    Input("chat-submit", "n_clicks"),
    Input("chat-clear", "n_clicks"),
    Input("chip-q1", "n_clicks"),
    Input("chip-q2", "n_clicks"),
    Input("chip-q3", "n_clicks"),
    dash.dependencies.State("chat-input", "value"),
    dash.dependencies.State("chat-history-store", "data"),
    dash.dependencies.State("load-forecast-store", "data"),
    dash.dependencies.State("iso-selector", "value"),
    prevent_initial_call=True,
)
def handle_chat(submit_clicks, clear_clicks, _c1, _c2, _c3,
                user_input, history, forecast_data, iso):
    from dash import ctx
    triggered = ctx.triggered_id
    if triggered == "chat-clear":
        return [], ""
    if triggered in _CHIP_QUESTIONS:
        user_input = _CHIP_QUESTIONS[triggered]
    if not user_input or not user_input.strip():
        return history or [], ""

    history = history or []
    response = generate_chat_response(iso, forecast_data or {}, user_input.strip(), history)
    updated = history + [
        {"role": "user",      "content": user_input.strip()},
        {"role": "assistant", "content": response},
    ]
    return updated, ""


@app.callback(
    Output("chat-history", "children"),
    Input("chat-history-store", "data"),
)
def render_chat(history):
    if not history:
        return html.Div("Ask anything about the current forecast, methodology, or market context.",
                        style={"color": "#94a3b8", "fontSize": "12px", "padding": "4px 0"})

    bubbles = []
    for msg in history:
        is_user = msg["role"] == "user"
        bubbles.append(html.Div(
            style={
                "display": "flex",
                "justifyContent": "flex-end" if is_user else "flex-start",
                "marginBottom": "10px",
            },
            children=[
                html.Div(
                    msg["content"],
                    style={
                        "maxWidth": "80%",
                        "background":   "#2563eb" if is_user else "#f1f5f9",
                        "color":        "#ffffff"  if is_user else "#1e293b",
                        "borderRadius": "12px 12px 2px 12px" if is_user else "12px 12px 12px 2px",
                        "padding":      "10px 14px",
                        "fontSize":     "13px",
                        "lineHeight":   "1.55",
                    },
                ),
            ],
        ))
    return bubbles


# ---------------------------------------------------------------------------
# (Datacenter impact panel removed — underlying data kept in api_cache/)


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
server = app.server  # expose Flask server for gunicorn


# ---------------------------------------------------------------------------
# Mount the FastAPI REST layer (api.py) at /fastapi on the same Flask server.
# Wrapped defensively — a problem here must not take down the dashboard.
# ---------------------------------------------------------------------------
try:
    from a2wsgi import ASGIMiddleware
    from werkzeug.middleware.dispatcher import DispatcherMiddleware
    from api import app as _fastapi_app

    server.wsgi_app = DispatcherMiddleware(
        server.wsgi_app, {"/fastapi": ASGIMiddleware(_fastapi_app)}
    )
    logging.getLogger(__name__).info("FastAPI layer mounted at /fastapi")
except Exception:
    logging.getLogger(__name__).exception("FastAPI layer failed to mount — dashboard continues without it")


# ---------------------------------------------------------------------------
# REST API routes (served by the same Flask server as the dashboard)
# ---------------------------------------------------------------------------
from flask import jsonify, request as flask_request

_VALID_ISOS = {"pjm", "caiso", "ercot", "miso", "nyiso", "isone", "spp"}


@server.route("/api/v1/forecast/<iso>")
def api_forecast(iso):
    """
    GET /api/v1/forecast/<iso>?force=0
    Returns the 15-day load forecast for the given ISO as JSON.

    Response schema:
    {
      "iso": "pjm",
      "generated": "2025-07-27T10:00:00",
      "load": [{date, mean_load_gw, low_load_gw, high_load_gw, hdd, cdd, avg_temp_f}],
      "price_forecast": [{date, forecast_price, low_price, high_price}],
      "net_load": [{date, solar_gw, wind_gw, renewable_gw}],
      "backtest": {mape_test, ...}
    }
    """
    iso = iso.lower()
    if iso not in _VALID_ISOS:
        return jsonify({"error": f"Unknown ISO. Valid: {sorted(_VALID_ISOS)}"}), 400

    force = flask_request.args.get("force", "0") == "1"
    data  = _fetch_iso_forecast(iso, force=force)
    if not data:
        return jsonify({"error": f"Forecast unavailable for {iso.upper()} — model may not be trained yet"}), 503

    return jsonify({
        "iso":            iso.upper(),
        "generated":      datetime.now().isoformat(timespec="seconds"),
        "load":           data.get("load", []),
        "price_forecast": data.get("price_forecast", []),
        "net_load":       data.get("net_load", []),
        "backtest":       data.get("backtest", {}),
    })


@server.route("/api/v1/isos")
def api_isos():
    """GET /api/v1/isos — list available ISOs and their model status."""
    status = []
    for iso in sorted(_VALID_ISOS):
        cfg   = _ISO_CONFIGS.get(iso, {})
        model = cfg.get("model_ref", lambda: None)() if cfg else None
        status.append({
            "iso":         iso.upper(),
            "model_ready": model is not None,
            "cache_file":  str(cfg.get("cache_file", "")) if cfg else "",
        })
    return jsonify({"isos": status})


@server.route("/api/v1/verification/<iso>")
def api_verification(iso):
    """GET /api/v1/verification/<iso> — forecast vs actuals statistics."""
    iso = iso.lower()
    if iso not in _VALID_ISOS:
        return jsonify({"error": f"Unknown ISO. Valid: {sorted(_VALID_ISOS)}"}), 400
    stats = load_verification_stats(iso)
    return jsonify({"iso": iso.upper(), **stats})


@server.route("/api/v1/health")
def api_health():
    """GET /api/v1/health — liveness check."""
    return jsonify({"status": "ok", "timestamp": datetime.now().isoformat()})


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, port=port)
