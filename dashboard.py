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
from dash import dcc, html, Input, Output, ALL
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
from temperature_modeling.datacenter_agent import load_datacenter_projects
from temperature_modeling.verification import record_forecast, load_verification_stats
from temperature_modeling.net_load import fetch_net_load_forecast
from temperature_modeling.price_forecast import forecast_prices
from temperature_modeling.capacity_market import get_capacity_market_data, get_reserve_margin_color
from temperature_modeling.ensemble import get_ensemble_forecast

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
# Datacenter projects — loaded from agent-maintained cache (falls back to baseline)
# ---------------------------------------------------------------------------
_DC_PROJECTS = load_datacenter_projects()

_STATUS_COLOR = {
    "Operating": "#22c55e", "Operating / Expanding": "#22c55e",
    "Under Construction": "#f97316", "Approved": "#2563eb",
    "Announced": "#94a3b8",
}


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
            for d2, avg_f in avg_f_hist.items():
                lag1  = avg_f_hist.get(d2 - timedelta(days=1))
                lag2  = avg_f_hist.get(d2 - timedelta(days=2))
                lag7  = avg_f_hist.get(d2 - timedelta(days=7))
                rv    = [avg_f_hist.get(d2 - timedelta(days=k)) for k in range(7)]
                roll7 = sum(v for v in rv if v) / max(sum(1 for v in rv if v), 1)
                feats, _, _ = _bf(avg_f, avg_f + 5, avg_f - 5, d2, lag1, lag2, lag7, roll7)
                hindcast[d2.isoformat()] = round(model.predict([feats])[0] / 1000, 2)
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
    )
    load_data = [{"date": lf.valid_date.isoformat(),
                  "mean_load_gw": round(lf.mean_load_mw / 1000, 2),
                  "low_load_gw":  round(lf.low_load_mw  / 1000, 2),
                  "high_load_gw": round(lf.high_load_mw / 1000, 2),
                  "avg_temp_f":   round(lf.avg_temp_f, 1) if lf.avg_temp_f else None,
                  "hdd":          round(lf.hdd, 1),
                  "cdd":          round(lf.cdd, 1)}
                 for lf in load_forecasts]

    # Official benchmark data
    comparison: dict = {"actual": {}, "da_fcst": {}}
    try:
        comparison = cfg["comparison_fn"](session)
    except Exception:
        log.exception("%s: official comparison fetch failed — chart will show no actuals", iso.upper())

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

    result = {
        "load":            load_data,
        "dates":           forecast_dates_strs,
        "comparison":      comparison,
        cfg["bench_key"]:  bench_data,
        "backtest":        backtest,
        "hindcast":        hindcast,
    }
    # Enrich result with net load and price forecast before caching
    net_load = fetch_net_load_forecast(iso, forecast_dates_list, session)
    result["net_load"] = net_load

    prices = forecast_prices(iso, load_data)
    result["price_forecast"] = prices

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


def _prefetch_comparisons():
    """Populate comparison caches for CAISO, ERCOT, MISO at startup."""
    import time as _t
    from temperature_modeling.caiso_load import fetch_caiso_official_comparison
    from temperature_modeling.ercot_load import fetch_ercot_official_comparison
    from temperature_modeling.miso_load  import fetch_miso_official_comparison

    _s = requests.Session()
    _s.headers["User-Agent"] = "grid-dashboard/startup"
    for label, fn in [
        ("CAISO", lambda: fetch_caiso_official_comparison(_s)),
        ("ERCOT", lambda: fetch_ercot_official_comparison(_s)),
        ("MISO",  lambda: fetch_miso_official_comparison(_s)),
    ]:
        try:
            result = fn()
            n = len(result.get("actual", {}))
            log.info("%s comparison: %d days cached", label, n)
        except Exception:
            log.exception("%s comparison prefetch failed", label)
        _t.sleep(1)


import threading as _threading
_threading.Thread(target=_prefetch_comparisons, daemon=True).start()

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

METHODOLOGY_NOTE = """
**Methodology** — Separate machine learning models are trained for each ISO (PJM, CAISO, ERCOT, MISO) on 2 years of \
hourly EIA demand data aggregated to daily averages, paired with ERA5 reanalysis temperatures. Temperature inputs are \
population-weighted across 12 representative monitoring locations per ISO footprint. Features include \
heating/cooling degree-days (HDD/CDD) from daily average, high, and low temperatures; day-of-week encoding; \
US federal holiday and holiday-week flags; bridge-day indicators; T−1, T−2, and T−7 temperature lags; and a 7-day \
rolling average temperature to capture heat-wave persistence and population acclimatisation (27 features total). \
Forward forecasts use GFS NWP output via Open-Meteo (15-day horizon). Historical hindcast uses ERA5 reanalysis. \
**ISO coverage:** PJM (Eastern US, ~65 GW peak) — 12 locations from Chicago to Washington DC; \
CAISO (California, ~45 GW peak) — 12 locations from San Diego to Sacramento; \
ERCOT (Texas, ~80 GW peak) — 12 locations from Houston to Amarillo; \
MISO (Midcontinent, ~120 GW peak) — 12 locations from New Orleans to Fargo. \
**Benchmarks:** PJM uses PJM DataMiner official 7-day forecast; CAISO uses CAISO OASIS 7-day system forecast; \
ERCOT uses ERCOT public reports API when available, otherwise EIA day-ahead; MISO uses EIA day-ahead demand forecast. \
**Accuracy note:** The hindcast MAPE shown (~0.4–0.5%) is measured under ERA5 observed temperatures on the \
held-out 20% test set — it reflects model fit given known weather, not live forecast skill. \
Forward forecast error, driven by GFS temperature uncertainty, is higher and consistent with industry norms of \
1–3% day-ahead and 3–5% week-ahead for temperature-driven load models.
"""


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------
app = dash.Dash(
    __name__,
    title="Grid Load Forecast — PJM & CAISO",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
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
                html.Button(
                    "⟳ Refresh", id="refresh-btn",
                    style={"background": "#f1f5f9", "border": "1px solid #e2e8f0",
                           "color": "#475569", "padding": "6px 16px",
                           "borderRadius": "6px", "cursor": "pointer", "fontSize": "13px"},
                ),
            ],
        ),

        # ── Body ──────────────────────────────────────────────────────────────
        html.Div(
            style={"maxWidth": "1400px", "margin": "0 auto", "padding": "20px 24px"},
            children=[

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

                # ── Datacenter impact analysis ─────────────────────────────────
                html.Div(
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div("Proposed Datacenter — Grid Impact Analysis",
                                 style={"fontSize": "13px", "fontWeight": 600,
                                        "color": "#0f172a", "marginBottom": "4px"}),
                        html.Div(
                            "Multi-dimensional impact analysis for approved and announced large datacenter "
                            "projects in the selected ISO territory. Select one or more projects to model "
                            "their combined effect on load, prices, emissions, and reserve margin. "
                            "Assumes 90% capacity factor.",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "16px"},
                        ),
                        html.Div(id="dc-project-table", style={"marginBottom": "16px"}),
                        html.Div(id="dc-cards",
                                 style={"display": "flex", "gap": "10px",
                                        "flexWrap": "wrap", "marginBottom": "20px"}),
                        # Load impact
                        html.Div("Load Impact — 15-Day Forecast",
                                 style={"fontSize": "11px", "fontWeight": 600,
                                        "color": "#64748b", "textTransform": "uppercase",
                                        "letterSpacing": "0.05em", "marginBottom": "6px"}),
                        dcc.Graph(id="dc-chart", style={"height": "260px"},
                                  config={"displayModeBar": False}),
                        # Non-linear price impact
                        html.Div("Price Impact — Non-Linear Merit Order Response",
                                 style={"fontSize": "11px", "fontWeight": 600,
                                        "color": "#64748b", "textTransform": "uppercase",
                                        "letterSpacing": "0.05em",
                                        "marginTop": "18px", "marginBottom": "4px"}),
                        html.Div(
                            "Price sensitivity rises non-linearly as load approaches grid capacity — "
                            "peaking units are progressively more expensive. High-load days carry "
                            "disproportionately higher marginal cost.",
                            style={"fontSize": "11px", "color": "#94a3b8", "marginBottom": "8px"},
                        ),
                        dcc.Graph(id="dc-price-chart", style={"height": "240px"},
                                  config={"displayModeBar": False}),
                        # Emissions impact
                        html.Div("Emissions Impact",
                                 style={"fontSize": "11px", "fontWeight": 600,
                                        "color": "#64748b", "textTransform": "uppercase",
                                        "letterSpacing": "0.05em",
                                        "marginTop": "18px", "marginBottom": "4px"}),
                        html.Div(id="dc-emissions-div",
                                 style={"fontSize": "12px", "color": "#475569",
                                        "lineHeight": "1.7"}),

                        # Research benchmark comparison
                        html.Div("How Our Estimates Compare to Published Research",
                                 style={"fontSize": "11px", "fontWeight": 600,
                                        "color": "#64748b", "textTransform": "uppercase",
                                        "letterSpacing": "0.05em",
                                        "marginTop": "24px", "marginBottom": "10px"}),
                        html.Div(id="dc-benchmark-table"),
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
                            "Separate machine learning models are trained for each ISO (PJM, CAISO, ERCOT, MISO) "
                            "on 2 years of hourly EIA demand data aggregated to daily averages, paired with ERA5 "
                            "reanalysis temperatures population-weighted across 12 representative locations per ISO. "
                            "Features include HDD/CDD from daily average, high, and low temperatures; day-of-week "
                            "encoding; US federal holiday flags; T−1, T−2, and T−7 temperature lags; and a 7-day "
                            "rolling average temperature to capture heat-wave persistence (27 features total). "
                            "Forward forecasts use GFS NWP output via Open-Meteo (15-day horizon); historical "
                            "hindcast uses ERA5 reanalysis.",
                            style={"fontSize": "12px", "color": "#64748b",
                                   "lineHeight": "1.7", "margin": "0 0 8px 0"},
                        ),
                        html.P(
                            "ISO coverage — "
                            "PJM (Eastern US, ~65 GW peak): 12 locations from Chicago to Washington DC. "
                            "CAISO (California, ~45 GW peak): 12 locations from San Diego to Sacramento. "
                            "ERCOT (Texas, ~80 GW peak): 12 locations weighted by population from Houston to Amarillo. "
                            "MISO (Midcontinent, ~120 GW peak): 12 locations from New Orleans to Fargo. "
                            "Benchmarks — PJM: PJM DataMiner official 7-day forecast. "
                            "CAISO: CAISO OASIS 7-day system forecast. "
                            "ERCOT: ERCOT public reports API (ERCOT_API_KEY) or EIA day-ahead. "
                            "MISO: EIA day-ahead demand forecast. "
                            "Accuracy note: the hindcast MAPE shown (~0.4–0.5%) is measured under ERA5 "
                            "observed temperatures on the held-out 20% test set — it reflects model fit "
                            "given known weather, not live forecast skill. Forward forecast error (GFS "
                            "temperature uncertainty) is higher, consistent with industry norms of "
                            "1–3% day-ahead and 3–5% week-ahead for temperature-driven load models.",
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
        dcc.Store(id="ensemble-store"),
        dcc.Store(id="chat-history-store", data=[]),
        dcc.Store(id="dc-selected-mw", data=0),
        dcc.Interval(id="daily-refresh", interval=24 * 60 * 60 * 1000, n_intervals=0),
        dcc.Interval(id="forecast-refresh", interval=15 * 60 * 1000, n_intervals=0),
    ],
)


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
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
)
def render(data, iso):
    if not data or not data.get("load"):
        return ([], "", _empty_fig())

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

    summary_cards = [
        card("Today (GW)", f"{today_gw:.1f}", "GFS-based", load_color(today_gw)),
        card("15-Day Peak", f"{peak_gw:.1f}", f"on {peak_lbl}", load_color(peak_gw)),
        card("15-Day Avg", f"{avg_gw:.1f}", "GW baseline", "#475569"),
        card("Hindcast MAPE", mape_str, "ERA5 obs. temps", mape_col),
        card("Verified MAPE 7d", v7, f"30d: {v30}  bias: {vbias}", v_col),
    ]

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

    return summary_cards, subtitle, fig


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
        fig = go.Figure()
        fig.add_annotation(text="Price data unavailable (EIA_API_KEY required)",
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
    price_s = f"${price:,.0f}/MW-year" if price else "Energy-only market"
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
# Datacenter impact callback
# ---------------------------------------------------------------------------
@app.callback(
    Output("dc-project-table", "children"),
    Input("iso-selector", "value"),
)
def render_project_table(iso):
    projects = _DC_PROJECTS.get(iso, [])

    def _badge(status):
        col = _STATUS_COLOR.get(status, "#94a3b8")
        return html.Span(status, style={
            "backgroundColor": col + "18", "color": col,
            "border": f"1px solid {col}40",
            "borderRadius": "4px", "padding": "1px 7px",
            "fontSize": "10px", "fontWeight": 600,
        })

    rows = []
    for p in projects:
        rows.append(html.Tr([
            html.Td(
                dcc.Checklist(
                    id={"type": "dc-proj-check", "id": p["id"]},
                    options=[{"label": "", "value": p["id"]}],
                    value=[p["id"]],  # default: all selected
                    inputStyle={"cursor": "pointer"},
                ),
                style={"width": "30px", "verticalAlign": "middle"},
            ),
            html.Td(html.Div([
                html.Div(p["name"],
                         style={"fontSize": "12px", "fontWeight": 600, "color": "#0f172a"}),
                html.Div(p["operator"],
                         style={"fontSize": "10px", "color": "#94a3b8"}),
            ]), style={"verticalAlign": "middle", "paddingRight": "12px"}),
            html.Td(p["location"],
                    style={"fontSize": "11px", "color": "#64748b", "verticalAlign": "middle"}),
            html.Td(f"{p['mw']:,} MW",
                    style={"fontSize": "12px", "fontWeight": 700, "color": "#0f172a",
                           "textAlign": "right", "verticalAlign": "middle",
                           "paddingRight": "12px"}),
            html.Td(_badge(p["status"]), style={"verticalAlign": "middle"}),
        ], style={"borderBottom": "1px solid #f1f5f9"}))

    return html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(html.Tr([
                html.Th("", style={"width": "30px"}),
                html.Th("Project", style=_th),
                html.Th("Location", style=_th),
                html.Th("Capacity", style={**_th, "textAlign": "right"}),
                html.Th("Status", style=_th),
            ])),
            html.Tbody(rows),
        ],
    )


_th = {"fontSize": "10px", "color": "#94a3b8", "textTransform": "uppercase",
       "letterSpacing": "0.05em", "padding": "4px 0 8px 0", "fontWeight": 600}


@app.callback(
    Output("dc-cards", "children"),
    Output("dc-chart", "figure"),
    Output("dc-price-chart", "figure"),
    Output("dc-emissions-div", "children"),
    Output("dc-benchmark-table", "children"),
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
    Input({"type": "dc-proj-check", "id": ALL}, "value"),
)
def render_datacenter(data, iso, check_values):
    projects  = _DC_PROJECTS.get(iso, [])
    selected  = {v[0] for v in (check_values or []) if v}
    total_mw  = sum(p["mw"] for p in projects if p["id"] in selected)
    dc_mw     = total_mw or 0

    empty = ([], _empty_fig("Select projects above to model their grid impact"),
             _empty_fig(), "", "")
    if not data or not data.get("load") or dc_mw == 0:
        return empty

    load_list   = data["load"]
    dates       = [d["date"]         for d in load_list]
    means       = [d["mean_load_gw"] for d in load_list]
    actual_dict = data.get("comparison", {}).get("actual", {})
    pm          = data.get("price_model", {})
    p_slope     = pm.get("slope")
    p_int       = pm.get("intercept")

    CF      = 0.90
    dc_gw   = dc_mw * CF / 1000
    ann_gwh = dc_gw * 8760            # annual energy, GWh

    avg_load = sum(means) / len(means) if means else 1
    peak_historical = max(actual_dict.values()) if actual_dict else max(means) * 1.15
    peak_forecast   = max(means)
    pct_increase    = dc_gw / avg_load * 100
    augmented       = [gw + dc_gw for gw in means]

    # ── Non-linear merit-order price sensitivity ───────────────────────────────
    # Calibrated from ISO market data: price sensitivity rises exponentially
    # as load approaches grid capacity (peakers progressively more expensive).
    # ERCOT uses scarcity pricing (ORDC) so sensitivity is higher near peak.
    _base_sens_map = {"pjm": 5.5, "caiso": 7.0, "ercot": 8.0, "miso": 5.0}
    base_sens = _base_sens_map.get(iso, 5.5)   # $/MWh per GW at reference load
    nl_exp    = 4.4                              # exponential steepness
    ref_ratio = 0.70                             # reference load ratio

    def _price_sens(load_gw):
        ratio = load_gw / peak_historical
        return base_sens * (2.718 ** (nl_exp * max(0, ratio - ref_ratio)))

    # Daily price delta (non-linear)
    daily_price_delta_base = [round(_price_sens(gw) * dc_gw, 1) for gw in means]
    daily_price_delta_aug  = [round(_price_sens(gw + dc_gw) * dc_gw, 1) for gw in means]

    # Baseline + augmented projected prices
    base_prices = (
        [round(p_slope * gw + p_int, 1) for gw in means]
        if p_slope is not None else []
    )
    aug_prices = (
        [round(p_slope * (gw + dc_gw) + p_int, 1) for gw in means]
        if p_slope is not None else []
    )

    avg_delta = sum(daily_price_delta_base) / len(daily_price_delta_base)
    peak_delta = max(daily_price_delta_base)

    # ── Emissions (EPA eGRID 2023 marginal rates) ─────────────────────────────
    # PJM: 1,264 | CAISO: 876 | ERCOT (ERCT): 1,050 | MISO (MROW/SRSO): 1,580
    _lbs_map = {"pjm": 1264, "caiso": 876, "ercot": 1050, "miso": 1580}
    lbs_per_mwh   = _lbs_map.get(iso, 1264)
    tons_per_mwh  = lbs_per_mwh / 2204.62
    ann_co2_ktons = ann_gwh * 1000 * tons_per_mwh / 1000   # kilotonnes
    ann_co2_mtons = ann_co2_ktons / 1000                   # megatonnes

    # ── Reserve margin impact ─────────────────────────────────────────────────
    # PJM target 20% (current 14.8%) | CAISO 15% (current 16.0%)
    # ERCOT target 10.75% (current ~10.5%, historically tight)
    # MISO target 18.1% (current ~20.3%)
    _rm_target_map  = {"pjm": 20.0, "caiso": 15.0, "ercot": 10.75, "miso": 18.1}
    _rm_current_map = {"pjm": 14.8, "caiso": 16.0, "ercot": 10.5,  "miso": 20.3}
    rm_target  = _rm_target_map.get(iso, 20.0)
    rm_current = _rm_current_map.get(iso, 14.8)
    rm_delta   = dc_gw / (peak_historical + dc_gw) * 100   # % reduction in reserve

    # ── Capacity market cost ──────────────────────────────────────────────────
    # PJM Dec-2025: $333.44/MW-day | CAISO RA: ~$80/MW-day
    # ERCOT: energy-only market, no capacity payment ($0) but high scarcity pricing
    # MISO Planning Resource Auction: ~$35/MW-day (zone-dependent)
    _cap_map = {"pjm": 333.44, "caiso": 80.0, "ercot": 0.0, "miso": 35.0}
    cap_price_per_mw_day = _cap_map.get(iso, 333.44)
    ann_cap_cost_m       = dc_mw * cap_price_per_mw_day * 365 / 1e6

    # ── Interconnection cost estimate ─────────────────────────────────────────
    # PJM 2024: ~$145M/GW | CAISO: ~$145M/GW
    # ERCOT: large queue backlog, ~$160M/GW (ERCOT nodal interconnection)
    # MISO: less congested queue, ~$120M/GW (MISO Generator Interconnection Process)
    _intercon_map = {"pjm": 145, "caiso": 145, "ercot": 160, "miso": 120}
    intercon_cost_m = dc_gw * _intercon_map.get(iso, 145)

    # ── REC / Carbon neutrality cost ─────────────────────────────────────────
    # PJM Class I RECs: ~$40/MWh | CAISO: ~$15/MWh (abundant solar)
    # ERCOT: ~$8/MWh (abundant wind, cheapest RECs in US)
    # MISO: ~$20/MWh (growing wind but coal-heavy grid → higher offset needed)
    _rec_map = {"pjm": 40, "caiso": 15, "ercot": 8, "miso": 20}
    rec_mwh = _rec_map.get(iso, 40)
    ann_rec_cost_m = ann_gwh * rec_mwh / 1e3

    # ── Summary cards ─────────────────────────────────────────────────────────
    n_proj = len(selected)
    dc_cards = [
        card("Projects Selected", str(n_proj),            f"{dc_mw:,} MW total",     "#0f172a"),
        card("Continuous Load",   f"{dc_gw:.2f} GW",      "90% CF, 24/7",            "#2563eb"),
        card("Load Increase",     f"+{pct_increase:.1f}%","vs 15-day avg",            "#f97316"),
        card("Avg Price Impact",  f"+${avg_delta:.1f}",   "$/MWh (non-linear avg)",   "#7c3aed"),
        card("Peak Day Price",    f"+${peak_delta:.1f}",  "$/MWh on peak day",        "#ef4444"),
        card("Annual CO₂",        f"{ann_co2_ktons:.0f}", "kt CO₂/yr (marginal)",     "#64748b"),
        card("Reserve Margin",    f"−{rm_delta:.2f}%",    f"from {rm_current:.1f}% current", "#dc2626"),
        card("Capacity Cost",     f"${ann_cap_cost_m:.1f}M","$/yr (market rate)",    "#0284c7"),
        card("Interconnection",   f"~${intercon_cost_m:.0f}M","one-time est.",       "#475569"),
    ]

    # ── Load impact chart ─────────────────────────────────────────────────────
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=means, mode="lines+markers", name="Baseline",
        line=dict(color="#94a3b8", width=2), marker=dict(size=4),
        hovertemplate="<b>%{x|%d %b}</b><br>Baseline: %{y:.1f} GW<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=augmented, mode="lines+markers",
        name=f"With {dc_mw:,} MW DC",
        line=dict(color="#2563eb", width=2.5), marker=dict(size=4),
        fill="tonexty", fillcolor="rgba(37,99,235,0.08)",
        hovertemplate="<b>%{x|%d %b}</b><br>With DC: %{y:.1f} GW<extra></extra>",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.30, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", x=0),
        margin=dict(l=60, r=20, t=10, b=80),
        xaxis=dict(type="date", tickformat="%d %b", gridcolor="#f8fafc",
                   tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0", dtick="D1"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".1f", ticksuffix=" GW",
                   title=dict(text="Load (GW)", font=dict(size=11, color="#64748b")),
                   linecolor="#e2e8f0"),
        height=260,
    )

    # ── Non-linear price impact chart ─────────────────────────────────────────
    # Show daily price delta as bars, colored by load-ratio tier
    bar_colors = []
    for gw in means:
        ratio = gw / peak_historical
        if ratio > 0.90:
            bar_colors.append("#ef4444")   # red — high scarcity
        elif ratio > 0.80:
            bar_colors.append("#f97316")   # amber — elevated
        else:
            bar_colors.append("#7c3aed")   # purple — normal

    dc_pfig = go.Figure()
    dc_pfig.add_trace(go.Bar(
        x=dates, y=daily_price_delta_base,
        name="Price delta ($/MWh)",
        marker_color=bar_colors,
        hovertemplate="<b>%{x|%d %b}</b><br>Price impact: +$%{y:.1f}/MWh<extra></extra>",
    ))
    if base_prices:
        dc_pfig.add_trace(go.Scatter(
            x=dates, y=base_prices, mode="lines", name="Baseline price",
            line=dict(color="#94a3b8", width=1.5, dash="dot"),
            yaxis="y2",
            hovertemplate="<b>%{x|%d %b}</b><br>Base: $%{y:.0f}/MWh<extra></extra>",
        ))
        dc_pfig.update_layout(yaxis2=dict(
            title=dict(text="Price ($/MWh)", font=dict(size=10, color="#94a3b8")),
            overlaying="y", side="right", showgrid=False,
            tickformat=".0f", tickprefix="$",
            tickfont=dict(color="#94a3b8", size=9),
        ))

    dc_pfig.add_hline(y=avg_delta, line_dash="dot", line_color="#7c3aed",
                      line_width=1.5,
                      annotation_text=f"avg +${avg_delta:.1f}/MWh",
                      annotation_font_size=9, annotation_font_color="#7c3aed",
                      annotation_position="top right")
    dc_pfig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", x=0),
        margin=dict(l=60, r=55 if base_prices else 20, t=10, b=85),
        xaxis=dict(type="date", tickformat="%d %b", gridcolor="#f8fafc",
                   tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0", dtick="D1"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".1f", tickprefix="+$",
                   title=dict(text="Marginal Price Increase ($/MWh)",
                              font=dict(size=11, color="#64748b")),
                   linecolor="#e2e8f0"),
        height=240,
        barmode="overlay",
    )
    # Tier legend annotation
    dc_pfig.add_annotation(
        x=1.0, y=1.06, xref="paper", yref="paper", showarrow=False,
        text="<span style='color:#7c3aed'>■</span> Normal  "
             "<span style='color:#f97316'>■</span> Elevated  "
             "<span style='color:#ef4444'>■</span> Peak scarcity",
        font=dict(size=9, color="#64748b"), align="right",
    )

    # ── Emissions text block ───────────────────────────────────────────────────
    carbon_price_usd = 50   # $/tonne CO2 (EU ETS reference)
    social_cost_m = ann_co2_ktons * carbon_price_usd / 1e3  # $M/yr social cost
    _dc_iso_labels = {"pjm": "PJM", "caiso": "CAISO", "ercot": "ERCOT", "miso": "MISO",
                      "nyiso": "NYISO", "isone": "ISO-NE", "spp": "SPP"}
    iso_label = _dc_iso_labels.get(iso, iso.upper())
    emissions_text = [
        html.Span(f"Annual marginal CO₂ emissions: "),
        html.Strong(f"{ann_co2_ktons:.0f} kt CO₂/yr"),
        html.Span(f" ({ann_co2_mtons:.2f} Mt) — based on EPA eGRID 2023 marginal "
                  f"rate for {iso_label} ({lbs_per_mwh:,} lbs/MWh). "),
        html.Br(),
        html.Span(f"Social cost of carbon (~$50/t): "),
        html.Strong(f"~${social_cost_m:.1f}M/yr"),
        html.Span(f".  Carbon neutrality via RECs (~${rec_mwh}/MWh): "),
        html.Strong(f"~${ann_rec_cost_m:.1f}M/yr"),
        html.Span("."),
        html.Br(),
        html.Span(
            f"Note: marginal emission rates reflect the actual generators dispatched "
            f"to meet incremental load — higher than average grid rates because "
            f"baseload (nuclear, hydro) is already committed.",
            style={"color": "#94a3b8", "fontSize": "11px"},
        ),
    ]

    # ── Research benchmark comparison table ───────────────────────────────────
    per_gw_dc = max(dc_gw, 0.001)
    our_co2   = ann_co2_ktons / per_gw_dc          # kt/GW
    our_cap   = ann_cap_cost_m / (dc_mw / 1000)    # $M/GW/yr
    our_intercon = intercon_cost_m / per_gw_dc      # $M/GW

    def _align(ours, pub_lo, pub_hi, unit, note=""):
        mid = (pub_lo + pub_hi) / 2
        pct = (ours - mid) / mid * 100 if mid else 0
        if abs(pct) < 15:
            verdict, col = "Aligned", "#22c55e"
        elif pct < 0:
            verdict, col = "Conservative", "#f97316"
        else:
            verdict, col = "Above research", "#ef4444"
        return [
            html.Span(f"{ours:.0f} {unit}",
                      style={"fontWeight": 700, "color": "#0f172a"}),
            html.Span(f"  vs  {pub_lo:.0f}–{pub_hi:.0f} {unit} (published)",
                      style={"color": "#64748b", "fontSize": "11px"}),
            html.Span(f"  {verdict}" + (f"  — {note}" if note else ""),
                      style={"color": col, "fontSize": "11px", "fontWeight": 600}),
        ]

    bm_rows = [
        ("Interconnection cost", _align(
            our_intercon, 145, 240, "$M/GW",
            "UCS: $147M/GW; LBNL mean: $240M/GW")),
        ("Capacity market cost", _align(
            our_cap, 110, 125, "$M/GW/yr",
            "PJM Dec-2025 auction $333.44/MW-day — exact match")),
        ("CO₂ — marginal emissions", _align(
            our_co2, 450, 580, "kt CO₂/GW/yr",
            "EPA eGRID 2023 marginal rate; MIT/CMU confirm MEF is correct method")),
        ("Price sensitivity (avg)", [
            html.Span(f"${avg_delta / dc_gw:.1f}/MWh per GW",
                      style={"fontWeight": 700, "color": "#0f172a"}),
            html.Span("  vs  IEEFA: 'factor of 10×' in capacity prices; no published $/MWh/GW spot figure",
                      style={"color": "#64748b", "fontSize": "11px"}),
            html.Span("  Conservative — spot price impact understudied",
                      style={"color": "#f97316", "fontSize": "11px", "fontWeight": 600}),
        ]),
        ("Reserve margin impact", [
            html.Span(f"−{rm_delta:.2f}% per {dc_gw:.1f} GW",
                      style={"fontWeight": 700, "color": "#0f172a"}),
            html.Span("  Harvard Belfer Centre confirms thinning margins; no published $/GW figure",
                      style={"color": "#64748b", "fontSize": "11px"}),
            html.Span("  Directionally correct — no benchmark to validate magnitude",
                      style={"color": "#94a3b8", "fontSize": "11px", "fontWeight": 600}),
        ]),
    ]

    bm_table = html.Table(
        style={"width": "100%", "borderCollapse": "collapse"},
        children=[
            html.Thead(html.Tr([
                html.Th("Metric", style=_th),
                html.Th("Our estimate vs published range", style=_th),
            ])),
            html.Tbody([
                html.Tr([
                    html.Td(label,
                            style={"fontSize": "12px", "color": "#475569",
                                   "fontWeight": 500, "padding": "8px 12px 8px 0",
                                   "verticalAlign": "top", "whiteSpace": "nowrap",
                                   "borderBottom": "1px solid #f1f5f9"}),
                    html.Td(cells,
                            style={"fontSize": "12px", "padding": "8px 0",
                                   "borderBottom": "1px solid #f1f5f9",
                                   "lineHeight": "1.6"}),
                ])
                for label, cells in bm_rows
            ]),
        ],
    )

    return dc_cards, fig, dc_pfig, emissions_text, bm_table


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
server = app.server  # expose Flask server for gunicorn


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
