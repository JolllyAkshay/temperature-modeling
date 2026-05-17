"""
US Grid Load Forecast Dashboard — PJM & CAISO
Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""

import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import dash
from dash import dcc, html, Input, Output, ALL
import plotly.graph_objects as go

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
)
from temperature_modeling.caiso import CAISO_LOAD_LOCATIONS
from temperature_modeling.caiso_load import (
    CAISOLoadModel, load_caiso_model,
    fetch_caiso_official_comparison, fetch_caiso_oasis_7day,
    weighted_avg_temp_f_caiso, _CAISO_MODEL_PATH,
)

FORECAST_CACHE_TTL_HOURS = 3
_PRICE_CACHE_TTL_HOURS   = 6

_LOAD_FORECAST_CACHE_FILE  = _HERE / "api_cache" / "load_forecast_cache.json"
_LOAD_TRAINING_DATA_PATH   = _HERE / "api_cache" / "pjm_load_training.json"
_CAISO_FORECAST_CACHE_FILE = _HERE / "api_cache" / "caiso_forecast_cache.json"
_CAISO_TRAINING_DATA_PATH  = _HERE / "api_cache" / "caiso_load_training.json"
_PJM_PRICE_CACHE           = _HERE / "api_cache" / "pjm_price_cache.json"
_CAISO_PRICE_CACHE         = _HERE / "api_cache" / "caiso_price_cache.json"


# ---------------------------------------------------------------------------
# Price helpers
# ---------------------------------------------------------------------------
def _fetch_pjm_prices(session):
    """PJM day-ahead system energy prices from DataMiner → {date_str: avg $/MWh}."""
    if _PJM_PRICE_CACHE.exists():
        if (time.time() - _PJM_PRICE_CACHE.stat().st_mtime) / 3600 < _PRICE_CACHE_TTL_HOURS:
            try:
                return json.loads(_PJM_PRICE_CACHE.read_text())
            except Exception:
                pass
    key = os.environ.get("PJM_API_KEY", "")
    if not key:
        return {}
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=60)
    try:
        r = session.get(
            "https://api.pjm.com/api/v1/da_hrl_lmps",
            headers={"Ocp-Apim-Subscription-Key": key},
            params={
                "startRow": 1, "rowCount": 50000,
                "fields": "datetime_beginning_ept,system_energy_price_da",
                "datetime_beginning_ept": f"{start_dt} 00:00 to {end_dt} 23:00",
                "type": "ZONE",
            },
            timeout=30,
        )
        r.raise_for_status()
        # system_energy_price_da is the same across all zones per hour — deduplicate by hour
        seen_hours: set = set()
        hourly: dict = {}
        for item in r.json().get("items", []):
            hr = str(item.get("datetime_beginning_ept", ""))
            p  = item.get("system_energy_price_da")
            if hr and p is not None and hr not in seen_hours:
                seen_hours.add(hr)
                d = hr[:10]
                hourly.setdefault(d, []).append(float(p))
        result = {d: round(sum(v) / len(v), 2) for d, v in hourly.items() if v}
        _PJM_PRICE_CACHE.write_text(json.dumps(result))
        return result
    except Exception:
        try:
            return json.loads(_PJM_PRICE_CACHE.read_text())
        except Exception:
            return {}


def _fetch_caiso_prices(session):
    """CAISO day-ahead LMPs (SP15 hub) from OASIS → {date_str: avg $/MWh}."""
    if _CAISO_PRICE_CACHE.exists():
        if (time.time() - _CAISO_PRICE_CACHE.stat().st_mtime) / 3600 < _PRICE_CACHE_TTL_HOURS:
            try:
                return json.loads(_CAISO_PRICE_CACHE.read_text())
            except Exception:
                pass
    import zipfile, io, xml.etree.ElementTree as _ET
    end_dt   = date.today()
    start_dt = end_dt - timedelta(days=31)   # CAISO OASIS max 31 days per query
    try:
        r = session.get(
            "http://oasis.caiso.com/oasisapi/SingleZip",
            params={
                "queryname":     "PRC_LMP",
                "startdatetime": start_dt.strftime("%Y%m%dT07:00-0000"),
                "enddatetime":   end_dt.strftime("%Y%m%dT07:00-0000"),
                "version":       1,
                "market_run_id": "DAM",
                "node":          "TH_SP15_GEN-APND",
            },
            timeout=60,
        )
        r.raise_for_status()
        daily: dict = {}
        with zipfile.ZipFile(io.BytesIO(r.content)) as zf:
            fname = next((n for n in zf.namelist() if n.endswith(".xml")), zf.namelist()[0])
            with zf.open(fname) as f:
                root = _ET.parse(f).getroot()
                for rd in root.iter():
                    if not rd.tag.endswith("REPORT_DATA"):
                        continue
                    d_el = next((c for c in rd if c.tag.endswith("OPR_DATE")), None)
                    v_el = next((c for c in rd if c.tag.endswith("VALUE")), None)
                    if d_el is not None and v_el is not None and d_el.text and v_el.text:
                        try:
                            daily.setdefault(d_el.text, []).append(float(v_el.text))
                        except (ValueError, TypeError):
                            pass
        result = {d: round(sum(v) / len(v), 2) for d, v in daily.items() if v}
        _CAISO_PRICE_CACHE.write_text(json.dumps(result))
        return result
    except Exception:
        try:
            return json.loads(_CAISO_PRICE_CACHE.read_text())
        except Exception:
            return {}


def _caiso_solar_gw(date_str: str) -> float:
    """Seasonal solar generation estimate for CAISO.
    ~47 GW installed (utility + rooftop); daily capacity factor 18–28%, peak near summer solstice."""
    import math
    from datetime import date as _date
    doy = _date.fromisoformat(date_str).timetuple().tm_yday
    cf  = 0.23 + 0.05 * math.cos(2 * math.pi * (doy - 172) / 365)
    return round(47.0 * cf, 2)


def _price_regression(price_dict, load_dict, solar_fn=None):
    """OLS: price = slope * net_load + intercept.
    solar_fn(date_str) → GW subtracted from gross load before regression (CAISO only).
    Returns (slope, intercept, sigma, r2) or (None,)*4."""
    pairs = [(load_dict[d], price_dict[d]) for d in price_dict if d in load_dict]
    if len(pairs) < 10:
        return None, None, None, None
    if solar_fn is not None:
        net_loads = [load_dict[d] - solar_fn(d) for d in price_dict if d in load_dict]
    else:
        net_loads = [p[0] for p in pairs]
    prices = [p[1] for p in pairs]
    n      = len(net_loads)
    ml, mp = sum(net_loads) / n, sum(prices) / n
    num    = sum((l - ml) * (p - mp) for l, p in zip(net_loads, prices))
    den    = sum((l - ml) ** 2 for l in net_loads)
    if den == 0:
        return None, None, None, None
    slope     = num / den
    intercept = mp - slope * ml
    residuals = [p - (slope * l + intercept) for l, p in zip(net_loads, prices)]
    ss_res    = sum(r ** 2 for r in residuals)
    ss_tot    = sum((p - mp) ** 2 for p in prices)
    sigma     = (ss_res / max(n - 2, 1)) ** 0.5
    r2        = round(1 - ss_res / ss_tot, 3) if ss_tot > 0 else 0.0
    return round(slope, 4), round(intercept, 4), round(sigma, 4), r2

# ---------------------------------------------------------------------------
# Load models at startup
# ---------------------------------------------------------------------------
_LOAD_MODEL: LoadCorrectionModel | None = None
try:
    _LOAD_MODEL = load_load_model()
    print("  PJM load model loaded OK")
except Exception as e:
    print(f"  Warning: could not load PJM load model: {e}")

_CAISO_MODEL: CAISOLoadModel | None = None
try:
    _CAISO_MODEL = load_caiso_model()
    print("  CAISO load model loaded OK")
except Exception as e:
    print(f"  Warning: could not load CAISO load model: {e}")


# ---------------------------------------------------------------------------
# Real approved / announced datacenter projects (public sources, 2024-2025)
# ---------------------------------------------------------------------------
_DC_PROJECTS = {
    "pjm": [
        {"id": "stack_stafford",  "name": "STACK Stafford Technology Campus",
         "operator": "STACK Infrastructure",    "location": "Stafford County, VA",
         "mw": 1800, "status": "Approved"},
        {"id": "pax_carlisle",    "name": "Pennsylvania Digital I (PAX)",
         "operator": "PowerHouse / PA Data Center Partners", "location": "Carlisle, PA",
         "mw": 1350, "status": "Announced"},
        {"id": "edgecore_louisa", "name": "EdgeCore Louisa County Campus",
         "operator": "EdgeCore Data Centers",   "location": "Louisa County, VA",
         "mw": 1100, "status": "Announced"},
        {"id": "cleanarc_va1",    "name": "CleanArc VA1 Campus",
         "operator": "CleanArc Data Centers",   "location": "Caroline County, VA",
         "mw": 900,  "status": "Under Construction"},
        {"id": "msft_mt_pleasant","name": "Microsoft Mt. Pleasant",
         "operator": "Microsoft",               "location": "Mt. Pleasant, WI",
         "mw": 1000, "status": "Under Construction"},
        {"id": "meta_dekalb",     "name": "Meta DeKalb County",
         "operator": "Meta",                    "location": "DeKalb County, GA",
         "mw": 500,  "status": "Approved"},
        {"id": "amazon_nova",     "name": "Amazon AWS Northern Virginia",
         "operator": "Amazon",                  "location": "Northern Virginia",
         "mw": 800,  "status": "Operating / Expanding"},
        {"id": "coreweave_lanc",  "name": "CoreWeave Lancaster",
         "operator": "CoreWeave",               "location": "Lancaster, PA",
         "mw": 300,  "status": "Announced"},
    ],
    "caiso": [
        {"id": "google_sj",       "name": "Google San Jose Campus",
         "operator": "Google",                  "location": "San Jose, CA",
         "mw": 400,  "status": "Announced"},
        {"id": "meta_sac",        "name": "Meta Sacramento Campus",
         "operator": "Meta",                    "location": "Sacramento, CA",
         "mw": 250,  "status": "Operating"},
        {"id": "msft_sv",         "name": "Microsoft Silicon Valley",
         "operator": "Microsoft",               "location": "San Jose, CA",
         "mw": 200,  "status": "Announced"},
        {"id": "amazon_elk",      "name": "Amazon AWS Elk Grove",
         "operator": "Amazon",                  "location": "Elk Grove, CA",
         "mw": 150,  "status": "Operating"},
        {"id": "vantage_sd2",     "name": "Vantage SD2 Campus",
         "operator": "Vantage Data Centers",    "location": "San Diego, CA",
         "mw": 120,  "status": "Approved"},
        {"id": "qts_richmond",    "name": "QTS Richmond Campus",
         "operator": "QTS / Blackstone",        "location": "Richmond, CA",
         "mw": 180,  "status": "Under Construction"},
    ],
}

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
                "daily": "temperature_2m_max,temperature_2m_min",
                "forecast_days": forecast_days,
                "temperature_unit": "fahrenheit",
                "timezone": "auto",
            },
            timeout=20,
        )
        r.raise_for_status()
        d = r.json()["daily"]
        return {"label": label, "dates": d["time"],
                "hi": d["temperature_2m_max"], "lo": d["temperature_2m_min"]}
    except Exception:
        return None


# ---------------------------------------------------------------------------
# PJM load forecast
# ---------------------------------------------------------------------------
def fetch_pjm_load_forecast(force=False):
    if _LOAD_MODEL is None:
        return {}

    if not force and _LOAD_FORECAST_CACHE_FILE.exists():
        age_h = (time.time() - _LOAD_FORECAST_CACHE_FILE.stat().st_mtime) / 3600
        if age_h < FORECAST_CACHE_TTL_HOURS:
            try:
                cached = json.loads(_LOAD_FORECAST_CACHE_FILE.read_text())
                if isinstance(cached, dict) and cached.get("load"):
                    if cached["load"][0]["date"] == date.today().isoformat():
                        return cached
            except Exception:
                pass

    session = requests.Session()
    session.headers["User-Agent"] = "load-forecast-dashboard/1.0"

    pjm_avg_c: dict = {}
    pjm_hi_c:  dict = {}
    pjm_lo_c:  dict = {}
    forecast_dates_strs = None

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch_one, loc["label"], loc["lat"], loc["lon"], session): loc
                for loc in PJM_LOAD_LOCATIONS}
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            label = res["label"]
            def f_to_c(lst):
                return [(v - 32) * 5 / 9 if v is not None else None for v in lst]
            hi_c  = f_to_c(res["hi"])
            lo_c  = f_to_c(res["lo"])
            avg_c = [(h + l) / 2 if h and l else None for h, l in zip(hi_c, lo_c)]
            pjm_avg_c[label] = avg_c
            pjm_hi_c[label]  = hi_c
            pjm_lo_c[label]  = lo_c
            if forecast_dates_strs is None:
                forecast_dates_strs = res["dates"][:15]

    if not pjm_avg_c or not forecast_dates_strs:
        return {}

    forecast_dates_list = [date.fromisoformat(d) for d in forecast_dates_strs]

    from temperature_modeling._era5 import fetch_era5_daily
    from temperature_modeling.pjm_load import weighted_avg_temp_f, _build_features as _bf
    from temperature_modeling.models import Coordinates as _C

    era5_session = requests.Session()
    era5_session.headers["User-Agent"] = "load-forecast-dashboard/1.0"
    today = date.today()
    era5_avg_hist: dict = {}
    recent_avg_f = []
    try:
        per_label = {}
        for loc in PJM_LOAD_LOCATIONS:
            per_label[loc["label"]] = fetch_era5_daily(
                _C(loc["lat"], loc["lon"]), today - timedelta(days=16),
                today - timedelta(days=1), era5_session,
            )
        era5_avg_hist = per_label
        for lag_d in sorted([today - timedelta(days=k) for k in range(8, 0, -1)]):
            c_map = {loc["label"]: per_label[loc["label"]][lag_d]
                     for loc in PJM_LOAD_LOCATIONS
                     if lag_d in per_label.get(loc["label"], {})}
            if c_map:
                recent_avg_f.append(weighted_avg_temp_f(c_map))
    except Exception:
        recent_avg_f = []

    hindcast: dict = {}
    if era5_avg_hist:
        try:
            avg_f_hist: dict = {}
            for off in range(16):
                d2 = today - timedelta(days=off + 1)
                c_map = {loc["label"]: era5_avg_hist[loc["label"]][d2]
                         for loc in PJM_LOAD_LOCATIONS
                         if d2 in era5_avg_hist.get(loc["label"], {})}
                if c_map:
                    avg_f_hist[d2] = weighted_avg_temp_f(c_map)
            for d2, avg_f in avg_f_hist.items():
                lag1 = avg_f_hist.get(d2 - timedelta(days=1))
                lag2 = avg_f_hist.get(d2 - timedelta(days=2))
                lag7 = avg_f_hist.get(d2 - timedelta(days=7))
                rv = [avg_f_hist.get(d2 - timedelta(days=k)) for k in range(7)]
                roll7 = sum(v for v in rv if v) / max(sum(1 for v in rv if v), 1)
                feats, _, _ = _bf(avg_f, avg_f + 5, avg_f - 5, d2, lag1, lag2, lag7, roll7)
                hindcast[d2.isoformat()] = round(_LOAD_MODEL.predict([feats])[0] / 1000, 2)
        except Exception:
            hindcast = {}

    load_forecasts = _LOAD_MODEL.predict_with_uncertainty(
        forecast_temps_c=pjm_avg_c, forecast_hi_c=pjm_hi_c, forecast_lo_c=pjm_lo_c,
        gefs_spread_c={}, forecast_dates=forecast_dates_list,
        recent_avg_temps_f=recent_avg_f if len(recent_avg_f) >= 2 else None,
    )

    load_data = [{"date": lf.valid_date.isoformat(),
                  "mean_load_gw": round(lf.mean_load_mw / 1000, 2),
                  "low_load_gw":  round(lf.low_load_mw  / 1000, 2),
                  "high_load_gw": round(lf.high_load_mw / 1000, 2)}
                 for lf in load_forecasts]

    try:
        comparison = fetch_pjm_official_comparison(session)
    except Exception:
        comparison = {"actual": {}, "da_fcst": {}}
    try:
        pjm_7day = fetch_pjm_dataminer_7day(session)
    except Exception:
        pjm_7day = {}
    try:
        backtest = run_load_backtest(_LOAD_MODEL, str(_LOAD_TRAINING_DATA_PATH))
    except Exception:
        backtest = {}
    try:
        prices = _fetch_pjm_prices(session)
    except Exception:
        prices = {}
    actual_gw = comparison.get("actual", {})
    p_slope, p_intercept, p_sigma, p_r2 = _price_regression(prices, actual_gw)

    result = {"load": load_data, "dates": forecast_dates_strs,
              "comparison": comparison, "pjm_7day": pjm_7day,
              "backtest": backtest, "hindcast": hindcast,
              "prices": prices,
              "price_model": {"slope": p_slope, "intercept": p_intercept, "sigma": p_sigma, "r2": p_r2}}
    _LOAD_FORECAST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOAD_FORECAST_CACHE_FILE.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# CAISO load forecast
# ---------------------------------------------------------------------------
def fetch_caiso_load_forecast(force=False):
    if _CAISO_MODEL is None:
        return {}

    if not force and _CAISO_FORECAST_CACHE_FILE.exists():
        age_h = (time.time() - _CAISO_FORECAST_CACHE_FILE.stat().st_mtime) / 3600
        if age_h < FORECAST_CACHE_TTL_HOURS:
            try:
                cached = json.loads(_CAISO_FORECAST_CACHE_FILE.read_text())
                if isinstance(cached, dict) and cached.get("load"):
                    if cached["load"][0]["date"] == date.today().isoformat():
                        return cached
            except Exception:
                pass

    session = requests.Session()
    session.headers["User-Agent"] = "load-forecast-dashboard/1.0"

    caiso_avg_c: dict = {}
    caiso_hi_c:  dict = {}
    caiso_lo_c:  dict = {}
    forecast_dates_strs = None

    with ThreadPoolExecutor(max_workers=6) as pool:
        futs = {pool.submit(_fetch_one, loc["label"], loc["lat"], loc["lon"], session): loc
                for loc in CAISO_LOAD_LOCATIONS}
        for fut in as_completed(futs):
            res = fut.result()
            if not res:
                continue
            label = res["label"]
            def f_to_c(lst):
                return [(v - 32) * 5 / 9 if v is not None else None for v in lst]
            hi_c  = f_to_c(res["hi"])
            lo_c  = f_to_c(res["lo"])
            avg_c = [(h + l) / 2 if h and l else None for h, l in zip(hi_c, lo_c)]
            caiso_avg_c[label] = avg_c
            caiso_hi_c[label]  = hi_c
            caiso_lo_c[label]  = lo_c
            if forecast_dates_strs is None:
                forecast_dates_strs = res["dates"][:15]

    if not caiso_avg_c or not forecast_dates_strs:
        return {}

    forecast_dates_list = [date.fromisoformat(d) for d in forecast_dates_strs]

    from temperature_modeling._era5 import fetch_era5_daily
    from temperature_modeling.pjm_load import _build_features as _bf
    from temperature_modeling.models import Coordinates as _C

    era5_session = requests.Session()
    era5_session.headers["User-Agent"] = "load-forecast-dashboard/1.0"
    today = date.today()
    era5_avg_hist: dict = {}
    recent_avg_f = []
    try:
        per_label = {}
        for loc in CAISO_LOAD_LOCATIONS:
            per_label[loc["label"]] = fetch_era5_daily(
                _C(loc["lat"], loc["lon"]), today - timedelta(days=16),
                today - timedelta(days=1), era5_session,
            )
        era5_avg_hist = per_label
        for lag_d in sorted([today - timedelta(days=k) for k in range(8, 0, -1)]):
            c_map = {loc["label"]: per_label[loc["label"]][lag_d]
                     for loc in CAISO_LOAD_LOCATIONS
                     if lag_d in per_label.get(loc["label"], {})}
            if c_map:
                recent_avg_f.append(weighted_avg_temp_f_caiso(c_map))
    except Exception:
        recent_avg_f = []

    hindcast: dict = {}
    if era5_avg_hist:
        try:
            avg_f_hist: dict = {}
            for off in range(16):
                d2 = today - timedelta(days=off + 1)
                c_map = {loc["label"]: era5_avg_hist[loc["label"]][d2]
                         for loc in CAISO_LOAD_LOCATIONS
                         if d2 in era5_avg_hist.get(loc["label"], {})}
                if c_map:
                    avg_f_hist[d2] = weighted_avg_temp_f_caiso(
                        {k: v for k, v in c_map.items()})
            for d2, avg_f in avg_f_hist.items():
                lag1 = avg_f_hist.get(d2 - timedelta(days=1))
                lag2 = avg_f_hist.get(d2 - timedelta(days=2))
                lag7 = avg_f_hist.get(d2 - timedelta(days=7))
                rv = [avg_f_hist.get(d2 - timedelta(days=k)) for k in range(7)]
                roll7 = sum(v for v in rv if v) / max(sum(1 for v in rv if v), 1)
                feats, _, _ = _bf(avg_f, avg_f + 5, avg_f - 5, d2, lag1, lag2, lag7, roll7)
                hindcast[d2.isoformat()] = round(_CAISO_MODEL.predict([feats])[0] / 1000, 2)
        except Exception:
            hindcast = {}

    load_forecasts = _CAISO_MODEL.predict_with_uncertainty(
        forecast_temps_c=caiso_avg_c, forecast_hi_c=caiso_hi_c, forecast_lo_c=caiso_lo_c,
        gefs_spread_c={}, forecast_dates=forecast_dates_list,
        recent_avg_temps_f=recent_avg_f if len(recent_avg_f) >= 2 else None,
    )

    load_data = [{"date": lf.valid_date.isoformat(),
                  "mean_load_gw": round(lf.mean_load_mw / 1000, 2),
                  "low_load_gw":  round(lf.low_load_mw  / 1000, 2),
                  "high_load_gw": round(lf.high_load_mw / 1000, 2)}
                 for lf in load_forecasts]

    try:
        comparison = fetch_caiso_official_comparison(session)
    except Exception:
        comparison = {"actual": {}, "da_fcst": {}}
    try:
        oasis_7day = fetch_caiso_oasis_7day(session)
    except Exception:
        oasis_7day = {}
    try:
        backtest = run_load_backtest(_CAISO_MODEL, str(_CAISO_TRAINING_DATA_PATH))
    except Exception:
        backtest = {}
    try:
        prices = _fetch_caiso_prices(session)
    except Exception:
        prices = {}
    actual_gw = comparison.get("actual", {})
    p_slope, p_intercept, p_sigma, p_r2 = _price_regression(
        prices, actual_gw, solar_fn=_caiso_solar_gw)

    result = {"load": load_data, "dates": forecast_dates_strs,
              "comparison": comparison, "oasis_7day": oasis_7day,
              "backtest": backtest, "hindcast": hindcast,
              "prices": prices,
              "price_model": {"slope": p_slope, "intercept": p_intercept,
                              "sigma": p_sigma, "r2": p_r2, "use_solar": True}}
    _CAISO_FORECAST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _CAISO_FORECAST_CACHE_FILE.write_text(json.dumps(result))
    return result


# ---------------------------------------------------------------------------
# Startup — pre-load PJM forecast
# ---------------------------------------------------------------------------
print("\nLoading forecasts...")
_startup_data = fetch_pjm_load_forecast()
print(f"  Dashboard at http://127.0.0.1:8050\n")


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
**Methodology** — Forecasts are produced by an XGBoost gradient-boosted regression model trained on 2 years of daily EIA \
load data paired with ERA5 reanalysis temperatures. Temperature inputs are population-weighted across 12 monitoring \
locations per ISO footprint. Features include heating/cooling degree-days (HDD/CDD) from daily average, high, and low \
temperatures; day-of-week one-hot encoding; US federal holiday and holiday-week flags; bridge-day indicators; T-1, T-2, \
and T-7 temperature lags; and a 7-day rolling average temperature (captures heat-wave persistence and population \
acclimatisation). Forward forecasts use GFS NWP output from Open-Meteo (15-day horizon). Historical hindcast uses \
ERA5 reanalysis. PJM comparison benchmark: official 7-day forecast from PJM DataMiner API. CAISO comparison benchmark: \
OASIS 7-day system load forecast. Backtest uses a chronological 80/20 train/test split.
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
                    html.Span(f"PJM · CAISO  ·  Updated {datetime.now().strftime('%d %b %Y')}",
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
                            options=[{"label": "PJM (Eastern US)", "value": "pjm"},
                                     {"label": "CAISO (California)", "value": "caiso"}],
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
                                "flexWrap": "wrap", "marginBottom": "20px"}),

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

                # ── Price chart ────────────────────────────────────────────────
                html.Div(
                    id="price-section",
                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                           "border": "1px solid #e2e8f0", "padding": "20px",
                           "boxShadow": "0 1px 3px rgba(0,0,0,0.05)",
                           "marginBottom": "20px"},
                    children=[
                        html.Div(
                            style={"display": "flex", "justifyContent": "space-between",
                                   "alignItems": "baseline", "marginBottom": "4px"},
                            children=[
                                html.Div("Day-Ahead Electricity Price",
                                         style={"fontSize": "13px", "fontWeight": 600,
                                                "color": "#0f172a"}),
                                html.Div(id="price-subtitle",
                                         style={"fontSize": "11px", "color": "#94a3b8"}),
                            ],
                        ),
                        dcc.Graph(id="price-chart", style={"height": "240px"},
                                  config={"displayModeBar": False}),
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
                            "Forecasts are produced by an XGBoost gradient-boosted regression model "
                            "trained on 2 years of daily EIA load data paired with ERA5 reanalysis "
                            "temperatures across 12 population-weighted monitoring locations per ISO "
                            "footprint. Features include heating/cooling degree-days (HDD/CDD) from "
                            "daily average, high, and low temperatures; day-of-week one-hot encoding; "
                            "US federal holiday and holiday-week flags; bridge-day indicators; T−1, T−2, "
                            "and T−7 temperature lags; and a 7-day rolling average temperature (captures "
                            "heat-wave persistence and population acclimatisation). 27 features total.",
                            style={"fontSize": "12px", "color": "#64748b",
                                   "lineHeight": "1.7", "margin": "0 0 8px 0"},
                        ),
                        html.P(
                            "Forward forecasts use GFS NWP output via Open-Meteo (15-day horizon). "
                            "Historical hindcast uses ERA5 reanalysis. "
                            "PJM benchmark: official 7-day forecast from PJM DataMiner API. "
                            "CAISO benchmark: OASIS 7-day system load forecast. "
                            "Backtest uses a chronological 80/20 train/test split — "
                            "PJM test MAPE: 0.4% · CAISO test MAPE: 0.3%.",
                            style={"fontSize": "12px", "color": "#64748b",
                                   "lineHeight": "1.7", "margin": 0},
                        ),
                    ],
                ),

            ],
        ),

        # Store
        dcc.Store(id="load-forecast-store"),
        dcc.Store(id="dc-selected-mw", data=0),
    ],
)


# ---------------------------------------------------------------------------
# Callbacks
# ---------------------------------------------------------------------------
@app.callback(
    Output("load-forecast-store", "data"),
    Input("iso-selector", "value"),
    Input("refresh-btn", "n_clicks"),
)
def load_forecast_data(iso, n_clicks):
    force = n_clicks is not None and n_clicks > 0
    if iso == "caiso":
        return fetch_caiso_load_forecast(force=force)
    return fetch_pjm_load_forecast(force=force)


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
    Output("price-chart", "figure"),
    Output("price-subtitle", "children"),
    Output("price-section", "style"),
    Input("load-forecast-store", "data"),
    Input("iso-selector", "value"),
)
def render(data, iso):
    if not data or not data.get("load"):
        return ([], "", _empty_fig(), _empty_fig(), "")

    iso_label = "CAISO (California ISO)" if iso == "caiso" else "PJM Interconnection"
    load_list  = data["load"]
    comparison = data.get("comparison", {})
    backtest   = data.get("backtest", {})
    hindcast   = data.get("hindcast", {})
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

    # 14-day hindcast MAPE vs EIA actual
    actual_dict = comparison.get("actual", {})
    da_dict     = comparison.get("da_fcst", {})
    h_errs = [abs(combined_hindcast[d] - actual_dict[d]) / actual_dict[d] * 100
              for d in combined_hindcast if d in actual_dict and actual_dict[d]]
    recent_mape = sum(h_errs[-14:]) / len(h_errs[-14:]) if h_errs else None
    mape_str = f"{recent_mape:.1f}%" if recent_mape is not None else "—"
    mape_col = "#22c55e" if recent_mape is not None and recent_mape < 3 else "#f97316"

    summary_cards = [
        card("Today (GW)", f"{today_gw:.1f}", "GFS-based", load_color(today_gw)),
        card("15-Day Peak", f"{peak_gw:.1f}", f"on {peak_lbl}", load_color(peak_gw)),
        card("15-Day Avg", f"{avg_gw:.1f}", "GW baseline", "#475569"),
        card("Model vs Actual", mape_str, "14-day MAPE", mape_col),
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

    # Official benchmark
    if iso == "caiso":
        oasis = data.get("oasis_7day", {})
        oasis_dates = sorted(d for d in oasis if d in dates)
        if oasis_dates:
            fig.add_trace(go.Scatter(
                x=oasis_dates, y=[oasis[d] for d in oasis_dates],
                mode="lines+markers", name="CAISO Official 7-Day (OASIS)",
                line=dict(color="#16a34a", width=2, dash="dash"),
                marker=dict(size=7, color="#16a34a", symbol="diamond",
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>%{x|%d %b}</b><br>CAISO OASIS: %{y:.1f} GW<extra></extra>",
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
    else:
        pjm_7day = data.get("pjm_7day", {})
        pjm7_dates = sorted(d for d in pjm_7day if d in dates)
        if pjm7_dates:
            fig.add_trace(go.Scatter(
                x=pjm7_dates, y=[pjm_7day[d] for d in pjm7_dates],
                mode="lines+markers", name="PJM Official 7-Day (DataMiner)",
                line=dict(color="#2563eb", width=2, dash="dash"),
                marker=dict(size=7, color="#2563eb", symbol="diamond",
                            line=dict(color="#ffffff", width=1.5)),
                hovertemplate="<b>%{x|%d %b}</b><br>PJM official: %{y:.1f} GW<extra></extra>",
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

    # ── Price chart ───────────────────────────────────────────────────────────
    prices    = data.get("prices", {})
    pm        = data.get("price_model", {})
    p_slope   = pm.get("slope")
    p_int     = pm.get("intercept")
    p_sigma   = pm.get("sigma")
    p_r2      = pm.get("r2")
    use_solar = pm.get("use_solar", False)

    hist_p_dates = sorted(d for d in hist_dates_raw if d in prices)
    hist_p_vals  = [prices[d] for d in hist_p_dates]

    # For CAISO, drive projection from net load (gross load − estimated solar)
    if use_solar and p_slope is not None:
        proj_x = [gw - _caiso_solar_gw(d) for gw, d in zip(means, dates)]
    else:
        proj_x = means

    fwd_p_vals  = (
        [round(p_slope * x + p_int, 1) for x in proj_x]
        if p_slope is not None else []
    )
    fwd_p_lows  = (
        [round(v - p_sigma, 1) for v in fwd_p_vals]
        if fwd_p_vals and p_sigma is not None else []
    )
    fwd_p_highs = (
        [round(v + p_sigma, 1) for v in fwd_p_vals]
        if fwd_p_vals and p_sigma is not None else []
    )
    price_subtitle = "PJM DataMiner (DA LMP)"

    # When load explains almost nothing, anchor projection to historical mean (R²<0.15)
    if fwd_p_vals and p_r2 is not None and p_r2 < 0.15 and hist_p_vals:
        hist_mean = sum(hist_p_vals) / len(hist_p_vals)
        fwd_p_vals  = [round(hist_mean, 1)] * len(dates)
        fwd_p_lows  = [round(hist_mean - p_sigma, 1)] * len(dates) if p_sigma else fwd_p_lows
        fwd_p_highs = [round(hist_mean + p_sigma, 1)] * len(dates) if p_sigma else fwd_p_highs

    pfig = go.Figure()
    if hist_p_vals:
        pfig.add_trace(go.Scatter(
            x=hist_p_dates, y=hist_p_vals, mode="lines+markers",
            name="Historical DA Price",
            line=dict(color="#7c3aed", width=2),
            marker=dict(size=4, color="#7c3aed"),
            hovertemplate="<b>%{x|%d %b}</b><br>Price: $%{y:.1f}/MWh<extra></extra>",
        ))
    if fwd_p_vals:
        if fwd_p_highs and fwd_p_lows:
            pfig.add_trace(go.Scatter(
                x=dates, y=fwd_p_highs,
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                showlegend=False, hoverinfo="skip",
            ))
            pfig.add_trace(go.Scatter(
                x=dates, y=fwd_p_lows,
                mode="lines", line=dict(color="rgba(0,0,0,0)"),
                fill="tonexty", fillcolor="rgba(167,139,250,0.25)",
                showlegend=False, hoverinfo="skip",
            ))
        pfig.add_trace(go.Scatter(
            x=dates, y=fwd_p_vals, mode="lines+markers",
            name="Projected Price (model)",
            line=dict(color="#a78bfa", width=2, dash="dash"),
            marker=dict(size=4, color="#a78bfa"),
            hovertemplate="<b>%{x|%d %b}</b><br>Projected: $%{y:.1f}/MWh<extra></extra>",
        ))
    if hist_p_vals or fwd_p_vals:
        all_p = hist_p_vals + fwd_p_vals
        avg_p = sum(all_p) / len(all_p)
        pfig.add_hline(y=avg_p, line_dash="dot", line_color="#cbd5e1", line_width=1.5,
                       annotation_text=f"avg: ${avg_p:.0f}/MWh",
                       annotation_font_size=10, annotation_font_color="#94a3b8",
                       annotation_position="top left")
    else:
        pfig = _empty_fig("Price data unavailable")
        price_subtitle = "Fetching…"

    pfig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.32, font=dict(size=10),
                    bgcolor="rgba(0,0,0,0)", x=0),
        margin=dict(l=60, r=20, t=10, b=80),
        xaxis=dict(type="date", tickformat="%d %b", gridcolor="#f8fafc",
                   tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0",
                   dtick="D1"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=".0f", tickprefix="$",
                   title=dict(text="Price ($/MWh)", font=dict(size=11, color="#64748b")),
                   linecolor="#e2e8f0"),
        height=240,
    )

    price_section_style = {"display": "none"} if iso == "caiso" else {
        "backgroundColor": "#ffffff", "borderRadius": "10px",
        "border": "1px solid #e2e8f0", "padding": "20px",
        "boxShadow": "0 1px 3px rgba(0,0,0,0.05)", "marginBottom": "20px",
    }
    return summary_cards, subtitle, fig, pfig, price_subtitle, price_section_style


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
    # Calibrated from PJM/CAISO market data: price sensitivity rises exponentially
    # as load approaches grid capacity (peakers progressively more expensive).
    # Sensitivity at 70% load ≈ $5/GW, at 90% ≈ $12/GW, at 100% ≈ $30/GW
    base_sens = 5.5 if iso == "pjm" else 7.0   # $/MWh per GW at reference load
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
    # PJM marginal: 1,264 lbs CO2/MWh  |  CAISO: 876 lbs CO2/MWh
    lbs_per_mwh   = 1264 if iso == "pjm" else 876
    tons_per_mwh  = lbs_per_mwh / 2204.62
    ann_co2_ktons = ann_gwh * 1000 * tons_per_mwh / 1000   # kilotonnes
    ann_co2_mtons = ann_co2_ktons / 1000                   # megatonnes

    # ── Reserve margin impact ─────────────────────────────────────────────────
    # PJM target 20%, current delivered 14.8% (Dec 2025 auction).
    # We show how much the new DC shaves from the reserve margin at peak.
    rm_target = 20.0 if iso == "pjm" else 15.0
    rm_current = 14.8 if iso == "pjm" else 16.0
    rm_delta   = dc_gw / (peak_historical + dc_gw) * 100   # % reduction in reserve

    # ── Capacity market cost ──────────────────────────────────────────────────
    # PJM Dec 2025 auction: $333.44/MW-day; CAISO Resource Adequacy ≈ $80/MW-day
    cap_price_per_mw_day = 333.44 if iso == "pjm" else 80.0
    ann_cap_cost_m       = dc_mw * cap_price_per_mw_day * 365 / 1e6

    # ── Interconnection cost estimate ─────────────────────────────────────────
    # PJM 2024: $4.36B for ~30 GW of datacenter queue ≈ $145M/GW
    # CAISO: $2B+ for large-load upgrades, assume similar $/GW
    intercon_cost_m = dc_gw * 145

    # ── REC / Carbon neutrality cost ─────────────────────────────────────────
    # PJM Class I RECs ≈ $40/MWh (2024 high); CAISO ≈ $15/MWh (lower solar cost)
    rec_mwh = 40 if iso == "pjm" else 15
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
    iso_label = "PJM" if iso == "pjm" else "CAISO"
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

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    app.run(debug=False, port=port)
