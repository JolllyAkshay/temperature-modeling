"""
Net load = gross load - (solar + wind) generation.

Critical for CAISO (duck curve) and ERCOT (high wind penetration).
Solar and wind are estimated from Open-Meteo using:
  - direct_radiation + diffuse_radiation  → solar PV proxy (GHI)
  - wind_speed_100m                       → wind power proxy

Actual generation scales are calibrated per ISO from historical capacity factors.
This gives a relative shape that matches real net-load patterns even if the
absolute MW numbers are approximate.

Public API
----------
fetch_net_load_forecast(iso, forecast_dates, session) -> list[dict]
    Returns list of {date, gross_gw, solar_gw, wind_gw, net_gw} for each date.
"""

import logging
from datetime import date, timedelta

import requests

from ._era5 import ERA5_ARCHIVE_URL, get_json

log = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# Representative lat/lon and installed capacity (GW) per ISO
# Sources: EIA, CAISO/ERCOT grid data (approximate 2024 figures)
_ISO_RENEWABLES: dict = {
    "caiso": {
        "lat": 36.5, "lon": -119.5,
        "solar_capacity_gw": 23.0,   # ~23 GW utility solar
        "wind_capacity_gw":  7.5,    # ~7.5 GW wind
        "solar_cf_peak":     0.85,   # fraction of capacity at peak GHI
        "wind_cf_peak":      0.45,
    },
    "ercot": {
        "lat": 31.0, "lon": -99.0,
        "solar_capacity_gw": 22.0,
        "wind_capacity_gw":  38.0,   # ~38 GW wind (largest US wind fleet)
        "solar_cf_peak":     0.80,
        "wind_cf_peak":      0.42,
    },
    "pjm": {
        "lat": 39.5, "lon": -80.0,
        "solar_capacity_gw": 12.0,
        "wind_capacity_gw":  10.0,
        "solar_cf_peak":     0.75,
        "wind_cf_peak":      0.35,
    },
    "miso": {
        "lat": 43.0, "lon": -90.0,
        "solar_capacity_gw": 8.0,
        "wind_capacity_gw":  30.0,
        "solar_cf_peak":     0.72,
        "wind_cf_peak":      0.40,
    },
}

# Max GHI (W/m²) used to normalise to capacity factor
_MAX_GHI_W_M2 = 900.0
# Max wind speed (m/s at 100m) mapped to full capacity factor
_MAX_WIND_MS  = 12.0


def _daily_renewable_gw(cfg: dict, ghi_vals: list, wind_vals: list) -> tuple[float, float]:
    """Shared solar/wind capacity-factor calibration — same formula for forecast and historical."""
    # Solar: daily average of GHI relative to peak → capacity factor → GW
    avg_ghi  = sum(ghi_vals) / len(ghi_vals) if ghi_vals else 0
    solar_cf = min(avg_ghi / _MAX_GHI_W_M2, 1.0) * cfg["solar_cf_peak"]
    solar_gw = round(solar_cf * cfg["solar_capacity_gw"], 2)

    # Wind: daily P75 wind speed (renewable output tracks higher winds)
    wv       = sorted(wind_vals) if wind_vals else [0]
    p75_wind = wv[int(len(wv) * 0.75)]
    wind_cf  = min(p75_wind / _MAX_WIND_MS, 1.0) * cfg["wind_cf_peak"]
    wind_gw  = round(wind_cf * cfg["wind_capacity_gw"], 2)
    return solar_gw, wind_gw


def fetch_historical_renewable_generation(iso: str, start: str, end: str) -> dict[str, float]:
    """
    Return {date_str: renewable_gw} (solar + wind) for a historical date
    range, using the same capacity-factor calibration as the forecast
    version but sourced from ERA5 archive data (get_json already caches to
    disk and retries on rate limits — no separate caching needed here).

    Used to give the price model a net-load (load - renewables) feature
    instead of raw load for ISOs where renewable penetration materially
    drives price (CAISO's duck curve, ERCOT's wind fleet) — raw load
    alone can't explain a price collapse on a sunny/windy day at the same
    load level as a still/cloudy one.
    """
    cfg = _ISO_RENEWABLES.get(iso)
    if not cfg:
        return {}

    session = requests.Session()
    session.headers["User-Agent"] = "net-load-history/1.0"
    params = {
        "latitude":   cfg["lat"],
        "longitude":  cfg["lon"],
        "hourly":     ["direct_radiation", "diffuse_radiation", "wind_speed_100m"],
        "start_date": start,
        "end_date":   end,
        "timezone":   "America/Chicago",
        "wind_speed_unit": "ms",
    }
    try:
        data = get_json(ERA5_ARCHIVE_URL, params, session, "renewable-history")
    except Exception:
        log.warning("%s: historical renewable generation fetch failed", iso.upper())
        return {}

    hourly = data.get("hourly", {})
    times  = hourly.get("time", [])
    h_dir  = hourly.get("direct_radiation", [])
    h_dif  = hourly.get("diffuse_radiation", [])
    h_wind = hourly.get("wind_speed_100m", [])

    by_date: dict = {}
    for i, t in enumerate(times):
        d = t[:10]
        by_date.setdefault(d, {"ghi": [], "wind": []})
        by_date[d]["ghi"].append((h_dir[i] or 0) + (h_dif[i] or 0))
        by_date[d]["wind"].append(h_wind[i] or 0)

    result: dict[str, float] = {}
    for d, vals in by_date.items():
        solar_gw, wind_gw = _daily_renewable_gw(cfg, vals["ghi"], vals["wind"])
        result[d] = round(solar_gw + wind_gw, 2)
    return result


def fetch_net_load_forecast(
    iso: str,
    forecast_dates: list,
    session: requests.Session | None = None,
) -> list:
    """
    Fetch solar + wind generation proxies for the forecast window and compute
    daily net load by subtracting from the gross load forecast.

    Parameters
    ----------
    iso            : ISO code
    forecast_dates : list of date objects (15-day window)
    session        : optional requests.Session

    Returns
    -------
    list of dicts: {date, solar_gw, wind_gw, renewable_gw}
    Empty list on failure.
    """
    cfg = _ISO_RENEWABLES.get(iso)
    if not cfg:
        return []

    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "net-load-forecast/1.0"

    start = forecast_dates[0].isoformat()
    end   = forecast_dates[-1].isoformat()

    params = {
        "latitude":            cfg["lat"],
        "longitude":           cfg["lon"],
        "daily":               ["shortwave_radiation_sum", "wind_speed_10m_max"],
        "hourly":              ["direct_radiation", "diffuse_radiation", "wind_speed_100m"],
        "start_date":          start,
        "end_date":            end,
        "timezone":            "America/Chicago",
        "wind_speed_unit":     "ms",
    }

    try:
        r = session.get(_OPEN_METEO_URL, params=params, timeout=20)
        r.raise_for_status()
        data = r.json()
    except Exception:
        log.warning("Open-Meteo renewable fetch failed for %s", iso.upper())
        return []

    daily   = data.get("daily", {})
    hourly  = data.get("hourly", {})

    dates_d = daily.get("time", [])
    sw_sum  = daily.get("shortwave_radiation_sum", [])  # MJ/m²

    # Hourly → aggregate to daily peak capacity factors
    h_times = hourly.get("time", [])
    h_dir   = hourly.get("direct_radiation", [])
    h_dif   = hourly.get("diffuse_radiation", [])
    h_wind  = hourly.get("wind_speed_100m", [])

    # Group hourly values by date
    hourly_by_date: dict = {}
    for i, t in enumerate(h_times):
        d = t[:10]
        hourly_by_date.setdefault(d, {"ghi": [], "wind": []})
        ghi  = (h_dir[i] or 0) + (h_dif[i] or 0)
        wind = h_wind[i] or 0
        hourly_by_date[d]["ghi"].append(ghi)
        hourly_by_date[d]["wind"].append(wind)

    results = []
    for i, d in enumerate(dates_d):
        hd = hourly_by_date.get(d, {})
        solar_gw, wind_gw = _daily_renewable_gw(cfg, hd.get("ghi", [0]), hd.get("wind", [0]))
        results.append({
            "date":          d,
            "solar_gw":      solar_gw,
            "wind_gw":       wind_gw,
            "renewable_gw":  round(solar_gw + wind_gw, 2),
        })

    log.info("%s: net load computed — avg solar %.1f GW, avg wind %.1f GW",
             iso.upper(),
             sum(r["solar_gw"] for r in results) / len(results) if results else 0,
             sum(r["wind_gw"]  for r in results) / len(results) if results else 0)
    return results
