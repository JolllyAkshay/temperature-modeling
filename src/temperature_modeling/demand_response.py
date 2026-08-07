"""
Demand-response opportunity windows.

Identifies the best 4-hour blocks for shifting flexible load (EV charging,
industrial demand, HVAC pre-conditioning) over the next 48 hours based on:
  - Estimated hourly carbon intensity (lbs CO2/MWh)
  - Estimated hourly grid cost (relative price index)

Both are derived from:
  - Hourly GHI + wind from Open-Meteo (same source as net_load.py)
  - Daily load forecast scaled with a typical ISO hourly profile
  - Current carbon intensity baseline from EIA (via carbon_intensity module)
"""

import logging
from datetime import datetime, date, timedelta, timezone
from typing import Dict, List, Optional, Tuple

import requests

log = logging.getLogger(__name__)

_OPEN_METEO_URL = "https://api.open-meteo.com/v1/forecast"

# ISO renewable capacity (same calibration as net_load.py)
_ISO_CFG: dict = {
    "pjm":   {"lat": 39.5, "lon": -80.0, "solar_gw": 12.0,  "wind_gw": 10.0,
               "solar_cf": 0.75, "wind_cf": 0.35, "tz": "America/New_York"},
    "caiso": {"lat": 36.5, "lon": -119.5, "solar_gw": 23.0, "wind_gw": 7.5,
               "solar_cf": 0.85, "wind_cf": 0.45, "tz": "America/Los_Angeles"},
    "ercot": {"lat": 31.0, "lon": -99.0,  "solar_gw": 22.0, "wind_gw": 38.0,
               "solar_cf": 0.80, "wind_cf": 0.42, "tz": "America/Chicago"},
    "miso":  {"lat": 43.0, "lon": -90.0,  "solar_gw": 8.0,  "wind_gw": 30.0,
               "solar_cf": 0.72, "wind_cf": 0.40, "tz": "America/Chicago"},
    "nyiso": {"lat": 42.5, "lon": -76.0,  "solar_gw": 6.0,  "wind_gw": 5.5,
               "solar_cf": 0.72, "wind_cf": 0.35, "tz": "America/New_York"},
    "isone": {"lat": 42.4, "lon": -72.0,  "solar_gw": 5.0,  "wind_gw": 4.5,
               "solar_cf": 0.70, "wind_cf": 0.38, "tz": "America/New_York"},
    "spp":   {"lat": 37.5, "lon": -97.0,  "solar_gw": 9.0,  "wind_gw": 32.0,
               "solar_cf": 0.78, "wind_cf": 0.45, "tz": "America/Chicago"},
}

# Typical weekday hourly load profile (normalized; 1.0 = daily mean)
# Shape: valley 3-4 AM, morning ramp, afternoon peak 4-7 PM, evening decline
_WEEKDAY_PROFILE = [
    0.83, 0.79, 0.76, 0.74, 0.75, 0.80,   # 00-05
    0.88, 0.97, 1.05, 1.09, 1.11, 1.12,   # 06-11
    1.11, 1.10, 1.11, 1.13, 1.16, 1.18,   # 12-17
    1.16, 1.12, 1.07, 1.00, 0.93, 0.87,   # 18-23
]
_WEEKEND_PROFILE = [
    0.80, 0.77, 0.74, 0.73, 0.74, 0.77,   # 00-05
    0.82, 0.88, 0.95, 1.01, 1.06, 1.09,   # 06-11
    1.10, 1.10, 1.10, 1.10, 1.11, 1.12,   # 12-17
    1.10, 1.07, 1.03, 0.98, 0.91, 0.85,   # 18-23
]

_MAX_GHI = 900.0   # W/m² reference peak
_MAX_WIND = 12.0   # m/s at 100m reference peak


def compute_dr_windows(
    iso: str,
    daily_load_gw: Dict[str, float],   # {date_str: gw} next 2 days
    current_ci: dict,                  # result from fetch_carbon_intensity()
    session: Optional[requests.Session] = None,
) -> dict:
    """
    Compute hourly load / carbon / cost estimates for the next 48 h and
    identify the best 4-hour blocks for shifting flexible demand.

    Returns
    -------
    {
      "hours": [{"ts": "2026-07-28T14:00", "hour": 14, "date": "2026-07-28",
                 "load_gw": float, "solar_gw": float, "wind_gw": float,
                 "net_load_gw": float, "ci_score": float,   # 0=clean, 1=dirty
                 "cost_score": float}, ...],   # 0=cheap, 1=expensive
      "low_carbon_window": {"start": 13, "end": 16, "label": "1–4 PM",
                            "solar_peak_gw": 8.5, "carbon_reduction_pct": 28},
      "low_cost_window":   {"start": 2,  "end": 5,  "label": "2–5 AM",
                            "cost_reduction_pct": 35},
      "best_window":       {"start": 13, "end": 16, "label": "1–4 PM",
                            "reason": "Solar peak + mild load"},
    }
    Empty dict on failure.
    """
    cfg = _ISO_CFG.get(iso)
    if not cfg:
        return {}

    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "grid-dr/1.0"

    today     = date.today()
    end_date  = today + timedelta(days=1)

    # ── Fetch hourly GHI + wind ──────────────────────────────────────────────
    params = {
        "latitude":        cfg["lat"],
        "longitude":       cfg["lon"],
        "hourly":          ["direct_radiation", "diffuse_radiation", "wind_speed_100m"],
        "start_date":      today.isoformat(),
        "end_date":        end_date.isoformat(),
        "timezone":        cfg["tz"],
        "wind_speed_unit": "ms",
    }
    try:
        r = session.get(_OPEN_METEO_URL, params=params, timeout=20)
        r.raise_for_status()
        hourly = r.json().get("hourly", {})
    except Exception:
        log.warning("DR: Open-Meteo fetch failed for %s", iso.upper())
        return {}

    h_times = hourly.get("time", [])
    h_dir   = hourly.get("direct_radiation",  [None] * len(h_times))
    h_dif   = hourly.get("diffuse_radiation", [None] * len(h_times))
    h_wind  = hourly.get("wind_speed_100m",   [None] * len(h_times))

    if not h_times:
        return {}

    # ── Build per-hour estimates ─────────────────────────────────────────────
    # Base carbon intensity (lbs/MWh) from latest EIA data
    base_ci_lbs = current_ci.get("lbs_co2_per_mwh", 700.0)
    base_clean   = current_ci.get("clean_pct", 30.0) / 100.0

    hours: List[dict] = []
    for i, ts in enumerate(h_times):
        d_str  = ts[:10]
        hr     = int(ts[11:13])
        d_obj  = date.fromisoformat(d_str)
        is_wkd = d_obj.weekday() >= 5

        daily_gw = daily_load_gw.get(d_str, 0)
        if not daily_gw:
            continue

        profile  = _WEEKEND_PROFILE if is_wkd else _WEEKDAY_PROFILE
        load_gw  = daily_gw * profile[hr]

        ghi  = (h_dir[i] or 0) + (h_dif[i] or 0)
        wind = h_wind[i] or 0

        solar_cf  = min(ghi  / _MAX_GHI,  1.0) * cfg["solar_cf"]
        wind_cf   = min(wind / _MAX_WIND, 1.0) * cfg["wind_cf"]
        solar_gw  = solar_cf  * cfg["solar_gw"]
        wind_gw   = wind_cf   * cfg["wind_gw"]

        net_load_gw = max(load_gw - solar_gw - wind_gw, load_gw * 0.3)

        # Carbon score: lower renewable fraction at this hour → higher CI
        renewable_gw  = solar_gw + wind_gw
        renew_fraction = min(renewable_gw / max(load_gw, 1), 0.95)
        # Scale from base clean fraction — more solar/wind → lower CI
        ci_factor     = (1.0 - renew_fraction) / max(1.0 - base_clean, 0.1)
        est_ci_lbs    = base_ci_lbs * ci_factor

        # Scores: 0 = best, 1 = worst
        hours.append({
            "ts":           ts,
            "hour":         hr,
            "date":         d_str,
            "load_gw":      round(load_gw, 2),
            "solar_gw":     round(solar_gw, 2),
            "wind_gw":      round(wind_gw, 2),
            "net_load_gw":  round(net_load_gw, 2),
            "est_ci_lbs":   round(est_ci_lbs, 0),
        })

    if len(hours) < 8:
        return {}

    # Normalise scores to [0, 1] across all hours
    all_ci   = [h["est_ci_lbs"] for h in hours]
    all_load = [h["net_load_gw"] for h in hours]
    ci_min, ci_max     = min(all_ci),   max(all_ci)
    ld_min, ld_max     = min(all_load), max(all_load)

    for h in hours:
        h["ci_score"]   = (h["est_ci_lbs"]   - ci_min) / max(ci_max - ci_min, 1)
        h["cost_score"] = (h["net_load_gw"] - ld_min) / max(ld_max - ld_min, 1)

    # ── Find best 4-hour blocks ──────────────────────────────────────────────
    def _best_window(score_key: str) -> dict:
        best_score = float("inf")
        best_start = 0
        for i in range(len(hours) - 3):
            window_score = sum(hours[i + j][score_key] for j in range(4)) / 4
            if window_score < best_score:
                best_score = window_score
                best_start = i
        i = best_start
        start_h = hours[i]["hour"]
        end_h   = hours[i + 3]["hour"]
        return {
            "start_idx": i,
            "start_h":   start_h,
            "end_h":     end_h,
            "date":      hours[i]["date"],
            "label":     _hour_label(start_h, hours[i + 3]["hour"]),
            "avg_score": best_score,
        }

    low_carbon = _best_window("ci_score")
    low_cost   = _best_window("cost_score")

    # Compute reduction percentages
    i_c = low_carbon["start_idx"]
    avg_ci_window   = sum(h["est_ci_lbs"]   for h in hours[i_c:i_c+4]) / 4
    overall_avg_ci  = sum(all_ci) / len(all_ci)
    ci_reduction    = round((1 - avg_ci_window / overall_avg_ci) * 100) if overall_avg_ci else 0

    i_l = low_cost["start_idx"]
    avg_ld_window   = sum(h["net_load_gw"] for h in hours[i_l:i_l+4]) / 4
    overall_avg_ld  = sum(all_load) / len(all_load)
    cost_reduction  = round((1 - avg_ld_window / overall_avg_ld) * 100) if overall_avg_ld else 0

    # Best overall = combined score
    def _best_combined() -> dict:
        best_score = float("inf")
        best_i = 0
        for i in range(len(hours) - 3):
            s = sum(0.6 * hours[i+j]["ci_score"] + 0.4 * hours[i+j]["cost_score"]
                    for j in range(4)) / 4
            if s < best_score:
                best_score, best_i = s, i
        i = best_i
        # Pick a human reason
        solar_peak = max(hours[i+j]["solar_gw"] for j in range(4))
        wind_peak  = max(hours[i+j]["wind_gw"]  for j in range(4))
        if solar_peak > 3:
            reason = f"Solar peak ({solar_peak:.0f} GW) lowers carbon intensity"
        elif wind_peak > 5:
            reason = f"High wind output ({wind_peak:.0f} GW) reduces grid emissions"
        else:
            reason = "Off-peak demand keeps prices and emissions low"
        return {
            "start_h": hours[i]["hour"],
            "end_h":   hours[i+3]["hour"],
            "date":    hours[i]["date"],
            "label":   _hour_label(hours[i]["hour"], hours[i+3]["hour"]),
            "reason":  reason,
        }

    best = _best_combined()

    low_carbon["carbon_reduction_pct"] = max(ci_reduction, 0)
    low_cost["cost_reduction_pct"]     = max(cost_reduction, 0)
    low_carbon.pop("start_idx", None)
    low_cost.pop("start_idx", None)

    return {
        "hours":            hours,
        "low_carbon_window": low_carbon,
        "low_cost_window":   low_cost,
        "best_window":       best,
    }


def _hour_label(start: int, end: int) -> str:
    def _fmt(h):
        if h == 0:   return "12 AM"
        if h < 12:   return f"{h} AM"
        if h == 12:  return "12 PM"
        return f"{h - 12} PM"
    return f"{_fmt(start)} – {_fmt(end + 1)}"
