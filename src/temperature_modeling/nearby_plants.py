"""
Nearest operating power plants to a location, from EIA-860 (via the EIA v2
electricity/operating-generator-capacity route — confirmed live and
queryable, unlike EIA-861's reliability/SAIDI-SAIFI data, which has no
REST API and is only published as bulk annual Excel files).

Public API
----------
find_nearest_plants(lat: float, lon: float, state: str, n=5, session=None) -> list[dict]
"""

import logging
import math
import os
import time
from typing import Optional

import requests

log = logging.getLogger(__name__)

_EIA_KEY = os.environ.get("EIA_API_KEY", "")
_EIA_GEN_URL = "https://api.eia.gov/v2/electricity/operating-generator-capacity/data/"

_STATE_CACHE: dict = {}  # {state: (timestamp, plants)}
_STATE_CACHE_TTL = 24 * 3600  # plant fleets change slowly — daily cache is plenty


def _haversine_miles(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    r = 3958.8
    p1, p2 = math.radians(lat1), math.radians(lat2)
    dphi = math.radians(lat2 - lat1)
    dlambda = math.radians(lon2 - lon1)
    a = math.sin(dphi / 2) ** 2 + math.cos(p1) * math.cos(p2) * math.sin(dlambda / 2) ** 2
    return 2 * r * math.asin(math.sqrt(a))


def _fetch_state_plants(state: str, session: Optional[requests.Session] = None) -> list:
    """
    Latest-period operating generators for `state`, aggregated to one row
    per plant (nameplate capacity summed across a plant's generators,
    primary fuel taken as the fuel with the most capacity at that plant).
    A single state/period query tops out around 3,000 rows even for
    California/Texas — well inside one 5000-row page, so no pagination.
    """
    state = (state or "").strip().upper()
    if not state or not _EIA_KEY:
        return []

    cached_ts, cached_val = _STATE_CACHE.get(state, (0, None))
    if cached_val is not None and time.time() - cached_ts < _STATE_CACHE_TTL:
        return cached_val

    sess = session or requests.Session()
    try:
        latest_r = sess.get(_EIA_GEN_URL, params={
            "api_key": _EIA_KEY, "frequency": "monthly", "data[0]": "latitude",
            "facets[stateid][]": state, "facets[status][]": "OP",
            "sort[0][column]": "period", "sort[0][direction]": "desc", "length": 1,
        }, timeout=20)
        latest_r.raise_for_status()
        latest_rows = latest_r.json().get("response", {}).get("data", [])
        if not latest_rows:
            return []
        latest_period = latest_rows[0]["period"]

        r = sess.get(_EIA_GEN_URL, params={
            "api_key": _EIA_KEY, "frequency": "monthly",
            "data[0]": "latitude", "data[1]": "longitude", "data[2]": "nameplate-capacity-mw",
            "facets[stateid][]": state, "facets[status][]": "OP",
            "start": latest_period, "end": latest_period, "length": 5000,
        }, timeout=25)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
    except Exception as exc:
        log.warning("EIA-860 plant fetch failed for %s: %s", state, exc)
        return []

    by_plant: dict = {}
    for row in rows:
        lat, lon, cap = row.get("latitude"), row.get("longitude"), row.get("nameplate-capacity-mw")
        if lat is None or lon is None or cap is None:
            continue
        try:
            lat, lon, cap = float(lat), float(lon), float(cap)
        except (TypeError, ValueError):
            continue
        pid = row.get("plantid")
        entry = by_plant.setdefault(pid, {
            "plant_name": row.get("plantName") or "Unknown",
            "owner": row.get("entityName") or "",
            "lat": lat, "lon": lon,
            "capacity_mw": 0.0,
            "fuels": {},
            "balancing_authority": row.get("balancing_authority_code"),
        })
        entry["capacity_mw"] += cap
        fuel = row.get("energy-source-desc") or "Other"
        entry["fuels"][fuel] = entry["fuels"].get(fuel, 0.0) + cap

    plants = []
    for entry in by_plant.values():
        primary_fuel = max(entry["fuels"], key=entry["fuels"].get) if entry["fuels"] else "Unknown"
        plants.append({
            "plant_name": entry["plant_name"], "owner": entry["owner"],
            "lat": entry["lat"], "lon": entry["lon"],
            "capacity_mw": round(entry["capacity_mw"], 1),
            "primary_fuel": primary_fuel,
            "balancing_authority": entry["balancing_authority"],
        })

    _STATE_CACHE[state] = (time.time(), plants)
    return plants


def find_nearest_plants(lat: float, lon: float, state: str, n: int = 5,
                         session: Optional[requests.Session] = None) -> list:
    """
    Return the `n` operating power plants closest to (lat, lon), searching
    within `state` — a zip's own state, which occasionally misses a plant
    just over a state line, but keeps the query correct and cheap for the
    overwhelming majority of zips rather than fetching every neighboring
    state on every lookup.

    Each result: {plant_name, owner, distance_miles, capacity_mw,
    primary_fuel, balancing_authority}. Empty list on failure, missing
    EIA key, or no plants found for the state.
    """
    plants = _fetch_state_plants(state, session=session)
    if not plants:
        return []
    for p in plants:
        p["distance_miles"] = round(_haversine_miles(lat, lon, p["lat"], p["lon"]), 1)
    plants.sort(key=lambda p: p["distance_miles"])
    return [{k: v for k, v in p.items() if k not in ("lat", "lon")} for p in plants[:n]]
