"""
NASA POWER API client for satellite-informed surface state variables.

Provides three MERRA-2 / MODIS-derived fields per day:
  GWETROOT  -- Root Zone Soil Wetness (fraction 0-1), assimilates SMAP
  SNODP     -- Snow Depth (m), assimilates satellite snow products
  SNODP     -- Snow Depth (m), assimilates satellite snow products

API documentation: https://power.larc.nasa.gov/docs/services/api/
No authentication required.  Responses are cached to disk via the shared
cache layer in _era5.py.
"""

from datetime import date, datetime
from typing import Dict, Optional

import requests

from ._era5 import get_json
from .models import Coordinates

NASA_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"

# NASA POWER missing-data sentinel value
_FILL_VALUE = -999.0


def fetch_nasa_power_daily(
    coords: Coordinates,
    start: date,
    end: date,
    session: requests.Session,
) -> Dict[date, Dict[str, Optional[float]]]:
    """
    Fetch GWETROOT and SNODP from the NASA POWER daily point API for a date
    range and return a mapping of ``{date: {"smap_soil_wetness": float|None,
    "modis_snow_m": float|None}}``.

    Batch the full date range into a single HTTP request; responses are cached
    to disk so repeat calls are free.

    Parameters
    ----------
    coords:
        Target latitude/longitude.
    start, end:
        Inclusive date range (must not exceed ~20 years per POWER API limits).
    session:
        Shared requests.Session.

    Returns
    -------
    dict[date, dict]
        Daily values.  Entries are ``None`` when the API returns the fill
        value (-999) or no data for that date.
    """
    params = {
        "parameters": "GWETROOT,SNODP",
        "community": "AG",
        "longitude": coords.lon,
        "latitude": coords.lat,
        "start": start.strftime("%Y%m%d"),
        "end": end.strftime("%Y%m%d"),
        "format": "JSON",
    }
    data = get_json(NASA_POWER_URL, params, session, "NASA POWER", _backoff=10.0)

    param_block = data.get("properties", {}).get("parameter", {})
    gwetroot: dict = param_block.get("GWETROOT", {})
    snodp: dict    = param_block.get("SNODP", {})
    ndvi: dict     = {}

    result: Dict[date, Dict[str, Optional[float]]] = {}
    for date_str in set(gwetroot) | set(snodp):
        try:
            d = datetime.strptime(date_str, "%Y%m%d").date()
        except ValueError:
            continue
        w = gwetroot.get(date_str)
        s = snodp.get(date_str)
        result[d] = {
            "smap_soil_wetness": (
                None if (w is None or float(w) < _FILL_VALUE + 1) else round(float(w), 4)
            ),
            "modis_snow_m": (
                None if (s is None or float(s) < _FILL_VALUE + 1) else round(float(s), 3)
            ),
        }
    return result
