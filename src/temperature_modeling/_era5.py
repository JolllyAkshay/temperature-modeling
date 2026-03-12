"""
Internal helpers for fetching ERA5 reanalysis data from Open-Meteo's archive
API and reducing hourly output to daily means.

Shared by verification.py and features.py.  Not part of the public API.
"""

from collections import defaultdict
from datetime import date, datetime
from typing import Dict, List

import requests

from .exceptions import SatelliteAPIError
from .models import Coordinates

ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"


def fetch_era5_daily(
    coords: Coordinates,
    start: date,
    end: date,
    session: requests.Session,
    variable: str = "temperature_2m",
) -> Dict[date, float]:
    """
    Fetch an ERA5 hourly variable from the Open-Meteo archive and return
    daily means keyed by date.

    Parameters
    ----------
    coords:
        Target latitude/longitude.
    start, end:
        Inclusive date range.
    session:
        Shared requests.Session.
    variable:
        Open-Meteo hourly variable name (default ``"temperature_2m"``).

    Returns
    -------
    dict[date, float]
        Daily mean values.  Days with no non-null hours are omitted.

    Raises
    ------
    SatelliteAPIError
        On network error, non-200 HTTP status, or unexpected JSON shape.
    """
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": variable,
        "timezone": "UTC",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    data = get_json(ERA5_ARCHIVE_URL, params, session, "ERA5")
    return hourly_to_daily_mean(data, variable)


def get_json(
    url: str,
    params: dict,
    session: requests.Session,
    label: str,
) -> dict:
    """GET *url* with *params*, parse JSON, raise SatelliteAPIError on failure."""
    try:
        resp = session.get(url, params=params)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"{label} request failed: {exc}") from exc
    if resp.status_code != 200:
        raise SatelliteAPIError(
            f"{label} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise SatelliteAPIError(f"{label} JSON parse error: {exc}") from exc


def hourly_to_daily_mean(data: dict, variable: str) -> Dict[date, float]:
    """
    Average hourly Open-Meteo values into daily means.

    Parameters
    ----------
    data:
        Parsed JSON response containing ``data["hourly"]["time"]`` and
        ``data["hourly"][variable]``.
    variable:
        The hourly variable key to aggregate.

    Returns
    -------
    dict[date, float]
        Daily mean values.  Hours with ``None`` are skipped; days where all
        hours are ``None`` are omitted entirely.
    """
    try:
        times = data["hourly"]["time"]
        values = data["hourly"][variable]
    except KeyError as exc:
        raise SatelliteAPIError(
            f"Unexpected Open-Meteo response shape (missing {exc})"
        ) from exc

    daily: Dict[date, List[float]] = defaultdict(list)
    for ts_str, val in zip(times, values):
        if val is None:
            continue
        daily[datetime.fromisoformat(ts_str).date()].append(val)

    return {d: sum(vals) / len(vals) for d, vals in daily.items()}
