import math
from datetime import date, datetime, timedelta, timezone
from typing import Optional

import requests

from .exceptions import NCEIAPIError, NCEIAuthError, NoStationFoundError
from .models import Coordinates, TemperatureObservation

NCEI_BASE = "https://www.ncei.noaa.gov/cdo-web/api/v2"
STATION_SEARCH_RADIUS_DEG = 0.5  # ~55 km bounding box half-width
_MAX_DAYS_PER_REQUEST = 365       # NCEI silently truncates beyond this


def find_nearest_station(
    coords: Coordinates,
    token: str,
    session: requests.Session,
) -> str:
    """
    Return the GHCND station ID closest to coords.

    Searches within a bounding box of ±STATION_SEARCH_RADIUS_DEG degrees and
    picks the station with smallest Euclidean distance to coords.

    Raises
    ------
    NCEIAuthError        on HTTP 401
    NoStationFoundError  if no stations are found in the search area
    NCEIAPIError         on other HTTP errors
    """
    extent = (
        f"{coords.lat - STATION_SEARCH_RADIUS_DEG},"
        f"{coords.lon - STATION_SEARCH_RADIUS_DEG},"
        f"{coords.lat + STATION_SEARCH_RADIUS_DEG},"
        f"{coords.lon + STATION_SEARCH_RADIUS_DEG}"
    )
    try:
        response = session.get(
            f"{NCEI_BASE}/stations",
            params={
                "extent": extent,
                "datatypeid": "TMAX,TMIN",
                "datasetid": "GHCND",
                "limit": 100,
            },
            headers={"token": token},
        )
    except requests.RequestException as exc:
        raise NCEIAPIError(f"NCEI station search request failed: {exc}") from exc

    _check_ncei_response(response, "station search")

    data = response.json()
    stations = data.get("results", [])
    if not stations:
        raise NoStationFoundError(
            f"No GHCND stations found within {STATION_SEARCH_RADIUS_DEG}° of "
            f"({coords.lat}, {coords.lon}). Try a location with more weather stations."
        )

    # Pick the station closest to the requested coordinates
    def distance(station: dict) -> float:
        return math.hypot(
            station["latitude"] - coords.lat,
            station["longitude"] - coords.lon,
        )

    nearest = min(stations, key=distance)
    return nearest["id"]


def get_historical_temperatures(
    station_id: str,
    start_date: date,
    end_date: date,
    token: str,
    session: requests.Session,
) -> list:
    """
    Fetch TMAX and TMIN records from NCEI GHCND for a station.

    Automatically pages the request if the date range exceeds 365 days,
    since NCEI silently truncates longer requests.

    Parameters
    ----------
    station_id:
        Full GHCND station ID, e.g. "GHCND:USW00014735".
    start_date, end_date:
        Inclusive date range.
    token:
        NCEI CDO API token.
    session:
        requests.Session with appropriate headers.

    Returns
    -------
    list[TemperatureObservation]
        NCEI values are in tenths of °C; converted to °C and °F here.

    Raises
    ------
    NCEIAuthError, NCEIAPIError
    """
    observations = []
    chunk_start = start_date

    while chunk_start <= end_date:
        chunk_end = min(
            chunk_start + timedelta(days=_MAX_DAYS_PER_REQUEST - 1),
            end_date,
        )
        observations.extend(
            _fetch_chunk(station_id, chunk_start, chunk_end, token, session)
        )
        chunk_start = chunk_end + timedelta(days=1)

    return observations


def _fetch_chunk(
    station_id: str,
    start_date: date,
    end_date: date,
    token: str,
    session: requests.Session,
) -> list:
    """Fetch a single ≤365-day chunk of GHCND data."""
    try:
        response = session.get(
            f"{NCEI_BASE}/data",
            params={
                "datasetid": "GHCND",
                "stationid": station_id,
                "startdate": start_date.isoformat(),
                "enddate": end_date.isoformat(),
                "datatypeid": "TMAX,TMIN",
                "units": "metric",
                "limit": 1000,
            },
            headers={"token": token},
        )
    except requests.RequestException as exc:
        raise NCEIAPIError(f"NCEI data request failed: {exc}") from exc

    _check_ncei_response(response, "data fetch")

    records = response.json().get("results", [])
    observations = []
    for rec in records:
        # NCEI returns value in tenths of °C when units=metric is NOT set,
        # but with units=metric it returns °C directly. We use metric above.
        temp_c = rec["value"]
        temp_f = _c_to_f(temp_c)
        ts = datetime.fromisoformat(rec["date"]).replace(tzinfo=timezone.utc)
        observations.append(
            TemperatureObservation(
                timestamp=ts,
                temp_f=temp_f,
                temp_c=temp_c,
                source="NCEI",
            )
        )
    return observations


def _check_ncei_response(response: requests.Response, context: str) -> None:
    if response.status_code == 401:
        raise NCEIAuthError()
    if response.status_code != 200:
        raise NCEIAPIError(
            f"NCEI {context} returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )


def _c_to_f(temp_c: float) -> float:
    return round(temp_c * 9 / 5 + 32, 2)
