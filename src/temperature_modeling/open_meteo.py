from datetime import date, datetime, timedelta, timezone

import requests

from .exceptions import SatelliteAPIError
from .models import Coordinates, SatelliteObservation

_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"

# Open-Meteo archive has a ~5-day lag; use forecast endpoint within that window.
_ARCHIVE_LAG_DAYS = 5


def get_surface_temperatures(
    coords: Coordinates,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> list:
    """
    Fetch hourly surface skin temperature from Open-Meteo.

    Uses the archive endpoint (ERA5) for historical data and the forecast
    endpoint for dates within the last 5 days or in the future.

    Parameters
    ----------
    coords:
        Latitude/longitude of the location.
    start_date, end_date:
        Inclusive date range.
    session:
        requests.Session with appropriate headers.

    Returns
    -------
    list[SatelliteObservation]
        Hourly surface skin temperature readings (skin_temperature).

    Raises
    ------
    SatelliteAPIError
        On non-200 response or unexpected JSON shape.
    """
    cutoff = date.today() - timedelta(days=_ARCHIVE_LAG_DAYS)

    observations = []

    # Split range: archive for older dates, forecast for recent/future
    if start_date <= cutoff:
        archive_end = min(end_date, cutoff)
        observations.extend(
            _fetch_chunk(coords, start_date, archive_end, _ARCHIVE_URL, session)
        )

    if end_date > cutoff:
        forecast_start = max(start_date, cutoff + timedelta(days=1))
        observations.extend(
            _fetch_chunk(coords, forecast_start, end_date, _FORECAST_URL, session)
        )

    return sorted(observations, key=lambda o: o.timestamp)


def _fetch_chunk(
    coords: Coordinates,
    start_date: date,
    end_date: date,
    url: str,
    session: requests.Session,
) -> list:
    """Fetch one chunk from a single Open-Meteo endpoint."""
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": "skin_temperature",
        "timezone": "UTC",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    try:
        response = session.get(url, params=params)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"Open-Meteo request failed: {exc}") from exc

    if response.status_code != 200:
        raise SatelliteAPIError(
            f"Open-Meteo returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        data = response.json()
        times = data["hourly"]["time"]
        temps_c = data["hourly"]["skin_temperature"]
    except (KeyError, ValueError) as exc:
        raise SatelliteAPIError(
            f"Unexpected Open-Meteo response shape: {exc}"
        ) from exc

    observations = []
    for ts_str, temp_c in zip(times, temps_c):
        if temp_c is None:
            continue
        ts = datetime.fromisoformat(ts_str).replace(tzinfo=timezone.utc)
        observations.append(
            SatelliteObservation(
                timestamp=ts,
                surface_temp_c=round(temp_c, 2),
                surface_temp_f=round(temp_c * 9 / 5 + 32, 2),
                source="Open-Meteo",
            )
        )
    return observations
