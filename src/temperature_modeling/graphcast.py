from datetime import date, datetime, timedelta, timezone

import requests

from .exceptions import SatelliteAPIError
from .models import Coordinates, SatelliteObservation

# GraphCast (gfs_graphcast025) is served through two Open-Meteo endpoints:
#   - historical-forecast-api for archived model runs (available from 2024-02-05)
#   - forecast API for recent/future dates
_HISTORICAL_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_FORECAST_URL = "https://api.open-meteo.com/v1/forecast"
_MODEL = "gfs_graphcast025"

# Archive has a ~2-day lag before runs are committed to the historical store.
_ARCHIVE_LAG_DAYS = 2

# GraphCast archive starts on 2024-02-05.
_ARCHIVE_START = date(2024, 2, 5)


def get_surface_temperatures(
    coords: Coordinates,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> list:
    """
    Fetch hourly 2-metre air temperature from Google's GraphCast model via
    Open-Meteo (model id: ``gfs_graphcast025``).

    GraphCast does not output a land surface skin temperature field; the
    closest available surface variable is ``temperature_2m``.  Readings are
    stored in ``SatelliteObservation.surface_temp_c / surface_temp_f`` with
    ``source="GraphCast"`` so callers can distinguish them from ERA5 or
    MERRA-2 observations.

    Parameters
    ----------
    coords:
        Latitude/longitude of the location.
    start_date, end_date:
        Inclusive date range.  Dates before 2024-02-05 are not available and
        will return no observations.
    session:
        requests.Session with appropriate headers.

    Returns
    -------
    list[SatelliteObservation]
        Hourly 2-metre air temperature readings sorted by timestamp.

    Raises
    ------
    SatelliteAPIError
        On non-200 response or unexpected JSON shape.
    """
    cutoff = date.today() - timedelta(days=_ARCHIVE_LAG_DAYS)

    # Clamp start to the GraphCast archive's earliest available date.
    effective_start = max(start_date, _ARCHIVE_START)
    if effective_start > end_date:
        return []

    observations = []

    if effective_start <= cutoff:
        archive_end = min(end_date, cutoff)
        observations.extend(
            _fetch_chunk(coords, effective_start, archive_end, _HISTORICAL_URL, session)
        )

    if end_date > cutoff:
        forecast_start = max(effective_start, cutoff + timedelta(days=1))
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
    """Fetch one date chunk from a single Open-Meteo GraphCast endpoint."""
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": "temperature_2m",
        "models": _MODEL,
        "timezone": "UTC",
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
    }

    try:
        response = session.get(url, params=params)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"GraphCast (Open-Meteo) request failed: {exc}") from exc

    if response.status_code != 200:
        raise SatelliteAPIError(
            f"GraphCast (Open-Meteo) returned HTTP {response.status_code}: "
            f"{response.text[:200]}"
        )

    try:
        data = response.json()
        times = data["hourly"]["time"]
        temps_c = data["hourly"]["temperature_2m"]
    except (KeyError, ValueError) as exc:
        raise SatelliteAPIError(
            f"Unexpected GraphCast (Open-Meteo) response shape: {exc}"
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
                source="GraphCast",
            )
        )
    return observations
