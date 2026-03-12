from datetime import date, datetime, timezone

import requests

from .exceptions import SatelliteAPIError
from .models import Coordinates, SatelliteObservation

_POWER_URL = "https://power.larc.nasa.gov/api/temporal/daily/point"


def get_surface_temperatures(
    coords: Coordinates,
    start_date: date,
    end_date: date,
    session: requests.Session,
) -> list:
    """
    Fetch daily Earth Skin Temperature (TS) from NASA POWER MERRA-2.

    Parameters
    ----------
    coords:
        Latitude/longitude of the location.
    start_date, end_date:
        Inclusive date range (datetime.date objects).
    session:
        requests.Session with appropriate headers.

    Returns
    -------
    list[SatelliteObservation]
        Daily surface skin temperature readings. Missing days are skipped.

    Raises
    ------
    SatelliteAPIError
        On non-200 response or unexpected JSON shape.
    """
    params = {
        "parameters": "TS",
        "community": "RE",
        "latitude": coords.lat,
        "longitude": coords.lon,
        "start": start_date.strftime("%Y%m%d"),
        "end": end_date.strftime("%Y%m%d"),
        "format": "JSON",
    }

    try:
        response = session.get(_POWER_URL, params=params)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"NASA POWER request failed: {exc}") from exc

    if response.status_code != 200:
        raise SatelliteAPIError(
            f"NASA POWER returned HTTP {response.status_code}: {response.text[:200]}"
        )

    try:
        ts_data = response.json()["properties"]["parameter"]["TS"]
    except (KeyError, ValueError) as exc:
        raise SatelliteAPIError(
            f"Unexpected NASA POWER response shape: {exc}"
        ) from exc

    observations = []
    for date_str, temp_c in ts_data.items():
        if temp_c is None or temp_c < -990:
            continue
        # date_str format: "YYYYMMDD"
        ts = datetime.strptime(date_str, "%Y%m%d").replace(tzinfo=timezone.utc)
        observations.append(
            SatelliteObservation(
                timestamp=ts,
                surface_temp_c=round(temp_c, 2),
                surface_temp_f=round(temp_c * 9 / 5 + 32, 2),
                source="NASA-POWER",
            )
        )

    return sorted(observations, key=lambda o: o.timestamp)
