from datetime import datetime, timezone

import requests

from .exceptions import NWSAPIError
from .models import Coordinates, ForecastPeriod, HourlyForecast
from .pjm import STATE_NAME_TO_ABBR, validate_pjm_state

NWS_BASE = "https://api.weather.gov"


def get_gridpoint_metadata(coords: Coordinates, session: requests.Session) -> dict:
    """
    Call /points/{lat},{lon} and return the raw 'properties' dict.

    Also validates that the resolved state is within PJM territory.

    Raises
    ------
    NWSAPIError
        On non-200 response or unexpected JSON shape.
    LocationNotInPJMError
        If the NWS-resolved state is outside PJM territory.
    """
    url = f"{NWS_BASE}/points/{coords.lat:.4f},{coords.lon:.4f}"
    try:
        response = session.get(url)
    except requests.RequestException as exc:
        raise NWSAPIError(f"NWS request failed: {exc}") from exc

    if response.status_code != 200:
        detail = _extract_nws_detail(response)
        raise NWSAPIError(
            f"NWS /points returned HTTP {response.status_code}: {detail}"
        )

    try:
        props = response.json()["properties"]
    except (KeyError, ValueError) as exc:
        raise NWSAPIError(f"Unexpected NWS /points response shape: {exc}") from exc

    # NWS returns full state name in relativeLocation
    state_full = (
        props.get("relativeLocation", {})
        .get("properties", {})
        .get("state", "")
    )
    state_abbr = STATE_NAME_TO_ABBR.get(state_full, state_full)
    validate_pjm_state(state_abbr)

    return props


def get_forecast_periods(
    gridpoint_metadata: dict,
    session: requests.Session,
) -> list:
    """
    Fetch the 7-day period forecast from the forecastUrl in metadata.

    Returns
    -------
    list[ForecastPeriod]
    """
    forecast_url = gridpoint_metadata.get("forecast")
    if not forecast_url:
        raise NWSAPIError("No forecast URL in NWS gridpoint metadata.")

    try:
        response = session.get(forecast_url)
    except requests.RequestException as exc:
        raise NWSAPIError(f"NWS forecast request failed: {exc}") from exc

    if response.status_code != 200:
        detail = _extract_nws_detail(response)
        raise NWSAPIError(f"NWS forecast returned HTTP {response.status_code}: {detail}")

    try:
        periods = response.json()["properties"]["periods"]
    except (KeyError, ValueError) as exc:
        raise NWSAPIError(f"Unexpected NWS forecast response shape: {exc}") from exc

    result = []
    for p in periods:
        result.append(
            ForecastPeriod(
                name=p["name"],
                start_time=_parse_iso(p["startTime"]),
                end_time=_parse_iso(p["endTime"]),
                is_daytime=p["isDaytime"],
                temp_f=p["temperature"],
                short_forecast=p["shortForecast"],
            )
        )
    return result


def get_hourly_forecast(
    gridpoint_metadata: dict,
    session: requests.Session,
) -> list:
    """
    Fetch hourly forecast from the forecastHourly URL in metadata.

    Returns
    -------
    list[HourlyForecast]
    """
    hourly_url = gridpoint_metadata.get("forecastHourly")
    if not hourly_url:
        raise NWSAPIError("No forecastHourly URL in NWS gridpoint metadata.")

    try:
        response = session.get(hourly_url)
    except requests.RequestException as exc:
        raise NWSAPIError(f"NWS hourly forecast request failed: {exc}") from exc

    if response.status_code != 200:
        detail = _extract_nws_detail(response)
        raise NWSAPIError(
            f"NWS hourly forecast returned HTTP {response.status_code}: {detail}"
        )

    try:
        periods = response.json()["properties"]["periods"]
    except (KeyError, ValueError) as exc:
        raise NWSAPIError(f"Unexpected NWS hourly response shape: {exc}") from exc

    return [
        HourlyForecast(
            timestamp=_parse_iso(p["startTime"]),
            temp_f=p["temperature"],
            short_forecast=p["shortForecast"],
        )
        for p in periods
    ]


def _parse_iso(ts: str) -> datetime:
    """Parse an ISO 8601 timestamp string to a timezone-aware datetime."""
    return datetime.fromisoformat(ts)


def _extract_nws_detail(response: requests.Response) -> str:
    """Extract the 'detail' field from an NWS error JSON body, or return raw text."""
    try:
        return response.json().get("detail", response.text[:200])
    except ValueError:
        return response.text[:200]
