from datetime import date
from typing import Optional, Tuple, Union

import requests

from .geocoding import geocode_location
from .models import Coordinates, Location, WeatherResult
from .ncei import find_nearest_station, get_historical_temperatures
from .nws import get_forecast_periods, get_gridpoint_metadata, get_hourly_forecast
from .pjm import STATE_NAME_TO_ABBR, validate_pjm_state

_USER_AGENT = (
    "temperature-modeling-library/0.1.0 "
    "(https://github.com/example/temperature-modeling)"
)


class WeatherClient:
    """
    High-level client for pulling PJM-territory temperature data.

    Parameters
    ----------
    ncei_token:
        Free API token from https://www.ncei.noaa.gov/cdo-web/token
        Required only when fetching historical data.
    timeout:
        HTTP timeout in seconds applied to all requests (default 10).

    Examples
    --------
    Forecast only (no token needed):

    >>> client = WeatherClient()
    >>> result = client.get_weather("Columbus, OH", include_historical=False)
    >>> result.location.state
    'OH'

    Historical + forecast:

    >>> client = WeatherClient(ncei_token="YOUR_TOKEN")
    >>> from datetime import date
    >>> result = client.get_weather(
    ...     (39.9526, -75.1652),
    ...     start_date=date(2024, 1, 1),
    ...     end_date=date(2024, 1, 31),
    ... )
    >>> len(result.historical) > 0
    True
    """

    def __init__(
        self,
        ncei_token: Optional[str] = None,
        timeout: int = 10,
    ) -> None:
        self._ncei_token = ncei_token
        self._timeout = timeout
        self._session = requests.Session()
        self._session.headers.update({
            "User-Agent": _USER_AGENT,
            "Accept": "application/json",
        })
        self._session.request = self._wrap_timeout(self._session.request)

    def get_weather(
        self,
        location: Union[str, Tuple[float, float]],
        start_date: Optional[date] = None,
        end_date: Optional[date] = None,
        include_forecast: bool = True,
        include_historical: bool = True,
    ) -> WeatherResult:
        """
        Fetch temperature data for a PJM-territory location.

        Parameters
        ----------
        location:
            Either a "City, ST" string or a (lat, lon) tuple.
        start_date, end_date:
            Date range for historical data. Both must be provided together
            when fetching historical data.
        include_forecast:
            Whether to fetch NWS 7-day + hourly forecast.
        include_historical:
            Whether to fetch NCEI historical data. Requires ncei_token and
            both start_date and end_date.

        Returns
        -------
        WeatherResult

        Raises
        ------
        GeocodingError
            Location string could not be resolved.
        LocationNotInPJMError
            Location is outside PJM ISO territory.
        NWSAPIError
            NWS API request failed.
        NCEIAPIError / NCEIAuthError / NoStationFoundError
            NCEI API request failed.
        ValueError
            include_historical=True but ncei_token, start_date, or
            end_date is missing.
        """
        if include_historical:
            if not self._ncei_token:
                raise ValueError(
                    "ncei_token is required for historical data. "
                    "Register at https://www.ncei.noaa.gov/cdo-web/token"
                )
            if start_date is None or end_date is None:
                raise ValueError(
                    "Both start_date and end_date are required for historical data."
                )

        # Resolve location; for tuple inputs also fetch NWS metadata once so
        # forecast fetches can reuse it without an extra /points call.
        resolved_location, nws_metadata = self._resolve_location(location)

        result = WeatherResult(location=resolved_location)

        if include_forecast:
            if nws_metadata is None:
                nws_metadata = get_gridpoint_metadata(
                    resolved_location.coordinates, self._session
                )
            result.forecast_periods = get_forecast_periods(nws_metadata, self._session)
            result.hourly_forecast = get_hourly_forecast(nws_metadata, self._session)

        if include_historical:
            coords = resolved_location.coordinates
            station_id = find_nearest_station(
                coords, self._ncei_token, self._session  # type: ignore[arg-type]
            )
            result.historical = get_historical_temperatures(
                station_id, start_date, end_date,  # type: ignore[arg-type]
                self._ncei_token, self._session     # type: ignore[arg-type]
            )

        return result

    def _resolve_location(
        self,
        location: Union[str, Tuple[float, float]],
    ) -> Tuple[Location, Optional[dict]]:
        """
        Resolve a string or (lat, lon) tuple to a Location.

        For tuple inputs, also fetches NWS /points metadata (needed for state
        validation) and returns it so the caller can reuse it for forecasts.

        Returns
        -------
        (Location, nws_metadata | None)
            nws_metadata is returned for tuple inputs; None for string inputs.
        """
        if isinstance(location, tuple):
            lat, lon = location
            coords = Coordinates(lat=lat, lon=lon)
            metadata = get_gridpoint_metadata(coords, self._session)
            state_full = (
                metadata.get("relativeLocation", {})
                .get("properties", {})
                .get("state", "")
            )
            state_abbr = STATE_NAME_TO_ABBR.get(state_full, state_full)
            validate_pjm_state(state_abbr)
            loc = Location(
                coordinates=coords,
                state=state_abbr,
                display_name=f"{lat:.4f}, {lon:.4f}",
            )
            return loc, metadata
        else:
            loc = geocode_location(location, self._session)
            return loc, None

    def _wrap_timeout(self, original_request):
        """Wrap session.request to inject a default timeout."""
        timeout = self._timeout

        def request_with_timeout(method, url, **kwargs):
            kwargs.setdefault("timeout", timeout)
            return original_request(method, url, **kwargs)

        return request_with_timeout
