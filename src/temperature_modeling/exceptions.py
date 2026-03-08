class TemperatureModelingError(Exception):
    """Base exception for the temperature-modeling library."""


class GeocodingError(TemperatureModelingError):
    """Location string could not be resolved to coordinates."""


class LocationNotInPJMError(TemperatureModelingError):
    """Resolved location is outside PJM ISO territory."""

    def __init__(self, state: str) -> None:
        super().__init__(
            f"State '{state}' is not within PJM ISO territory. "
            "PJM covers DE, IL, IN, KY, MD, MI, NJ, NC, OH, PA, TN, VA, WV, and DC."
        )
        self.state = state


class NWSAPIError(TemperatureModelingError):
    """NWS API returned an unexpected or error response."""


class NCEIAPIError(TemperatureModelingError):
    """NCEI CDO API returned an unexpected or error response."""


class NCEIAuthError(NCEIAPIError):
    """NCEI token is missing or invalid (HTTP 401)."""

    def __init__(self) -> None:
        super().__init__(
            "NCEI API token is missing or invalid. "
            "Register for a free token at https://www.ncei.noaa.gov/cdo-web/token"
        )


class NoStationFoundError(NCEIAPIError):
    """No NCEI weather station found near the given coordinates."""
