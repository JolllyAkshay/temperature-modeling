from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional


@dataclass
class Coordinates:
    lat: float
    lon: float


@dataclass
class Location:
    coordinates: Coordinates
    state: str        # two-letter abbreviation, e.g. "PA"
    display_name: str  # human-readable, e.g. "Philadelphia, PA"


@dataclass
class TemperatureObservation:
    """A single temperature reading (historical or current)."""
    timestamp: datetime
    temp_f: Optional[float]
    temp_c: Optional[float]
    source: str  # "NWS" or "NCEI"


@dataclass
class ForecastPeriod:
    """One period from the NWS 7-day forecast."""
    name: str           # e.g. "Tonight", "Monday"
    start_time: datetime
    end_time: datetime
    is_daytime: bool
    temp_f: int
    short_forecast: str  # e.g. "Mostly Cloudy"


@dataclass
class HourlyForecast:
    timestamp: datetime
    temp_f: int
    short_forecast: str


@dataclass
class SatelliteObservation:
    """A surface skin temperature reading from Open-Meteo or NASA POWER."""
    timestamp: datetime
    surface_temp_c: float
    surface_temp_f: float
    source: str  # "Open-Meteo" or "NASA-POWER"


@dataclass
class WeatherResult:
    """Top-level container returned to the caller."""
    location: Location
    historical: list = field(default_factory=list)        # list[TemperatureObservation]
    forecast_periods: list = field(default_factory=list)  # list[ForecastPeriod]
    hourly_forecast: list = field(default_factory=list)   # list[HourlyForecast]
    satellite: list = field(default_factory=list)         # list[SatelliteObservation]
