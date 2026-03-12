__version__ = "0.1.0"

from .client import WeatherClient
from .exceptions import (
    GeocodingError,
    LocationNotInPJMError,
    NCEIAPIError,
    NCEIAuthError,
    NoStationFoundError,
    NWSAPIError,
    SatelliteAPIError,
    TemperatureModelingError,
)
from .models import (
    Coordinates,
    ForecastPeriod,
    ForecastSample,
    HourlyForecast,
    LeadTimeSkill,
    Location,
    SatelliteObservation,
    TemperatureObservation,
    WeatherResult,
)
from .verification import collect_verification_records, score_by_lead
from .features import (
    ClimateNormals,
    FeatureVector,
    build_climate_normals,
    extract_features,
)

__all__ = [
    "WeatherClient",
    "WeatherResult",
    "Location",
    "Coordinates",
    "TemperatureObservation",
    "ForecastPeriod",
    "HourlyForecast",
    "SatelliteObservation",
    "ForecastSample",
    "LeadTimeSkill",
    "collect_verification_records",
    "score_by_lead",
    "ClimateNormals",
    "FeatureVector",
    "build_climate_normals",
    "extract_features",
    "TemperatureModelingError",
    "GeocodingError",
    "LocationNotInPJMError",
    "NWSAPIError",
    "NCEIAPIError",
    "NCEIAuthError",
    "NoStationFoundError",
    "SatelliteAPIError",
]
