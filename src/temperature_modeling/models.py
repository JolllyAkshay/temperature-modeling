from dataclasses import dataclass, field
from datetime import date, datetime
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
class ForecastSample:
    """One forecast–observation pair from a historical GraphCast run."""
    init_date: date       # date the model was initialized
    valid_date: date      # date being verified
    lead_days: int        # valid_date - init_date
    forecast_temp_c: float   # GraphCast temperature_2m prediction (°C)
    observed_temp_c: float   # ERA5 temperature_2m reanalysis truth (°C)
    error_c: float           # forecast - observed; positive = warm bias
    init_skin_temp_c: Optional[float] = None  # ERA5 skin temp at init date (°C)
    init_z500_m: Optional[float] = None       # ERA5 500hPa geopotential height at init date (m)
    init_t850_c: Optional[float] = None       # ERA5 850hPa temperature at init date (°C)
    init_soil_m3: Optional[float] = None      # ERA5 surface soil moisture at init date (m³/m³)
    init_snow_m: Optional[float] = None       # ERA5 snow depth at init date (m)
    init_smap_soil_wetness: Optional[float] = None  # NASA POWER GWETROOT (0–1, assimilates SMAP)
    init_modis_snow_m: Optional[float] = None       # NASA POWER SNODP (m, assimilates satellite snow)
    init_ndvi: Optional[float] = None               # NASA POWER surface albedo from MODIS (0–1); independent of ERA5 land model
    init_mjo_amplitude: Optional[float] = None      # MJO RMM amplitude at init date
    init_mjo_sin_phase: Optional[float] = None      # sin(2pi * MJO_phase/8)
    init_mjo_cos_phase: Optional[float] = None      # cos(2pi * MJO_phase/8)
    init_nao: Optional[float] = None                # NAO index at init date
    init_ao: Optional[float] = None                 # AO index at init date
    init_pna: Optional[float] = None                # PNA index at init date
    ens_spread_c: Optional[float] = None            # ECMWF ENS spread (std dev across 51 members) at this lead
    ens_mean_c: Optional[float] = None              # ECMWF ENS mean at this lead


@dataclass
class LeadTimeSkill:
    """Verification statistics aggregated for a single forecast lead time."""
    lead_days: int
    n: int          # number of samples
    rmse: float     # root-mean-square error (°C)
    mae: float      # mean absolute error (°C)
    bias: float     # mean error; positive = model runs warm (°C)


@dataclass
class WeatherResult:
    """Top-level container returned to the caller."""
    location: Location
    historical: list = field(default_factory=list)        # list[TemperatureObservation]
    forecast_periods: list = field(default_factory=list)  # list[ForecastPeriod]
    hourly_forecast: list = field(default_factory=list)   # list[HourlyForecast]
    satellite: list = field(default_factory=list)         # list[SatelliteObservation]


@dataclass
class LoadObservation:
    """One day of historical PJM load paired with temperature-derived features."""
    date: date
    hdd: float              # HDD from daily avg temp (base 65°F)
    cdd: float              # CDD from daily avg temp (base 65°F)
    avg_temp_f: float       # Population-weighted avg temp across PJM grid (°F)
    hi_temp_f: float        # Population-weighted daily HIGH temp (°F)
    lo_temp_f: float        # Population-weighted daily LOW temp (°F)
    actual_load_mw: float   # PJM RTO daily mean load (MW)
    is_weekend: bool
    day_of_week: int        # 0=Monday … 6=Sunday
    is_holiday: bool        # US federal holiday
    day_of_year: int
    temp_lag1_f: Optional[float] = None   # yesterday's avg temp (°F)
    temp_lag2_f: Optional[float] = None   # 2 days ago avg temp (°F)


@dataclass
class LoadForecast:
    """One day of PJM load forecast derived from temperature projections."""
    valid_date: date
    lead_days: int
    mean_load_mw: float    # point forecast
    low_load_mw: float     # 10th-percentile uncertainty bound
    high_load_mw: float    # 90th-percentile uncertainty bound
    hdd: float
    cdd: float
    avg_temp_f: float
