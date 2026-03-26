"""
Feature extraction for GraphCast post-processing.

Converts raw ``ForecastSample`` records (forecast + observation pairs) into
``FeatureVector`` objects ready for training an error-correction model.

Feature set
-----------
Each vector has ten predictors and one target:

Temporal
  ``lead_days``           — integer lead time (primary error driver)
  ``lead_days_sq``        — squared lead time (captures non-linear growth)
  ``valid_sin_doy``       — sin(2π × doy/365) of the valid date
  ``valid_cos_doy``       — cos(2π × doy/365) of the valid date
  ``init_sin_doy``        — sin(2π × doy/365) of the init date
  ``init_cos_doy``        — cos(2π × doy/365) of the init date

Forecast value
  ``forecast_temp_c``     — raw GraphCast temperature_2m prediction

Climatological
  ``clim_mean_c``         — ERA5 long-term mean temperature for valid doy
  ``clim_std_c``          — ERA5 long-term std for valid doy (spread proxy)
  ``forecast_anomaly_c``  — forecast_temp_c − clim_mean_c; large anomalies
                            tend to be over-predicted by NWP models

Target
  ``error_c``             — forecast − observed (positive = warm bias)

Climatological normals are pre-computed via :func:`build_climate_normals`
and stored in a :class:`ClimateNormals` object that can be reused across
many calls to :func:`extract_features`.
"""

import math
from collections import defaultdict
from dataclasses import dataclass
from datetime import date, timedelta
from typing import Dict, List

import requests

from ._era5 import fetch_era5_daily
from .models import Coordinates, ForecastSample

# Number of ERA5 years used to compute the climatological baseline.
_DEFAULT_CLIM_YEARS = 5

# Ordered list of FeatureVector predictor field names (excludes the target
# ``error_c``).  Used by correction.py to build X matrices reproducibly.
FEATURE_FIELDS: List[str] = [
    "lead_days",
    "lead_days_sq",
    "valid_sin_doy",
    "valid_cos_doy",
    "init_sin_doy",
    "init_cos_doy",
    "forecast_temp_c",
    "clim_mean_c",
    "clim_std_c",
    "forecast_anomaly_c",
    # Rolling recent-error signal: 7-day trailing mean of day-1 GC errors
    # before init_date.  Captures regime-persistent warm/cold biases.
    # Computed from already-collected records — no extra API calls needed.
    "recent_gc_error_mean",
    # MJO state at init time — dominant sub-seasonal (10-30 day) signal.
    # Amplitude: strength of MJO (>1 = active); 0 when inactive.
    # sin/cos phase: circular encoding of MJO phase (1-8); 0 when inactive.
    "init_mjo_amplitude",
    "init_mjo_sin_phase",
    "init_mjo_cos_phase",
    # NAO, AO, and PNA teleconnection indices at init time.
    # NAO: controls eastern US winter storm tracks; 1-3 week persistence.
    # AO:  polar vortex strength; negative -> cold-air outbreaks.
    # PNA: ridge/trough pattern over North America; most direct US temp predictor.
    "init_nao",
    "init_ao",
    "init_pna",
    # Location (normalised lat/lon) — populated in pooled-model mode.
    "norm_lat",
    "norm_lon",
    # Interaction features.
    "error_x_lead",
    "anomaly_x_lead",
    "mjo_x_lead",
]

# Extended feature set that adds ERA5 skin temperature at init time.
# Use this when ForecastSamples were collected with include_satellite_features=True.
FEATURE_FIELDS_SAT: List[str] = FEATURE_FIELDS + [
    "init_skin_temp_c",
    "init_skin_anomaly_c",
]

# Extended feature set that adds four synoptic ERA5 variables at init time:
#   - 500 hPa geopotential height anomaly (deviation from monthly mean)
#   - 500 hPa geopotential height (absolute, for context)
#   - 850 hPa temperature        (lower-troposphere thermal state)
#   - Surface soil moisture      (land surface moisture memory)
#   - Snow depth                 (surface albedo / energy budget)
# Use when ForecastSamples were collected with include_era5_extra=True.
FEATURE_FIELDS_ERA5: List[str] = FEATURE_FIELDS + [
    "init_z500_m",
    "init_z500_anom_m",
    "init_t850_c",
    "init_soil_m3",
    "init_snow_m",
]

# Satellite-enhanced feature set: ERA5 synoptic + NASA POWER delta features.
#
# Rather than raw SMAP/MODIS values (which are collinear with ERA5 soil/snow),
# we use z-score deltas: how much does each satellite product disagree with ERA5,
# in standardised units?  This strips the shared signal and gives XGBoost only
# the genuinely new information each satellite adds.
#
#   smap_soil_delta  = z(SMAP_soil_wetness) - z(ERA5_soil_m3)
#   modis_snow_delta = z(MODIS_snow_m)      - z(ERA5_snow_m)
#
# A large positive delta means the satellite sees much wetter/deeper than ERA5;
# large negative means the opposite.  ERA5 alone still provides the baseline.
#
# Use when ForecastSamples were collected with include_era5_extra=True AND
# include_nasa_power=True.
FEATURE_FIELDS_SAT: List[str] = FEATURE_FIELDS_ERA5 + [
    "smap_soil_delta",
    "modis_snow_delta",
]

# ECMWF ensemble-enhanced feature set: SAT features + ENS uncertainty signals.
#
#   ens_spread_c     — std dev across 51 ECMWF members at this (init, lead) pair
#                      High spread = high forecast uncertainty; correction model
#                      should shrink toward climatology.
#   ens_mean_c       — ECMWF ensemble mean; often more accurate than deterministic GC
#   gc_vs_ens_delta  — GraphCast minus ECMWF mean.  Large disagreement often indicates
#                      GC is wrong (ECMWF consensus is the stronger prior).
#   ens_spread_x_lead — spread × lead_days interaction: uncertainty growth with lead.
#
# Use when ForecastSamples were collected with include_ecmwf_ens=True.
FEATURE_FIELDS_ENS: List[str] = FEATURE_FIELDS_SAT + [
    "ens_spread_c",
    "ens_mean_c",
    "gc_vs_ens_delta",
    "ens_spread_x_lead",
]


@dataclass
class FeatureVector:
    """Ten predictors + target error for one ForecastSample.

    Optional satellite fields (``init_skin_temp_c``, ``init_skin_anomaly_c``)
    are populated when records were collected with
    ``include_satellite_features=True``; otherwise they default to 0.0.
    """
    # Temporal
    lead_days: float
    lead_days_sq: float
    valid_sin_doy: float
    valid_cos_doy: float
    init_sin_doy: float
    init_cos_doy: float
    # Forecast value
    forecast_temp_c: float
    # Climatological
    clim_mean_c: float
    clim_std_c: float
    forecast_anomaly_c: float
    # Target
    error_c: float
    # Rolling recent-error signal (computed from records; 0.0 for earliest init dates)
    recent_gc_error_mean: float = 0.0
    # Satellite surface state at init time (optional; 0.0 when not available)
    init_skin_temp_c: float = 0.0
    init_skin_anomaly_c: float = 0.0
    # Synoptic ERA5 state at init time (optional; 0.0 when not available)
    init_z500_m: float = 0.0        # 500 hPa geopotential height (m)
    init_z500_anom_m: float = 0.0   # z500 deviation from monthly climatology (m)
    init_t850_c: float = 0.0        # 850 hPa temperature (C)
    init_soil_m3: float = 0.0       # surface soil moisture (m3/m3)
    init_snow_m: float = 0.0        # snow depth (m)
    # NASA POWER satellite-informed state at init time (optional; 0.0 when not available)
    init_smap_soil_wetness: float = 0.0  # GWETROOT root-zone soil wetness (0-1)
    init_modis_snow_m: float = 0.0       # SNODP snow depth (m)
    # Satellite–ERA5 delta features (z-score of satellite minus z-score of ERA5 equivalent).
    # Captures the independent information each satellite product adds beyond ERA5.
    smap_soil_delta: float = 0.0   # z(SMAP_wetness) - z(ERA5_soil_m3)
    modis_snow_delta: float = 0.0  # z(MODIS_snow_m) - z(ERA5_snow_m)
    # MODIS surface albedo — genuinely independent of ERA5 land model.
    # Low albedo = green canopy or dark wet soil; high albedo = snow or dry bare soil.
    init_ndvi: float = 0.0  # field name kept as init_ndvi; stores ALLSKY_SRF_ALB (0–1)
    # MJO state at init time (0.0 when inactive or unavailable)
    init_mjo_amplitude: float = 0.0
    init_mjo_sin_phase: float = 0.0
    init_mjo_cos_phase: float = 0.0
    # Teleconnection indices at init time (0.0 when unavailable)
    init_nao: float = 0.0
    init_ao: float = 0.0
    init_pna: float = 0.0
    # Location features (normalised; 0.0 when not set — single-location mode)
    norm_lat: float = 0.0   # (lat - 37) / 10  →  roughly centred on CONUS
    norm_lon: float = 0.0   # (lon + 95) / 20  →  roughly centred on CONUS
    # Interaction features
    error_x_lead: float = 0.0        # recent_gc_error_mean × lead_days
    anomaly_x_lead: float = 0.0      # forecast_anomaly_c  × lead_days
    mjo_x_lead: float = 0.0          # mjo_amplitude       × lead_days
    # ECMWF ensemble uncertainty features (0.0 when not collected)
    ens_spread_c: float = 0.0        # std dev across 51 ECMWF members (°C)
    ens_mean_c: float = 0.0          # ECMWF ensemble mean (°C)
    gc_vs_ens_delta: float = 0.0     # GraphCast - ECMWF mean (°C)
    ens_spread_x_lead: float = 0.0   # ens_spread × lead_days


class ClimateNormals:
    """
    ERA5 climatological mean and standard deviation by day-of-year (1–366).

    Build with :func:`build_climate_normals`; pass to :func:`extract_features`.
    """

    def __init__(
        self,
        doy_mean: Dict[int, float],
        doy_std: Dict[int, float],
    ) -> None:
        self._mean = doy_mean
        self._std = doy_std

    @property
    def doy_mean(self) -> Dict[int, float]:
        """Mean temperature_2m by day-of-year."""
        return self._mean

    @property
    def doy_std(self) -> Dict[int, float]:
        """Std of temperature_2m by day-of-year."""
        return self._std

    def mean(self, d: date) -> float:
        """Return climatological mean for the given date's day-of-year."""
        return self._mean[d.timetuple().tm_yday]

    def std(self, d: date) -> float:
        """Return climatological std for the given date's day-of-year."""
        return self._std.get(d.timetuple().tm_yday, 1.0)


def build_climate_normals(
    coords: Coordinates,
    anchor_date: date,
    session: requests.Session,
    clim_years: int = _DEFAULT_CLIM_YEARS,
) -> ClimateNormals:
    """
    Fetch ERA5 daily temperature_2m for *clim_years* years ending on
    *anchor_date* and return per-DOY mean and standard deviation.

    Parameters
    ----------
    coords:
        Target latitude/longitude.
    anchor_date:
        Last date of the climatological period (typically the start of your
        evaluation window, so future observations do not leak into training).
    session:
        Shared requests.Session.
    clim_years:
        Number of years of ERA5 data to use (default 5).

    Returns
    -------
    ClimateNormals

    Raises
    ------
    SatelliteAPIError
        If the ERA5 fetch fails.
    """
    end = anchor_date
    start = date(end.year - clim_years, end.month, end.day)

    daily = fetch_era5_daily(coords, start, end, session)

    # Group values by day-of-year.
    doy_values: Dict[int, List[float]] = defaultdict(list)
    for d, temp in daily.items():
        doy_values[d.timetuple().tm_yday].append(temp)

    doy_mean: Dict[int, float] = {}
    doy_std: Dict[int, float] = {}

    for doy, vals in doy_values.items():
        mean = sum(vals) / len(vals)
        doy_mean[doy] = mean
        if len(vals) > 1:
            variance = sum((v - mean) ** 2 for v in vals) / (len(vals) - 1)
            doy_std[doy] = math.sqrt(variance)
        else:
            doy_std[doy] = 1.0  # fallback when only one year of data

    return ClimateNormals(doy_mean=doy_mean, doy_std=doy_std)


def _compute_rolling_error(records: List[ForecastSample]) -> Dict[date, float]:
    """
    Compute the 7-day trailing mean of day-1 GC errors per init_date.

    For init_date ``d``, averages the day-1 forecast error from init_dates
    ``d-1`` through ``d-7`` (whichever are present in records).  Returns 0.0
    for init_dates where fewer than 1 prior day-1 error is available.
    """
    day1_errors: Dict[date, float] = {
        r.init_date: r.error_c for r in records if r.lead_days == 1
    }
    result: Dict[date, float] = {}
    for d in {r.init_date for r in records}:
        past = [
            day1_errors[d - timedelta(days=k)]
            for k in range(1, 8)
            if (d - timedelta(days=k)) in day1_errors
        ]
        result[d] = round(sum(past) / len(past), 3) if past else 0.0
    return result


def extract_features(
    records: List[ForecastSample],
    normals: ClimateNormals,
    include_satellite: bool = False,
    include_era5_extra: bool = False,
    include_nasa_power: bool = False,
    include_ecmwf_ens: bool = False,
    lat: float = 0.0,
    lon: float = 0.0,
) -> List[FeatureVector]:
    """
    Convert a list of ``ForecastSample`` records into ``FeatureVector``
    objects using pre-computed climatological normals.

    Parameters
    ----------
    records:
        Output of :func:`~temperature_modeling.verification.collect_verification_records`.
    normals:
        Output of :func:`build_climate_normals`.
    include_satellite:
        If True, populate ``init_skin_temp_c`` and ``init_skin_anomaly_c``
        from ``r.init_skin_temp_c`` when available.  Requires records
        collected with ``include_satellite_features=True``.
    include_era5_extra:
        If True, populate ``init_z500_m``, ``init_t850_c``,
        ``init_soil_m3``, and ``init_snow_m`` from the corresponding
        ForecastSample fields.  Requires records collected with
        ``include_era5_extra=True``.
    include_nasa_power:
        If True, populate ``init_smap_soil_wetness`` and
        ``init_modis_snow_m`` from NASA POWER fields on the sample.
        Requires records collected with ``include_nasa_power=True``.

    Returns
    -------
    list[FeatureVector]
        One vector per record, in the same order.
    """
    rolling_errors = _compute_rolling_error(records)

    # Compute z500 monthly climatology from records (no API calls needed).
    # Used to derive init_z500_anom_m = z500 - monthly_mean(z500).
    z500_by_month: Dict[int, List[float]] = defaultdict(list)
    if include_era5_extra:
        for r in records:
            if r.init_z500_m is not None:
                z500_by_month[r.init_date.month].append(r.init_z500_m)
    z500_clim_monthly: Dict[int, float] = {
        m: sum(vs) / len(vs) for m, vs in z500_by_month.items()
    }

    # Compute z-score stats for satellite delta features (non-zero values only,
    # since 0.0 is the missing-data sentinel for both ERA5 and NASA POWER fields).
    def _zstats(vals: List[float]):
        """Return (mean, std) of non-zero values; std=1 if fewer than 2 points."""
        nz = [v for v in vals if v != 0.0]
        if not nz:
            return 0.0, 1.0
        mu = sum(nz) / len(nz)
        if len(nz) < 2:
            return mu, 1.0
        var = sum((v - mu) ** 2 for v in nz) / (len(nz) - 1)
        return mu, math.sqrt(var) if var > 0 else 1.0

    smap_mu, smap_std   = _zstats([r.init_smap_soil_wetness or 0.0 for r in records])
    era5s_mu, era5s_std = _zstats([r.init_soil_m3 or 0.0 for r in records])
    modis_mu, modis_std = _zstats([r.init_modis_snow_m or 0.0 for r in records])
    era5n_mu, era5n_std = _zstats([r.init_snow_m or 0.0 for r in records])

    vectors = []
    for r in records:
        valid_doy = r.valid_date.timetuple().tm_yday
        init_doy = r.init_date.timetuple().tm_yday
        clim_mean = normals.mean(r.valid_date)
        clim_std = normals.std(r.valid_date)

        # Satellite surface state features at init time.
        if include_satellite and r.init_skin_temp_c is not None:
            init_skin_temp_c = r.init_skin_temp_c
            init_skin_anomaly_c = round(r.init_skin_temp_c - normals.mean(r.init_date), 3)
        else:
            init_skin_temp_c = 0.0
            init_skin_anomaly_c = 0.0

        # Synoptic ERA5 state features at init time.
        if include_era5_extra:
            init_z500_m = r.init_z500_m if r.init_z500_m is not None else 0.0
            z500_clim = z500_clim_monthly.get(r.init_date.month, 0.0)
            init_z500_anom_m = round(init_z500_m - z500_clim, 1) if init_z500_m else 0.0
            init_t850_c = r.init_t850_c if r.init_t850_c is not None else 0.0
            init_soil_m3 = r.init_soil_m3 if r.init_soil_m3 is not None else 0.0
            init_snow_m = r.init_snow_m if r.init_snow_m is not None else 0.0
        else:
            init_z500_m = init_z500_anom_m = init_t850_c = init_soil_m3 = init_snow_m = 0.0

        # NASA POWER satellite-informed features at init time.
        if include_nasa_power:
            init_smap_soil_wetness = r.init_smap_soil_wetness if r.init_smap_soil_wetness is not None else 0.0
            init_modis_snow_m = r.init_modis_snow_m if r.init_modis_snow_m is not None else 0.0
            # Delta features: satellite z-score minus ERA5 z-score.
            # Non-zero check ensures we don't use missing-data sentinels in the delta.
            if init_smap_soil_wetness != 0.0 and init_soil_m3 != 0.0:
                smap_z  = (init_smap_soil_wetness - smap_mu)  / smap_std
                era5s_z = (init_soil_m3            - era5s_mu) / era5s_std
                smap_soil_delta = round(smap_z - era5s_z, 4)
            else:
                smap_soil_delta = 0.0
            if init_modis_snow_m != 0.0 and init_snow_m != 0.0:
                modis_z = (init_modis_snow_m - modis_mu) / modis_std
                era5n_z = (init_snow_m       - era5n_mu) / era5n_std
                modis_snow_delta = round(modis_z - era5n_z, 4)
            else:
                modis_snow_delta = 0.0
        else:
            init_smap_soil_wetness = init_modis_snow_m = 0.0
            smap_soil_delta = modis_snow_delta = 0.0
        init_ndvi = (r.init_ndvi if r.init_ndvi is not None else 0.0) if include_nasa_power else 0.0

        # ECMWF ensemble features
        if include_ecmwf_ens and r.ens_spread_c is not None and r.ens_mean_c is not None:
            ens_spread_c = r.ens_spread_c
            ens_mean_c = r.ens_mean_c
            gc_vs_ens_delta = round(r.forecast_temp_c - r.ens_mean_c, 3)
            ens_spread_x_lead = round(r.ens_spread_c * r.lead_days, 4)
        else:
            ens_spread_c = ens_mean_c = gc_vs_ens_delta = ens_spread_x_lead = 0.0

        vectors.append(
            FeatureVector(
                lead_days=float(r.lead_days),
                lead_days_sq=float(r.lead_days ** 2),
                valid_sin_doy=round(math.sin(2 * math.pi * valid_doy / 365), 6),
                valid_cos_doy=round(math.cos(2 * math.pi * valid_doy / 365), 6),
                init_sin_doy=round(math.sin(2 * math.pi * init_doy / 365), 6),
                init_cos_doy=round(math.cos(2 * math.pi * init_doy / 365), 6),
                forecast_temp_c=r.forecast_temp_c,
                clim_mean_c=round(clim_mean, 3),
                clim_std_c=round(clim_std, 3),
                forecast_anomaly_c=round(r.forecast_temp_c - clim_mean, 3),
                recent_gc_error_mean=rolling_errors.get(r.init_date, 0.0),
                error_c=r.error_c,
                init_skin_temp_c=init_skin_temp_c,
                init_skin_anomaly_c=init_skin_anomaly_c,
                init_z500_m=init_z500_m,
                init_z500_anom_m=init_z500_anom_m,
                init_t850_c=init_t850_c,
                init_soil_m3=init_soil_m3,
                init_snow_m=init_snow_m,
                init_smap_soil_wetness=init_smap_soil_wetness,
                init_modis_snow_m=init_modis_snow_m,
                smap_soil_delta=smap_soil_delta,
                modis_snow_delta=modis_snow_delta,
                init_ndvi=init_ndvi,
                init_mjo_amplitude=r.init_mjo_amplitude if r.init_mjo_amplitude is not None else 0.0,
                init_mjo_sin_phase=r.init_mjo_sin_phase if r.init_mjo_sin_phase is not None else 0.0,
                init_mjo_cos_phase=r.init_mjo_cos_phase if r.init_mjo_cos_phase is not None else 0.0,
                init_nao=r.init_nao if r.init_nao is not None else 0.0,
                init_ao=r.init_ao if r.init_ao is not None else 0.0,
                init_pna=r.init_pna if r.init_pna is not None else 0.0,
                norm_lat=round((lat - 37.0) / 10.0, 4),
                norm_lon=round((lon + 95.0) / 20.0, 4),
                error_x_lead=round(rolling_errors.get(r.init_date, 0.0) * r.lead_days, 4),
                anomaly_x_lead=round((r.forecast_temp_c - clim_mean) * r.lead_days, 4),
                mjo_x_lead=round((r.init_mjo_amplitude or 0.0) * r.lead_days, 4),
                ens_spread_c=ens_spread_c,
                ens_mean_c=ens_mean_c,
                gc_vs_ens_delta=gc_vs_ens_delta,
                ens_spread_x_lead=ens_spread_x_lead,
            )
        )
    return vectors
