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
]


@dataclass
class FeatureVector:
    """Ten predictors + target error for one ForecastSample."""
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


def extract_features(
    records: List[ForecastSample],
    normals: ClimateNormals,
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

    Returns
    -------
    list[FeatureVector]
        One vector per record, in the same order.
    """
    vectors = []
    for r in records:
        valid_doy = r.valid_date.timetuple().tm_yday
        init_doy = r.init_date.timetuple().tm_yday
        clim_mean = normals.mean(r.valid_date)
        clim_std = normals.std(r.valid_date)

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
                error_c=r.error_c,
            )
        )
    return vectors
