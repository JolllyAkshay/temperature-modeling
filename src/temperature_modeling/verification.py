"""
Verification framework: collect historical GraphCast forecast–observation pairs
and compute skill scores by lead time.

How it works
------------
For each initialization date ``init_date`` in the supplied list:

1. Fetch the GraphCast 16-day forecast (``gfs_graphcast025``) issued on
   ``init_date`` from the Open-Meteo historical-forecast API.  Hourly
   ``temperature_2m`` values are averaged to daily means.

2. Fetch ERA5 reanalysis ``temperature_2m`` for the corresponding valid
   dates from the Open-Meteo archive API.  ERA5 serves as the independent
   observational truth.

3. Pair forecasts with observations at each lead time (1–16 days) and
   record the signed error (forecast − observed).

The resulting ``ForecastSample`` records can be passed to ``score_by_lead``
to obtain RMSE, MAE, and bias at every lead time — providing the baseline
needed for training a post-processing correction model.

Typical usage
-------------
>>> import requests
>>> from datetime import date, timedelta
>>> from temperature_modeling.models import Coordinates
>>> from temperature_modeling.verification import (
...     collect_verification_records, score_by_lead
... )
>>> session = requests.Session()
>>> coords = Coordinates(lat=39.96, lon=-82.99)
>>> init_dates = [date(2024, 6, 1) + timedelta(days=i) for i in range(30)]
>>> records = collect_verification_records(coords, init_dates, session)
>>> skill = score_by_lead(records)
>>> for s in skill:
...     print(f"Day {s.lead_days:>2}: RMSE={s.rmse:.2f}°C  bias={s.bias:+.2f}°C  n={s.n}")
"""

import math
import time
from collections import defaultdict
from datetime import date, timedelta
from typing import Dict, List, Optional

import requests

from ._era5 import (
    ERA5_ARCHIVE_URL,
    fetch_era5_daily,
    fetch_era5_bulk,
    fetch_era5_init_state,
    fetch_ecmwf_ens_spread,
    fetch_gefs_spread,
    get_json,
    hourly_to_daily_mean,
)
from ._satellite import fetch_nasa_power_daily
from ._mjo import fetch_mjo_daily
from ._teleconnections import fetch_nao_daily, fetch_ao_daily, fetch_pna_daily
from .exceptions import SatelliteAPIError
from .models import Coordinates, ForecastSample, LeadTimeSkill

_GC_HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_GC_MODEL = "gfs_graphcast025"

# GraphCast archive is available from this date onward via Open-Meteo.
_GC_ARCHIVE_START = date(2024, 2, 5)

# ERA5 pressure-level and surface variables fetched at init time when
# include_era5_extra=True.  Order must match the mapping in
# collect_verification_records.
_ERA5_EXTRA_VARS = [
    "geopotential_height_500hPa",  # large-scale circulation pattern
    "temperature_850hPa",          # lower-troposphere thermal state
    "soil_moisture_0_to_7cm",      # land surface moisture memory
    "snow_depth",                  # surface albedo / energy budget
]


def collect_verification_records(
    coords: Coordinates,
    init_dates: List[date],
    session: requests.Session,
    max_lead_days: int = 16,
    include_satellite_features: bool = False,
    include_era5_extra: bool = False,
    include_nasa_power: bool = False,
    include_mjo: bool = True,
    include_ecmwf_ens: bool = False,
) -> List[ForecastSample]:
    """
    Collect forecast–observation pairs for a list of GraphCast init dates.

    For each ``init_date``, the function fetches the GraphCast forecast
    initialized on that date (lead days 1 through ``max_lead_days``) and
    pairs each day's prediction against the ERA5 reanalysis truth.

    Parameters
    ----------
    coords:
        Latitude/longitude of the target location.
    init_dates:
        List of model initialization dates to verify.  Dates before
        2024-02-05 (GraphCast archive start) are silently skipped.
    session:
        requests.Session (headers, timeout) shared with the caller.
    max_lead_days:
        Maximum lead time to verify (default 16, the GraphCast horizon).
    include_satellite_features:
        If True, fetch ERA5 skin_temperature for each init_date and attach
        it to ForecastSample as ``init_skin_temp_c``.  Used to build
        satellite-enhanced feature vectors for the correction model.
    include_era5_extra:
        If True, fetch four additional ERA5 variables at each init_date:
        500 hPa geopotential height, 850 hPa temperature, surface soil
        moisture, and snow depth.  Stored as ``init_z500_m``,
        ``init_t850_c``, ``init_soil_m3``, and ``init_snow_m``.
    include_nasa_power:
        If True, fetch GWETROOT (SMAP-calibrated root-zone soil wetness)
        and SNODP (satellite-assimilated snow depth) from the NASA POWER
        API for the full init_date range in a single batched request.
        Stored as ``init_smap_soil_wetness`` and ``init_modis_snow_m``.

    Returns
    -------
    list[ForecastSample]
        All paired records, sorted by (init_date, lead_days).

    Raises
    ------
    SatelliteAPIError
        If any individual API call fails.  Callers may want to catch this
        per init_date in long batch runs.
    """
    records: List[ForecastSample] = []

    # Fetch global teleconnection indices once (cached to disk after first run).
    mjo_data: Dict[date, dict] = {}
    if include_mjo:
        try:
            mjo_data = fetch_mjo_daily(session)
        except Exception as exc:
            print(f"    [MJO] fetch failed ({exc}); MJO features will be 0", flush=True)

    nao_data: Dict[date, float] = {}
    ao_data:  Dict[date, float] = {}
    pna_data: Dict[date, float] = {}
    try:
        nao_data = fetch_nao_daily(session)
    except Exception as exc:
        print(f"    [NAO] fetch failed ({exc}); NAO features will be 0", flush=True)
    try:
        ao_data = fetch_ao_daily(session)
    except Exception as exc:
        print(f"    [AO] fetch failed ({exc}); AO features will be 0", flush=True)
    try:
        pna_data = fetch_pna_daily(session)
    except Exception as exc:
        print(f"    [PNA] fetch failed ({exc}); PNA features will be 0", flush=True)

    valid_init_dates = sorted(d for d in init_dates if d >= _GC_ARCHIVE_START)

    # ── Bulk pre-fetches (1 API call each, covers full date range) ────────────

    # ERA5 observed temperatures for ALL valid dates in one call.
    # Valid dates span from first_init+1 through last_init+max_lead_days.
    era5_observed_bulk: Dict[date, float] = {}
    if valid_init_dates:
        obs_start = valid_init_dates[0] + timedelta(days=1)
        obs_end   = valid_init_dates[-1] + timedelta(days=max_lead_days)
        era5_observed_bulk = fetch_era5_daily(coords, obs_start, obs_end, session)

    # ERA5 synoptic init-state variables for ALL init dates in one call.
    era5_init_bulk: Dict[str, Dict[date, float]] = {}
    if include_era5_extra and valid_init_dates:
        era5_init_bulk = fetch_era5_bulk(
            coords,
            valid_init_dates[0],
            valid_init_dates[-1],
            _ERA5_EXTRA_VARS,
            session,
        )

    # ERA5 skin temperature for ALL init dates in one call.
    skin_bulk: Dict[date, float] = {}
    if include_satellite_features and valid_init_dates:
        skin_bulk = fetch_era5_daily(
            coords, valid_init_dates[0], valid_init_dates[-1], session,
            variable="skin_temperature",
        )

    # NASA POWER: one batch call for full init_date range.
    nasa_power_data: Dict[date, dict] = {}
    if include_nasa_power and valid_init_dates:
        nasa_power_data = fetch_nasa_power_daily(
            coords,
            valid_init_dates[0],
            valid_init_dates[-1],
            session,
        )

    # ── Per-init-date loop (only GraphCast calls remain here) ─────────────────

    for init_date in init_dates:
        if init_date < _GC_ARCHIVE_START:
            continue

        gc_daily = _fetch_gc_daily(coords, init_date, max_lead_days, session)
        if not gc_daily:
            continue
        time.sleep(0.15)  # ~6-7 req/s — stays well under Open-Meteo free tier

        # Ensemble spread: try GEFS (full archive) first, fall back to ECMWF.
        ens_daily: Dict[date, dict] = {}
        if include_ecmwf_ens:
            try:
                ens_daily = fetch_gefs_spread(coords, init_date, max_lead_days, session)
                time.sleep(0.05)
            except Exception:
                pass  # GEFS failed
            if not ens_daily:
                try:
                    ens_daily = fetch_ecmwf_ens_spread(coords, init_date, max_lead_days, session)
                    time.sleep(0.15)
                except Exception:
                    pass  # ENS fetch failed; features will be 0 for this init date

        # Look up pre-fetched ERA5 observed temperatures.
        era5_daily = era5_observed_bulk

        # Look up pre-fetched skin temperature.
        init_skin_temp: Optional[float] = skin_bulk.get(init_date) if include_satellite_features else None

        # Look up pre-fetched synoptic ERA5 init state.
        era5_extra: dict = {}
        if include_era5_extra:
            era5_extra = {
                var: era5_init_bulk.get(var, {}).get(init_date)
                for var in _ERA5_EXTRA_VARS
            }

        # Look up pre-fetched NASA POWER values for this init date.
        pwr = nasa_power_data.get(init_date, {})
        smap_wetness: Optional[float] = pwr.get("smap_soil_wetness")
        modis_snow: Optional[float] = pwr.get("modis_snow_m")
        ndvi: Optional[float] = pwr.get("ndvi")

        # Look up MJO, NAO, AO state for this init date.
        mjo = mjo_data.get(init_date, {})
        mjo_amplitude: Optional[float] = mjo.get("mjo_amplitude")
        mjo_sin_phase: Optional[float] = mjo.get("mjo_sin_phase")
        mjo_cos_phase: Optional[float] = mjo.get("mjo_cos_phase")
        nao: Optional[float] = nao_data.get(init_date)
        ao:  Optional[float] = ao_data.get(init_date)
        pna: Optional[float] = pna_data.get(init_date)

        for lead in range(1, max_lead_days + 1):
            valid_date = init_date + timedelta(days=lead)
            gc_temp = gc_daily.get(valid_date)
            obs_temp = era5_daily.get(valid_date)
            if gc_temp is not None and obs_temp is not None:
                z500 = era5_extra.get("geopotential_height_500hPa")
                t850 = era5_extra.get("temperature_850hPa")
                soil = era5_extra.get("soil_moisture_0_to_7cm")
                snow = era5_extra.get("snow_depth")
                ens_info = ens_daily.get(valid_date, {})
                records.append(
                    ForecastSample(
                        init_date=init_date,
                        valid_date=valid_date,
                        lead_days=lead,
                        forecast_temp_c=round(gc_temp, 2),
                        observed_temp_c=round(obs_temp, 2),
                        error_c=round(gc_temp - obs_temp, 2),
                        init_skin_temp_c=round(init_skin_temp, 2) if init_skin_temp is not None else None,
                        init_z500_m=round(z500, 1) if z500 is not None else None,
                        init_t850_c=round(t850, 2) if t850 is not None else None,
                        init_soil_m3=round(soil, 4) if soil is not None else None,
                        init_snow_m=round(snow, 3) if snow is not None else None,
                        init_smap_soil_wetness=smap_wetness,
                        init_modis_snow_m=modis_snow,
                        init_ndvi=ndvi,
                        init_mjo_amplitude=mjo_amplitude,
                        init_mjo_sin_phase=mjo_sin_phase,
                        init_mjo_cos_phase=mjo_cos_phase,
                        init_nao=nao,
                        init_ao=ao,
                        init_pna=pna,
                        ens_spread_c=ens_info.get("ens_spread_c"),
                        ens_mean_c=ens_info.get("ens_mean_c"),
                    )
                )

    return sorted(records, key=lambda r: (r.init_date, r.lead_days))


def score_by_lead(records: List[ForecastSample]) -> List[LeadTimeSkill]:
    """
    Compute RMSE, MAE, and bias for each distinct lead time in *records*.

    Parameters
    ----------
    records:
        Output of :func:`collect_verification_records`.

    Returns
    -------
    list[LeadTimeSkill]
        One entry per lead time, sorted ascending by ``lead_days``.
    """
    by_lead: Dict[int, List[float]] = defaultdict(list)
    for r in records:
        by_lead[r.lead_days].append(r.error_c)

    result = []
    for lead, errors in sorted(by_lead.items()):
        n = len(errors)
        bias = sum(errors) / n
        mae = sum(abs(e) for e in errors) / n
        rmse = math.sqrt(sum(e ** 2 for e in errors) / n)
        result.append(
            LeadTimeSkill(
                lead_days=lead,
                n=n,
                rmse=round(rmse, 3),
                mae=round(mae, 3),
                bias=round(bias, 3),
            )
        )
    return result


# ── Internal helpers ─────────────────────────────────────────────────────────

def _fetch_gc_daily(
    coords: Coordinates,
    init_date: date,
    max_lead_days: int,
    session: requests.Session,
) -> Dict[date, float]:
    """
    Fetch the GraphCast forecast initialized on *init_date* and return
    daily mean temperature_2m keyed by valid date.
    """
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": "temperature_2m",
        "models": _GC_MODEL,
        "timezone": "UTC",
        "start_date": init_date.isoformat(),
        "end_date": (init_date + timedelta(days=max_lead_days)).isoformat(),
    }
    data = get_json(_GC_HIST_URL, params, session, "GraphCast verification")
    return hourly_to_daily_mean(data, "temperature_2m")


def _fetch_era5_daily(
    coords: Coordinates,
    start: date,
    end: date,
    session: requests.Session,
) -> Dict[date, float]:
    return fetch_era5_daily(coords, start, end, session)


def _fetch_skin_temp_daily(
    coords: Coordinates,
    d: date,
    session: requests.Session,
) -> Dict[date, float]:
    """Fetch ERA5 skin_temperature for a single date and return its daily mean."""
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": "skin_temperature",
        "timezone": "UTC",
        "start_date": d.isoformat(),
        "end_date": d.isoformat(),
    }
    data = get_json(ERA5_ARCHIVE_URL, params, session, "ERA5 skin temperature")
    return hourly_to_daily_mean(data, "skin_temperature")
