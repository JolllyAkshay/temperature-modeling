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
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List

import requests

from .exceptions import SatelliteAPIError
from .models import Coordinates, ForecastSample, LeadTimeSkill

_GC_HIST_URL = "https://historical-forecast-api.open-meteo.com/v1/forecast"
_ERA5_URL = "https://archive-api.open-meteo.com/v1/archive"
_GC_MODEL = "gfs_graphcast025"

# GraphCast archive is available from this date onward via Open-Meteo.
_GC_ARCHIVE_START = date(2024, 2, 5)


def collect_verification_records(
    coords: Coordinates,
    init_dates: List[date],
    session: requests.Session,
    max_lead_days: int = 16,
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

    for init_date in init_dates:
        if init_date < _GC_ARCHIVE_START:
            continue

        gc_daily = _fetch_gc_daily(coords, init_date, max_lead_days, session)
        if not gc_daily:
            continue

        valid_start = init_date + timedelta(days=1)
        valid_end = init_date + timedelta(days=max_lead_days)
        era5_daily = _fetch_era5_daily(coords, valid_start, valid_end, session)

        for lead in range(1, max_lead_days + 1):
            valid_date = init_date + timedelta(days=lead)
            gc_temp = gc_daily.get(valid_date)
            obs_temp = era5_daily.get(valid_date)
            if gc_temp is not None and obs_temp is not None:
                records.append(
                    ForecastSample(
                        init_date=init_date,
                        valid_date=valid_date,
                        lead_days=lead,
                        forecast_temp_c=round(gc_temp, 2),
                        observed_temp_c=round(obs_temp, 2),
                        error_c=round(gc_temp - obs_temp, 2),
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
    data = _get_json(_GC_HIST_URL, params, session, "GraphCast verification")
    return _hourly_to_daily_mean(data, "temperature_2m")


def _fetch_era5_daily(
    coords: Coordinates,
    start: date,
    end: date,
    session: requests.Session,
) -> Dict[date, float]:
    """
    Fetch ERA5 reanalysis temperature_2m for *start*–*end* and return
    daily means keyed by date.
    """
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": "temperature_2m",
        "timezone": "UTC",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    data = _get_json(_ERA5_URL, params, session, "ERA5 verification")
    return _hourly_to_daily_mean(data, "temperature_2m")


def _get_json(
    url: str,
    params: dict,
    session: requests.Session,
    label: str,
) -> dict:
    try:
        resp = session.get(url, params=params)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"{label} request failed: {exc}") from exc
    if resp.status_code != 200:
        raise SatelliteAPIError(
            f"{label} returned HTTP {resp.status_code}: {resp.text[:200]}"
        )
    try:
        return resp.json()
    except ValueError as exc:
        raise SatelliteAPIError(f"{label} JSON parse error: {exc}") from exc


def _hourly_to_daily_mean(data: dict, variable: str) -> Dict[date, float]:
    """Average hourly values into daily means, skipping None entries."""
    try:
        times = data["hourly"]["time"]
        values = data["hourly"][variable]
    except KeyError as exc:
        raise SatelliteAPIError(
            f"Unexpected response shape (missing {exc})"
        ) from exc

    daily: Dict[date, List[float]] = defaultdict(list)
    for ts_str, val in zip(times, values):
        if val is None:
            continue
        d = datetime.fromisoformat(ts_str).date()
        daily[d].append(val)

    return {d: sum(vals) / len(vals) for d, vals in daily.items()}
