"""
Internal helpers for fetching ERA5 reanalysis data from Open-Meteo's archive
API and reducing hourly output to daily means.

Shared by verification.py and features.py.  Not part of the public API.
"""

import hashlib
import json
import math
import os
import time
from collections import defaultdict
from datetime import date, datetime, timedelta
from typing import Dict, List, Optional

import requests

from .exceptions import SatelliteAPIError
from .models import Coordinates

ERA5_ARCHIVE_URL = "https://archive-api.open-meteo.com/v1/archive"
_ENS_URL = "https://ensemble-api.open-meteo.com/v1/ensemble"
_ENS_MODEL = "ecmwf_ifs025"
_ENS_MEMBERS = 51  # members 00–50

# Disk cache for API responses.  Set to None to disable.
# Resolved relative to this file: src/temperature_modeling/../../api_cache/
_CACHE_DIR: Optional[str] = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache"
)


def _cache_path(url: str, params: dict) -> str:
    key = json.dumps({"url": url, "params": sorted(params.items())}, sort_keys=True)
    h = hashlib.md5(key.encode()).hexdigest()
    return os.path.join(_CACHE_DIR, f"{h}.json")  # type: ignore[arg-type]


def _cache_load(url: str, params: dict) -> Optional[dict]:
    if _CACHE_DIR is None:
        return None
    path = _cache_path(url, params)
    if os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            # Corrupt cache file — delete it so we re-fetch from API
            try:
                os.unlink(path)
            except OSError:
                pass
    return None


def _cache_save(url: str, params: dict, data: dict) -> None:
    if _CACHE_DIR is None:
        return
    os.makedirs(_CACHE_DIR, exist_ok=True)
    path = _cache_path(url, params)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, path)


def fetch_era5_daily(
    coords: Coordinates,
    start: date,
    end: date,
    session: requests.Session,
    variable: str = "temperature_2m",
    retries: int = 5,
    backoff: float = 65.0,
) -> Dict[date, float]:
    """
    Fetch an ERA5 hourly variable from the Open-Meteo archive and return
    daily means keyed by date.

    Parameters
    ----------
    coords:
        Target latitude/longitude.
    start, end:
        Inclusive date range.
    session:
        Shared requests.Session.
    variable:
        Open-Meteo hourly variable name (default ``"temperature_2m"``).
    retries, backoff:
        Passed through to get_json's 429 retry policy. The 65s-start,
        5-retry default can block for up to ~33 minutes in the worst case
        (65+130+260+520+1040s) — fine for a one-off backtest, but far too
        long for a best-effort call inside a time-boxed batch job. Callers
        like precache_forecasts.py's hindcast loop (which already has a
        GFS-based fallback if this fails) should pass a much tighter budget.

    Returns
    -------
    dict[date, float]
        Daily mean values.  Days with no non-null hours are omitted.

    Raises
    ------
    SatelliteAPIError
        On network error, non-200 HTTP status, or unexpected JSON shape.
    """
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": variable,
        "timezone": "UTC",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    data = get_json(ERA5_ARCHIVE_URL, params, session, "ERA5", _retries=retries, _backoff=backoff)
    return hourly_to_daily_mean(data, variable)


def fetch_era5_init_state(
    coords: Coordinates,
    d: date,
    variables: List[str],
    session: "requests.Session",
) -> Dict[str, Optional[float]]:
    """
    Fetch daily means for multiple ERA5 variables on a single date in one
    API call.  Returns ``{variable_name: daily_mean}`` for each requested
    variable; ``None`` when no valid hourly values exist for that variable.

    Parameters
    ----------
    coords:
        Target latitude/longitude.
    d:
        The single date to query (start_date == end_date).
    variables:
        List of Open-Meteo hourly variable names, e.g.
        ``["geopotential_height_500hPa", "temperature_850hPa",
           "soil_moisture_0_to_7cm", "snow_depth"]``.
    session:
        Shared requests.Session.

    Returns
    -------
    dict[str, float | None]

    Raises
    ------
    SatelliteAPIError
        On network error, non-200 HTTP status, or unexpected JSON shape.
    """
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "start_date": d.isoformat(),
        "end_date": d.isoformat(),
    }
    data = get_json(ERA5_ARCHIVE_URL, params, session, "ERA5 init state")
    return {var: _daily_mean_for_date(data, var, d) for var in variables}


def fetch_era5_bulk(
    coords: Coordinates,
    start: date,
    end: date,
    variables: List[str],
    session: "requests.Session",
) -> Dict[str, Dict[date, float]]:
    """
    Fetch daily means for multiple ERA5 variables over a date range in ONE
    API call and return ``{variable: {date: daily_mean}}``.

    Use this instead of calling :func:`fetch_era5_init_state` per init_date.
    One bulk request covers the entire backtest period, reducing API calls
    from N_init_dates → 1.

    Parameters
    ----------
    coords:
        Target latitude/longitude.
    start, end:
        Inclusive date range.
    variables:
        List of Open-Meteo hourly variable names.
    session:
        Shared requests.Session.

    Returns
    -------
    dict[str, dict[date, float]]
        Outer key = variable name; inner key = date; value = daily mean.
        Dates with no valid hourly values are omitted from the inner dict.
    """
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": ",".join(variables),
        "timezone": "UTC",
        "start_date": start.isoformat(),
        "end_date": end.isoformat(),
    }
    data = get_json(ERA5_ARCHIVE_URL, params, session, "ERA5 bulk init state")
    return {var: hourly_to_daily_mean(data, var) for var in variables}


def _daily_mean_for_date(data: dict, variable: str, d: date) -> Optional[float]:
    """Return the daily mean of *variable* for date *d* from an Open-Meteo response."""
    try:
        times = data["hourly"]["time"]
        values = data["hourly"][variable]
    except KeyError:
        return None
    vals = [
        v for ts_str, v in zip(times, values)
        if v is not None and datetime.fromisoformat(ts_str).date() == d
    ]
    return sum(vals) / len(vals) if vals else None


def get_json(
    url: str,
    params: dict,
    session: requests.Session,
    label: str,
    _retries: int = 5,
    _backoff: float = 65.0,
) -> dict:
    """
    GET *url* with *params*, parse JSON, raise SatelliteAPIError on failure.

    Responses are cached to disk (``api_cache/`` in the project root) keyed
    by a hash of the URL and parameters.  Subsequent calls with the same
    arguments are served from disk without hitting the network.

    Retries automatically on HTTP 429 (rate limit) with exponential back-off
    starting at *_backoff* seconds (15 s, 30 s, 60 s, …).
    """
    cached = _cache_load(url, params)
    if cached is not None:
        return cached

    for attempt in range(_retries):
        try:
            resp = session.get(url, params=params, timeout=30)
        except requests.RequestException as exc:
            raise SatelliteAPIError(f"{label} request failed: {exc}") from exc

        if resp.status_code == 429:
            wait = _backoff * (2 ** attempt)
            print(f"    [rate limit] waiting {wait:.0f}s before retry {attempt + 1}/{_retries}...", flush=True)
            time.sleep(wait)
            continue

        if resp.status_code != 200:
            raise SatelliteAPIError(
                f"{label} returned HTTP {resp.status_code}: {resp.text[:200]}"
            )
        try:
            data = resp.json()
        except ValueError as exc:
            raise SatelliteAPIError(f"{label} JSON parse error: {exc}") from exc

        _cache_save(url, params, data)
        return data

    raise SatelliteAPIError(f"{label} rate-limited after {_retries} retries")


def fetch_ecmwf_ens_spread(
    coords: Coordinates,
    init_date: date,
    max_lead_days: int,
    session: "requests.Session",
) -> Dict[date, dict]:
    """
    Fetch ECMWF IFS ensemble forecast (51 members) initialized on *init_date*
    and return per-valid-date ensemble mean and spread (std dev across members).

    Returns
    -------
    dict[date, {"ens_mean_c": float, "ens_spread_c": float}]
        Keyed by valid date.  Dates with fewer than 2 members are omitted.
    """
    member_vars = [f"temperature_2m_member{i:02d}" for i in range(_ENS_MEMBERS)]
    params = {
        "latitude": coords.lat,
        "longitude": coords.lon,
        "hourly": ",".join(member_vars),
        "models": _ENS_MODEL,
        "start_date": init_date.isoformat(),
        "end_date": (init_date + timedelta(days=max_lead_days)).isoformat(),
    }
    data = get_json(_ENS_URL, params, session, "ECMWF ENS")
    times = data.get("hourly", {}).get("time", [])

    # Accumulate hourly values per (valid_date, member)
    daily_by_date: Dict[date, List[float]] = defaultdict(list)
    for member_var in member_vars:
        member_vals = data.get("hourly", {}).get(member_var, [])
        day_accum: Dict[date, List[float]] = defaultdict(list)
        for ts_str, val in zip(times, member_vals):
            if val is not None:
                day_accum[datetime.fromisoformat(ts_str).date()].append(val)
        for d, vals in day_accum.items():
            daily_by_date[d].append(sum(vals) / len(vals))  # member daily mean

    result: Dict[date, dict] = {}
    for d, member_means in daily_by_date.items():
        if len(member_means) < 2:
            continue
        mean = sum(member_means) / len(member_means)
        variance = sum((v - mean) ** 2 for v in member_means) / (len(member_means) - 1)
        result[d] = {
            "ens_mean_c": round(mean, 3),
            "ens_spread_c": round(math.sqrt(variance), 3),
        }
    return result


# ---------------------------------------------------------------------------
# GEFS ensemble spread  (NOAA AWS S3 byte-range downloads)
# ---------------------------------------------------------------------------

import threading as _threading

_GEFS_S3_BASE = "https://noaa-gefs-pds.s3.amazonaws.com"
# Sub-directory with 0.5° spread/mean files (available out to f840 = 35 days)
_GEFS_SUBDIR = "atmos/pgrb2ap5"
_GEFS_SPREAD_CACHE_DIR: Optional[str] = (
    os.path.join(os.path.dirname(__file__), "..", "..", "api_cache", "gefs_spread")
    if _CACHE_DIR is not None
    else None
)
# Forecast hours that cover leads 1-16 days (4 steps per day × 16 = 64 values)
_GEFS_LEAD_HOURS: List[int] = [h for h in range(6, 385, 6)]  # 6,12,...,384

# eccodes MEMFS definitions parser is not thread-safe; serialize all decodes.
_ECCODES_LOCK = _threading.Lock()
# Cache for the imported eccodes module so we load DLLs once.
_ECCODES_MODULE = None


def _gefs_cleanup_tmp_grib2() -> None:
    """Delete leftover *.grib2 temp files from previous crashed processes."""
    import glob as _glob
    import tempfile as _tempfile
    pattern = os.path.join(_tempfile.gettempdir(), "tmp*.grib2")
    for f in _glob.glob(pattern):
        try:
            os.unlink(f)
        except OSError:
            pass  # in use by another process — skip


_gefs_cleanup_tmp_grib2()


def _gefs_eccodes_dir() -> Optional[str]:
    """Return the directory containing eccodes.dll, or None if not found."""
    try:
        import site
        for sp in site.getsitepackages():
            d = os.path.join(sp, "eccodes")
            if os.path.isfile(os.path.join(d, "eccodes.dll")):
                return d
        return None
    except Exception:
        return None


def _gefs_load_eccodes():
    """Import eccodes once (thread-safe), adding the DLL directory first."""
    global _ECCODES_MODULE
    if _ECCODES_MODULE is not None:
        return _ECCODES_MODULE
    with _ECCODES_LOCK:
        if _ECCODES_MODULE is None:
            dll_dir = _gefs_eccodes_dir()
            if dll_dir and hasattr(os, "add_dll_directory"):
                os.add_dll_directory(dll_dir)
            import eccodes as _eccodes  # type: ignore[import-not-found]
            _ECCODES_MODULE = _eccodes
    return _ECCODES_MODULE


def _gefs_cache_path(init_date: date) -> str:
    assert _GEFS_SPREAD_CACHE_DIR is not None
    os.makedirs(_GEFS_SPREAD_CACHE_DIR, exist_ok=True)
    return os.path.join(_GEFS_SPREAD_CACHE_DIR, f"{init_date.strftime('%Y%m%d')}.json")


def _gefs_cache_load(init_date: date) -> Optional[dict]:
    if _GEFS_SPREAD_CACHE_DIR is None:
        return None
    p = _gefs_cache_path(init_date)
    if os.path.exists(p):
        try:
            with open(p, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, ValueError, OSError):
            try:
                os.unlink(p)
            except OSError:
                pass
    return None


def _gefs_cache_save(init_date: date, data: dict) -> None:
    if _GEFS_SPREAD_CACHE_DIR is None:
        return
    p = _gefs_cache_path(init_date)
    tmp = p + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(data, f)
    os.replace(tmp, p)


def _loc_key(lat: float, lon: float) -> str:
    return f"{lat:.4f}_{lon:.4f}"


def _gefs_fetch_grib_bytes(
    session: "requests.Session",
    init_date: date,
    fhour: int,
    product: str,  # "gespr" or "geavg"
) -> Optional[bytes]:
    """
    Download the TMP 2m above ground byte range from a GEFS S3 GRIB2 file.
    Returns raw GRIB2 bytes for just that message, or None on failure.
    """
    yyyymmdd = init_date.strftime("%Y%m%d")
    fname_base = f"{product}.t00z.pgrb2a.0p50.f{fhour:03d}"
    idx_url = f"{_GEFS_S3_BASE}/gefs.{yyyymmdd}/00/{_GEFS_SUBDIR}/{fname_base}.idx"
    data_url = f"{_GEFS_S3_BASE}/gefs.{yyyymmdd}/00/{_GEFS_SUBDIR}/{fname_base}"

    try:
        r_idx = session.get(idx_url, timeout=20)
        if r_idx.status_code != 200:
            return None
        lines = r_idx.text.strip().split("\n")
        # Find TMP 2m line
        tmp_idx = next(
            (i for i, l in enumerate(lines) if ":TMP:2 m above ground:" in l), None
        )
        if tmp_idx is None:
            return None
        start = int(lines[tmp_idx].split(":")[1])
        end = int(lines[tmp_idx + 1].split(":")[1]) - 1 if tmp_idx + 1 < len(lines) else start + 300000

        r_data = session.get(data_url, headers={"Range": f"bytes={start}-{end}"}, timeout=30)
        if r_data.status_code not in (200, 206):
            return None
        return r_data.content
    except Exception:
        return None


def _gefs_extract_points(
    eccodes,
    grib_bytes: bytes,
    locs: List[tuple],  # [(lat, lon), ...]
    kelvin_to_celsius: bool = False,
) -> Dict[tuple, float]:
    """
    Extract values at multiple (lat, lon) points from a GRIB2 byte string.

    Decodes the field once and samples all requested locations, which is much
    faster than calling eccodes once per location.

    Parameters
    ----------
    kelvin_to_celsius : bool
        If True subtract 273.15 (use for geavg temperature mean).
    """
    import tempfile
    tmp = tempfile.NamedTemporaryFile(suffix=".grib2", delete=False)
    tmpname = tmp.name
    tmp.write(grib_bytes)
    tmp.flush()
    tmp.close()
    msg = None
    fh = None
    try:
        fh = open(tmpname, "rb")
        with _ECCODES_LOCK:  # serialize eccodes MEMFS definitions access
            msg = eccodes.codes_grib_new_from_file(fh)
            if msg is None:
                return {}
            ni: int = eccodes.codes_get(msg, "Ni")
            nj: int = eccodes.codes_get(msg, "Nj")
            lat1: float = eccodes.codes_get(msg, "latitudeOfFirstGridPointInDegrees")
            lon1: float = eccodes.codes_get(msg, "longitudeOfFirstGridPointInDegrees")
            dj = 180.0 / (nj - 1)
            di = 360.0 / ni
            vals = eccodes.codes_get_values(msg)  # decode once
            eccodes.codes_release(msg)
            msg = None
        result: Dict[tuple, float] = {}
        offset = -273.15 if kelvin_to_celsius else 0.0
        for lat, lon in locs:
            lat_idx = max(0, min(round((lat1 - lat) / dj), nj - 1))
            lon_idx = max(0, min(round((lon % 360.0 - lon1) / di), ni - 1))
            result[(lat, lon)] = float(vals[lat_idx * ni + lon_idx]) + offset
        return result
    except Exception:
        return {}
    finally:
        if msg is not None:
            try:
                eccodes.codes_release(msg)
            except Exception:
                pass
        if fh is not None:
            fh.close()
        try:
            os.unlink(tmpname)
        except Exception:
            pass


def _gefs_build_result(
    init_date: date,
    day_spread: Dict[int, List[float]],
    day_mean: Dict[int, List[float]],
) -> Dict[date, dict]:
    result: Dict[date, dict] = {}
    for lead_day in sorted(set(day_spread) | set(day_mean)):
        sp_list = day_spread.get(lead_day, [])
        mn_list = day_mean.get(lead_day, [])
        result[init_date + timedelta(days=lead_day)] = {
            "ens_spread_c": round(sum(sp_list) / len(sp_list), 3) if sp_list else 0.0,
            "ens_mean_c": round(sum(mn_list) / len(mn_list), 3) if mn_list else 0.0,
        }
    return result


def fetch_gefs_spread_bulk(
    coords_list: List[Coordinates],
    init_date: date,
    max_lead_days: int,
    session: "requests.Session",
) -> Dict[str, Dict[date, dict]]:
    """
    Fetch GEFS ensemble spread for *multiple* locations from a single init_date.

    Downloads each 6-hourly GRIB2 file once and extracts all locations in one
    decode pass.  Saves results for all locations into one cache file.

    Returns
    -------
    dict[loc_key, dict[valid_date, {"ens_spread_c", "ens_mean_c"}]]
    """
    try:
        eccodes = _gefs_load_eccodes()
    except Exception:
        return {}

    locs_tuples = [(c.lat, c.lon) for c in coords_list]
    keys = [_loc_key(c.lat, c.lon) for c in coords_list]

    # Per-location daily accumulators
    day_spread: Dict[str, Dict[int, List[float]]] = {k: defaultdict(list) for k in keys}
    day_mean: Dict[str, Dict[int, List[float]]] = {k: defaultdict(list) for k in keys}

    lead_hours = [h for h in _GEFS_LEAD_HOURS if h <= max_lead_days * 24]

    for fhour in lead_hours:
        spread_bytes = _gefs_fetch_grib_bytes(session, init_date, fhour, "gespr")
        mean_bytes = _gefs_fetch_grib_bytes(session, init_date, fhour, "geavg")
        lead_day = math.ceil(fhour / 24)
        time.sleep(0.05)

        if spread_bytes:
            sv_map = _gefs_extract_points(eccodes, spread_bytes, locs_tuples, False)
            for (lat, lon), sv in sv_map.items():
                k = _loc_key(lat, lon)
                day_spread[k][lead_day].append(sv)

        if mean_bytes:
            mv_map = _gefs_extract_points(eccodes, mean_bytes, locs_tuples, True)
            for (lat, lon), mv in mv_map.items():
                k = _loc_key(lat, lon)
                day_mean[k][lead_day].append(mv)

    # Build per-location results
    out: Dict[str, Dict[date, dict]] = {}
    for k in keys:
        out[k] = _gefs_build_result(init_date, day_spread[k], day_mean[k])

    # Merge into cache
    if _GEFS_SPREAD_CACHE_DIR is not None:
        existing = _gefs_cache_load(init_date) or {}
        for k, res in out.items():
            existing[k] = {d.isoformat(): v for d, v in res.items()}
        _gefs_cache_save(init_date, existing)

    return out


def fetch_gefs_spread(
    coords: Coordinates,
    init_date: date,
    max_lead_days: int,
    session: "requests.Session",
) -> Dict[date, dict]:
    """
    Fetch GEFS ensemble spread and mean for one location on *init_date*.

    Checks api_cache/gefs_spread/ first.  If a cache miss, downloads only
    this location's data (for pre-computation use fetch_gefs_spread_bulk).

    Returns
    -------
    dict[date, {"ens_spread_c": float, "ens_mean_c": float}]
    """
    key = _loc_key(coords.lat, coords.lon)
    cache = _gefs_cache_load(init_date)
    if cache is not None and key in cache:
        return {date.fromisoformat(k): v for k, v in cache[key].items()}

    # Single-location fallback (downloads files once for just this coord)
    try:
        eccodes = _gefs_load_eccodes()
    except Exception:
        return {}

    locs_tuples = [(coords.lat, coords.lon)]
    day_spread: Dict[int, List[float]] = defaultdict(list)
    day_mean: Dict[int, List[float]] = defaultdict(list)

    for fhour in [h for h in _GEFS_LEAD_HOURS if h <= max_lead_days * 24]:
        spread_bytes = _gefs_fetch_grib_bytes(session, init_date, fhour, "gespr")
        mean_bytes = _gefs_fetch_grib_bytes(session, init_date, fhour, "geavg")
        lead_day = math.ceil(fhour / 24)
        time.sleep(0.05)

        if spread_bytes:
            sv_map = _gefs_extract_points(eccodes, spread_bytes, locs_tuples, False)
            for _, sv in sv_map.items():
                day_spread[lead_day].append(sv)

        if mean_bytes:
            mv_map = _gefs_extract_points(eccodes, mean_bytes, locs_tuples, True)
            for _, mv in mv_map.items():
                day_mean[lead_day].append(mv)

    result = _gefs_build_result(init_date, day_spread, day_mean)

    if result and _GEFS_SPREAD_CACHE_DIR is not None:
        existing = cache or {}
        existing[key] = {d.isoformat(): v for d, v in result.items()}
        _gefs_cache_save(init_date, existing)

    return result


def hourly_to_daily_mean(data: dict, variable: str) -> Dict[date, float]:
    """
    Average hourly Open-Meteo values into daily means.

    Parameters
    ----------
    data:
        Parsed JSON response containing ``data["hourly"]["time"]`` and
        ``data["hourly"][variable]``.
    variable:
        The hourly variable key to aggregate.

    Returns
    -------
    dict[date, float]
        Daily mean values.  Hours with ``None`` are skipped; days where all
        hours are ``None`` are omitted entirely.
    """
    try:
        times = data["hourly"]["time"]
        values = data["hourly"][variable]
    except KeyError as exc:
        raise SatelliteAPIError(
            f"Unexpected Open-Meteo response shape (missing {exc})"
        ) from exc

    daily: Dict[date, List[float]] = defaultdict(list)
    for ts_str, val in zip(times, values):
        if val is None:
            continue
        daily[datetime.fromisoformat(ts_str).date()].append(val)

    return {d: sum(vals) / len(vals) for d, vals in daily.items()}
