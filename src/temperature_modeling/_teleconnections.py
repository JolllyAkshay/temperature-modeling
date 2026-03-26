"""
NOAA CPC daily teleconnection index client: NAO, AO, and PNA.

All indices are downloaded as plain-text ASCII files from NOAA CPC and
cached to disk.  No authentication required.

NAO (North Atlantic Oscillation)
---------------------------------
Controls winter storm tracks and temperature patterns across the eastern US
and Europe.  Positive NAO → mild, wet eastern US winters; negative → cold,
snowy.  Persistence timescale: 1–3 weeks.

AO (Arctic Oscillation / Northern Annular Mode)
-------------------------------------------------
Reflects strength of the polar vortex.  Negative AO → weakened vortex,
cold-air outbreaks in the mid-latitudes.  Closely related to NAO but
captures more of the stratospheric variability (10–30 day timescale).

PNA (Pacific-North American pattern)
--------------------------------------
Reflects the large-scale ridge/trough configuration over the North Pacific
and North America.  Positive PNA → amplified ridge over western North
America → warm western US, cold central/eastern US.  Most direct
teleconnection predictor of US temperature anomalies at 5–15 day lead.

Data sources (NOAA CPC — public, no auth):
  NAO daily: https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.nao.index.b500101.current.ascii
  AO  daily: https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.ao.index.b500101.current.ascii
  PNA daily: https://ftp.cpc.ncep.noaa.gov/cwlinks/norm.daily.pna.index.b500101.current.ascii
"""

import json
import os
from datetime import date, datetime
from typing import Dict, Optional

import requests

from ._era5 import _CACHE_DIR
from .exceptions import SatelliteAPIError

_NAO_URL = (
    "https://ftp.cpc.ncep.noaa.gov/cwlinks/"
    "norm.daily.nao.index.b500101.current.ascii"
)
_AO_URL = (
    "https://ftp.cpc.ncep.noaa.gov/cwlinks/"
    "norm.daily.ao.index.b500101.current.ascii"
)
_PNA_URL = (
    "https://ftp.cpc.ncep.noaa.gov/cwlinks/"
    "norm.daily.pna.index.b500101.current.ascii"
)

_NAO_CACHE = "teleconnections_nao.json"
_AO_CACHE  = "teleconnections_ao.json"
_PNA_CACHE = "teleconnections_pna.json"


def _cache_path(filename: str) -> Optional[str]:
    if _CACHE_DIR is None:
        return None
    return os.path.join(_CACHE_DIR, filename)


def _load_cache(filename: str) -> Optional[Dict[date, float]]:
    path = _cache_path(filename)
    if path and os.path.exists(path):
        try:
            with open(path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {datetime.strptime(k, "%Y-%m-%d").date(): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, OSError):
            try:
                os.unlink(path)
            except OSError:
                pass
    return None


def _save_cache(filename: str, data: Dict[date, float]) -> None:
    path = _cache_path(filename)
    if not path:
        return
    os.makedirs(os.path.dirname(path), exist_ok=True)
    tmp = path + ".tmp"
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump({k.isoformat(): v for k, v in data.items()}, f)
    os.replace(tmp, path)


def _parse_cpc_ascii(text: str) -> Dict[date, float]:
    """
    Parse CPC ASCII teleconnection index files.

    Two common formats are handled:
      Format A (NAO): year month day value  (one row per day)
      Format B (AO):  year month d1 d2 ... d31  (one row per month)
    """
    result: Dict[date, float] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or not line[0].isdigit():
            continue
        parts = line.split()
        if len(parts) < 4:
            continue
        try:
            year = int(parts[0])
            month = int(parts[1])
        except ValueError:
            continue

        # Format A: year month day value
        if len(parts) == 4:
            try:
                day = int(parts[2])
                val = float(parts[3])
                if val < -90:   # missing sentinel
                    continue
                result[date(year, month, day)] = round(val, 3)
            except (ValueError, OverflowError):
                continue

        # Format B: year month v1 v2 ... v31 (daily values across row)
        elif len(parts) >= 14:
            for day_idx, val_str in enumerate(parts[2:], start=1):
                try:
                    val = float(val_str)
                    if val < -90:
                        continue
                    result[date(year, month, day_idx)] = round(val, 3)
                except (ValueError, OverflowError):
                    continue

    return result


def _fetch_index(url: str, cache_file: str, label: str,
                 session: requests.Session) -> Dict[date, float]:
    cached = _load_cache(cache_file)
    if cached is not None:
        return cached
    try:
        resp = session.get(url, timeout=60)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"{label} fetch failed: {exc}") from exc
    if resp.status_code != 200:
        raise SatelliteAPIError(f"{label} returned HTTP {resp.status_code}")
    data = _parse_cpc_ascii(resp.text)
    if not data:
        raise SatelliteAPIError(f"{label} parsed to zero records — check URL/format")
    _save_cache(cache_file, data)
    return data


def fetch_nao_daily(session: requests.Session) -> Dict[date, float]:
    """Return daily NAO index values keyed by date."""
    return _fetch_index(_NAO_URL, _NAO_CACHE, "NAO", session)


def fetch_ao_daily(session: requests.Session) -> Dict[date, float]:
    """Return daily AO index values keyed by date."""
    return _fetch_index(_AO_URL, _AO_CACHE, "AO", session)


def fetch_pna_daily(session: requests.Session) -> Dict[date, float]:
    """Return daily PNA index values keyed by date."""
    return _fetch_index(_PNA_URL, _PNA_CACHE, "PNA", session)
