"""
Wheeler-Hendon Real-time Multivariate MJO (RMM) index client.

Fetches the daily RMM1/RMM2 indices from the Australian Bureau of
Meteorology and returns per-date amplitude + circular phase encoding.

Data source: http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt
No authentication required.  The full historical file (~50 years) is
fetched in a single request and cached to disk; subsequent runs are free.

MJO phases 1-8 are circularly encoded as sin/cos so the model treats
phase 8 → 1 transitions as continuous rather than a discontinuity.
When MJO amplitude < 1.0 (weak/inactive MJO), sin_phase and cos_phase
are set to 0.0 (phase is undefined at low amplitudes).
"""

import json
import math
import os
from datetime import date, datetime
from typing import Dict, Optional

import requests

from ._era5 import _CACHE_DIR
from .exceptions import SatelliteAPIError

BOM_MJO_URL = "http://www.bom.gov.au/climate/mjo/graphics/rmm.74toRealtime.txt"
_MJO_CACHE_FILE = "mjo_rmm_bom.json"

# MJO is considered active when amplitude >= this threshold.
_ACTIVE_THRESHOLD = 1.0


def _mjo_cache_path() -> Optional[str]:
    if _CACHE_DIR is None:
        return None
    return os.path.join(_CACHE_DIR, _MJO_CACHE_FILE)


def fetch_mjo_daily(
    session: requests.Session,
) -> Dict[date, Dict[str, float]]:
    """
    Return daily MJO state for every date in the BOM RMM archive.

    Returns
    -------
    dict[date, dict]
        Keys: ``mjo_amplitude``, ``mjo_sin_phase``, ``mjo_cos_phase``.
        sin/cos are 0.0 when amplitude < 1.0 (inactive MJO).

    Raises
    ------
    SatelliteAPIError
        On network error or unexpected file format.
    """
    cache_path = _mjo_cache_path()
    if cache_path and os.path.exists(cache_path):
        try:
            with open(cache_path, "r", encoding="utf-8") as f:
                raw = json.load(f)
            return {datetime.strptime(k, "%Y-%m-%d").date(): v for k, v in raw.items()}
        except (json.JSONDecodeError, ValueError, OSError):
            try:
                os.unlink(cache_path)
            except OSError:
                pass

    try:
        resp = session.get(BOM_MJO_URL, timeout=60)
    except requests.RequestException as exc:
        raise SatelliteAPIError(f"MJO fetch failed: {exc}") from exc

    if resp.status_code != 200:
        raise SatelliteAPIError(f"MJO returned HTTP {resp.status_code}: {resp.text[:200]}")

    result = _parse_rmm(resp.text)
    if not result:
        raise SatelliteAPIError("MJO file parsed to zero records — unexpected format")

    if cache_path:
        os.makedirs(os.path.dirname(cache_path), exist_ok=True)
        tmp = cache_path + ".tmp"
        with open(tmp, "w", encoding="utf-8") as f:
            json.dump({k.isoformat(): v for k, v in result.items()}, f)
        os.replace(tmp, cache_path)

    return result


def _parse_rmm(text: str) -> Dict[date, Dict[str, float]]:
    """
    Parse the BOM RMM text file.

    Expected column order (space-separated):
        year  month  day  RMM1  RMM2  phase  amplitude  [source_flag]
    Lines starting with '#' or containing non-numeric year are skipped.
    """
    result: Dict[date, Dict[str, float]] = {}
    for line in text.splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        parts = line.split()
        if len(parts) < 7:
            continue
        try:
            year, month, day = int(parts[0]), int(parts[1]), int(parts[2])
            phase = int(parts[5])
            amplitude = float(parts[6])
        except (ValueError, IndexError):
            continue
        try:
            d = date(year, month, day)
        except ValueError:
            continue

        if amplitude >= _ACTIVE_THRESHOLD and 1 <= phase <= 8:
            # Circular encoding: phase 8 → 1 is continuous
            angle = 2 * math.pi * phase / 8
            sin_phase = round(math.sin(angle), 6)
            cos_phase = round(math.cos(angle), 6)
        else:
            sin_phase = cos_phase = 0.0

        result[d] = {
            "mjo_amplitude": round(amplitude, 3),
            "mjo_sin_phase": sin_phase,
            "mjo_cos_phase": cos_phase,
        }
    return result
