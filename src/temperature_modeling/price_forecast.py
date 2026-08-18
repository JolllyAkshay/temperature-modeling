"""
Day-ahead electricity price fetching and regression-based forecast.

Data sources:
  - CAISO: CAISO OASIS PRC_LMP (DAM, TH_NP15_GEN-APND + TH_SP15_GEN-APND) + EIA load
  - MISO: docs.misoenergy.org YYYYMMDD_da_exante_lmp.csv (8 hub nodes) + EIA load
  - PJM:   PJM DataMiner2 DA hub LMPs (PJM_API_KEY, free at pjm.com) + EIA load
  - ERCOT: ERCOT public NP reports (np6-86-cd DA settlement prices) + EIA load
  - NYISO: NYISO public day-ahead zone LMP CSVs (mis.nyiso.com) + EIA load
  - ISO-NE: ISO-NE webservices REST API (configure _ISONE_API_USER / _ISONE_API_PASS)
  - SPP: SPP Marketplace public DA-LMP-SL files (SPPNORTH_HUB + SPPSOUTH_HUB) + EIA load
  - NG price: EIA Henry Hub daily spot price ($/MMBtu) as regression co-variate

Forecast: multivariate OLS — log(price) ~ log(load) + weekday + seasonal + log(ng_price)
"""

import csv
import io
import json
import logging
import math
import os
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Disk cache for slow, rate-limited price-history fetches (CAISO OASIS
# chunking, SPP per-day file downloads). Without this, every dashboard
# forecast request re-runs the full external fetch — 30-50s each time.
# ---------------------------------------------------------------------------
_PRICE_HISTORY_CACHE_TTL_H = 6


def _read_price_history_cache(path: str) -> list | None:
    if not os.path.exists(path):
        return None
    try:
        age_h = (time.time() - os.path.getmtime(path)) / 3600
        if age_h >= _PRICE_HISTORY_CACHE_TTL_H:
            return None
        with open(path, encoding="utf-8") as f:
            data = json.load(f)
        return data or None
    except Exception:
        return None


def _write_price_history_cache(path: str, data: list) -> None:
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f)
    except OSError:
        log.warning("Could not write price history cache to %s", path)

_EIA_LOAD_URL   = "https://api.eia.gov/v2/electricity/rto/region-data/data/"
_EIA_NG_URL     = "https://api.eia.gov/v2/natural-gas/pri/fut/data/"   # daily Henry Hub spot
_NYISO_LMP_URL  = "https://mis.nyiso.com/public/csv/damlbmp/{date}damlbmp_zone.csv"
_ISONE_API_BASE = "https://webservices.iso-ne.com/api/v1.1"

# SPP Marketplace DA LMP — public, no auth needed
_SPP_FILE_LIST_URL  = "https://marketplace.spp.org/file-browser-api/"
_SPP_DOWNLOAD_URL   = "https://marketplace.spp.org/file-browser-api/download/da-lmp-by-settlement-location"
_SPP_HUB_LOCATIONS  = {"SPPNORTH_HUB", "SPPSOUTH_HUB"}

# CAISO OASIS DA LMP — public, no auth needed
_CAISO_OASIS_URL   = "http://oasis.caiso.com/oasisapi/SingleZip"
_CAISO_HUB_NODES   = {"TH_NP15_GEN-APND", "TH_SP15_GEN-APND"}

# MISO docs DA ExAnte LMP — public, no auth needed
_MISO_LMP_URL = "https://docs.misoenergy.org/marketreports/{date}_da_exante_lmp.csv"

# Credentials for ISO-NE webservices (free registration at iso-ne.com)
_ISONE_API_USER = os.environ.get("ISONE_API_USER", "")
_ISONE_API_PASS = os.environ.get("ISONE_API_PASS", "")

# PJM DataMiner2 DA LMPs — free API key registration at pjm.com
_PJM_DATAMINER_URL   = "https://dataminer2.pjm.com/feed/da_hrl_lmps/excel"
_PJM_DATAMINER_KEY   = os.environ.get("PJM_API_KEY", "")
_PJM_WESTERN_HUB_ID  = 1   # pnode_id for PJM Western Hub (main reference)

# ERCOT DAM Settlement Point Prices — NP4-190-CD (register at api.ercot.com)
_ERCOT_NP_URL         = "https://api.ercot.com/api/public-reports/np4-190-cd/dam_stlmnt_pnt_prices"
_ERCOT_API_KEY        = os.environ.get("ERCOT_API_KEY", "")
_ERCOT_USERNAME       = os.environ.get("ERCOT_USERNAME", "")
_ERCOT_PASSWORD       = os.environ.get("ERCOT_PASSWORD", "")
_ERCOT_HUB_NODES      = {"HB_BUSAVG", "HB_WEST", "HB_NORTH", "HB_SOUTH", "HB_HOUSTON"}
# Azure B2C ROPC token endpoint for api.ercot.com
_ERCOT_TOKEN_URL = (
    "https://ercotb2c.b2clogin.com/ercotb2c.onmicrosoft.com"
    "/B2C_1_PUBAPI-ROPC-FLOW/oauth2/v2.0/token"
)
_ERCOT_CLIENT_ID = "fec253ea-0d06-4272-a5e6-b478baeecd70"
_ercot_bearer_cache: dict = {}   # {token, expires_at}


def _ercot_bearer_token() -> str:
    """Fetch (or return cached) ERCOT OAuth id_token via ROPC flow."""
    import time as _time
    cached = _ercot_bearer_cache
    if cached.get("token") and _time.time() < cached.get("expires_at", 0) - 60:
        return cached["token"]
    if not (_ERCOT_USERNAME and _ERCOT_PASSWORD):
        return ""
    try:
        r = requests.post(
            _ERCOT_TOKEN_URL,
            data={
                "grant_type":    "password",
                "username":      _ERCOT_USERNAME,
                "password":      _ERCOT_PASSWORD,
                "client_id":     _ERCOT_CLIENT_ID,
                "scope":         f"openid {_ERCOT_CLIENT_ID} offline_access",
                "response_type": "id_token",
            },
            timeout=15,
        )
        if r.status_code == 200:
            j = r.json()
            token = j.get("id_token") or j.get("access_token", "")
            cached["token"]      = token
            cached["expires_at"] = _time.time() + int(j.get("expires_in", 3600))
            log.info("ERCOT: OAuth token obtained (expires in %ds)", j.get("expires_in", 3600))
            return token
        log.warning("ERCOT token fetch HTTP %d: %s", r.status_code, r.text[:300])
    except Exception as exc:
        log.warning("ERCOT token fetch failed: %s", exc)
    return ""


# ---------------------------------------------------------------------------
# EIA data fetching
# ---------------------------------------------------------------------------

def _eia_get(url: str, params: dict) -> list:
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        return []
    params["api_key"] = api_key
    try:
        r = requests.get(url, params=params, timeout=30)
        r.raise_for_status()
        return r.json().get("response", {}).get("data", [])
    except Exception:
        log.warning("EIA request failed: %s", url)
        return []


def _eia_load_daily(respondent: str, start: str, end: str) -> dict:
    """Return {date_str: avg_mw} from EIA hourly demand. Paginates for long date ranges."""
    daily: dict = {}
    offset = 0
    page_size = 5000   # EIA hard limit per request

    while True:
        rows = _eia_get(_EIA_LOAD_URL, {
            "frequency":            "hourly",
            "data[0]":              "value",
            "facets[respondent][]": respondent,
            "facets[type][]":       "D",
            "start":  start,
            "end":    end,
            "length": page_size,
            "offset": offset,
        })
        if not rows:
            break
        for row in rows:
            d = row.get("period", "")[:10]
            v = row.get("value")
            if d and v is not None:
                daily.setdefault(d, []).append(float(v))
        if len(rows) < page_size:
            break
        offset += page_size

    return {d: sum(vs) / len(vs) for d, vs in daily.items()}


def fetch_henry_hub_daily(start: str, end: str) -> dict:
    """
    Return {date_str: ng_price_per_mmbtu} for Henry Hub natural gas spot prices.
    Source: EIA natural-gas/pri/fut, series RNGWHHD (daily spot, $/MMBtu).
    Weekend / holiday gaps are forward-filled from the preceding trading day.
    Returns empty dict on failure (model gracefully omits the NG feature).
    """
    rows = _eia_get(_EIA_NG_URL, {
        "frequency":        "daily",
        "data[0]":          "value",
        "facets[series][]": "RNGWHHD",
        "start":            start,
        "end":              end,
        "length":           1000,  # supports ~2 years of daily NG spot prices
        "sort[0][column]":  "period",
        "sort[0][direction]": "asc",
    })
    raw: dict = {}
    for row in rows:
        d = row.get("period", "")[:10]
        v = row.get("value")
        if d and v is not None:
            try:
                raw[d] = float(v)
            except (TypeError, ValueError):
                pass

    if not raw:
        return {}

    # Forward-fill gaps (weekends / holidays have no NG trading)
    start_dt = date.fromisoformat(start)
    end_dt   = date.fromisoformat(end)
    result: dict = {}
    last_val: float | None = None
    cur = start_dt
    while cur <= end_dt:
        ds = cur.isoformat()
        if ds in raw:
            last_val = raw[ds]
        if last_val is not None:
            result[ds] = last_val
        cur += timedelta(days=1)
    return result


# ---------------------------------------------------------------------------
# ISO-specific price fetchers
# ---------------------------------------------------------------------------

_NYISO_MONTHLY_ZIP_URL = "http://mis.nyiso.com/public/csv/damlbmp/{ym}01damlbmp_zone_csv.zip"
_NYISO_RECENT_CSV_URL  = "https://mis.nyiso.com/public/csv/damlbmp/{date}damlbmp_zone.csv"
_NYISO_RECENT_DAYS     = 7   # use individual CSVs only for the most recent days


def _parse_nyiso_csv(text: str) -> dict[str, list[float]]:
    """
    Parse NYISO DA LMP CSV → {date_str: [lmp, ...]}.
    NYISO timestamps are 'MM/DD/YYYY HH:MM' — convert to YYYY-MM-DD.
    """
    price_by_date: dict[str, list[float]] = {}
    for row in csv.DictReader(io.StringIO(text)):
        ts = row.get("Time Stamp", "")
        if not ts:
            continue
        # Parse MM/DD/YYYY HH:MM → YYYY-MM-DD
        try:
            parts = ts.split()[0].split("/")   # ['MM', 'DD', 'YYYY']
            d = f"{parts[2]}-{parts[0]:0>2}-{parts[1]:0>2}"
        except (IndexError, ValueError):
            continue
        try:
            price_by_date.setdefault(d, []).append(float(row["LBMP ($/MWHr)"]))
        except (KeyError, ValueError, TypeError):
            pass
    return price_by_date


def _fetch_nyiso_price_history(days: int = 90) -> list:
    """
    Fetch NYISO day-ahead zone LMP history.
    - History (> 7 days old): monthly zip archives from mis.nyiso.com
    - Recent (≤ 7 days): individual daily CSV files
    No authentication required.
    """
    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.headers["User-Agent"] = "grid-dashboard/1.0"

    price_by_date: dict[str, list[float]] = {}

    # --- Monthly zip archives for bulk history ---
    archive_end = end_dt - timedelta(days=_NYISO_RECENT_DAYS)
    if start_dt <= archive_end:
        months: list[tuple[int, int]] = []
        cur = start_dt.replace(day=1)
        while cur <= archive_end:
            months.append((cur.year, cur.month))
            cur = (cur.replace(day=28) + timedelta(days=4)).replace(day=1)

        def _fetch_month_zip(ym: tuple[int, int]) -> dict[str, list[float]]:
            year, month = ym
            url = _NYISO_MONTHLY_ZIP_URL.format(ym=f"{year}{month:02d}")
            try:
                r = session.get(url, timeout=60)
                if r.status_code != 200 or not r.content:
                    return {}
                result: dict[str, list[float]] = {}
                with zipfile.ZipFile(io.BytesIO(r.content)) as z:
                    for name in z.namelist():
                        if not name.endswith(".csv"):
                            continue
                        text = z.read(name).decode("utf-8", errors="replace")
                        for d, prices in _parse_nyiso_csv(text).items():
                            result.setdefault(d, []).extend(prices)
                return result
            except Exception:
                return {}

        with ThreadPoolExecutor(max_workers=8) as pool:
            for month_data in pool.map(_fetch_month_zip, months):
                for d, prices in month_data.items():
                    price_by_date.setdefault(d, []).extend(prices)

    # --- Individual CSVs for recent days ---
    recent_start = max(start_dt, end_dt - timedelta(days=_NYISO_RECENT_DAYS - 1))

    def _fetch_day(d: date):
        url = _NYISO_RECENT_CSV_URL.format(date=d.strftime("%Y%m%d"))
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                return {}
            return _parse_nyiso_csv(r.text)
        except Exception:
            return {}

    recent_dates = [recent_start + timedelta(days=i)
                    for i in range((end_dt - recent_start).days + 1)]
    with ThreadPoolExecutor(max_workers=8) as pool:
        for day_data in pool.map(_fetch_day, recent_dates):
            for d, prices in day_data.items():
                price_by_date.setdefault(d, []).extend(prices)

    # Average prices per day and filter to requested window
    target = {(end_dt - timedelta(days=i)).isoformat() for i in range(days)}
    daily_price = {
        d: sum(vs) / len(vs)
        for d, vs in price_by_date.items()
        if vs and d in target
    }

    daily_load = _eia_load_daily("NYIS", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(daily_price) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(daily_price[d], 2),
        })

    log.info("NYISO: fetched %d days of price history (monthly zips + EIA load)", len(result))
    return result


def _fetch_isone_price_history(days: int = 90) -> list:
    """
    Fetch ISO-NE day-ahead hub LMP from ISO-NE webservices REST API.
    Requires free account at iso-ne.com — set ISONE_API_USER + ISONE_API_PASS env vars
    (your ISO Express login works directly as HTTP Basic Auth, no separate key).
    Hub: Massachusetts Hub (node 4000).

    The API is per-day-per-location only (no date-range endpoint) —
    GET /hourlylmp/da/final/day/{yyyymmdd}/location/4000.json — so this
    fetches one day at a time, in parallel.
    """
    if not _ISONE_API_USER or not _ISONE_API_PASS:
        log.info("ISO-NE: no API credentials — set ISONE_API_USER + ISONE_API_PASS")
        return []

    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.auth = (_ISONE_API_USER, _ISONE_API_PASS)
    session.headers["Accept"] = "application/json"

    def _fetch_day(d: date) -> tuple[str, float | None]:
        url = f"{_ISONE_API_BASE}/hourlylmp/da/final/day/{d.strftime('%Y%m%d')}/location/4000.json"
        try:
            r = session.get(url, timeout=30)
            if r.status_code != 200:
                return d.isoformat(), None
            lmps = [
                float(entry["LmpTotal"])
                for entry in r.json().get("HourlyLmps", {}).get("HourlyLmp", [])
                if entry.get("LmpTotal") is not None
            ]
            avg = (sum(lmps) / len(lmps)) if lmps else None
            return d.isoformat(), avg
        except Exception:
            return d.isoformat(), None

    dates = [end_dt - timedelta(days=i) for i in range(days)]
    price_by_date: dict = {}
    with ThreadPoolExecutor(max_workers=5) as pool:
        for d_str, avg in pool.map(_fetch_day, dates):
            if avg is not None:
                price_by_date[d_str] = [avg]

    if not price_by_date:
        log.warning("ISO-NE: no price data returned for any requested day")
        return []

    daily_load = _eia_load_daily("ISNE", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(price_by_date) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(sum(price_by_date[d]) / len(price_by_date[d]), 2),
        })

    log.info("ISO-NE: fetched %d days of price history (ISO-NE webservices)", len(result))
    return result


_SPP_PRICE_HISTORY_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "spp_price_history_90d.json"
)


def _fetch_spp_price_history(days: int = 90) -> list:
    cached = _read_price_history_cache(_SPP_PRICE_HISTORY_CACHE)
    if cached is not None:
        return cached
    result = _fetch_spp_price_history_live(days)
    if result:
        _write_price_history_cache(_SPP_PRICE_HISTORY_CACHE, result)
    return result


def _fetch_spp_price_history_live(days: int = 90) -> list:
    """
    Fetch SPP day-ahead hub LMP from SPP Marketplace public DA-LMP-SL CSV files.
    No authentication required.
    Uses average of SPPNORTH_HUB and SPPSOUTH_HUB as the system-level price.
    """
    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.headers["User-Agent"] = "grid-dashboard/1.0"

    def _list_month(year: int, month: int) -> list[str]:
        """Return list of file paths available for a given year/month."""
        try:
            r = session.get(
                _SPP_FILE_LIST_URL,
                params={
                    "fsName": "da-lmp-by-settlement-location",
                    "path":   f"/{year}/{month:02d}/By_Day/",
                    "type":   "folder",
                },
                timeout=20,
            )
            if r.status_code != 200:
                return []
            return [item["path"] for item in r.json() if item.get("type") == "file"]
        except Exception:
            return []

    def _fetch_day_file(path: str) -> tuple[str, float | None]:
        """Download one DA-LMP-SL file and return (date_str, avg_hub_lmp)."""
        try:
            r = session.get(_SPP_DOWNLOAD_URL, params={"path": path}, timeout=60)
            if r.status_code != 200:
                return "", None
            reader = csv.DictReader(io.StringIO(r.text))
            lmps: list[float] = []
            date_str = ""
            for row in reader:
                if row.get("Settlement Location", "") not in _SPP_HUB_LOCATIONS:
                    continue
                lmp = row.get("LMP", "")
                try:
                    lmps.append(float(lmp))
                except (ValueError, TypeError):
                    pass
                if not date_str:
                    raw = row.get("Interval", "")
                    if raw:
                        # "07/25/2026 01:00:00" → "2026-07-25"
                        try:
                            parts = raw.split()[0].split("/")
                            date_str = f"{parts[2]}-{parts[0]}-{parts[1]}"
                        except Exception:
                            pass
            avg = (sum(lmps) / len(lmps)) if lmps else None
            return date_str, avg
        except Exception:
            return "", None

    # Collect file paths covering the date range
    months_needed: set[tuple[int, int]] = set()
    cur = start_dt.replace(day=1)
    while cur <= end_dt:
        months_needed.add((cur.year, cur.month))
        # advance one month
        if cur.month == 12:
            cur = cur.replace(year=cur.year + 1, month=1)
        else:
            cur = cur.replace(month=cur.month + 1)

    all_paths: list[str] = []
    for (yr, mo) in sorted(months_needed):
        all_paths.extend(_list_month(yr, mo))

    # Filter to paths within our date window
    target_dates = {
        (end_dt - timedelta(days=i)).isoformat()
        for i in range(days)
    }

    def _path_date(p: str) -> str:
        """Extract YYYY-MM-DD from path like /2026/07/By_Day/DA-LMP-SL-202607250100.csv"""
        try:
            name = p.rsplit("/", 1)[-1]            # DA-LMP-SL-202607250100.csv
            ds   = name.split("-")[3][:8]           # "20260725"
            return f"{ds[:4]}-{ds[4:6]}-{ds[6:8]}"
        except Exception:
            return ""

    paths_to_fetch = [p for p in all_paths if _path_date(p) in target_dates]

    # Parallel download
    price_by_date: dict[str, float] = {}
    with ThreadPoolExecutor(max_workers=6) as pool:
        for d_str, avg in pool.map(_fetch_day_file, paths_to_fetch):
            if d_str and avg is not None:
                price_by_date[d_str] = round(avg, 2)

    # Pair with EIA SWPP demand
    daily_load = _eia_load_daily("SWPP", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(price_by_date) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": price_by_date[d],
        })

    log.info("SPP: fetched %d days of price history (Marketplace DA-LMP-SL + EIA load)", len(result))
    return result


def _fetch_miso_price_history(days: int = 90) -> list:
    """
    Fetch MISO day-ahead hub LMP from MISO docs.misoenergy.org market reports.
    No authentication required. File: YYYYMMDD_da_exante_lmp.csv.
    Averages all 8 MISO hub nodes (ILLINOIS.HUB, INDIANA.HUB, etc.) across all
    24 hours to get a daily system-level DA price. Pairs with EIA MISO load data.
    """
    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)

    session = requests.Session()
    session.headers["User-Agent"] = "grid-dashboard/1.0"

    def _fetch_day(d: date) -> tuple:
        url = _MISO_LMP_URL.format(date=d.strftime("%Y%m%d"))
        try:
            r = session.get(url, timeout=30)
            if r.status_code != 200 or not r.content:
                return d.isoformat(), None
            lines = r.text.splitlines()
            # Header row starts with "Node" — skip the 4-line preamble
            header_idx = next(
                (i for i, l in enumerate(lines) if l.startswith("Node,")), None
            )
            if header_idx is None:
                return d.isoformat(), None
            lmps: list[float] = []
            for row in csv.DictReader(lines[header_idx:]):
                if row.get("Type", "").strip() != "Hub":
                    continue
                if row.get("Value", "").strip() != "LMP":
                    continue
                for k, v in row.items():
                    if k.startswith("HE "):
                        try:
                            lmps.append(float(v))
                        except (ValueError, TypeError):
                            pass
            avg = (sum(lmps) / len(lmps)) if lmps else None
            return d.isoformat(), avg
        except Exception:
            return d.isoformat(), None

    dates = [end_dt - timedelta(days=i) for i in range(days)]
    price_by_date: dict = {}
    # Cap at 5 workers — MISO files are ~1MB each; too many concurrent requests
    # overwhelm both the server and the local connection
    with ThreadPoolExecutor(max_workers=5) as pool:
        for d_str, price in pool.map(_fetch_day, dates):
            if price is not None:
                price_by_date[d_str] = price

    daily_load = _eia_load_daily("MISO", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(price_by_date) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(price_by_date[d], 2),
        })

    log.info("MISO: fetched %d days of price history (docs.misoenergy.org + EIA load)", len(result))
    return result


_CAISO_PRICE_HISTORY_CACHE = os.path.join(
    os.path.dirname(__file__), "..", "..", "api_cache", "caiso_price_history_90d.json"
)


def _fetch_caiso_price_history(days: int = 90) -> list:
    cached = _read_price_history_cache(_CAISO_PRICE_HISTORY_CACHE)
    if cached is not None:
        return cached
    result = _fetch_caiso_price_history_live(days)
    if result:
        _write_price_history_cache(_CAISO_PRICE_HISTORY_CACHE, result)
    return result


def _fetch_caiso_price_history_live(days: int = 90) -> list:
    """
    Fetch CAISO day-ahead LMP history from CAISO OASIS (PRC_LMP, DAM market).
    No authentication required. Averages TH_NP15_GEN-APND and TH_SP15_GEN-APND
    hub nodes across all 24 hours to get a daily system-level DA price.
    Pairs with EIA CISO (CAL respondent) load data.
    """
    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)

    session = requests.Session()
    session.headers["User-Agent"] = "grid-dashboard/1.0"

    # CAISO OASIS limits responses to ~31 days; use 28-day chunks to stay safe
    CHUNK_DAYS = 28
    price_by_date: dict = {}

    chunk_start = start_dt
    while chunk_start <= end_dt:
        chunk_end = min(chunk_start + timedelta(days=CHUNK_DAYS), end_dt + timedelta(days=1))
        start_str = chunk_start.strftime("%Y%m%dT00:00-0000")
        end_str   = chunk_end.strftime("%Y%m%dT00:00-0000")
        for attempt in range(3):
            try:
                r = session.get(
                    _CAISO_OASIS_URL,
                    params={
                        "queryname":     "PRC_LMP",
                        "market_run_id": "DAM",
                        "node":          ",".join(_CAISO_HUB_NODES),
                        "startdatetime": start_str,
                        "enddatetime":   end_str,
                        "version":       1,
                        "resultformat":  6,
                    },
                    timeout=60,
                )
                if r.status_code == 429:
                    wait = 10 * (attempt + 1)
                    log.info("CAISO OASIS: rate-limited, waiting %ds", wait)
                    time.sleep(wait)
                    continue
                if r.status_code == 200 and r.content:
                    z = zipfile.ZipFile(io.BytesIO(r.content))
                    raw = z.read(z.namelist()[0]).decode("utf-8")
                    for row in csv.DictReader(raw.splitlines()):
                        if row.get("LMP_TYPE", "").strip() != "LMP":
                            continue
                        d = row.get("OPR_DT", "")[:10]
                        try:
                            price_by_date.setdefault(d, []).append(float(row.get("MW", "nan")))
                        except (ValueError, TypeError):
                            pass
                break
            except Exception:
                log.warning("CAISO OASIS: request failed for %s -> %s", start_str, end_str)
                break
        time.sleep(5)   # polite inter-chunk delay to avoid 429s
        chunk_start = chunk_end

    target = {(end_dt - timedelta(days=i)).isoformat() for i in range(days)}
    daily_price = {
        d: sum(vs) / len(vs)
        for d, vs in price_by_date.items()
        if vs and d in target
    }

    daily_load = _eia_load_daily("CAL", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(daily_price) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(daily_price[d], 2),
        })

    log.info("CAISO: fetched %d days of price history (OASIS PRC_LMP + EIA load)", len(result))
    return result


def _fetch_pjm_price_history(days: int = 90) -> list:
    """
    Fetch PJM day-ahead LMPs from PJM DataMiner2 API (Western Hub, pnode_id=1).
    Requires PJM_API_KEY env var — free registration at pjm.com/data/dataminer.
    Pairs with EIA PJM demand for the regression model.
    """
    if not _PJM_DATAMINER_KEY:
        log.info("PJM: no DataMiner2 key — set PJM_API_KEY (free at pjm.com)")
        return []

    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.headers["Ocp-Apim-Subscription-Key"] = _PJM_DATAMINER_KEY
    session.headers["User-Agent"] = "grid-dashboard/1.0"

    price_by_date: dict = {}
    start_row = 1
    page_size = 50000

    while True:
        try:
            r = session.get(
                _PJM_DATAMINER_URL,
                params={
                    "startRow":               start_row,
                    "endRow":                 start_row + page_size - 1,
                    "pnode_id":               _PJM_WESTERN_HUB_ID,
                    "datetime_beginning_ept": start_dt.strftime("%Y-%m-%d 00:00"),
                    "datetime_ending_ept":    end_dt.strftime("%Y-%m-%d 23:00"),
                },
                timeout=60,
            )
            if r.status_code != 200:
                log.warning("PJM DataMiner2: HTTP %d", r.status_code)
                break
            reader = csv.DictReader(io.StringIO(r.text))
            rows = list(reader)
            if not rows:
                break
            for row in rows:
                period = (row.get("datetime_beginning_ept") or row.get("DateTime Beginning (EPT)") or "")[:10]
                lmp_raw = row.get("total_lmp_da") or row.get("Total LMP DA") or row.get("LMP")
                if period and lmp_raw:
                    try:
                        price_by_date.setdefault(period, []).append(float(lmp_raw))
                    except (ValueError, TypeError):
                        pass
            if len(rows) < page_size:
                break
            start_row += page_size
        except Exception:
            log.exception("PJM DataMiner2: price fetch failed")
            break

    if not price_by_date:
        return []

    daily_load = _eia_load_daily("PJM", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(price_by_date) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(sum(price_by_date[d]) / len(price_by_date[d]), 2),
        })

    log.info("PJM: fetched %d days of price history (DataMiner2 + EIA load)", len(result))
    return result


def _fetch_ercot_price_history(days: int = 90) -> list:
    """
    Fetch ERCOT day-ahead settlement point prices from ERCOT's report API.
    Requires ERCOT_API_KEY + ERCOT_USERNAME + ERCOT_PASSWORD (api.ercot.com).
    Uses HB_BUSAVG hub (load-weighted average) when available, falls back to hub average.
    Pairs with EIA ERCOT (TEX) demand data for the regression model.
    """
    if not _ERCOT_API_KEY:
        log.info("ERCOT: no API key — set ERCOT_API_KEY (register at api.ercot.com)")
        return []

    bearer = _ercot_bearer_token()
    if not bearer:
        log.info("ERCOT: no OAuth token — set ERCOT_USERNAME + ERCOT_PASSWORD")
        return []

    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.headers["User-Agent"]                = "grid-dashboard/1.0"
    session.headers["Ocp-Apim-Subscription-Key"] = _ERCOT_API_KEY
    session.headers["Authorization"]             = f"Bearer {bearer}"

    # NP4-190-CD returns array-format rows: [deliveryDate, hourEnding, settlementPoint, price, DSTFlag]
    # Filter at API level for HB_BUSAVG to avoid fetching all ~26k rows/day.
    # Fall back to HB_HUBAVG if HB_BUSAVG returns nothing.
    def _fetch_hub(hub: str) -> dict[str, list[float]]:
        by_date: dict[str, list[float]] = {}
        page = 1
        page_size = 8760  # 365 days × 24h fits in one page
        while True:
            try:
                r = session.get(
                    _ERCOT_NP_URL,
                    params={
                        "deliveryDateFrom": start_dt.strftime("%Y-%m-%d"),
                        "deliveryDateTo":   end_dt.strftime("%Y-%m-%d"),
                        "settlementPoint":  hub,
                        "size":             page_size,
                        "page":             page,
                    },
                    timeout=60,
                )
                if r.status_code != 200:
                    log.warning("ERCOT NP API: HTTP %d for %s", r.status_code, hub)
                    break
                payload = r.json()
                rows = payload.get("data", [])
                for row in rows:
                    # row = [deliveryDate, hourEnding, settlementPoint, price, DSTFlag]
                    if not isinstance(row, (list, tuple)) or len(row) < 4:
                        continue
                    d = str(row[0])[:10]
                    try:
                        by_date.setdefault(d, []).append(float(row[3]))
                    except (TypeError, ValueError):
                        pass
                total = payload.get("_meta", {}).get("totalRecords", len(rows))
                if page * page_size >= total or len(rows) < page_size:
                    break
                page += 1
            except Exception as exc:
                log.warning("ERCOT NP API: page %d failed: %s", page, exc)
                break
        return by_date

    price_by_date = _fetch_hub("HB_BUSAVG")
    if not price_by_date:
        log.info("ERCOT: HB_BUSAVG empty, trying HB_HUBAVG")
        price_by_date = _fetch_hub("HB_HUBAVG")

    daily_prices = {d: sum(v) / len(v) for d, v in price_by_date.items() if v}

    if not daily_prices:
        log.info("ERCOT: NP4-190-CD returned no hub price data")
        return []

    daily_load = _eia_load_daily("TEX", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(daily_prices) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(daily_prices[d], 2),
        })

    log.info("ERCOT: fetched %d days of price history (NP4-190-CD + EIA load)", len(result))
    return result


def fetch_price_history(iso: str, days: int = 90) -> list:
    """
    Return [{date, load_mw, price_usd_mwh}] for the last `days` days.
    Dispatches to ISO-specific fetchers for all supported ISOs.
    """
    if iso == "caiso":
        return _fetch_caiso_price_history(days)
    if iso == "miso":
        return _fetch_miso_price_history(days)
    if iso == "nyiso":
        return _fetch_nyiso_price_history(days)
    if iso == "isone":
        return _fetch_isone_price_history(days)
    if iso == "spp":
        return _fetch_spp_price_history(days)
    if iso == "pjm":
        return _fetch_pjm_price_history(days)
    if iso == "ercot":
        return _fetch_ercot_price_history(days)

    log.info("%s: no price fetcher configured", iso.upper())
    return []


def price_unavailable_reason(iso: str) -> str:
    """Return a user-facing explanation of why price data is unavailable for this ISO."""
    if iso == "pjm":
        if not os.environ.get("PJM_API_KEY"):
            return "Set PJM_API_KEY for price data (free at pjm.com/data/dataminer)"
        return "PJM price data temporarily unavailable"
    if iso == "ercot":
        if not os.environ.get("ERCOT_API_KEY"):
            return "Set ERCOT_API_KEY for price data (register at api.ercot.com)"
        return "ERCOT price data temporarily unavailable"
    if iso == "isone":
        if not os.environ.get("ISONE_API_USER"):
            return "Set ISONE_API_USER + ISONE_API_PASS for price data (iso-ne.com)"
        return "ISO-NE price data temporarily unavailable"
    return "Price data unavailable for this ISO"


def _attach_ng_prices(history: list) -> list:
    """
    Attach Henry Hub NG spot price to each history record.
    Records without a matching NG price are kept with ng_price=None
    (the model falls back to the 5-feature version for those rows).
    """
    if not history:
        return history
    dates = sorted(r["date"] for r in history)
    ng = fetch_henry_hub_daily(dates[0], dates[-1])
    for r in history:
        r["ng_price"] = ng.get(r["date"])
    return history


# ---------------------------------------------------------------------------
# Regression model (log-linear load → price, optional NG co-variate)
# ---------------------------------------------------------------------------

def _fit_price_model(history: list) -> dict | None:
    """
    Fit OLS: log(price) ~ log(load) + weekday + sin_month + cos_month [+ log(ng_price)] + intercept.
    If Henry Hub NG prices are available for ≥ half the rows the model uses 6 features;
    otherwise falls back to 5 features (no NG term).
    Returns {"coeffs": list, "rmse": float, "use_ng": bool} or None if insufficient data.
    """
    rows = [r for r in history if r["price_usd_mwh"] > 0 and r["load_mw"] > 0]
    if len(rows) < 7:
        log.warning("Insufficient price history for regression (%d rows)", len(rows))
        return None

    # Winsorise extreme price spikes before fitting — scarcity events ($500+/MWh)
    # bias the OLS log-linear seasonal coefficients upward, producing winter base
    # forecasts that look like scarcity, not normal weather.
    # Cap: 3× the 95th percentile of the raw price distribution.
    prices_sorted = sorted(r["price_usd_mwh"] for r in rows)
    p95_idx  = max(0, int(0.95 * len(prices_sorted)) - 1)
    p95      = prices_sorted[p95_idx]
    winsor_cap = 3.0 * p95
    n_winsorised = sum(1 for r in rows if r["price_usd_mwh"] > winsor_cap)
    if n_winsorised:
        log.info("Price model: winsorising %d rows above $%.0f/MWh (3×p95 $%.0f)",
                 n_winsorised, winsor_cap, p95)
    rows = [{**r, "price_usd_mwh": min(r["price_usd_mwh"], winsor_cap)} for r in rows]

    rows_with_ng = [r for r in rows if r.get("ng_price") and r["ng_price"] > 0]
    use_ng = len(rows_with_ng) >= len(rows) // 2

    fit_rows = rows_with_ng if use_ng else rows

    X, y_vals = [], []
    for r in fit_rows:
        d = date.fromisoformat(r["date"])
        month = d.month
        row_x = [
            math.log(max(r["load_mw"], 1)),
            float(d.weekday() < 5),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
        ]
        if use_ng:
            row_x.append(math.log(max(r["ng_price"], 0.01)))
        row_x.append(1.0)  # intercept always last
        X.append(row_x)
        y_vals.append(math.log(max(r["price_usd_mwh"], 1)))

    X_arr = np.array(X)
    y_arr = np.array(y_vals)
    coeffs, _, _, _ = np.linalg.lstsq(X_arr, y_arr, rcond=None)

    y_pred = X_arr @ coeffs
    resid = np.exp(y_pred) - np.exp(y_arr)
    rmse = float(np.sqrt(np.mean(resid ** 2)))

    log.info("Price model: %d features (%s NG), RMSE $%.2f/MWh",
             len(coeffs), "with" if use_ng else "without", rmse)
    return {"coeffs": coeffs.tolist(), "rmse": round(rmse, 2), "use_ng": use_ng}


def forecast_prices(iso: str, load_forecast: list) -> list:
    """
    Forecast day-ahead prices from the load forecast.

    Parameters
    ----------
    iso           : ISO code
    load_forecast : list of {date, mean_load_gw, ...} from the load model store

    Returns
    -------
    list of {date, forecast_price, low_price, high_price, model_rmse}
    Empty list if data is unavailable or regression fails.
    """
    history = fetch_price_history(iso, days=90)
    if not history:
        return []

    history = _attach_ng_prices(history)
    model = _fit_price_model(history)
    if not model:
        return []

    coeffs  = model["coeffs"]
    rmse    = model["rmse"]
    use_ng  = model.get("use_ng", False)

    # For forecasting future days, use a 7-day trailing average of NG spot as proxy
    ng_vals = [r["ng_price"] for r in history if r.get("ng_price") and r["ng_price"] > 0]
    ng_forecast = sum(ng_vals[-7:]) / len(ng_vals[-7:]) if ng_vals else None

    # Sanity bounds: reject forecast if regression wildly extrapolates
    train_prices = [r["price"] for r in history if r.get("price") and r["price"] > 0]
    max_train_price = max(train_prices) if train_prices else 500.0
    price_ceiling = max(max_train_price * 20, 2000.0)   # 20× max training price or $2000

    # Reject entire forecast if training data spans < 30 days (seasonal regression unreliable)
    unique_months = {date.fromisoformat(r["date"]).month for r in history if r.get("date")}
    if len(unique_months) < 2:
        log.warning("%s: price history spans only 1 calendar month — forecast suppressed to avoid extrapolation",
                    iso.upper())
        return []

    results = []
    for day in load_forecast:
        load_mw = day.get("mean_load_gw", 0) * 1000
        if load_mw <= 0:
            continue
        d = date.fromisoformat(day["date"])
        month = d.month
        x = [
            math.log(max(load_mw, 1)),
            float(d.weekday() < 5),
            math.sin(2 * math.pi * month / 12),
            math.cos(2 * math.pi * month / 12),
        ]
        if use_ng and ng_forecast:
            x.append(math.log(max(ng_forecast, 0.01)))
        x.append(1.0)

        # Only predict if feature count matches (handles use_ng mismatch gracefully)
        if len(x) != len(coeffs):
            continue

        price = math.exp(sum(c * xi for c, xi in zip(coeffs, x)))

        if price > price_ceiling:
            log.warning("%s: predicted price $%.0f/MWh exceeds sanity ceiling $%.0f — suppressing forecast",
                        iso.upper(), price, price_ceiling)
            return []

        results.append({
            "date":           day["date"],
            "forecast_price": round(price, 2),
            "low_price":      round(max(price - rmse, 0), 2),
            "high_price":     round(price + rmse, 2),
            "model_rmse":     rmse,
        })

    log.info("%s: price forecast complete — %d days, model RMSE $%.2f/MWh",
             iso.upper(), len(results), rmse)
    return results
