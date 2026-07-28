"""
Day-ahead electricity price fetching and regression-based forecast.

Data sources:
  - CAISO: CAISO OASIS PRC_LMP (DAM, TH_NP15_GEN-APND + TH_SP15_GEN-APND) + EIA load
  - MISO: docs.misoenergy.org YYYYMMDD_da_exante_lmp.csv (8 hub nodes) + EIA load
  - PJM / ERCOT: EIA v2 wholesale-markets/prices + rto/region-data (endpoint defunct as of 2026)
  - NYISO: NYISO public day-ahead zone LMP CSVs (mis.nyiso.com) + EIA load
  - ISO-NE: ISO-NE webservices REST API (configure _ISONE_API_USER / _ISONE_API_PASS)
  - SPP: SPP Marketplace public DA-LMP-SL files (SPPNORTH_HUB + SPPSOUTH_HUB) + EIA load
  - NG price: EIA Henry Hub daily spot price ($/MMBtu) as regression co-variate

Forecast: multivariate OLS — log(price) ~ log(load) + weekday + seasonal + log(ng_price)
"""

import csv
import io
import logging
import math
import os
import zipfile
from concurrent.futures import ThreadPoolExecutor
from datetime import date, timedelta

import numpy as np
import requests

log = logging.getLogger(__name__)

_EIA_PRICE_URL  = "https://api.eia.gov/v2/electricity/wholesale-markets/prices/data/"
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

# EIA respondent codes for load and price (EIA wholesale prices endpoint)
# SPP is NOT listed here — EIA does not collect SPP wholesale prices; use SPP Marketplace instead
_ISO_EIA = {
    "pjm":   {"load_region": "PJM",  "price_region": "PJM",  "price_type": "DA"},
    "caiso": {"load_region": "CAL",  "price_region": "CAL",  "price_type": "DA"},
    "ercot": {"load_region": "TEX",  "price_region": "TEX",  "price_type": "DA"},
    "miso":  {"load_region": "MISO", "price_region": "MIDW", "price_type": "DA"},
}


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
    """Return {date_str: avg_mw} from EIA hourly demand."""
    rows = _eia_get(_EIA_LOAD_URL, {
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": respondent,
        "facets[type][]":       "D",
        "start": start, "end": end, "length": 5000,
    })
    daily: dict = {}
    for row in rows:
        d = row.get("period", "")[:10]
        v = row.get("value")
        if d and v is not None:
            daily.setdefault(d, []).append(float(v))
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
        "length":           500,
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

def _fetch_nyiso_price_history(days: int = 90) -> list:
    """
    Fetch NYISO day-ahead zone LMP history from NYISO's public CSV files.
    No authentication required. Pairs with EIA load for the regression model.
    """
    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.headers["User-Agent"] = "grid-dashboard/1.0"

    def _fetch_day(d: date):
        url = _NYISO_LMP_URL.format(date=d.strftime("%Y%m%d"))
        try:
            r = session.get(url, timeout=15)
            if r.status_code != 200:
                return d.isoformat(), None
            reader = csv.DictReader(io.StringIO(r.text))
            prices = []
            for row in reader:
                try:
                    prices.append(float(row["LBMP ($/MWHr)"]))
                except (KeyError, ValueError, TypeError):
                    pass
            return d.isoformat(), (sum(prices) / len(prices)) if prices else None
        except Exception:
            return d.isoformat(), None

    dates = [end_dt - timedelta(days=i) for i in range(days)]
    price_by_date: dict = {}
    with ThreadPoolExecutor(max_workers=8) as pool:
        for d_str, price in pool.map(_fetch_day, dates):
            if price is not None:
                price_by_date[d_str] = price

    daily_load = _eia_load_daily("NYIS", start_dt.isoformat(), end_dt.isoformat())

    result = []
    for d in sorted(set(price_by_date) & set(daily_load)):
        result.append({
            "date":          d,
            "load_mw":       round(daily_load[d]),
            "price_usd_mwh": round(price_by_date[d], 2),
        })

    log.info("NYISO: fetched %d days of price history (NYISO CSV + EIA load)", len(result))
    return result


def _fetch_isone_price_history(days: int = 90) -> list:
    """
    Fetch ISO-NE day-ahead hub LMP from ISO-NE webservices REST API.
    Requires free account at iso-ne.com — set ISONE_API_USER + ISONE_API_PASS env vars.
    Hub: Massachusetts Hub (node 4000).
    """
    if not _ISONE_API_USER or not _ISONE_API_PASS:
        log.info("ISO-NE: no API credentials — set ISONE_API_USER + ISONE_API_PASS")
        return []

    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days - 1)
    session  = requests.Session()
    session.auth = (_ISONE_API_USER, _ISONE_API_PASS)
    session.headers["Accept"] = "application/json"

    price_by_date: dict = {}
    try:
        url = (f"{_ISONE_API_BASE}/hourlylmp/da/final/hourly/"
               f"{start_dt.strftime('%Y%m%d')}/{end_dt.strftime('%Y%m%d')}.json")
        r = session.get(url, timeout=30)
        r.raise_for_status()
        for entry in r.json().get("HourlyLmps", {}).get("HourlyLmp", []):
            if str(entry.get("Location", {}).get("@LocId", "")) != "4000":
                continue
            period = entry.get("BeginDate", "")[:10]
            lmp = entry.get("LmpTotal")
            if period and lmp is not None:
                price_by_date.setdefault(period, []).append(float(lmp))
    except Exception:
        log.exception("ISO-NE: price fetch failed")
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


def _fetch_spp_price_history(days: int = 90) -> list:
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


def _fetch_caiso_price_history(days: int = 90) -> list:
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
        except Exception:
            log.warning("CAISO OASIS: request failed for %s -> %s", start_str, end_str)
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


def fetch_price_history(iso: str, days: int = 90) -> list:
    """
    Return [{date, load_mw, price_usd_mwh}] for the last `days` days.
    Dispatches to ISO-specific fetchers for CAISO, NYISO, ISO-NE, and SPP.
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

    cfg = _ISO_EIA.get(iso)
    if not cfg:
        return []

    end_dt   = date.today() - timedelta(days=1)
    start_dt = end_dt - timedelta(days=days)
    start, end = start_dt.isoformat(), end_dt.isoformat()

    load_rows = _eia_get(_EIA_LOAD_URL, {
        "frequency":            "hourly",
        "data[0]":              "value",
        "facets[respondent][]": cfg["load_region"],
        "facets[type][]":       "D",
        "start": start, "end": end, "length": 5000,
    })
    price_rows = _eia_get(_EIA_PRICE_URL, {
        "frequency":            "hourly",
        "data[0]":              "price",
        "facets[respondent][]": cfg["price_region"],
        "facets[type][]":       cfg["price_type"],
        "start": start, "end": end, "length": 5000,
    })

    daily_load:  dict = {}
    daily_price: dict = {}

    for row in load_rows:
        d = row.get("period", "")[:10]
        v = row.get("value")
        if d and v is not None:
            daily_load.setdefault(d, []).append(float(v))

    for row in price_rows:
        d = row.get("period", "")[:10]
        v = row.get("price")
        if d and v is not None:
            try:
                daily_price.setdefault(d, []).append(float(v))
            except (TypeError, ValueError):
                pass

    result = []
    for d in sorted(set(daily_load) & set(daily_price)):
        load_vals  = daily_load[d]
        price_vals = daily_price[d]
        result.append({
            "date":          d,
            "load_mw":       round(sum(load_vals)  / len(load_vals)),
            "price_usd_mwh": round(sum(price_vals) / len(price_vals), 2),
        })

    log.info("%s: fetched %d days of load+price history from EIA", iso.upper(), len(result))
    return result


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
