"""
12-month forward electricity price curve with peak/off-peak split.

Methodology:
  1. NOAA 30-year climate normals (1991-2020) → monthly expected temperature by ISO
  2. Three weather scenarios via historical monthly temperature std dev:
       cold = normal - 1σ,  base = normal,  hot = normal + 1σ
  3. HDD/CDD load model → monthly average load estimate (GW)
  4. OLS log-linear price model fit from 90-day price history, applied forward
  5. Henry Hub forward curve from EIA STEO (free, published monthly, 18-month horizon)
  6. On-peak / off-peak split via seasonal price ratio lookup
       On-peak  = Mon-Fri HE07-22 (NERC definition, ≈45% of hours)
       Off-peak = all other hours; derived by weighted-average constraint
"""

from __future__ import annotations

import json
import logging
import math
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Any

import requests

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# NOAA 30-year climate normals (1991-2020), population-weighted avg °F by ISO
# ---------------------------------------------------------------------------
_NORMALS_F: dict[str, dict[int, float]] = {
    "pjm":   {1: 31.0, 2: 34.0, 3: 43.0, 4: 54.0, 5: 64.0, 6: 73.0,
              7: 78.0, 8: 76.0, 9: 69.0, 10: 57.0, 11: 46.0, 12: 35.0},
    "caiso": {1: 55.0, 2: 57.0, 3: 60.0, 4: 64.0, 5: 68.0, 6: 74.0,
              7: 81.0, 8: 81.0, 9: 76.0, 10: 68.0, 11: 59.0, 12: 54.0},
    "ercot": {1: 47.0, 2: 51.0, 3: 59.0, 4: 67.0, 5: 76.0, 6: 83.0,
              7: 87.0, 8: 87.0, 9: 80.0, 10: 69.0, 11: 58.0, 12: 49.0},
    "miso":  {1: 22.0, 2: 26.0, 3: 38.0, 4: 51.0, 5: 62.0, 6: 71.0,
              7: 76.0, 8: 73.0, 9: 65.0, 10: 53.0, 11: 39.0, 12: 27.0},
    "nyiso": {1: 32.0, 2: 35.0, 3: 43.0, 4: 53.0, 5: 63.0, 6: 72.0,
              7: 77.0, 8: 75.0, 9: 68.0, 10: 56.0, 11: 45.0, 12: 35.0},
    "isone": {1: 28.0, 2: 30.0, 3: 39.0, 4: 49.0, 5: 59.0, 6: 68.0,
              7: 74.0, 8: 72.0, 9: 64.0, 10: 53.0, 11: 42.0, 12: 31.0},
    "spp":   {1: 35.0, 2: 40.0, 3: 50.0, 4: 61.0, 5: 70.0, 6: 79.0,
              7: 85.0, 8: 84.0, 9: 75.0, 10: 63.0, 11: 50.0, 12: 38.0},
}

# Monthly temperature std dev (°F) — scenario spread; winter is more volatile
_TEMP_STD_F: dict[int, float] = {
    1: 4.8, 2: 4.5, 3: 3.8, 4: 3.0, 5: 2.3, 6: 2.0,
    7: 1.8, 8: 1.9, 9: 2.2, 10: 2.9, 11: 3.7, 12: 4.5,
}

# ---------------------------------------------------------------------------
# Load model: base_gw + cdd_coeff × max(temp-65,0) + hdd_coeff × max(65-temp,0)
# Calibrated to typical ISO winter/summer peak and shoulder minimums
# ---------------------------------------------------------------------------
_LOAD_PARAMS: dict[str, tuple[float, float, float]] = {
    #               base_gw  cdd_coeff  hdd_coeff
    "pjm":   (74.0,   0.80,     0.62),
    "caiso": (27.5,   0.38,     0.10),
    "ercot": (42.0,   0.72,     0.28),
    "miso":  (77.0,   0.72,     0.52),
    "nyiso": (18.2,   0.24,     0.17),
    "isone": (14.2,   0.17,     0.14),
    "spp":   (36.5,   0.50,     0.28),
}

# ---------------------------------------------------------------------------
# Peak/off-peak split
# On-peak fraction of total monthly hours: 16h × (5/7 days) / 24h ≈ 0.454
# Peak ratio = on_peak_price / monthly_avg_price  (calibrated to CME market data)
# Off-peak derived from weighted average constraint:
#   f × on_peak + (1-f) × off_peak = monthly_avg
# ---------------------------------------------------------------------------
_PEAK_FRACTION = 0.454

_PEAK_RATIO: dict[str, dict[str, float]] = {
    "pjm":   {"summer": 1.28, "winter": 1.18, "shoulder": 1.11},
    "caiso": {"summer": 1.42, "winter": 1.13, "shoulder": 1.19},
    "ercot": {"summer": 1.52, "winter": 1.24, "shoulder": 1.12},
    "miso":  {"summer": 1.22, "winter": 1.16, "shoulder": 1.09},
    "nyiso": {"summer": 1.32, "winter": 1.22, "shoulder": 1.13},
    "isone": {"summer": 1.28, "winter": 1.20, "shoulder": 1.11},
    "spp":   {"summer": 1.38, "winter": 1.18, "shoulder": 1.10},
}

# Hard cap to prevent regression extrapolation blowup
_PRICE_CAP = 500.0
_PRICE_FLOOR = 1.0

# ---------------------------------------------------------------------------
# Price history cache — long fetches can be slow; cache 24h on disk
# ---------------------------------------------------------------------------
_CACHE_DIR = Path(__file__).parent.parent.parent / "api_cache"
_CACHE_MAX_AGE_H = 24

# Per-ISO *bootstrap* history depth — used only for the very first fetch, when
# no archive exists yet anywhere (local disk or the Space repo). ISOs with
# fast bulk/paginated fetchers bootstrap at 730 days (2 full seasonal
# cycles) in one shot. ISOs with slow per-day/chunked fetchers bootstrap at
# 400 days to keep the first fetch under a few minutes, then grow toward
# the same 730-day depth over time via small incremental top-ups (see
# _load_price_history) — no need to ever repeat a slow multi-hundred-day
# live fetch once an archive exists.
_HISTORY_DAYS: dict[str, int] = {
    "nyiso":  730,   # monthly zip archives, ~21s
    "pjm":    730,   # single paginated API, chunked at 366d (requires PJM_API_KEY)
    "ercot":  730,   # paginated (requires ERCOT_API_KEY)
    "caiso":  400,   # 28-day chunks, 5s inter-chunk delay → ~3 min for 400d
    "miso":   400,   # per-day 1MB files → ~2 min for 400d at 5 workers
    "isone":  400,   # per-day API, no bulk/range endpoint → same profile as MISO
    # spp is per-day file downloads despite the folder-listing step being fast —
    # 730d measured at ~7.5 min live; 400d keeps this in the same ballpark as
    # the other per-day fetchers above.
    "spp":    400,
}

_MAX_ARCHIVE_DAYS   = 730   # cap on how far the growing archive is allowed to reach
_ARCHIVE_TOPUP_DAYS = 14    # incremental fetch window once an archive already exists —
                            # generous buffer over the 6-hourly refresh cadence so a
                            # missed run or two doesn't create a gap


_SPACE_REPO_ID = "JollyAkshay/grid-dashboard"


def _download_from_hub(filename: str) -> None:
    """
    Best-effort pull of a persisted cache file from this Space's own repo
    into the local api_cache/ dir. The 6-hourly refresh_forecast_cache.yml
    Action keeps these current; this is what lets a freshly-restarted
    container start warm instead of either finding nothing (cache miss) or
    doing a live fetch that can silently under-deliver under load/rate
    limits, as happened to PJM the first time this was tested live. Public
    repo, no token needed.
    """
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        hf_hub_download(repo_id=_SPACE_REPO_ID, repo_type="space",
                         filename=filename, local_dir=str(_CACHE_DIR.parent))
        log.info("Pulled %s from the Space repo", filename)
    except Exception as exc:
        log.info("Could not pull %s from the Space repo: %s", filename, exc)


def _archive_path(iso: str) -> Path:
    return _CACHE_DIR / f"{iso}_price_history_archive.json"


def _migrate_legacy_cache(iso: str) -> list | None:
    """
    One-time migration: seed the new growing archive from whatever
    fixed-window cache file this ISO already has (local disk or the Space
    repo), instead of discarding a slow bootstrap fetch that already ran
    this session. Old files used the {iso}_price_history_{days}d.json
    naming scheme, with `days` sometimes 400 and sometimes 730 depending
    on when the file was written (isone/spp moved 730 -> 400 mid-session).
    """
    candidates = sorted({_HISTORY_DAYS.get(iso, 400), 730, 400}, reverse=True)
    for days in candidates:
        legacy_path = _CACHE_DIR / f"{iso}_price_history_{days}d.json"
        if not legacy_path.exists():
            _download_from_hub(f"api_cache/{iso}_price_history_{days}d.json")
        if legacy_path.exists():
            try:
                data = json.loads(legacy_path.read_text())
                if data:
                    log.info("%s: migrated %d rows from legacy cache %s",
                              iso.upper(), len(data), legacy_path.name)
                    return data
            except Exception:
                continue
    return None


def _load_price_history(iso: str) -> list:
    """
    Return price history backed by a growing, persisted archive rather than
    a fixed rolling window. Once bootstrapped, each refresh only fetches the
    last _ARCHIVE_TOPUP_DAYS days and merges them in — so CAISO/MISO/ISO-NE/
    SPP (which bootstrap at 400 days, see _HISTORY_DAYS) grow toward the
    same 730-day depth PJM/ERCOT/NYISO get in one shot, without ever
    repeating a slow multi-hundred-day live fetch.

    First call per ISO may take 1-5 minutes (CAISO/MISO especially) unless
    an existing archive — or a legacy fixed-window cache to migrate from —
    can be pulled from the Space repo instead. Subsequent calls within 24h
    read from local disk in <1s.
    """
    _CACHE_DIR.mkdir(exist_ok=True)
    cache_path = _archive_path(iso)

    def _read_local(ignore_age: bool = False) -> list | None:
        if not cache_path.exists():
            return None
        age_h = (time.time() - cache_path.stat().st_mtime) / 3600
        if not ignore_age and age_h >= _CACHE_MAX_AGE_H:
            return None
        try:
            data = json.loads(cache_path.read_text())
            log.info("%s: price history archive loaded from local cache (%d rows, %.1fh old)",
                      iso.upper(), len(data), age_h)
            return data or None
        except Exception:
            return None

    data = _read_local()
    if data is not None:
        return data

    # Local archive is missing/stale (fresh container restart) — try the
    # Space repo's persisted copy before falling back to a live fetch.
    _download_from_hub(f"api_cache/{iso}_price_history_archive.json")
    data = _read_local()
    if data is not None:
        return data

    from .price_forecast import fetch_price_history, _attach_ng_prices, _attach_renewable_generation  # noqa: PLC0415

    # A stale local/Hub archive (or one seeded from the legacy fixed-window
    # cache) is still a far better top-up base than starting from nothing —
    # only fall through to a full bootstrap fetch if no archive exists
    # anywhere at all.
    existing = _read_local(ignore_age=True)
    if existing is None:
        existing = _migrate_legacy_cache(iso)

    if existing:
        log.info("%s: topping up price history archive (%d existing rows)", iso.upper(), len(existing))
        fresh = fetch_price_history(iso, days=_ARCHIVE_TOPUP_DAYS)
        merged = {r["date"]: r for r in existing}
        for r in fresh:
            merged[r["date"]] = r  # fresh data wins on overlapping dates
        history = sorted(merged.values(), key=lambda r: r["date"])[-_MAX_ARCHIVE_DAYS:]
    else:
        days = _HISTORY_DAYS.get(iso, 400)
        log.info("%s: bootstrapping %d-day price history archive (cache miss)", iso.upper(), days)
        history = fetch_price_history(iso, days=days)

    if history:
        history = _attach_ng_prices(history, iso)
        history = _attach_renewable_generation(history, iso)
        try:
            cache_path.write_text(json.dumps(history))
            log.info("%s: price history archive saved (%d rows)", iso.upper(), len(history))
        except Exception:
            pass
    return history


def get_recent_settled_prices(iso: str, days: int = 90) -> list[dict]:
    """
    Return the trailing `days` of real, already-settled daily wholesale
    prices for `iso` — distinct from build_forward_curve's forward-looking
    strip, which is model-predicted. Reads from the same persisted archive
    _load_price_history maintains, so no extra network fetch on a warm cache.

    Each row: {date, price_usd_mwh, on_peak_usd_mwh, off_peak_usd_mwh}.
    Returns [] if the archive can't be loaded.
    """
    history = _load_price_history(iso)
    if not history:
        return []
    recent = sorted(history, key=lambda r: r["date"])[-days:]
    return [
        {
            "date": r["date"],
            "price_usd_mwh": r.get("price_usd_mwh"),
            "on_peak_usd_mwh": r.get("on_peak_usd_mwh"),
            "off_peak_usd_mwh": r.get("off_peak_usd_mwh"),
        }
        for r in recent
    ]


def _season(month: int) -> str:
    if month in (6, 7, 8):
        return "summer"
    if month in (12, 1, 2):
        return "winter"
    return "shoulder"


def _monthly_load_gw(iso: str, temp_f: float) -> float:
    base, cdd_c, hdd_c = _LOAD_PARAMS.get(iso, (50.0, 0.5, 0.4))
    cdd = max(temp_f - 65.0, 0.0)
    hdd = max(65.0 - temp_f, 0.0)
    return base + cdd_c * cdd + hdd_c * hdd


def _split_peak_offpeak(monthly_avg: float, iso: str, month: int) -> tuple[float, float]:
    """Return (on_peak, off_peak) $/MWh satisfying weighted average = monthly_avg."""
    ratio = _PEAK_RATIO.get(iso, _PEAK_RATIO["pjm"])[_season(month)]
    f = _PEAK_FRACTION
    on_peak = monthly_avg * ratio
    off_peak = (monthly_avg - f * on_peak) / (1.0 - f)
    return round(max(on_peak, _PRICE_FLOOR), 2), round(max(off_peak, _PRICE_FLOOR), 2)


def _add_months(d: date, n: int) -> date:
    """Advance a date by n calendar months."""
    month = d.month - 1 + n
    year  = d.year + month // 12
    month = month % 12 + 1
    import calendar
    day = min(d.day, calendar.monthrange(year, month)[1])
    return date(year, month, day)


# ---------------------------------------------------------------------------
# Henry Hub forward curve — EIA STEO (free, monthly publication, 18m horizon)
# ---------------------------------------------------------------------------

_HH_FUTURES_CACHE = _CACHE_DIR / "hh_futures_cache.json"


def _write_hh_futures_cache(result: dict[str, float]) -> None:
    try:
        _CACHE_DIR.mkdir(exist_ok=True)
        _HH_FUTURES_CACHE.write_text(json.dumps(result))
    except OSError:
        log.warning("Could not write HH futures cache to %s", _HH_FUTURES_CACHE)


def fetch_hh_futures(n_months: int = 14) -> dict[str, float]:
    """
    Return {YYYY-MM: $/MMBtu} from EIA STEO Henry Hub forecast.
    Falls back to latest EIA spot price held flat if STEO unavailable.

    Henry Hub is a single national price — identical for every ISO — so this
    is cached once and shared across all 7 ISOs instead of being refetched
    from EIA on every build_forward_curve() call. Only genuine EIA responses
    (STEO or spot-fallback) are cached; the last-resort $3.50 placeholder is
    never cached, so a transient EIA hiccup doesn't freeze a fake price for
    24 hours.
    """
    if _HH_FUTURES_CACHE.exists():
        try:
            age_h = (time.time() - _HH_FUTURES_CACHE.stat().st_mtime) / 3600
            if age_h < _CACHE_MAX_AGE_H:
                cached = json.loads(_HH_FUTURES_CACHE.read_text())
                if cached:
                    return cached
        except Exception:
            pass

    api_key = os.environ.get("EIA_API_KEY", "")
    today   = date.today()
    current_ym = today.strftime("%Y-%m")
    result: dict[str, float] = {}

    if api_key:
        try:
            # Request 6 months before current so we always have ≥1 actual price
            # for context; filter to >= current_ym in the loop
            lookback_ym = _add_months(today, -6).strftime("%Y-%m")
            r = requests.get(
                "https://api.eia.gov/v2/steo/data/",
                params={
                    "api_key":            api_key,
                    "frequency":          "monthly",
                    "data[0]":            "value",
                    "facets[seriesId][]": "NGHHUUS",   # HH spot + STEO forecast
                    "sort[0][column]":    "period",
                    "sort[0][direction]": "asc",
                    "start":              lookback_ym,
                    "length":             n_months + 8,
                },
                timeout=15,
            )
            for row in r.json().get("response", {}).get("data", []):
                period = row.get("period", "")
                value  = row.get("value")
                if period >= current_ym and value is not None:
                    result[period] = float(value)
            if result:
                log.info("HH futures: loaded %d months from EIA STEO", len(result))
                _write_hh_futures_cache(result)
                return result
        except Exception:
            log.warning("EIA STEO fetch failed — falling back to spot price")

    # Fallback: latest HH spot price, held flat
    if api_key:
        try:
            r = requests.get(
                "https://api.eia.gov/v2/natural-gas/pri/sum/dcu/nus/monthly/data/",
                params={
                    "api_key":            api_key,
                    "frequency":          "monthly",
                    "data[0]":            "value",
                    "sort[0][column]":    "period",
                    "sort[0][direction]": "desc",
                    "length":             3,
                },
                timeout=15,
            )
            rows = r.json().get("response", {}).get("data", [])
            if rows and rows[0].get("value") is not None:
                spot = float(rows[0]["value"])
                for i in range(n_months):
                    d = _add_months(today, i)
                    result[d.strftime("%Y-%m")] = spot
                log.info("HH futures: using flat spot price $%.2f from EIA", spot)
                _write_hh_futures_cache(result)
                return result
        except Exception:
            log.warning("EIA HH spot fallback also failed")

    # Last resort: hardcoded $3.50 flat — never cached, so the next call retries EIA
    log.warning("HH futures: no EIA data — using $3.50 placeholder")
    for i in range(n_months):
        d = _add_months(today, i)
        result[d.strftime("%Y-%m")] = 3.50
    return result


# ---------------------------------------------------------------------------
# Price prediction for a single month
# ---------------------------------------------------------------------------

def _predict_monthly_price(
    load_mw: float,
    month: int,
    ng_price: float,
    model: dict | None,
    iso: str,
    renewable_gw: float | None = None,
) -> tuple[float, float | None, float | None]:
    """
    Predict monthly average price ($/MWh).
    Returns (price, low, high) — low/high are CQR-calibrated quantile bounds
    when the model has them (None for the fallback heuristic, which has no
    statistical uncertainty estimate of its own).
    """
    if model:
        coeffs = model["coeffs"]
        use_ng = model.get("use_ng", False)
        load_feature = load_mw
        if model.get("use_net_load") and renewable_gw is not None:
            load_feature = max(load_mw - renewable_gw * 1000, 100.0)
        feature_values = {
            "log_load":     math.log(max(load_feature, 1.0)),
            "weekday_frac": 0.71,   # fraction of weekdays in a typical month
            "sin_month":    math.sin(2 * math.pi * month / 12),
            "cos_month":    math.cos(2 * math.pi * month / 12),
            "log_ng":       math.log(max(ng_price, 0.01)) if use_ng else None,
            "intercept":    1.0,
        }
        # feature_names records which columns the model was actually fit on
        # (e.g. weekday_frac is dropped when a training subset makes it a
        # constant, such as an on-peak-hours-only fit — see _fit_price_model)
        # — build x to match exactly, rather than assuming the full set.
        default_names = ["log_load", "weekday_frac", "sin_month", "cos_month"] + \
                         (["log_ng"] if use_ng else []) + ["intercept"]
        names = model.get("feature_names", default_names)
        x = [feature_values[n] for n in names]
        if len(x) == len(coeffs):
            raw = math.exp(sum(c * xi for c, xi in zip(coeffs, x)))
            price = min(max(raw, _PRICE_FLOOR), _PRICE_CAP)
            from .price_forecast import _price_bounds  # noqa: PLC0415
            low, high = _price_bounds(x, model, price, month)
            return price, min(max(low, _PRICE_FLOOR), _PRICE_CAP), min(max(high, _PRICE_FLOOR), _PRICE_CAP)

    # Fallback heuristic: base price + gas pass-through + load-above-base sensitivity
    # Uses the ISO's own typical base load (not a fixed 50 GW) so small ISOs respond correctly
    _BASE_PRICE = {"pjm": 35, "caiso": 42, "ercot": 33, "miso": 30, "nyiso": 46, "isone": 50, "spp": 28}
    base_price = _BASE_PRICE.get(iso, 35)
    base_gw    = _LOAD_PARAMS.get(iso, (50.0, 0.5, 0.4))[0]
    load_gw    = load_mw / 1000.0
    load_above = max(load_gw - base_gw, 0.0)
    # Marginal cost ramp: $1.50/MWh per GW above base load (approximates merit-order steepening)
    price = base_price + ng_price * 5.0 + load_above * 1.5
    return float(min(max(price, _PRICE_FLOOR), _PRICE_CAP)), None, None


# ---------------------------------------------------------------------------
# Main entry point
# ---------------------------------------------------------------------------

def build_forward_curve(iso: str, n_months: int = 12, history: list | None = None) -> dict[str, Any]:
    """
    Build a 12-month forward electricity price curve for the given ISO.

    Parameters
    ----------
    history:
        Pre-loaded price history, if the caller already has it (e.g. the
        dashboard also needs it for backtest_forward_curve in the same
        request — pass it to both instead of hitting _load_price_history
        twice). Loaded internally when omitted.

    Returns
    -------
    dict with keys:
        iso, horizon_months, model_source, gas_curve, curve

    Each curve entry has: month, season, climate_normal_f, scenario_spread_f,
    scenarios → {cold, base, hot} → {monthly_avg, on_peak, off_peak} ($/MWh)
    """
    # Fit OLS price model from 730-day cached price history
    model: dict | None = None
    try:
        from .price_forecast import _fit_price_model  # noqa: PLC0415
        if not history:
            history = _load_price_history(iso)
        if history:
            model = _fit_price_model(history)
    except Exception:
        log.warning("%s: price model fit failed — using fallback heuristic", iso.upper())
    history = history or []   # guarantee a list even if the fetch above raised

    # Reject model if training data spans < 6 calendar months — need full seasonal cycle
    if model and history:
        unique_months = {date.fromisoformat(r["date"]).month for r in history if r.get("date")}
        if len(unique_months) < 6:
            log.warning("%s: price history spans only %d months — seasonal OLS unreliable, using heuristic",
                        iso.upper(), len(unique_months))
            model = None

    # Separate on-peak/off-peak price models, fit from genuine NERC-peak-hour
    # vs off-peak-hour historical data (currently only PJM's history has the
    # on_peak_usd_mwh/off_peak_usd_mwh/on_peak_load_mw/off_peak_load_mw
    # fields this needs — see _fetch_pjm_price_history). Falls back to
    # _split_peak_offpeak's synthetic ratio (below) when unavailable, exactly
    # as before for every other ISO.
    peak_model = offpeak_model = None
    peak_load_ratio = offpeak_load_ratio = None
    peak_ceiling = offpeak_ceiling = None
    if history and any(r.get("on_peak_usd_mwh") is not None for r in history):
        try:
            from .price_forecast import _fit_peak_offpeak_price_models  # noqa: PLC0415
            peak_model, offpeak_model = _fit_peak_offpeak_price_models(history)
        except Exception:
            log.warning("%s: peak/off-peak price model fit failed — using synthetic ratio split", iso.upper())

        # Empirical peak-load / off-peak-load ratio to the blended monthly
        # load estimate _monthly_load_gw already produces — the fitted
        # models' own coefficients capture the peak-vs-off-peak price
        # relationship; this just gives them a peak-shaped (or off-peak
        # -shaped) load input to predict from, instead of the blended one.
        peak_ratios, offpeak_ratios = [], []
        for r in history:
            load = r.get("load_mw")
            if not load:
                continue
            if r.get("on_peak_load_mw") is not None:
                peak_ratios.append(r["on_peak_load_mw"] / load)
            if r.get("off_peak_load_mw") is not None:
                offpeak_ratios.append(r["off_peak_load_mw"] / load)
        peak_load_ratio = sum(peak_ratios) / len(peak_ratios) if peak_ratios else 1.0
        offpeak_load_ratio = sum(offpeak_ratios) / len(offpeak_ratios) if offpeak_ratios else 1.0

        peak_prices = [r["on_peak_usd_mwh"] for r in history if r.get("on_peak_usd_mwh")]
        offpeak_prices = [r["off_peak_usd_mwh"] for r in history if r.get("off_peak_usd_mwh")]
        peak_ceiling = max(max(peak_prices) * 5, 300.0) if peak_prices else 300.0
        offpeak_ceiling = max(max(offpeak_prices) * 5, 300.0) if offpeak_prices else 300.0

    # Seasonal renewable climatology (monthly average from the same history
    # the model was fit on) — the forward curve predicts a delivery month
    # 1-12 months out, not a specific day, so there's no "forecast" of
    # renewable output to use the way the 15-day dashboard panel can; the
    # historical monthly average is the equivalent of the weather normals
    # already used for temperature.
    renewable_by_month: dict[int, float] = {}
    if model and model.get("use_net_load") and history:
        from collections import defaultdict
        vals_by_month: dict[int, list] = defaultdict(list)
        for r in history:
            if r.get("renewable_gw") is not None:
                vals_by_month[date.fromisoformat(r["date"]).month].append(r["renewable_gw"])
        renewable_by_month = {m: sum(v) / len(v) for m, v in vals_by_month.items() if v}

    # Sanity ceiling: 5× max training price or $300, whichever is larger.
    # Was reading a "price" key that no history row has ever had (always
    # "price_usd_mwh") — train_prices was always empty, so every ISO got the
    # flat $300 fallback regardless of its real price range, silently
    # discarding legitimate high predictions (and their new CQR bounds) for
    # ISOs like ISO-NE that have genuinely settled above $400/MWh.
    train_prices = [r["price_usd_mwh"] for r in history if r.get("price_usd_mwh") and r["price_usd_mwh"] > 0]
    _price_ceiling = max(max(train_prices) * 5, 300.0) if train_prices else 300.0

    # Henry Hub forward curve
    gas_curve = fetch_hh_futures(n_months + 3)

    normals = _NORMALS_F.get(iso, _NORMALS_F["pjm"])
    today   = date.today()

    curve: list[dict] = []
    gas_assumptions: dict[str, float] = {}

    for i in range(n_months):
        delivery = _add_months(today, i + 1)   # first delivery month = next month
        ym    = delivery.strftime("%Y-%m")
        month = delivery.month

        normal_temp = normals.get(month, 55.0)
        std_temp    = _TEMP_STD_F.get(month, 3.0)

        ng_price = gas_curve.get(ym)
        if ng_price is None:
            # Use nearest available month if exact key missing
            ng_price = next(iter(gas_curve.values()), 3.50) if gas_curve else 3.50
        gas_assumptions[ym] = round(ng_price, 2)

        month_renewable_gw = renewable_by_month.get(month)

        scenarios: dict[str, dict] = {}
        for scenario_name, temp_offset in [("cold", -std_temp), ("base", 0.0), ("hot", +std_temp)]:
            temp    = normal_temp + temp_offset
            load_gw = _monthly_load_gw(iso, temp)
            price, low, high = _predict_monthly_price(
                load_gw * 1000, month, ng_price, model, iso, month_renewable_gw)
            # If OLS blows up for this month, fall back to heuristic
            if price > _price_ceiling:
                log.warning("%s %s: OLS price $%.0f exceeds ceiling $%.0f — using heuristic",
                            iso.upper(), ym, price, _price_ceiling)
                price, low, high = _predict_monthly_price(load_gw * 1000, month, ng_price, None, iso)

            on_peak_low = on_peak_high = off_peak_low = off_peak_high = None
            peak_split_method = "synthetic_ratio"
            if peak_model and offpeak_model:
                on_peak, on_peak_low, on_peak_high = _predict_monthly_price(
                    load_gw * 1000 * peak_load_ratio, month, ng_price, peak_model, iso, month_renewable_gw)
                if on_peak > peak_ceiling:
                    on_peak, on_peak_low, on_peak_high = _predict_monthly_price(
                        load_gw * 1000 * peak_load_ratio, month, ng_price, None, iso)
                off_peak, off_peak_low, off_peak_high = _predict_monthly_price(
                    load_gw * 1000 * offpeak_load_ratio, month, ng_price, offpeak_model, iso, month_renewable_gw)
                if off_peak > offpeak_ceiling:
                    off_peak, off_peak_low, off_peak_high = _predict_monthly_price(
                        load_gw * 1000 * offpeak_load_ratio, month, ng_price, None, iso)
                on_peak, off_peak = round(on_peak, 2), round(off_peak, 2)
                peak_split_method = "empirical_hourly"
            else:
                on_peak, off_peak = _split_peak_offpeak(price, iso, month)

            # Spark spread and implied heat rate — computed from displayed (rounded)
            # values so traders can reproduce every number independently:
            #   IHR (BTU/kWh) = monthly_avg × 1000 / ng_price_mmbtu
            #   spark_ccgt    = monthly_avg − 7.0  × ng_price_mmbtu
            #   spark_ct      = monthly_avg − 10.0 × ng_price_mmbtu
            p_r  = round(price, 2)
            ng_r = round(ng_price, 2)
            op_r = on_peak    # already rounded above (either path)
            fp_r = off_peak
            implied_hr = round(p_r  * 1000 / ng_r, 0) if ng_r > 0 else None
            spark_ccgt = round(p_r  - 7.0  * ng_r, 2)
            spark_ct   = round(p_r  - 10.0 * ng_r, 2)
            on_pk_hr   = round(op_r * 1000 / ng_r, 0) if ng_r > 0 else None
            off_pk_hr  = round(fp_r * 1000 / ng_r, 0) if ng_r > 0 else None

            scenarios[scenario_name] = {
                "avg_temp_f":          round(temp, 1),
                "load_gw":             round(load_gw, 2),
                "monthly_avg":         round(price, 2),
                "low_usd_mwh":         round(low, 2) if low is not None else None,
                "high_usd_mwh":        round(high, 2) if high is not None else None,
                "peak_split_method":   peak_split_method,
                "on_peak_low_usd_mwh":   round(on_peak_low, 2) if on_peak_low is not None else None,
                "on_peak_high_usd_mwh":  round(on_peak_high, 2) if on_peak_high is not None else None,
                "off_peak_low_usd_mwh":  round(off_peak_low, 2) if off_peak_low is not None else None,
                "off_peak_high_usd_mwh": round(off_peak_high, 2) if off_peak_high is not None else None,
                "on_peak":             on_peak,
                "off_peak":            off_peak,
                "implied_heat_rate":   implied_hr,   # BTU/kWh; compare to your plant HR
                "spark_spread_ccgt":   spark_ccgt,   # $/MWh at 7,000 BTU/kWh
                "spark_spread_ct":     spark_ct,     # $/MWh at 10,000 BTU/kWh
                "on_peak_heat_rate":   on_pk_hr,
                "off_peak_heat_rate":  off_pk_hr,
            }

        curve.append({
            "month":             ym,
            "season":            _season(month),
            "climate_normal_f":  normal_temp,
            "scenario_spread_f": round(std_temp, 1),
            "ng_price_mmbtu":    round(ng_price, 2),
            "scenarios":         scenarios,
        })

    log.info("%s: forward curve built — %d months, model=%s, HH months=%d",
             iso.upper(), len(curve), "OLS" if model else "heuristic", len(gas_curve))

    result = {
        "iso":            iso,
        "horizon_months": n_months,
        "model_source":   "ols-log-linear" if model else "fallback-heuristic",
        "gas_curve":      gas_assumptions,
        "curve":          curve,
    }

    _archive_snapshot(iso, result)
    return result


def _archive_snapshot(iso: str, result: dict) -> None:
    """Append today's forward curve to the per-ISO JSON Lines archive file."""
    _CACHE_DIR.mkdir(exist_ok=True)
    archive_path = _CACHE_DIR / f"{iso}_forward_curve_history.jsonl"
    snapshot_date = date.today().isoformat()
    record = {"snapshot_date": snapshot_date, **result}
    try:
        # Deduplicate: if today's snapshot already exists, overwrite that line
        existing: list[str] = []
        if archive_path.exists():
            existing = [
                line for line in archive_path.read_text().splitlines()
                if line.strip() and f'"snapshot_date": "{snapshot_date}"' not in line
            ]
        existing.append(json.dumps(record))
        archive_path.write_text("\n".join(existing) + "\n")
    except Exception:
        log.warning("%s: failed to write forward curve archive", iso.upper())


def load_forward_curve_history(iso: str, from_month: str = "", to_month: str = "") -> list[dict]:
    """
    Read archived forward curve snapshots for an ISO.

    Parameters
    ----------
    from_month : YYYY-MM  (inclusive, optional)
    to_month   : YYYY-MM  (inclusive, optional)

    Returns list of snapshot dicts, each containing the full forward curve
    as it was built on that date. Useful for backtesting model accuracy:
    compare what the curve predicted for delivery month M on snapshot day D
    against the actual settlement price.
    """
    archive_path = _CACHE_DIR / f"{iso}_forward_curve_history.jsonl"
    if not archive_path.exists():
        # Fresh container restart — local disk has nothing yet, but the
        # scheduled Action keeps a persisted copy on the Space repo itself.
        _download_from_hub(f"api_cache/{iso}_forward_curve_history.jsonl")
    if not archive_path.exists():
        return []

    snapshots = []
    for line in archive_path.read_text().splitlines():
        if not line.strip():
            continue
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        sd = rec.get("snapshot_date", "")
        if from_month and sd[:7] < from_month:
            continue
        if to_month and sd[:7] > to_month:
            continue
        snapshots.append(rec)

    return sorted(snapshots, key=lambda r: r.get("snapshot_date", ""))


def _month_diff(from_ym: str, to_ym: str) -> int:
    """Calendar months between two YYYY-MM strings (e.g. 2026-06 -> 2026-09 = 3)."""
    fy, fm = int(from_ym[:4]), int(from_ym[5:7])
    ty, tm = int(to_ym[:4]), int(to_ym[5:7])
    return (ty - fy) * 12 + (tm - fm)


def backtest_forward_curve(iso: str, history: list | None = None) -> dict[str, Any]:
    """
    Compare archived forward-curve snapshots against realized settlement
    prices, for delivery months that have since fully completed.

    Realized monthly averages come from the same cached price history used
    to fit the model. A month only counts as "realized" once it's fully in
    the past (never the current month) and has at least 20 days of price
    data — a handful of scattered days isn't a reliable settlement average.

    Parameters
    ----------
    history:
        Pre-loaded price history, if the caller already has it — see
        build_forward_curve's identical parameter. Loaded internally when
        omitted.

    Returns
    -------
    dict with:
        iso, n_snapshots, n_comparisons, mape, bias_usd_mwh,
        by_lead_months: {lead_months: {n, mape, bias_usd_mwh}} — accuracy
            broken out by how many months ahead the prediction was made,
            since a 1-month-ahead prediction and a 12-month-ahead one
            aren't the same claim and shouldn't be averaged together,
        records: the individual predicted-vs-actual comparisons.
    """
    snapshots = load_forward_curve_history(iso)
    empty = {"iso": iso, "n_snapshots": len(snapshots), "n_comparisons": 0,
             "mape": None, "bias_usd_mwh": None, "by_lead_months": {}, "records": []}
    if not snapshots:
        return empty

    if not history:
        history = _load_price_history(iso)
    monthly_actuals: dict[str, list[float]] = {}
    for row in history:
        d = row.get("date", "")
        price = row.get("price_usd_mwh")
        if d and price is not None:
            monthly_actuals.setdefault(d[:7], []).append(price)

    current_ym = date.today().strftime("%Y-%m")
    realized: dict[str, float] = {
        ym: sum(prices) / len(prices)
        for ym, prices in monthly_actuals.items()
        if ym < current_ym and len(prices) >= 20
    }

    records: list[dict] = []
    for snap in snapshots:
        snap_date = snap.get("snapshot_date", "")
        snap_ym = snap_date[:7]
        if not snap_ym:
            continue
        for entry in snap.get("curve", []):
            ym = entry.get("month")
            if not ym or ym not in realized:
                continue
            predicted = entry.get("scenarios", {}).get("base", {}).get("monthly_avg")
            if predicted is None:
                continue
            actual = realized[ym]
            error = predicted - actual
            records.append({
                "snapshot_date":     snap_date,
                "delivery_month":    ym,
                "lead_months":       _month_diff(snap_ym, ym),
                "predicted_usd_mwh": round(predicted, 2),
                "actual_usd_mwh":    round(actual, 2),
                "error_usd_mwh":     round(error, 2),
                "pct_error":         round(error / actual * 100, 1) if actual else None,
            })

    if not records:
        return empty

    def _mape(recs: list[dict]) -> float | None:
        pcts = [abs(r["pct_error"]) for r in recs if r["pct_error"] is not None]
        return round(sum(pcts) / len(pcts), 1) if pcts else None

    def _bias(recs: list[dict]) -> float:
        return round(sum(r["error_usd_mwh"] for r in recs) / len(recs), 2)

    by_lead: dict[int, list[dict]] = {}
    for r in records:
        by_lead.setdefault(r["lead_months"], []).append(r)

    return {
        "iso":             iso,
        "n_snapshots":     len(snapshots),
        "n_comparisons":   len(records),
        "mape":            _mape(records),
        "bias_usd_mwh":    _bias(records),
        "by_lead_months":  {
            lead: {"n": len(recs), "mape": _mape(recs), "bias_usd_mwh": _bias(recs)}
            for lead, recs in sorted(by_lead.items())
        },
        "records": sorted(records, key=lambda r: (r["snapshot_date"], r["delivery_month"])),
    }
