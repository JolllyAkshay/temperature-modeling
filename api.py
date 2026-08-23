"""
GridForecast REST API

Run:   uvicorn api:app --reload --port 8001
Docs:  http://localhost:8001/docs

Authentication: pass your API key in the X-API-Key request header.
Keys are stored in GRID_API_KEYS env var (comma-separated list).
"""

import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

import asyncio
import sys

from fastapi import BackgroundTasks, FastAPI, HTTPException, Security
from fastapi.middleware.cors import CORSMiddleware
from fastapi.security.api_key import APIKeyHeader

log = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# App and middleware
# ---------------------------------------------------------------------------

app = FastAPI(
    title="GridForecast API",
    description=(
        "15-day electricity load forecasts, day-ahead price outlooks, "
        "carbon intensity, and live model accuracy for all 7 major US ISOs."
    ),
    version="0.1.0",
    contact={"email": "akshaypradipjain@gmail.com"},
    license_info={"name": "Proprietary"},
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["GET"],
    allow_headers=["*"],
)

# ---------------------------------------------------------------------------
# Auth
# ---------------------------------------------------------------------------

_API_KEY_HEADER = APIKeyHeader(name="X-API-Key", auto_error=False)

_VALID_KEYS: set[str] = {
    k.strip()
    for k in os.environ.get("GRID_API_KEYS", "").split(",")
    if k.strip()
}


def _require_key(key: str | None = Security(_API_KEY_HEADER)) -> str:
    if not key or key not in _VALID_KEYS:
        raise HTTPException(status_code=401, detail="Invalid or missing X-API-Key header.")
    return key


# ---------------------------------------------------------------------------
# Cache helpers
# ---------------------------------------------------------------------------

_HERE            = Path(__file__).parent
_CACHE_DIR       = _HERE / "api_cache"
_CACHE_MAX_AGE_H = 24  # serve cache up to this old; refresh endpoint or dashboard regenerates it

_ISO_CACHE_FILES: dict[str, Path] = {
    iso: _CACHE_DIR / f"{iso}_forecast_cache.json"
    for iso in ("pjm", "caiso", "ercot", "miso", "nyiso", "isone", "spp")
}

_SUPPORTED_ISOS = sorted(_ISO_CACHE_FILES)


def _read_cache(iso: str) -> dict:
    """Return parsed cache dict, or raise 503 if missing/too stale."""
    path = _ISO_CACHE_FILES.get(iso)
    if not path or not path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Forecast for {iso.upper()} not yet available. "
                   "Start the dashboard once to populate the cache.",
        )
    age_h = (time.time() - path.stat().st_mtime) / 3600
    if age_h > _CACHE_MAX_AGE_H:
        raise HTTPException(
            status_code=503,
            detail=f"Cached forecast for {iso.upper()} is {age_h:.1f} h old (limit {_CACHE_MAX_AGE_H} h). "
                   "Restart the dashboard or call /v1/refresh/{iso}.",
        )
    try:
        return json.loads(path.read_text())
    except (json.JSONDecodeError, OSError) as exc:
        raise HTTPException(status_code=503, detail=f"Cache read error: {exc}")


def _cache_meta(iso: str) -> dict:
    path = _ISO_CACHE_FILES.get(iso)
    if not path or not path.exists():
        return {}
    mtime = path.stat().st_mtime
    return {
        "generated_at": datetime.fromtimestamp(mtime, tz=timezone.utc).isoformat(),
        "cache_age_hours": round((time.time() - mtime) / 3600, 2),
    }


def _validate_iso(iso: str) -> str:
    iso = iso.lower()
    if iso not in _ISO_CACHE_FILES:
        raise HTTPException(
            status_code=400,
            detail=f"Unknown ISO '{iso}'. Supported: {', '.join(_SUPPORTED_ISOS)}",
        )
    return iso


# ---------------------------------------------------------------------------
# Routes — public
# ---------------------------------------------------------------------------

@app.get("/", tags=["meta"])
def root():
    """API info and available endpoints."""
    return {
        "api":     "GridForecast",
        "version": "0.1.0",
        "docs":    "/docs",
        "isos":    _SUPPORTED_ISOS,
        "endpoints": {
            "forecast":               "/v1/forecast/{iso}",
            "prices":                 "/v1/prices/{iso}",
            "forward_curve":          "/v1/forward-curve/{iso}",
            "forward_curve_history":  "/v1/forward-curve/history/{iso}",
            "forward_curve_accuracy": "/v1/forward-curve/accuracy/{iso}",
            "futures_pricer":         "/v1/futures-pricer/{iso}",
            "net_load":               "/v1/net-load/{iso}",
            "carbon":                 "/v1/carbon/{iso}",
            "accuracy":               "/v1/accuracy/{iso}",
        },
    }


@app.get("/health", tags=["meta"])
def health():
    """Liveness probe."""
    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Routes — authenticated
# ---------------------------------------------------------------------------

@app.get("/v1/isos", tags=["meta"])
def list_isos(_key: str = Security(_require_key)):
    """List all supported ISOs and their cache status."""
    isos = []
    for iso in _SUPPORTED_ISOS:
        path = _ISO_CACHE_FILES[iso]
        if path.exists():
            age_h = (time.time() - path.stat().st_mtime) / 3600
            isos.append({
                "iso":             iso,
                "cache_age_hours": round(age_h, 2),
                "available":       age_h <= _CACHE_MAX_AGE_H,
            })
        else:
            isos.append({"iso": iso, "cache_age_hours": None, "available": False})
    return {"isos": isos}


@app.get("/v1/forecast/{iso}", tags=["forecast"])
def get_forecast(iso: str, _key: str = Security(_require_key)):
    """
    15-day day-ahead load forecast (GW) with 10th/90th percentile uncertainty bands.

    - **mean_gw**: ensemble point forecast (GW)
    - **low_gw**: 10th-percentile load (GW)
    - **high_gw**: 90th-percentile load (GW)
    - **avg_temp_f**: population-weighted average temperature (°F)
    - **hdd / cdd**: heating / cooling degree-days
    """
    iso  = _validate_iso(iso)
    data = _read_cache(iso)
    meta = _cache_meta(iso)

    load = data.get("load", [])
    bt   = data.get("backtest", {})

    return {
        "iso":          iso,
        **meta,
        "horizon_days": len(load),
        "forecast":     [
            {
                "date":       d["date"],
                "mean_gw":    d["mean_load_gw"],
                "low_gw":     d["low_load_gw"],
                "high_gw":    d["high_load_gw"],
                "avg_temp_f": d.get("avg_temp_f"),
                "hdd":        d.get("hdd", 0),
                "cdd":        d.get("cdd", 0),
            }
            for d in load
        ],
        "model": {
            "type":      "xgboost-autotheta-ensemble",
            "mape_test": bt.get("mape_test"),
            "rmse_test": bt.get("rmse_test"),
        },
    }


@app.get("/v1/prices/{iso}", tags=["forecast"])
def get_prices(iso: str, _key: str = Security(_require_key)):
    """
    Day-ahead price forecast ($/MWh) from OLS log-linear regression
    on load + seasonal features.

    Returns empty `forecast` list for ISOs where price data is unavailable
    (PJM requires PJM_API_KEY, ERCOT requires ERCOT_API_KEY).
    """
    from temperature_modeling.price_forecast import price_unavailable_reason

    iso  = _validate_iso(iso)
    data = _read_cache(iso)
    meta = _cache_meta(iso)

    prices = data.get("price_forecast", [])
    rmse   = prices[0].get("model_rmse") if prices else None

    return {
        "iso":      iso,
        **meta,
        "forecast": [
            {
                "date":              p["date"],
                "forecast_usd_mwh":  p["forecast_price"],
                "low_usd_mwh":       p["low_price"],
                "high_usd_mwh":      p["high_price"],
            }
            for p in prices
        ],
        "model": {
            "type":         "ols-log-linear",
            "rmse_usd_mwh": rmse,
        },
        "note": None if prices else price_unavailable_reason(iso),
    }


@app.get("/v1/net-load/{iso}", tags=["forecast"])
def get_net_load(iso: str, _key: str = Security(_require_key)):
    """
    15-day net load forecast (load minus renewable generation, GW).
    Net load represents the demand that must be met by dispatchable resources.
    """
    iso  = _validate_iso(iso)
    data = _read_cache(iso)
    meta = _cache_meta(iso)

    net = data.get("net_load", [])

    return {
        "iso":      iso,
        **meta,
        "forecast": net,
    }


@app.get("/v1/carbon/{iso}", tags=["grid"])
def get_carbon(iso: str, _key: str = Security(_require_key)):
    """
    Current grid carbon intensity (lbs CO₂/MWh) and fuel mix breakdown.
    Fetched live from WattTime / EIA; not cache-dependent.
    """
    from temperature_modeling.carbon_intensity import fetch_carbon_intensity

    iso = _validate_iso(iso)
    try:
        ci = fetch_carbon_intensity(iso)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Carbon data unavailable: {exc}")

    if not ci:
        raise HTTPException(status_code=503, detail="Carbon data unavailable for this ISO.")

    return {
        "iso":            iso,
        "fetched_at":     datetime.now(tz=timezone.utc).isoformat(),
        "lbs_co2_per_mwh": ci.get("lbs_co2_per_mwh"),
        "clean_pct":       ci.get("clean_pct"),
        "fuel_mix":        ci.get("fuel_mix", {}),
    }


@app.get("/v1/accuracy/{iso}", tags=["model"])
def get_accuracy(iso: str, _key: str = Security(_require_key)):
    """
    Live forecast verification: day-ahead predicted vs EIA reported actuals.
    MAPE computed over rolling 7-day and 30-day windows.
    Updated each day a new EIA actual becomes available.
    """
    from temperature_modeling.verification import load_verification_stats

    iso = _validate_iso(iso)
    try:
        stats = load_verification_stats(iso)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Verification data unavailable: {exc}")

    records = stats.get("records", [])
    last    = records[-1]["date"] if records else None

    return {
        "iso":                iso,
        "mape_7d":            stats.get("mape_7d"),
        "mape_30d":           stats.get("mape_30d"),
        "bias_mw":            stats.get("bias_mw"),
        "n_verified":         stats.get("n_verified", 0),
        "last_verified_date": last,
        "records":            records[-30:],   # last 30 verified days
    }


@app.get("/v1/summary/{iso}", tags=["forecast"])
def get_summary(iso: str, _key: str = Security(_require_key)):
    """
    Full summary for an ISO in one call: forecast, prices, model stats.
    Equivalent to calling /forecast + /prices + /accuracy together.
    """
    iso  = _validate_iso(iso)
    data = _read_cache(iso)
    meta = _cache_meta(iso)

    from temperature_modeling.price_forecast import price_unavailable_reason
    from temperature_modeling.verification import load_verification_stats

    load   = data.get("load", [])
    prices = data.get("price_forecast", [])
    bt     = data.get("backtest", {})

    vstats: dict = {}
    try:
        vstats = load_verification_stats(iso)
    except Exception:
        pass

    return {
        "iso": iso,
        **meta,
        "forecast": [
            {
                "date":       d["date"],
                "mean_gw":    d["mean_load_gw"],
                "low_gw":     d["low_load_gw"],
                "high_gw":    d["high_load_gw"],
                "avg_temp_f": d.get("avg_temp_f"),
            }
            for d in load
        ],
        "prices": [
            {
                "date":             p["date"],
                "forecast_usd_mwh": p["forecast_price"],
                "low_usd_mwh":      p["low_price"],
                "high_usd_mwh":     p["high_price"],
            }
            for p in prices
        ],
        "model": {
            "type":      "xgboost-autotheta-ensemble",
            "mape_test": bt.get("mape_test"),
            "mape_30d":  vstats.get("mape_30d"),
            "n_verified": vstats.get("n_verified", 0),
        },
        "price_note": None if prices else price_unavailable_reason(iso),
    }


# ---------------------------------------------------------------------------
# Forward curve — 12-month ahead price strip with peak/off-peak split
# ---------------------------------------------------------------------------

@app.get("/v1/forward-curve/{iso}", tags=["forecast"])
def get_forward_curve(
    iso: str,
    months: int = 12,
    _key: str = Security(_require_key),
):
    """
    12-month forward electricity price curve with peak/off-peak split.

    Returns three weather scenarios per delivery month:
    - **cold**: normal temperature minus 1 std dev (bearish load, lower price)
    - **base**: NOAA 30-year climate normal
    - **hot**: normal temperature plus 1 std dev (bullish load, higher price)

    Each scenario includes:
    - `monthly_avg` — settlement-equivalent average price ($/MWh)
    - `on_peak` — Mon-Fri HE08-23 EPT ex-NERC-holidays (ICE/NERC convention, 16h/day)
    - `off_peak` — remaining hours; satisfies weighted-average = monthly_avg

    For PJM, `on_peak`/`off_peak` are fit from genuine historical peak-hour
    vs off-peak-hour settlement data (`peak_split_method: "empirical_hourly"`
    in each scenario) — for every other ISO, and as a PJM fallback when
    there isn't enough peak/off-peak-specific history, it's a synthetic
    ratio applied to the blended price (`peak_split_method: "synthetic_ratio"`).

    Gas price assumptions come from EIA STEO Henry Hub forecasts (free,
    published monthly). The `gas_curve` field shows the assumed $/MMBtu
    for each delivery month so traders can substitute their own gas view.

    Query param `months` controls horizon (1–24, default 12).
    """
    from temperature_modeling.forward_curve import build_forward_curve

    iso    = _validate_iso(iso)
    months = min(max(months, 1), 24)

    try:
        result = build_forward_curve(iso, n_months=months)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Forward curve build failed: {exc}")

    return {
        "iso":          iso,
        "generated_at": datetime.now(tz=timezone.utc).isoformat(),
        **result,
    }


@app.get("/v1/forward-curve/history/{iso}", tags=["forecast"])
def get_forward_curve_history(
    iso: str,
    from_month: str = "",
    to_month: str = "",
    _key: str = Security(_require_key),
):
    """
    Historical archive of forward curve snapshots for an ISO.

    Each record is the full forward curve as it was built on a given date.
    Use this to backtest model accuracy: compare `curve[i].scenarios.base.monthly_avg`
    predicted on `snapshot_date` against the actual settlement price for that
    delivery month.

    Query params:
    - `from_month`: YYYY-MM start (inclusive, optional)
    - `to_month`:   YYYY-MM end   (inclusive, optional)

    Returns snapshots in chronological order.
    """
    from temperature_modeling.forward_curve import load_forward_curve_history

    iso = _validate_iso(iso)
    snapshots = load_forward_curve_history(iso, from_month=from_month, to_month=to_month)
    return {
        "iso":       iso,
        "snapshots": len(snapshots),
        "records":   snapshots,
    }


@app.get("/v1/forward-curve/accuracy/{iso}", tags=["model"])
def get_forward_curve_accuracy(iso: str, _key: str = Security(_require_key)):
    """
    Backtest: archived forward-curve predictions vs realized settlement
    prices, for delivery months that have since fully completed.

    `by_lead_months` breaks accuracy out by how many months ahead each
    prediction was made — a 1-month-ahead prediction and a 12-month-ahead
    one aren't the same claim and shouldn't be averaged together.

    Returns n_comparisons=0 until the archive has snapshots old enough
    to compare against a completed delivery month (accumulates daily via
    the scheduled cache-refresh job).
    """
    from temperature_modeling.forward_curve import backtest_forward_curve

    iso = _validate_iso(iso)
    try:
        return backtest_forward_curve(iso)
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Backtest failed: {exc}")


@app.get("/v1/futures-pricer/{iso}", tags=["forecast"])
def get_futures_price_comparison(
    iso: str,
    delivery_month: str,
    peak_type: str,
    quoted_price: float,
    scenario: str = "base",
    _key: str = Security(_require_key),
):
    """
    Compare a real power futures contract (e.g. quoted on ICE) against this
    project's own forward-curve prediction for the same delivery month and
    peak type — a decision-support signal for potential mispricing.

    Scoped to PJM Western Hub only for now. There's no live market data feed
    here (enterprise-only, out of reach) — `quoted_price` is always supplied
    by the caller.

    Query params:
    - `delivery_month`: "YYYY-MM", must be in the future, within 60 months out
    - `peak_type`: "monthly_avg" | "on_peak" | "off_peak"
    - `quoted_price`: the market-quoted $/MWh price to compare against
    - `scenario`: "cold" | "base" | "hot" (default "base")

    `signal` is one of "within_band" / "above_band" / "below_band" /
    "no_band_available". `confidence_notes` flags anything that should make
    you trust the comparison less — e.g. a delivery month beyond EIA STEO's
    real forecast horizon, or an on/off-peak split still using the synthetic
    ratio rather than a genuine peak-hour model fit.
    """
    from temperature_modeling.futures_pricer import price_contract

    try:
        return price_contract(
            iso=iso, delivery_month=delivery_month, peak_type=peak_type,
            quoted_price=quoted_price, scenario=scenario,
        )
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc))
    except Exception as exc:
        raise HTTPException(status_code=503, detail=f"Futures pricer failed: {exc}")


@app.post("/v1/forward-curve/warm/{iso}", tags=["admin"])
async def warm_forward_curve(
    iso: str,
    background_tasks: BackgroundTasks,
    _key: str = Security(_require_key),
):
    """
    Pre-warm the 730-day price history cache for this ISO in the background.
    The first call to /v1/forward-curve/{iso} can take 2-5 minutes for some ISOs
    (CAISO especially due to rate limiting). Call this endpoint first to build
    the cache; subsequent forward-curve requests will respond instantly.
    Poll the returned `cache_path` to check when the file appears.
    """
    import asyncio as _asyncio

    iso = _validate_iso(iso)

    async def _warm(iso: str) -> None:
        loop = _asyncio.get_event_loop()
        await loop.run_in_executor(None, _do_warm, iso)

    def _do_warm(iso: str) -> None:
        try:
            from temperature_modeling.forward_curve import _load_price_history
            _load_price_history(iso)
            log.info("Forward curve cache warmed for %s", iso.upper())
        except Exception:
            log.exception("Forward curve cache warm failed for %s", iso.upper())

    background_tasks.add_task(_warm, iso)
    return {
        "status":     "warming",
        "iso":        iso,
        "cache_path": f"api_cache/{iso}_price_history_archive.json",
    }


# ---------------------------------------------------------------------------
# Refresh — trigger background pipeline recompute
# ---------------------------------------------------------------------------

_refresh_lock: set[str] = set()   # ISOs currently refreshing


async def _run_refresh(iso: str) -> None:
    """Run the dashboard's forecast pipeline in a subprocess, updating the cache file."""
    if iso in _refresh_lock:
        return
    _refresh_lock.add(iso)
    try:
        proc = await asyncio.create_subprocess_exec(
            sys.executable, "refresh_cache.py", iso,
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
            cwd=str(_HERE),
        )
        _, stderr = await proc.communicate()
        if proc.returncode != 0:
            log.error("Refresh failed for %s: %s", iso, stderr.decode()[:500])
        else:
            log.info("Refresh complete for %s", iso)
    finally:
        _refresh_lock.discard(iso)


@app.post("/v1/refresh/{iso}", tags=["admin"])
async def refresh_iso(
    iso: str,
    background_tasks: BackgroundTasks,
    _key: str = Security(_require_key),
):
    """
    Trigger a background recompute of the forecast cache for this ISO.
    Returns immediately; the cache is updated within ~60 seconds.
    Poll `/v1/isos` to watch `cache_age_hours` drop.
    """
    iso = _validate_iso(iso)
    if iso in _refresh_lock:
        return {"status": "already_refreshing", "iso": iso}
    background_tasks.add_task(_run_refresh, iso)
    return {"status": "refresh_queued", "iso": iso}
