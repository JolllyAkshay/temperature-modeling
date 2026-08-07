"""
AI-powered forecast briefs and dashboard chat, backed by Claude.

Two public functions:
  generate_forecast_brief(iso, forecast_data) -> str
      3-5 sentence analyst narrative from the 15-day load forecast.
      Result is cached for BRIEF_CACHE_TTL_HOURS hours.

  generate_chat_response(iso, forecast_data, user_message, history) -> str
      Conversational Q&A about the forecast, with full conversation history.
"""

import json
import logging
import time
from datetime import date
from pathlib import Path

from ._llm import complete, is_available, provider_label
from .verification import load_verification_stats

log = logging.getLogger(__name__)

BRIEF_CACHE_TTL_HOURS: int = 3

_CACHE_DIR = Path(__file__).parent.parent.parent / "api_cache"
_BRIEF_CACHES = {
    iso: _CACHE_DIR / f"{iso}_brief_cache.json"
    for iso in ("pjm", "caiso", "ercot", "miso", "nyiso", "isone", "spp")
}

_ISO_CONTEXT = {
    "pjm":   "PJM Interconnection — Eastern US, ~65 GW peak. Covers IL, IN, KY, MD, MI, NJ, NC, OH, PA, TN, VA, WV, DC.",
    "caiso": "CAISO — California ISO, ~45 GW peak. Covers most of California; high solar penetration creates a pronounced duck-curve.",
    "ercot": "ERCOT — Texas ISO, ~80 GW peak. Energy-only market (no capacity payments); ORDC scarcity pricing kicks in above ~68 GW.",
    "miso":  "MISO — Midcontinent ISO, ~120 GW peak. Covers MN, WI, MI, IA, IL, MO, AR, LA, MS; historically coal-heavy, rapid wind build-out.",
    "nyiso": "NYISO — New York ISO, ~35 GW peak. Dense urban load; significant hydro imports from Quebec; NYC congestion drives zonal price spreads.",
    "isone": "ISO-NE — New England ISO, ~28 GW peak. Winter gas-electric fuel competition is the primary reliability concern; limited pipeline capacity.",
    "spp":   "SPP — Southwest Power Pool, ~90 GW peak. Wind-heavy (>50% penetration on strong days); covers Great Plains from Texas Panhandle to North Dakota.",
}

_BRIEF_SYSTEM = (
    "You are an expert energy market analyst specialising in US electricity grid operations. "
    "You receive a multi-signal intelligence package — load forecast, historical percentile rank, "
    "SHAP model attribution, real-time carbon intensity, demand-response windows, and price outlook. "
    "Your job is to synthesise these into ONE coherent 4-5 sentence analyst narrative that connects "
    "the signals and delivers a concrete, actionable insight. "
    "Do NOT list the signals separately. Weave them into a story: WHY is load at this level, "
    "WHAT does it mean for carbon and price, and WHAT should a grid operator or flexible load "
    "customer do about it. "
    "If load is above the 85th percentile, open with that as the lead risk. "
    "If a strong demand-response window exists (solar peak, wind surge, off-peak trough), "
    "close with that recommendation. "
    "Plain prose only — no markdown, no bullet points, no headers."
)

_CHAT_SYSTEM_TEMPLATE = (
    "You are an expert energy market analyst with deep knowledge of US electricity grid operations, "
    "load forecasting, power markets, grid regulations, and ISO/RTO rules. "
    "You have access to a live 15-day load forecast for {iso_label}, shown below. "
    "For forecast-specific questions (peak demand, load levels, temperature impacts) use the live data. "
    "For general energy market questions (market rules, capacity requirements, tariffs, regulations, "
    "grid concepts, ISO/RTO structures) answer from your domain knowledge — do NOT say the data "
    "doesn't contain it. Only say you don't know if the question is genuinely outside your expertise.\n\n"
    "=== LIVE FORECAST DATA ===\n{context_block}"
)


# ---------------------------------------------------------------------------
# Internal helpers
# ---------------------------------------------------------------------------

def _brief_cache_valid(iso: str) -> bool:
    path = _BRIEF_CACHES.get(iso)
    if not path or not path.exists():
        return False
    if (time.time() - path.stat().st_mtime) / 3600 >= BRIEF_CACHE_TTL_HOURS:
        return False
    try:
        data = json.loads(path.read_text())
        return (data.get("date") == date.today().isoformat()
                and data.get("version") == "v3"
                and bool(data.get("brief")))
    except (json.JSONDecodeError, OSError):
        return False


def _build_context_block(iso: str, forecast_data: dict) -> str:
    load_list  = forecast_data.get("load", [])
    comparison = forecast_data.get("comparison", {})
    backtest   = forecast_data.get("backtest", {})

    means = [d["mean_load_gw"] for d in load_list]
    dates = [d["date"]         for d in load_list]

    peak_gw   = max(means) if means else 0
    peak_date = dates[means.index(peak_gw)] if means else "—"
    avg_gw    = sum(means) / len(means) if means else 0
    today_gw  = means[0] if means else None

    da_fcst = comparison.get("da_fcst", {})
    daily = []
    for d in load_list[:7]:   # 7-day horizon is enough for the brief
        row: dict = {
            "date":        d["date"],
            "mean_gw":     d["mean_load_gw"],
            "band_gw":     round(d["high_load_gw"] - d["low_load_gw"], 1),
        }
        if d.get("avg_temp_f") is not None:
            row["avg_temp_f"] = round(d["avg_temp_f"], 1)
        if d["date"] in da_fcst:
            row["vs_iso_da_gw"] = round(d["mean_load_gw"] - da_fcst[d["date"]], 2)
        daily.append(row)

    # ── Percentile context ────────────────────────────────────────────────────
    percentile_block = {}
    pct_data = forecast_data.get("percentile", {})
    if pct_data and today_gw:
        percentile_block = {
            "today_percentile_for_month": pct_data.get("percentile"),
            "month_avg_gw": pct_data.get("month_avg_gw"),
            "month_max_gw": pct_data.get("month_max_gw"),
            "last_higher_date": pct_data.get("last_higher_date"),
            "interpretation": (
                "EXTREME demand event — top 10% for this calendar month"
                if (pct_data.get("percentile") or 0) >= 90
                else "Elevated demand" if (pct_data.get("percentile") or 0) >= 75
                else "Normal demand range"
            ),
        }

    # ── SHAP top driver ───────────────────────────────────────────────────────
    shap_block = {}
    shap = forecast_data.get("shap", {})
    if shap:
        groups = shap.get("groups", {})
        if groups:
            top_group = max(groups, key=lambda k: abs(groups[k]))
            shap_block = {
                "top_driver": top_group,
                "top_driver_mw": groups[top_group],
                "all_drivers": groups,
                "base_mw": shap.get("base_mw"),
            }

    # ── Carbon intensity ──────────────────────────────────────────────────────
    ci_block = {}
    ci = forecast_data.get("carbon", {})
    if ci:
        lbs = ci.get("lbs_co2_per_mwh", 0)
        ci_block = {
            "lbs_co2_per_mwh": lbs,
            "clean_pct": ci.get("clean_pct"),
            "top_fuels": dict(list(ci.get("fuel_mix", {}).items())[:3]),
            "label": (
                "very clean (<300)" if lbs < 300
                else "clean (300-500)" if lbs < 500
                else "mixed (500-750)" if lbs < 750
                else "carbon-heavy (>750)"
            ),
        }

    # ── Demand-response recommendation ───────────────────────────────────────
    dr_block = {}
    dr = forecast_data.get("demand_response", {})
    if dr:
        best = dr.get("best_window", {})
        low_ci = dr.get("low_carbon_window", {})
        low_cost = dr.get("low_cost_window", {})
        dr_block = {
            "best_window": best.get("label"),
            "best_window_reason": best.get("reason"),
            "low_carbon_window": low_ci.get("label"),
            "carbon_reduction_pct": low_ci.get("carbon_reduction_pct"),
            "low_cost_window": low_cost.get("label"),
            "cost_reduction_pct": low_cost.get("cost_reduction_pct"),
        }

    # ── Price outlook ─────────────────────────────────────────────────────────
    price_block = {}
    prices = forecast_data.get("price_forecast", [])
    if prices:
        p_vals = [p["forecast_price"] for p in prices[:7]]
        price_block = {
            "today_price": prices[0]["forecast_price"] if prices else None,
            "peak_price": max(p_vals) if p_vals else None,
            "peak_price_date": prices[p_vals.index(max(p_vals))]["date"] if p_vals else None,
            "7day_avg_price": round(sum(p_vals) / len(p_vals), 1) if p_vals else None,
        }

    # ── Live forecast verification ────────────────────────────────────────────
    live_accuracy = {}
    try:
        vstats = load_verification_stats(iso)
        if vstats["n_verified"] >= 7:
            live_accuracy = {
                "live_mape_7d":   vstats["mape_7d"],
                "live_mape_30d":  vstats["mape_30d"],
                "live_bias_mw":   vstats["bias_mw"],
                "n_verified_days": vstats["n_verified"],
                "note": "actual day-ahead error vs EIA reported actuals",
            }
    except Exception:
        pass

    return json.dumps({
        "iso":             iso.upper(),
        "iso_context":     _ISO_CONTEXT.get(iso, iso.upper()),
        "today":           date.today().isoformat(),
        "today_gw":        today_gw,
        "peak_gw":         round(peak_gw, 2),
        "peak_date":       peak_date,
        "avg_gw_15day":    round(avg_gw, 2),
        "model_mape_test": backtest.get("mape_test"),
        "7day_forecast":   daily,
        "demand_percentile": percentile_block,
        "load_driver_shap":  shap_block,
        "grid_carbon":       ci_block,
        "dr_opportunity":    dr_block,
        "price_outlook":     price_block,
        "live_accuracy":     live_accuracy,
    }, indent=2)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def generate_forecast_brief(iso: str, forecast_data: dict) -> str:
    """
    Return a 3-5 sentence analyst-style brief for the current forecast.
    Reads from cache if it was generated today within the TTL window.
    Returns empty string if API key is missing or call fails.
    """
    if _brief_cache_valid(iso):
        try:
            return json.loads(_BRIEF_CACHES[iso].read_text())["brief"]
        except Exception:
            pass

    if not is_available():
        return ""

    if not forecast_data.get("load"):
        return ""

    context = _build_context_block(iso, forecast_data)
    prompt = (
        f"Write a 4-5 sentence analyst brief for {iso.upper()} using the multi-signal intelligence below.\n\n"
        f"{context}\n\n"
        "Synthesise the signals into a coherent narrative — connect the load level (and its percentile rank "
        "if available) to the specific weather or calendar driver identified by SHAP, then connect that to "
        "the carbon and price implications, and close with the best demand-response window. "
        "If the percentile is >= 85, open with that as the headline risk. "
        "If live_accuracy is present and has n_verified_days >= 7, cite the live_mape_30d as the model's "
        "actual observed accuracy (e.g. 'this model has averaged X% error on day-ahead forecasts over the "
        "past 30 days, verified against EIA actuals'). "
        "Be specific: use the actual numbers. Plain prose only — no markdown."
    )

    brief = complete(_BRIEF_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=450)
    if not brief:
        return ""
    log.info("%s brief generated via %s (%d chars)", iso.upper(), provider_label(), len(brief))

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _BRIEF_CACHES[iso].write_text(
            json.dumps({"date": date.today().isoformat(), "version": "v3", "brief": brief})
        )
    except OSError:
        pass

    return brief


def generate_chat_response(
    iso: str,
    forecast_data: dict,
    user_message: str,
    history: list,
) -> str:
    """
    Generate a conversational response about the current forecast.

    Parameters
    ----------
    iso:          ISO code ("pjm", "caiso", "ercot", "miso")
    forecast_data: the dcc.Store payload
    user_message: the latest user turn
    history:      list of {"role": "user"|"assistant", "content": str} — prior turns only
    """
    if not is_available():
        return f"AI chat is unavailable — add GROQ_API_KEY or ANTHROPIC_API_KEY to your .env file."

    _ISO_LABELS = {
        "pjm":   "PJM Interconnection",
        "caiso": "CAISO (California ISO)",
        "ercot": "ERCOT (Texas)",
        "miso":  "MISO (Midcontinent ISO)",
        "nyiso": "NYISO (New York ISO)",
        "isone": "ISO-NE (New England ISO)",
        "spp":   "SPP (Southwest Power Pool)",
    }
    iso_label     = _ISO_LABELS.get(iso, iso.upper())
    context_block = _build_context_block(iso, forecast_data)
    system        = _CHAT_SYSTEM_TEMPLATE.format(iso_label=iso_label, context_block=context_block)

    messages = list(history) + [{"role": "user", "content": user_message}]
    reply = complete(system, messages, max_tokens=700)
    return reply or "Sorry, something went wrong. Please try again."
