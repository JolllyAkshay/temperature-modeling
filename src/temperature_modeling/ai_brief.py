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

log = logging.getLogger(__name__)

BRIEF_CACHE_TTL_HOURS: int = 3

_CACHE_DIR = Path(__file__).parent.parent.parent / "api_cache"
_BRIEF_CACHES = {
    iso: _CACHE_DIR / f"{iso}_brief_cache.json"
    for iso in ("pjm", "caiso", "ercot", "miso")
}

_ISO_CONTEXT = {
    "pjm":   "PJM Interconnection — Eastern US, ~65 GW peak. Covers IL, IN, KY, MD, MI, NJ, NC, OH, PA, TN, VA, WV, DC.",
    "caiso": "CAISO — California ISO, ~45 GW peak. Covers most of California; high solar penetration creates a pronounced duck-curve.",
    "ercot": "ERCOT — Texas ISO, ~80 GW peak. Energy-only market (no capacity payments); ORDC scarcity pricing kicks in above ~68 GW.",
    "miso":  "MISO — Midcontinent ISO, ~120 GW peak. Covers MN, WI, MI, IA, IL, MO, AR, LA, MS; historically coal-heavy, rapid wind build-out.",
}

_BRIEF_SYSTEM = (
    "You are an expert energy market analyst specialising in US electricity grid operations. "
    "Write concise, factual, analyst-style forecast briefs for grid operators and energy traders. "
    "Be precise with numbers. Lead with the most actionable insight. "
    "Write in flowing prose — no bullet points, no headers — 3 to 5 sentences maximum."
)

_CHAT_SYSTEM_TEMPLATE = (
    "You are an expert energy market analyst with deep knowledge of US electricity grid operations, "
    "load forecasting, and power markets. You have access to a live 15-day load forecast for {iso_label}. "
    "Answer questions concisely and precisely using the forecast data provided. "
    "If something is outside the provided data say so clearly.\n\n"
    "=== LIVE DATA ===\n{context_block}"
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
        return data.get("date") == date.today().isoformat() and bool(data.get("brief"))
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

    recent_actual = dict(list(comparison.get("actual", {}).items())[-7:])

    return json.dumps({
        "iso":              iso.upper(),
        "iso_context":      _ISO_CONTEXT.get(iso, iso.upper()),
        "today":            date.today().isoformat(),
        "forecast_days":    len(load_list),
        "today_gw":         means[0] if means else None,
        "peak_gw":          round(peak_gw, 2),
        "peak_date":        peak_date,
        "avg_gw_15day":     round(avg_gw, 2),
        "model_mape_test":  backtest.get("mape_test"),
        "daily_forecast":   [{"date": d["date"], "mean_gw": d["mean_load_gw"],
                               "low_gw": d["low_load_gw"], "high_gw": d["high_load_gw"]}
                              for d in load_list],
        "recent_actual_gw": recent_actual,
        "monthly_mape":     backtest.get("monthly_mape", {}),
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
        f"Generate a concise analyst-style forecast brief for the {iso.upper()} grid load outlook below.\n\n"
        f"{context}\n\n"
        "Lead with the headline risk or opportunity (peak load, heat event, cold snap, notable pattern). "
        "Include the peak date, magnitude, and any demand anomaly visible across the 15-day window. "
        "Plain prose only — no markdown."
    )

    brief = complete(_BRIEF_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=350)
    if not brief:
        return ""
    log.info("%s brief generated via %s (%d chars)", iso.upper(), provider_label(), len(brief))

    try:
        _CACHE_DIR.mkdir(parents=True, exist_ok=True)
        _BRIEF_CACHES[iso].write_text(
            json.dumps({"date": date.today().isoformat(), "brief": brief})
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
    }
    iso_label     = _ISO_LABELS.get(iso, iso.upper())
    context_block = _build_context_block(iso, forecast_data)
    system        = _CHAT_SYSTEM_TEMPLATE.format(iso_label=iso_label, context_block=context_block)

    messages = list(history) + [{"role": "user", "content": user_message}]
    reply = complete(system, messages, max_tokens=700)
    return reply or "Sorry, something went wrong. Please try again."
