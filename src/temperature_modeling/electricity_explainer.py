"""
Orchestration layer for the consumer "what powers my zip code" tool.
Ties together zip_lookup, carbon_intensity, capacity_market, and
market_competitiveness into one response — used by both the dashboard
page and the public API endpoint so the assembly logic exists once.

Public API
----------
explain_electricity(zip_code: str, session=None) -> dict
"""

import logging

from .zip_lookup import lookup_zip
from .carbon_intensity import fetch_carbon_intensity
from .capacity_market import get_capacity_market_data
from .market_competitiveness import get_market_competitiveness

log = logging.getLogger(__name__)

_WHOLESALE_DISCLAIMER = (
    "This is the wholesale price generators are paid — not your electric bill. "
    "Most residential customers pay a blended rate set by their utility or retail "
    "plan, which doesn't move hour-to-hour with this price. Only customers on a "
    "real-time-pricing plan see anything close to this directly."
)


def explain_electricity(zip_code: str, session=None) -> dict:
    """
    Assemble a consumer-facing explanation of the electricity serving a
    zip code.

    Returns a dict with:
        zip, found, utilities, data_vintage_year  (from zip_lookup)
        iso, non_rto (bool)
        fuel_mix:                {available, data} or {available: False, reason}
        capacity_auctions:       {available, data} or {available: False, reason}
        market_competitiveness:  {available, data} or {available: False, reason}
        wholesale_price_context: {available, data, disclaimer} or {available: False, reason}

    Raises ValueError for a malformed zip (propagated from zip_lookup).
    Never raises for a well-formed zip with no data, or a non-RTO zip —
    those are legitimate results, not errors.
    """
    zl = lookup_zip(zip_code)

    result = {
        "zip": zl["zip"],
        "found": zl["found"],
        "utilities": zl["utilities"],
        "data_vintage_year": zl["data_vintage_year"],
        "iso": zl["iso"],
        "non_rto": zl["found"] and zl["iso"] is None,
    }

    if not zl["found"]:
        reason = "No data for this zip code in our dataset."
        for key in ("fuel_mix", "capacity_auctions", "market_competitiveness", "wholesale_price_context"):
            result[key] = {"available": False, "reason": reason}
        return result

    iso = zl["iso"]
    if iso is None:
        reason = ("This zip code isn't served by one of the 7 wholesale electricity "
                   "markets this tool covers (PJM, CAISO, ERCOT, MISO, NYISO, ISO-NE, SPP) — "
                   "it's likely served by a vertically-integrated utility or federal power "
                   "authority instead. See the utility notes above for specifics.")
        for key in ("fuel_mix", "capacity_auctions", "market_competitiveness", "wholesale_price_context"):
            result[key] = {"available": False, "reason": reason}
        return result

    # Fuel mix
    try:
        fm = fetch_carbon_intensity(iso, session=session)
        result["fuel_mix"] = {"available": bool(fm), "data": fm} if fm else \
            {"available": False, "reason": "Fuel mix data temporarily unavailable."}
    except Exception:
        log.exception("%s: fuel mix fetch failed", iso.upper())
        result["fuel_mix"] = {"available": False, "reason": "Fuel mix data temporarily unavailable."}

    # Capacity auctions
    try:
        cm = get_capacity_market_data(iso)
        result["capacity_auctions"] = {"available": "error" not in cm, "data": cm}
    except Exception:
        log.exception("%s: capacity market lookup failed", iso.upper())
        result["capacity_auctions"] = {"available": False, "reason": "Capacity market data temporarily unavailable."}

    # Market competitiveness
    try:
        mc = get_market_competitiveness(iso)
        result["market_competitiveness"] = {"available": "error" not in mc, "data": mc}
    except Exception:
        log.exception("%s: market competitiveness lookup failed", iso.upper())
        result["market_competitiveness"] = {"available": False, "reason": "Market data temporarily unavailable."}

    # Wholesale price context — near-term forward strip, heavily caveated
    try:
        from .forward_curve import build_forward_curve  # noqa: PLC0415
        curve = build_forward_curve(iso, n_months=1)
        month = curve["curve"][0] if curve.get("curve") else None
        if month:
            result["wholesale_price_context"] = {
                "available": True,
                "data": {
                    "month": month["month"],
                    "monthly_avg_usd_mwh": month["scenarios"]["base"]["monthly_avg"],
                    "on_peak_usd_mwh": month["scenarios"]["base"]["on_peak"],
                    "off_peak_usd_mwh": month["scenarios"]["base"]["off_peak"],
                    "model_source": curve["model_source"],
                },
                "disclaimer": _WHOLESALE_DISCLAIMER,
            }
        else:
            result["wholesale_price_context"] = {"available": False, "reason": "Wholesale price data temporarily unavailable."}
    except Exception:
        log.exception("%s: wholesale price context failed", iso.upper())
        result["wholesale_price_context"] = {"available": False, "reason": "Wholesale price data temporarily unavailable."}

    return result
