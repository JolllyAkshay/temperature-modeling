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
from datetime import date, timedelta

from .zip_lookup import (
    lookup_zip, fetch_latest_state_residential_rate,
    fetch_state_residential_rate_history, geocode_zip,
)
from .carbon_intensity import fetch_carbon_intensity, fetch_carbon_intensity_history
from .demand_response import compute_dr_windows
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
        location:                {lat, lon, display_name} or None — for the map
        latest_state_rates:      {state: {period, res_rate_usd_mwh}, ...} —
                                   live EIA monthly state average, typically
                                   1-2 months old vs. the per-utility rate's
                                   annual snapshot; coarser (state-wide, not
                                   per-utility) but far fresher. Empty dict
                                   if the EIA key is missing or the fetch fails.
        retail_rate_history:      {available, data: {state, history: [{period,
                                   res_rate_usd_mwh}, ...]}} or {available: False,
                                   reason} — trailing 24 months, same state-average
                                   series as latest_state_rates, for a trend chart
        fuel_mix:                {available, data} or {available: False, reason}
        fuel_mix_history:        {available, data} or {available: False, reason} —
                                   trailing 24h clean_pct/fuel_mix_mw trend, for a chart
        best_time_to_use:        {available, data: {best_window, low_carbon_window,
                                   low_cost_window}} or {available: False, reason} —
                                   today/tomorrow demand-response windows (see
                                   demand_response.compute_dr_windows for field shapes)
        capacity_auctions:       {available, data} or {available: False, reason}
        market_competitiveness:  {available, data} or {available: False, reason}
        wholesale_price_context: {available, data, disclaimer, gap_explainer} or
                                   {available: False, reason} — data includes both
                                   annual_avg_usd_mwh (the figure comparable to the
                                   retail rate) and next_month_avg_usd_mwh (near-term,
                                   not meant for the retail comparison since a single
                                   month is often a cheap/expensive outlier)
        wholesale_price_history:  {available, data} or {available: False, reason} —
                                   real settled daily prices for the last 90 days
                                   (not model-predicted, unlike wholesale_price_context)

    Raises ValueError for a malformed zip (propagated from zip_lookup).
    Never raises for a well-formed zip with no data, or a non-RTO zip —
    those are legitimate results, not errors.
    """
    zl = lookup_zip(zip_code)

    # Best-effort — independent of whether the zip is in the utility
    # dataset, so a map location can still show even for a zip our
    # provider data doesn't cover. None on any geocoding failure; the map
    # is a visual nicety, never load-bearing for the rest of the response.
    try:
        location = geocode_zip(zl["zip"], session=session)
    except Exception:
        log.exception("%s: geocoding failed", zl["zip"])
        location = None

    result = {
        "zip": zl["zip"],
        "found": zl["found"],
        "utilities": zl["utilities"],
        "data_vintage_year": zl["data_vintage_year"],
        "iso": zl["iso"],
        "non_rto": zl["found"] and zl["iso"] is None,
        "location": location,
        "latest_state_rates": {},
        "retail_rate_history": {"available": False, "reason": "No data for this zip code in our dataset."},
    }

    if not zl["found"]:
        reason = "No data for this zip code in our dataset."
        for key in ("fuel_mix", "fuel_mix_history", "best_time_to_use", "capacity_auctions", "market_competitiveness", "wholesale_price_context", "wholesale_price_history"):
            result[key] = {"available": False, "reason": reason}
        return result

    # Best-effort — a live-rate miss should never take down the rest of
    # the response, and applies regardless of ISO/non-RTO status (a
    # utility's state-average rate exists whether or not it's in an ISO).
    for state in {u["state"] for u in zl["utilities"] if u.get("state")}:
        try:
            rate = fetch_latest_state_residential_rate(state, session=session)
        except Exception:
            log.exception("%s: state rate fetch failed", state)
            rate = None
        if rate:
            result["latest_state_rates"][state] = {
                "period": rate["period"], "res_rate_usd_mwh": rate["res_rate_usd_mwh"],
            }

    # Retail rate trend — trailing 24 months, one state (the first with
    # a mapped rate) since utilities at a single zip are almost always
    # in the same state; best-effort like the latest-rate fetch above.
    result["retail_rate_history"] = {"available": False, "reason": "No state rate data available."}
    for state in {u["state"] for u in zl["utilities"] if u.get("state")}:
        try:
            history = fetch_state_residential_rate_history(state, months=24, session=session)
        except Exception:
            log.exception("%s: state rate history fetch failed", state)
            history = []
        if history:
            result["retail_rate_history"] = {"available": True, "data": {"state": state, "history": history}}
            break

    iso = zl["iso"]
    if iso is None:
        reason = ("This zip code isn't served by one of the 7 wholesale electricity "
                   "markets this tool covers (PJM, CAISO, ERCOT, MISO, NYISO, ISO-NE, SPP) — "
                   "it's likely served by a vertically-integrated utility or federal power "
                   "authority instead. See the utility notes above for specifics.")
        for key in ("fuel_mix", "fuel_mix_history", "best_time_to_use", "capacity_auctions", "market_competitiveness", "wholesale_price_context", "wholesale_price_history"):
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

    # Fuel mix trend — trailing 24h, for a chart of how clean the mix has been
    try:
        fmh = fetch_carbon_intensity_history(iso, hours=24, session=session)
        result["fuel_mix_history"] = fmh
    except Exception:
        log.exception("%s: fuel mix history fetch failed", iso.upper())
        result["fuel_mix_history"] = {"available": False, "reason": "Fuel mix history temporarily unavailable."}

    # Best time to use electricity — today/tomorrow low-carbon and low-cost
    # windows, reusing demand_response.py's solar/wind-timing model (built
    # for the trader dashboard's DR panel). That page runs a full weather-
    # driven load forecast to size daily_load_gw precisely; this consumer
    # page doesn't have that pipeline available, so the daily level is
    # approximated from the current total generation (fm's total_mw) —
    # coarser, but good enough to scale the typical hourly demand shape
    # the model uses internally to time the solar/wind windows.
    try:
        if fm:
            today = date.today()
            tomorrow = today + timedelta(days=1)
            approx_gw = fm.get("total_mw", 0) / 1000
            daily_gw = {today.isoformat(): approx_gw, tomorrow.isoformat(): approx_gw}
            dr = compute_dr_windows(iso, daily_gw, fm, session=session)
            result["best_time_to_use"] = (
                {"available": True, "data": {
                    "best_window": dr["best_window"],
                    "low_carbon_window": dr["low_carbon_window"],
                    "low_cost_window": dr["low_cost_window"],
                }} if dr else
                {"available": False, "reason": "Best-time-to-use data temporarily unavailable."}
            )
        else:
            result["best_time_to_use"] = {"available": False, "reason": "Best-time-to-use data temporarily unavailable."}
    except Exception:
        log.exception("%s: demand-response window computation failed", iso.upper())
        result["best_time_to_use"] = {"available": False, "reason": "Best-time-to-use data temporarily unavailable."}

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

    # Wholesale price context — a 12-month forward strip. The annual average
    # is the number worth comparing against the retail rate (both are
    # annual figures); next month alone is often a shoulder month and can
    # make the wholesale-vs-retail gap look larger than it typically is.
    try:
        from .forward_curve import build_forward_curve  # noqa: PLC0415
        curve = build_forward_curve(iso, n_months=12)
        months = curve.get("curve") or []
        if months:
            next_month = months[0]
            annual_avg = sum(m["scenarios"]["base"]["monthly_avg"] for m in months) / len(months)
            result["wholesale_price_context"] = {
                "available": True,
                "data": {
                    "annual_avg_usd_mwh": round(annual_avg, 2),
                    "annual_avg_months": len(months),
                    "next_month": next_month["month"],
                    "next_month_avg_usd_mwh": next_month["scenarios"]["base"]["monthly_avg"],
                    "next_month_on_peak_usd_mwh": next_month["scenarios"]["base"]["on_peak"],
                    "next_month_off_peak_usd_mwh": next_month["scenarios"]["base"]["off_peak"],
                    "model_source": curve["model_source"],
                },
                "disclaimer": _WHOLESALE_DISCLAIMER,
                "gap_explainer": _retail_wholesale_gap_note(result["utilities"], annual_avg),
            }
        else:
            result["wholesale_price_context"] = {"available": False, "reason": "Wholesale price data temporarily unavailable."}
    except Exception:
        log.exception("%s: wholesale price context failed", iso.upper())
        result["wholesale_price_context"] = {"available": False, "reason": "Wholesale price data temporarily unavailable."}

    # Wholesale price history — real settled daily prices over the last 90
    # days, distinct from wholesale_price_context's forward-looking strip.
    try:
        from .forward_curve import get_recent_settled_prices  # noqa: PLC0415
        recent = get_recent_settled_prices(iso, days=90)
        result["wholesale_price_history"] = (
            {"available": True, "data": recent} if recent
            else {"available": False, "reason": "Wholesale price history temporarily unavailable."}
        )
    except Exception:
        log.exception("%s: wholesale price history failed", iso.upper())
        result["wholesale_price_history"] = {"available": False, "reason": "Wholesale price history temporarily unavailable."}

    return result


def _retail_wholesale_gap_note(utilities: list, annual_avg_wholesale: float) -> str | None:
    """
    A short, honestly-sourced explanation of why retail runs higher than
    wholesale — not a real per-utility cost breakdown (we don't have that
    data), just context so the gap reads as expected market structure
    rather than a data error. Only computed when a residential rate is
    actually available to compare against.
    """
    rates = [u["res_rate_usd_mwh"] for u in utilities if u.get("res_rate_usd_mwh")]
    if not rates or not annual_avg_wholesale:
        return None
    avg_retail = sum(rates) / len(rates)
    ratio = avg_retail / annual_avg_wholesale
    return (
        f"Your utility's average residential rate (${avg_retail:.0f}/MWh) runs about "
        f"{ratio:.1f}x the annual-average wholesale price (${annual_avg_wholesale:.0f}/MWh). "
        f"That's expected — per ISO-NE's own public breakdown, wholesale energy and "
        f"transmission together typically make up only about a third of a residential bill "
        f"in ISO-run markets; the rest is distribution infrastructure, capacity market "
        f"costs, state policy programs (like renewable energy requirements), and utility "
        f"margin, none of which show up in the wholesale price alone."
    )
