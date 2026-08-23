"""
Compare a real PJM Western Hub power futures contract (e.g. from ICE) against
this project's own forward-curve prediction for the same delivery month and
peak type, to surface potential mispricing as a decision-support signal.

Scoped to PJM Western Hub only — see build_forward_curve's own scope notes
for why (other PJM hubs and other ISOs are a natural extension later, but
each needs the same pnode-identity audit PJM just needed).

There is no live ICE data feed here (enterprise-only, out of reach) — the
market-quoted price is always supplied by the caller.

Public API
----------
price_contract(iso, delivery_month, peak_type, quoted_price,
                scenario="base", history=None) -> dict
"""

import logging
from datetime import date

from .forward_curve import build_forward_curve, fetch_hh_futures, _month_diff

log = logging.getLogger(__name__)

_VALID_PEAK_TYPES = ("monthly_avg", "on_peak", "off_peak")
_VALID_SCENARIOS  = ("cold", "base", "hot")
_MAX_LEAD_MONTHS  = 60
# EIA's STEO gas forecast realistically covers this far ahead — beyond it,
# fetch_hh_futures is effectively holding the last known price flat rather
# than using a genuine forward assumption. See build_forward_curve's own
# gas-curve handling for the same caveat.
_GAS_CURVE_CONFIDENT_MONTHS = 18

# scenario key -> (price key, low key, high key)
_PRICE_KEYS = {
    "monthly_avg": ("monthly_avg", "low_usd_mwh", "high_usd_mwh"),
    "on_peak":     ("on_peak", "on_peak_low_usd_mwh", "on_peak_high_usd_mwh"),
    "off_peak":    ("off_peak", "off_peak_low_usd_mwh", "off_peak_high_usd_mwh"),
}


def price_contract(
    iso: str,
    delivery_month: str,
    peak_type: str,
    quoted_price: float,
    scenario: str = "base",
    history: list | None = None,
) -> dict:
    """
    Compare `quoted_price` (a real market quote, e.g. from ICE) against this
    project's own forward-curve prediction for the same delivery month.

    Parameters
    ----------
    iso:            must be "pjm" — scope of this pass (see module docstring).
    delivery_month: "YYYY-MM", must be in the future and within 60 months out.
    peak_type:      "monthly_avg" | "on_peak" | "off_peak".
    quoted_price:   the market-quoted $/MWh price to compare against.
    scenario:       "cold" | "base" | "hot" — which weather scenario's curve
                     to compare against (default "base").
    history:        pre-loaded price history, if the caller already has it
                     (see build_forward_curve's identical parameter).

    Returns
    -------
    dict — see module docstring / plan for the full field list. Raises
    ValueError for invalid inputs rather than returning a partial result,
    since a malformed comparison is worse than no comparison.
    """
    iso = iso.lower()
    if iso != "pjm":
        raise ValueError(f"futures_pricer is scoped to PJM only for now, got iso={iso!r}")
    if peak_type not in _VALID_PEAK_TYPES:
        raise ValueError(f"peak_type must be one of {_VALID_PEAK_TYPES}, got {peak_type!r}")
    if scenario not in _VALID_SCENARIOS:
        raise ValueError(f"scenario must be one of {_VALID_SCENARIOS}, got {scenario!r}")
    if quoted_price is None or quoted_price <= 0:
        raise ValueError(f"quoted_price must be a positive number, got {quoted_price!r}")

    try:
        delivery_year, delivery_mon = int(delivery_month[:4]), int(delivery_month[5:7])
        if not (1 <= delivery_mon <= 12) or len(delivery_month) != 7 or delivery_month[4] != "-":
            raise ValueError
        date(delivery_year, delivery_mon, 1)   # validates the month itself
    except (ValueError, IndexError):
        raise ValueError(f"delivery_month must be 'YYYY-MM', got {delivery_month!r}") from None

    today = date.today()
    current_ym = today.strftime("%Y-%m")
    lead_months = _month_diff(current_ym, delivery_month)
    if lead_months < 1:
        raise ValueError(f"delivery_month {delivery_month!r} is not in the future "
                          f"(current month is {current_ym})")
    if lead_months > _MAX_LEAD_MONTHS:
        raise ValueError(f"delivery_month {delivery_month!r} is {lead_months} months out — "
                          f"more than the {_MAX_LEAD_MONTHS}-month cap on this tool "
                          f"(likely a typo, or genuinely too far out to price meaningfully)")

    result = build_forward_curve(iso, n_months=lead_months, history=history)
    entry = next((m for m in result["curve"] if m["month"] == delivery_month), None)
    if entry is None:
        raise ValueError(f"forward curve did not produce an entry for {delivery_month!r} "
                          f"(built {lead_months} months out)")

    scen = entry["scenarios"][scenario]
    price_key, low_key, high_key = _PRICE_KEYS[peak_type]
    model_price = scen[price_key]
    band_low, band_high = scen.get(low_key), scen.get(high_key)
    band_source = "cqr" if band_low is not None and band_high is not None else "none"

    spread = quoted_price - model_price
    spread_pct = (spread / model_price * 100) if model_price else None

    if band_source == "none":
        signal = "no_band_available"
    elif quoted_price < band_low:
        signal = "below_band"
    elif quoted_price > band_high:
        signal = "above_band"
    else:
        signal = "within_band"

    # Cheap — same 24h on-disk cache build_forward_curve itself just populated,
    # no extra network cost. Checked independently rather than trusting
    # result["gas_curve"] (already rounded/filled and can't distinguish a
    # real STEO month from a held-flat extrapolation).
    gas_curve = fetch_hh_futures(lead_months + 3)
    gas_curve_covers_month = delivery_month in gas_curve

    confidence_notes = []
    if result["model_source"] != "ols-log-linear":
        confidence_notes.append("no fitted price model for this ISO — using fallback heuristic, no real uncertainty band")
    if lead_months > _GAS_CURVE_CONFIDENT_MONTHS:
        confidence_notes.append(
            f"{lead_months} months out exceeds EIA STEO's realistic ~{_GAS_CURVE_CONFIDENT_MONTHS}-month "
            f"forecast horizon — the gas price assumption for this month may be held flat rather than forward-looking")
    elif not gas_curve_covers_month:
        confidence_notes.append("no EIA STEO gas price for this specific month — assumption may be extrapolated")
    if peak_type != "monthly_avg" and scen.get("peak_split_method") != "empirical_hourly":
        confidence_notes.append(
            "on/off-peak split for this prediction used the synthetic seasonal ratio, not a model fit on "
            "real peak-hour data — some of the comparison may reflect approximation error rather than market signal")

    confidence = "reduced" if confidence_notes else "normal"

    return {
        "iso":                    iso,
        "delivery_month":         delivery_month,
        "peak_type":               peak_type,
        "scenario":                scenario,
        "model_price":             model_price,
        "band_low":                band_low,
        "band_high":               band_high,
        "band_source":             band_source,
        "quoted_price":            round(quoted_price, 2),
        "spread_usd_mwh":          round(spread, 2),
        "spread_pct":              round(spread_pct, 1) if spread_pct is not None else None,
        "signal":                  signal,
        "lead_months":             lead_months,
        "model_source":            result["model_source"],
        "peak_split_method":       scen.get("peak_split_method"),
        "gas_curve_covers_month":  gas_curve_covers_month,
        "confidence":              confidence,
        "confidence_notes":        confidence_notes,
        "generated_at":            date.today().isoformat(),
    }
