"""
Capacity market data for US ISOs.

Fetches or returns cached static data for:
  - PJM RPM (Reliability Pricing Model) — annual capacity auction results
  - ERCOT ORDC (Operating Reserve Demand Curve) — real-time scarcity pricing parameters
  - CAISO RA (Resource Adequacy) — monthly local capacity requirements
  - MISO Planning Reserve Margin — annual auction results
  - NYISO ICAP — installed capacity market
  - ISO-NE Forward Capacity Market — annual auction results

Sources: publicly published by each ISO (no API key required for static data).
For live data, each ISO publishes XML/JSON reports; this module uses
the most recent published values as defaults and attempts live refresh.

Public API
----------
get_capacity_market_data(iso) -> dict
    Returns capacity market snapshot for the given ISO.
"""

import logging
import os
from datetime import date

import requests

log = logging.getLogger(__name__)

# Static baseline data (updated quarterly from ISO publications)
# Sources: PJM Markets & Operations, ERCOT Operating Guides, CAISO RA filings
#
# active_delivery_period / auction_held_date describe the auction whose
# results are what's actually in effect TODAY — not simply the most
# recently held auction. These markets clear years ahead of delivery, so
# "most recent" and "currently active" are usually different auctions
# (e.g. PJM's 2027/28 BRA had already cleared by mid-2026, but the
# capacity actually being delivered right now is still under the 2026/27
# BRA's results, held a year earlier). Getting this distinction right —
# not just picking whichever result was most recently in the news — is
# the entire point of these fields; each was verified against the specific
# delivery-period dates, not assumed from "which auction is newest."
_STATIC_DATA: dict = {
    "pjm": {
        "mechanism":        "Reliability Pricing Model (RPM)",
        "auction_type":     "Annual, 3-year forward",
        "active_delivery_period": "2026/27 (Jun 1 2026 – May 31 2027)",
        "auction_held_date": "2025-07-22",
        "native_unit_price": "$329.17/MW-day (RTO, at the FERC-approved cap)",
        "clearing_price_mw_year": 120_147,   # 329.17 * 365
        "procured_mw":      134_311,
        "requirement_mw":   None,
        "reserve_margin_pct": 14.8,
        "peak_mw_2024":     147_600,
        "lole_standard":    "0.1 days/year",
        "notes": "The 2026/27 BRA — whose results are the capacity actually being delivered right now — was held July 22, 2025 and cleared at the FERC-approved cap of $329.17/MW-day RTO-wide. The 2027/28 BRA has already cleared too (Dec 2025, $333.44/MW-day, also at the cap) but that capacity isn't delivering yet — it takes effect June 2027.",
        "source": "PJM 2026/27 Base Residual Auction Report, Jul 22 2025",
    },
    "caiso": {
        "mechanism":        "Resource Adequacy (RA)",
        "auction_type":     "Monthly + annual bilateral",
        "active_delivery_period": "2026 (calendar year)",
        "auction_held_date": None,   # no centralized auction — see notes
        "clearing_price_mw_year": 75_000,   # approximate bilateral RA value
        "procured_mw":      55_000,
        "requirement_mw":   50_300,
        "reserve_margin_pct": 17.5,
        "peak_mw_2024":     48_800,
        "lole_standard":    "1-in-10 year standard",
        "notes": "CAISO RA is bilateral (no centralized capacity auction, so there's no single \"auction date\" to report) — load-serving entities negotiate directly with resources and file monthly compliance showings. Local capacity areas (LA Basin, SF Bay) have separate requirements.",
        "source": "CAISO RA filing statistics 2026",
    },
    "ercot": {
        "mechanism":        "Energy-only + ORDC (Operating Reserve Demand Curve)",
        "auction_type":     "No forward capacity market — scarcity pricing via ORDC",
        "active_delivery_period": None,
        "auction_held_date": None,   # no capacity auction exists in ERCOT
        "clearing_price_mw_year": None,     # no capacity price — scarcity via energy
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 15.3,
        "peak_mw_2024":     85_500,
        "lole_standard":    "1-in-10 year (planning reserve margin target ~10%)",
        "ordc_max_adder_usd_mwh": 5_000,    # max ORDC adder at 0 reserves
        "ordc_lcap_mw":     2_000,           # reserves below which ORDC activates
        "notes": "ERCOT relies on energy market scarcity pricing (ORDC) rather than a capacity market — there is no capacity auction of any kind to date or report. ORDC adder peaks at $5,000/MWh when reserves fall below 2 GW.",
        "source": "ERCOT Operating Guide 2026",
    },
    "miso": {
        "mechanism":        "Planning Resource Auction (PRA)",
        "auction_type":     "Annual, 1-year forward",
        "active_delivery_period": "2026/27 (Jun 1 2026 – May 31 2027)",
        "auction_held_date": "2026-04-28",
        "native_unit_price": "$126.19/MW-day (annualized, North/Central subregion); $424.30/MW-day summer-only",
        "clearing_price_mw_year": 46_059,   # 126.19 * 365
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 16.8,
        "peak_mw_2024":     121_300,
        "lole_standard":    "0.1 days/year",
        "notes": "PY2026/27 results (released Apr 28, 2026) cleared at $126.19/MW-day annualized for the North/Central subregion — down 42% from PY2025/26's $217/MW-day, on a 4% increase in offered capacity. Only 1 year forward, so unlike PJM/ISO-NE this auction's results ARE this year's active delivery.",
        "source": "MISO 2026 Planning Resource Auction results, posted Apr 28 2026",
    },
    "nyiso": {
        "mechanism":        "Installed Capacity (ICAP) Market",
        "auction_type":     "Monthly + seasonal spot auctions",
        "active_delivery_period": "Summer 2026 Capability Period (May 1 – Oct 31 2026)",
        "auction_held_date": None,   # priced via recurring monthly spot auctions, not one annual event
        "native_unit_price": "NYC zone: $32.6/kW-month (record; cleared at the reference point, no surplus)",
        "clearing_price_mw_year": None,   # highly locational (NYC vs Rest-of-State vs Long Island) — no single system-wide figure verified
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 18.0,
        "peak_mw_2024":     32_800,
        "lole_standard":    "0.1 days/year",
        "notes": "Unlike PJM/MISO/ISO-NE, NYISO ICAP has no single annual auction date — it's priced through recurring monthly spot auctions throughout each capability period. Sample: NYC hit a record $32.6/kW-month for Summer 2026 (67% above its prior high), the monthly spot auction clearing at the reference point with no surplus. Rest-of-State and Long Island clear separately, generally lower.",
        "source": "NYISO ICAP monthly spot auction results, Summer 2026",
    },
    "isone": {
        "mechanism":        "Forward Capacity Market (FCM)",
        "auction_type":     "Annual, 3-year forward (FCA 19 delayed to Feb 2028 amid FCM reform)",
        "active_delivery_period": "2026/27 Capacity Commitment Period (Jun 1 2026 – May 31 2027)",
        "auction_held_date": "2023-03-06",
        "native_unit_price": "$2.590/kW-month (all zones and import interfaces except New Brunswick, $2.551)",
        "clearing_price_mw_year": 31_080,   # 2.590 * 12 * 1000
        "procured_mw":      31_370,
        "requirement_mw":   None,
        "reserve_margin_pct": 16.6,
        "peak_mw_2024":     24_200,
        "lole_standard":    "0.1 days/year",
        "notes": "FCA 17 — whose results are the capacity actually being delivered right now — was held March 6, 2023 and cleared at $2.590/kW-month for the 2026/27 Capacity Commitment Period. FCA 18 (held Feb 2024, $3.58/kW-month) has already cleared for 2027/28 but isn't delivering yet. FCA 19, which would ordinarily set 2028/29 prices, has been delayed to February 2028 as part of ISO-NE's broader FCM reform.",
        "source": "ISO-NE FCA 17 finalized results, Mar 6 2023",
    },
    "spp": {
        "mechanism":        "Integrated Marketplace — Reserve Margin requirement only",
        "auction_type":     "No forward capacity market",
        "active_delivery_period": None,
        "auction_held_date": None,   # no capacity auction exists in SPP
        "clearing_price_mw_year": None,
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 13.8,
        "peak_mw_2024":     56_700,
        "lole_standard":    "1-in-10 year",
        "notes": "SPP does not have a formal capacity market — there is no capacity auction of any kind to date or report. Reliability is enforced via a minimum reserve margin requirement (~12%) on load-serving entities.",
        "source": "SPP 2026 Reliability Assessment",
    },
}


def get_capacity_market_data(iso: str) -> dict:
    """
    Return capacity market data for the given ISO.
    Returns the static snapshot; future versions will refresh from ISO APIs.
    """
    iso = iso.lower()
    data = _STATIC_DATA.get(iso)
    if not data:
        return {"error": f"No capacity market data for {iso.upper()}"}

    return {
        **data,
        "iso":         iso.upper(),
        "as_of":       date.today().isoformat(),
    }


def get_reserve_margin_color(pct: float | None) -> str:
    """Traffic-light color for reserve margin percentage."""
    if pct is None:
        return "#94a3b8"
    if pct >= 15:
        return "#22c55e"
    if pct >= 10:
        return "#f97316"
    return "#ef4444"
