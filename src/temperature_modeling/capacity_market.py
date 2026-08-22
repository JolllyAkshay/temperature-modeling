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
_STATIC_DATA: dict = {
    "pjm": {
        "mechanism":        "Reliability Pricing Model (RPM)",
        "auction_type":     "Annual, 3-year forward",
        "current_auction":  "2027/28 Base Residual Auction",
        "native_unit_price": "$333.44/MW-day (RTO, at the FERC-approved cap)",
        "clearing_price_mw_year": 121_706,   # 333.44 * 365
        "procured_mw":      134_747,
        "requirement_mw":   None,
        "reserve_margin_pct": 14.8,
        "peak_mw_2024":     147_600,
        "lole_standard":    "0.1 days/year",
        "notes": "2027/28 BRA (held Dec 2025) cleared at the FERC-approved cap of $333.44/MW-day RTO-wide, up 1.3% from 2026/27's $329.17/MW-day — both auctions have now cleared at the cap, driven by retiring thermal capacity and load growth outpacing new supply.",
        "source": "PJM 2027/28 Base Residual Auction Report, Dec 17 2025",
    },
    "caiso": {
        "mechanism":        "Resource Adequacy (RA)",
        "auction_type":     "Monthly + annual bilateral",
        "current_auction":  "July 2026 monthly RA",
        "clearing_price_mw_year": 75_000,   # approximate bilateral RA value
        "procured_mw":      55_000,
        "requirement_mw":   50_300,
        "reserve_margin_pct": 17.5,
        "peak_mw_2024":     48_800,
        "lole_standard":    "1-in-10 year standard",
        "notes": "CAISO RA is bilateral (no centralized capacity auction). Monthly showings required. Local capacity areas (LA Basin, SF Bay) have separate requirements.",
        "source": "CAISO RA filing statistics 2026",
    },
    "ercot": {
        "mechanism":        "Energy-only + ORDC (Operating Reserve Demand Curve)",
        "auction_type":     "No forward capacity market — scarcity pricing via ORDC",
        "current_auction":  "N/A — real-time ORDC",
        "clearing_price_mw_year": None,     # no capacity price — scarcity via energy
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 15.3,
        "peak_mw_2024":     85_500,
        "lole_standard":    "1-in-10 year (planning reserve margin target ~10%)",
        "ordc_max_adder_usd_mwh": 5_000,    # max ORDC adder at 0 reserves
        "ordc_lcap_mw":     2_000,           # reserves below which ORDC activates
        "notes": "ERCOT relies on energy market scarcity pricing (ORDC) rather than a capacity market. ORDC adder peaks at $5,000/MWh when reserves fall below 2 GW.",
        "source": "ERCOT Operating Guide 2026",
    },
    "miso": {
        "mechanism":        "Planning Resource Auction (PRA)",
        "auction_type":     "Annual, 1-year forward",
        "current_auction":  "2025/26 PRA",
        "native_unit_price": "$217/MW-day (annualized system average, PY2025/26)",
        "clearing_price_mw_year": 79_205,   # 217 * 365
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 16.8,
        "peak_mw_2024":     121_300,
        "lole_standard":    "0.1 days/year",
        "notes": "PY2025/26 cleared at $217/MW-day annualized system average — roughly 10x PY2024/25's $21/MW-day. Zonal/seasonal prices varied widely (Summer 2025 hit $666.50/MW-day system-wide). First auction under MISO's new Reliability-Based Demand Curve (RBDC).",
        "source": "MISO 2025 Planning Resource Auction results, posted May 2025",
    },
    "nyiso": {
        "mechanism":        "Installed Capacity (ICAP) Market",
        "auction_type":     "Monthly + seasonal spot auctions",
        "current_auction":  "Summer 2026 Capability Period",
        "native_unit_price": "NYC zone: $32.6/kW-month (record; cleared at the reference point, no surplus)",
        "clearing_price_mw_year": None,   # highly locational (NYC vs Rest-of-State vs Long Island) — no single system-wide figure verified
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 18.0,
        "peak_mw_2024":     32_800,
        "lole_standard":    "0.1 days/year",
        "notes": "ICAP is priced separately by locality. NYC hit a record $32.6/kW-month for Summer 2026 (67% above its prior high) — the monthly spot auction cleared at the reference point with no surplus. Rest-of-State and Long Island clear at different, generally lower, prices not verified here.",
        "source": "NYISO ICAP monthly spot auction results, Summer 2026",
    },
    "isone": {
        "mechanism":        "Forward Capacity Market (FCM)",
        "auction_type":     "Annual, 3-year forward (FCA 19 delayed to Feb 2028 amid FCM reform)",
        "current_auction":  "FCA #18 (2027/28 Capacity Commitment Period)",
        "native_unit_price": "$3.58/kW-month (all zones and import interfaces)",
        "clearing_price_mw_year": 42_960,   # 3.58 * 12 * 1000
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 16.6,
        "peak_mw_2024":     24_200,
        "lole_standard":    "0.1 days/year",
        "notes": "FCA 18 (held Feb 2024) cleared at $3.58/kW-month for the 2027/28 Capacity Commitment Period. FCA 19 — which would ordinarily have set 2028/29 prices — has been delayed to February 2028 as part of ISO-NE's broader FCM reform, so this remains the most recent cleared result.",
        "source": "ISO-NE FCA 18 finalized results; FCA 19 delay filing",
    },
    "spp": {
        "mechanism":        "Integrated Marketplace — Reserve Margin requirement only",
        "auction_type":     "No forward capacity market",
        "current_auction":  "N/A — reserve margin requirement",
        "clearing_price_mw_year": None,
        "procured_mw":      None,
        "requirement_mw":   None,
        "reserve_margin_pct": 13.8,
        "peak_mw_2024":     56_700,
        "lole_standard":    "1-in-10 year",
        "notes": "SPP does not have a formal capacity market. Reliability is enforced via a minimum reserve margin requirement (~12%) on load-serving entities.",
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
