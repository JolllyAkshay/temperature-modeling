"""
How competitive is each ISO's wholesale market? A hand-curated snapshot
per ISO, in the same spirit as capacity_market.py: static data verified
against each ISO's own published Independent/Internal Market Monitor
annual report, not a live feed (none of these publish a structured API
for this).

This exists to honestly answer a version of "what did traders bid" —
individual bid data is never public in any ISO market (confirmed via this
project's own earlier research into ISO-NE's FCM auction data, which
masks participant identity). What IS public and genuinely informative is
each market monitor's own structural-competitiveness assessment.

Every ISO's monitor uses a different headline metric — confirmed by
reading each 2024 report directly, not assumed to be uniform:
  - PJM, MISO, SPP:  report HHI (Herfindahl-Hirschman Index)
  - ERCOT, ISO-NE:   report pivotal-supplier frequency (% of hours/intervals
                      where at least one supplier's output was needed to
                      meet demand — ERCOT's monitor explicitly states this
                      is a more reliable market-power indicator than HHI
                      for electricity markets)
  - CAISO:           reports Residual Supply Index (RSI) hours-below-1
  - NYISO:           reports a qualitative competitiveness conclusion,
                      not a single headline structural number, in its
                      2024 report

Public API
----------
get_market_competitiveness(iso) -> dict
"""

from datetime import date

_STATIC_DATA: dict = {
    "pjm": {
        "headline_metric": "Energy market HHI",
        "headline_value": "714 (2024 average)",
        "assessment": "Unconcentrated by FERC standards (min 553, max 983 over the year), "
                       "but the aggregate market structure was still assessed \"not competitive\" "
                       "on 49.5% of days in 2024 due to pivotal suppliers — a low HHI doesn't by "
                       "itself mean the market was competitive in every hour.",
        "year": 2024,
        "source": "Monitoring Analytics, 2024 State of the Market Report for PJM (Independent Market Monitor)",
    },
    "caiso": {
        "headline_metric": "Residual Supply Index (RSI3) hours below 1.0",
        "headline_value": "176 hours in 2024 (out of 8,760)",
        "assessment": "An RSI3 below 1.0 means even the three largest suppliers combined weren't "
                       "enough to make the market non-pivotal that hour. 176 hours (up from 129 in "
                       "2023) is a small fraction of the year — most hours were structurally competitive.",
        "year": 2024,
        "source": "CAISO Dept. of Market Monitoring, 2024 Annual Report on Market Issues and Performance",
    },
    "ercot": {
        "headline_metric": "Hours with at least one pivotal supplier",
        "headline_value": "63% of all hours in 2024",
        "assessment": "Up from 50% in 2023 and 57% in 2022 — a clear rising trend. Under high-load "
                       "conditions a supplier was pivotal roughly 90% of the time. ERCOT's own monitor "
                       "states pivotal-supplier analysis is a more reliable market-power indicator "
                       "than HHI for electricity markets, which is why ERCOT doesn't lead with HHI.",
        "year": 2024,
        "source": "Potomac Economics, 2024 State of the Market Report for the ERCOT Electricity Markets (Independent Market Monitor)",
    },
    "miso": {
        "headline_metric": "Generation HHI",
        "headline_value": "539 MISO-wide (low)",
        "assessment": "Low for the overall footprint, but very high in some sub-regions — 4,193 in "
                       "WUMS (Wisconsin/Upper Michigan) and 3,269 in the South Region, where a single "
                       "supplier operates nearly 60% of generation. The regional average masks real "
                       "local concentration.",
        "year": 2024,
        "source": "Potomac Economics, 2024 State of the Market Report for the MISO Markets (Independent Market Monitor)",
    },
    "nyiso": {
        "headline_metric": None,
        "headline_value": None,
        "assessment": "NYISO's 2024 report doesn't lead with a single structural number the way the "
                       "other ISOs' monitors do — its Executive Summary states the market \"performed "
                       "competitively in 2024\" and that supplier conduct was \"generally consistent "
                       "with expectations in a competitive market,\" with mitigation measures in NYC "
                       "and the G-J Locality (the two zones the monitor treats as most concentration-prone) "
                       "assessed as effective.",
        "year": 2024,
        "source": "Potomac Economics, 2024 State of the Market Report for the New York ISO Markets (Market Monitoring Unit)",
    },
    "isone": {
        "headline_metric": "Real-time intervals with at least one pivotal supplier",
        "headline_value": "33.3% of five-minute intervals in 2024",
        "assessment": "A sharp rise from 16.6% in 2020, though slightly below 2023's 37.3%. The IMM "
                       "attributes 2024's level to lower net imports leaving native generation a larger "
                       "share of load, making two suppliers frequently pivotal together.",
        "year": 2024,
        "source": "ISO-NE 2024 Annual Markets Report (Internal Market Monitor)",
    },
    "spp": {
        "headline_metric": "Hours moderately concentrated (HHI)",
        "headline_value": "2% of hours in 2024",
        "assessment": "Down from 3% in 2023, part of a steady decline since 2018 — SPP's monitor "
                       "concludes the market \"remained mostly unconcentrated.\" The largest online "
                       "supplier's real-time market share exceeded 20% in 14% of hours.",
        "year": 2024,
        "source": "SPP Market Monitoring Unit, State of the Market 2024",
    },
}


def get_market_competitiveness(iso: str) -> dict:
    """
    Return the market-concentration snapshot for the given ISO.

    Not "what traders bid" — that's never public data in any ISO market.
    This is each ISO's own Independent/Internal Market Monitor's published
    structural-competitiveness assessment, which is the closest honest
    answer to the underlying question ("is this market dominated by a
    few players, or genuinely competitive?").
    """
    iso = iso.lower()
    data = _STATIC_DATA.get(iso)
    if not data:
        return {"error": f"No market competitiveness data for {iso.upper()}"}
    return {**data, "iso": iso.upper(), "as_of": date.today().isoformat()}
