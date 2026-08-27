"""
Zip code -> electric utility -> ISO/RTO lookup.

Data source: NREL/OpenEI "U.S. Electric Utility Companies and Rates:
Look-up by Zip Code" (2024 vintage) — see data/zip_utility_rates.meta.json
for provenance. Maps a zip code to the utilities serving it plus each
utility's average residential/commercial/industrial rate.

That dataset has no ISO/RTO column, so utility -> ISO membership is a
separate, hand-curated map (_UTILITY_TO_ISO below) built from each utility's
actual, verified RTO membership status — not guessed from state boundaries,
since several states are split between ISOs (Illinois: PJM/MISO) or between
ISO and non-RTO utility territory (most of the Southeast, parts of the
Pacific Northwest and Mountain West). Every entry not in this map is
honestly reported as "not yet mapped", never silently defaulted to a guess.

Coverage is Pareto-style: the largest utilities by zip-code count first
(these cover the bulk of the US population), expanded over time rather
than attempting all ~1,200 distinct utility names in the source dataset
up front. See test_zip_lookup.py for the utilities whose ISO membership
has actually been verified against a real source.

Public API
----------
lookup_zip(zip_code: str) -> dict
fetch_latest_state_residential_rate(state: str, session=None) -> dict | None
"""

import csv
import json
import logging
import os
import time
from pathlib import Path

import requests

log = logging.getLogger(__name__)

_DATA_DIR = Path(__file__).parent / "data"
_ZIP_FILES = ["iou_zipcodes_2024.csv", "non_iou_zipcodes_2024.csv"]

_EIA_RETAIL_SALES_URL = "https://api.eia.gov/v2/electricity/retail-sales/data/"
_STATE_RATE_CACHE: dict = {}          # {state: (timestamp, result)}
_STATE_RATE_CACHE_TTL_S = 24 * 3600   # EIA publishes this monthly — daily cache is plenty fresh

# ---------------------------------------------------------------------------
# Utility -> ISO/RTO membership.
#
# None means "confirmed not a member of any of the 7 ISOs this project
# covers" — a common, legitimate answer (vertically-integrated Southeast
# utilities, federal power marketers, utilities that stay independent by
# choice), not a placeholder for "unknown." A utility simply absent from
# this dict is the actual "unknown, not yet researched" case — lookup_zip
# distinguishes the two (see iso_status in its return value).
#
# Every entry below was checked against a real source (RTO/utility company
# filings, RTO Insider, FERC dockets) during this pass — not inferred from
# state borders. Notable non-obvious cases, verified rather than assumed:
#   - Illinois is split: ComEd (northern IL) is PJM, Ameren Illinois
#     (southern IL) is MISO.
#   - Entergy Texas is MISO, not ERCOT, despite serving Texas customers.
#   - Kentucky Power (AEP) is PJM; Kentucky Utilities/LG&E are NOT — they
#     exited MISO in 2006 and remain independent (verified against their
#     own 2024 RTO Membership Analysis filed with the KY PSC).
#   - PacifiCorp and Xcel Energy-Colorado participate in CAISO's EIM/EDAM
#     and SPP's Markets+ respectively, but neither is a full RTO/capacity-
#     market member the way e.g. PG&E or OG&E are — treated as None here,
#     with the partial participation noted rather than glossed over.
#   - El Paso Electric and Rio Grande-area co-ops are in the Western
#     Interconnection, not ERCOT, despite being in Texas.
#   - City of Lubbock left SPP for ERCOT in 2021.
# ---------------------------------------------------------------------------
_UTILITY_TO_ISO: dict = {
    # --- PJM ---
    "Virginia Electric & Power Co":       "pjm",
    "Appalachian Power Co":               "pjm",
    "PPL Electric Utilities Corp":        "pjm",
    "Commonwealth Edison Co":             "pjm",
    "Ohio Power Co":                      "pjm",
    "Potomac Electric Power Co":          "pjm",
    "Monongahela Power Co":               "pjm",
    "Public Service Elec & Gas Co":       "pjm",
    "Ohio Edison Co":                     "pjm",
    "PECO Energy Co":                     "pjm",
    "Jersey Central Power & Lt Co":       "pjm",
    "Kentucky Power Co":                  "pjm",
    "Baltimore Gas & Electric Co":        "pjm",
    "Dayton Power & Light Co":            "pjm",

    # --- CAISO ---
    "Pacific Gas & Electric Co.":         "caiso",
    "Southern California Edison Co":      "caiso",

    # --- ERCOT ---
    "TXU Energy Retail Co, LLC":          "ercot",
    "Reliant Energy Retail Services":     "ercot",
    "Austin Energy":                      "ercot",
    "City of San Antonio - (TX)":         "ercot",
    "City of Lubbock - (TX)":             "ercot",   # left SPP for ERCOT in 2021
    "City of Garland - (TX)":             "ercot",
    "City of Denton - (TX)":              "ercot",
    "City of Bryan - (TX)":               "ercot",
    "City of College Station - (TX)":     "ercot",
    "City of Greenville - (TX)":          "ercot",
    "City of New Braunfels - (TX)":       "ercot",
    "City of Seguin - (TX)":              "ercot",
    "City of San Marcos - (TX)":          "ercot",
    "City of Brenham - (TX)":             "ercot",
    "City of Georgetown - (TX)":          "ercot",
    "Pedernales Electric Coop, Inc":      "ercot",
    "Bluebonnet Electric Coop, Inc":      "ercot",
    "Denton County Elec Coop, Inc":       "ercot",
    "Guadalupe Valley Elec Coop Inc":     "ercot",
    "Karnes Electric Coop Inc":           "ercot",
    "Bandera Electric Coop, Inc":         "ercot",
    "Fayette Electric Coop, Inc":         "ercot",
    "Wise Electric Coop Inc":             "ercot",

    # --- MISO ---
    "Ameren Illinois Company":            "miso",
    "Consumers Energy Co - (MI)":         "miso",
    "Interstate Power and Light Co":      "miso",
    "Duke Energy Indiana, LLC":           "miso",
    "Otter Tail Power Co":                "miso",
    "MidAmerican Energy Co":              "miso",
    "Northern States Power Co - Minnesota": "miso",
    "Union Electric Co - (MO)":           "miso",
    "DTE Electric Company":               "miso",
    "Entergy Arkansas LLC":               "miso",
    "Entergy Louisiana LLC":              "miso",
    "Entergy Mississippi LLC":            "miso",
    "Entergy Texas Inc.":                 "miso",   # MISO, not ERCOT, despite serving TX
    "Wisconsin Power & Light Co":         "miso",
    "Wisconsin Electric Power Co":        "miso",

    # --- NYISO ---
    "Niagara Mohawk Power Corp.":         "nyiso",
    "New York State Elec & Gas Corp":     "nyiso",
    "Consolidated Edison Co-NY Inc":      "nyiso",

    # --- ISO-NE ---
    "Massachusetts Electric Co":          "isone",
    "Connecticut Light & Power Co":       "isone",
    "Central Maine Power Co":             "isone",
    "NSTAR Electric Company":             "isone",
    "Public Service Co of NH":            "isone",
    "Green Mountain Power Corp":          "isone",

    # --- SPP ---
    "Oklahoma Gas & Electric Co":         "spp",
    "Southwestern Electric Power Co":     "spp",
    "Public Service Co of Oklahoma":      "spp",
    "Evergy Kansas Central, Inc":         "spp",
    "Evergy Metro":                       "spp",
    "Southwestern Public Service Co":     "spp",     # TX Panhandle — SPP, not ERCOT
    "North Plains Electric Coop Inc":     "spp",
    "Rita Blanca Electric Coop, Inc":     "spp",
    "Deaf Smith Electric Coop, Inc":      "spp",
    "Bailey County Elec Coop Assn":       "spp",
    "Lamb County Electric Coop, Inc":     "spp",

    # --- Confirmed non-RTO (verified, not a coverage gap) ---
    "Tennessee Valley Authority":         None,   # federal power authority, vertically integrated
    "Georgia Power Co":                   None,   # Southern Company, Southeast non-RTO
    "Alabama Power Co":                   None,   # Southern Company, Southeast non-RTO
    "Duke Energy Carolinas, LLC":         None,   # Southeast non-RTO
    "Duke Energy Progress - (NC)":        None,   # Southeast non-RTO
    "Duke Energy Florida, LLC":           None,   # Florida non-RTO
    "Florida Power & Light Co":           None,   # Florida non-RTO
    "Dominion Energy South Carolina, Inc": None,  # Southeast non-RTO
    "Arizona Public Service Co":          None,   # WECC, non-RTO
    "Los Angeles Department of Water & Power": None,  # municipal, not a CAISO member
    "Puget Sound Energy Inc":             None,   # Pacific NW, non-RTO
    "NorthWestern Energy LLC - (MT)":     None,   # Mountain West, non-RTO
    "El Paso Electric Co":                None,   # Western Interconnection, not ERCOT
    "Kentucky Utilities Co":              None,   # independent since exiting MISO in 2006
    "PacifiCorp":                         None,   # CAISO EIM/EDAM participant, not a full RTO member
    "Public Service Co of Colorado":      None,   # SPP Markets+ (day-ahead only) since 2025, not full RTO membership
}

# Utilities where the partial/evolving market participation above is worth
# surfacing to the user even though iso is None.
_NON_RTO_NOTES: dict = {
    "PacifiCorp": "Participates in CAISO's real-time (EIM) and day-ahead (EDAM) "
                  "markets, but is not a full RTO/capacity-market member.",
    "Public Service Co of Colorado": "Joined SPP's Markets+ day-ahead market in 2025; "
                                      "full RTO membership is a separate, still-pending decision.",
    "Tennessee Valley Authority": "A federal power marketing authority — vertically integrated, "
                                   "not a member of any RTO/ISO.",
}


def _load_zip_table() -> dict:
    """
    The source CSVs have one row per (zip, utility, service_type) — the
    same utility can appear more than once at a zip with different
    service_type values ("Bundled", "Delivery", "Energy") in deregulated
    states where customers can buy generation separately from delivery
    (e.g. PECO in PA offers both a full "Bundled" default rate and a
    delivery-only rate for customers who've picked a competitive
    generation supplier). Collapsing these naively would either silently
    drop real data or show the same utility name twice with different
    numbers and no explanation — instead, prefer the "Bundled" (all-in)
    rate when present, else sum "Delivery" + "Energy" into an estimated
    all-in rate, else fall back to whatever single row exists.
    """
    # {(zip, utility_name): {"Bundled": row, "Delivery": row, "Energy": row}}
    raw: dict = {}
    meta: dict = {}
    for fname in _ZIP_FILES:
        path = _DATA_DIR / fname
        if not path.exists():
            log.warning("zip_lookup: missing data file %s", path)
            continue
        with open(path, encoding="utf-8") as f:
            for row in csv.DictReader(f):
                zip_code = row["zip"].strip().zfill(5)
                name = row["utility_name"].strip()
                key = (zip_code, name)
                raw.setdefault(key, {})[row["service_type"].strip()] = row
                meta[key] = {"state": row["state"].strip(), "ownership": row["ownership"].strip()}

    table: dict = {}
    for (zip_code, name), by_type in raw.items():
        if "Bundled" in by_type:
            rates, rate_basis = by_type["Bundled"], "bundled"
            res, comm, ind = (_to_usd_mwh(rates.get(f)) for f in ("res_rate", "comm_rate", "ind_rate"))
        elif "Delivery" in by_type or "Energy" in by_type:
            rate_basis = "estimated bundled (delivery + energy)"
            res  = _sum_usd_mwh(by_type, "res_rate")
            comm = _sum_usd_mwh(by_type, "comm_rate")
            ind  = _sum_usd_mwh(by_type, "ind_rate")
        else:
            continue
        table.setdefault(zip_code, []).append({
            "name": name,
            "state": meta[(zip_code, name)]["state"],
            "ownership": meta[(zip_code, name)]["ownership"],
            "rate_basis": rate_basis,
            "res_rate_usd_mwh": res,
            "comm_rate_usd_mwh": comm,
            "ind_rate_usd_mwh": ind,
        })
    return table


def _sum_usd_mwh(by_type: dict, field: str) -> float | None:
    parts = [v for t in ("Delivery", "Energy") if t in by_type
             for v in [_to_usd_mwh(by_type[t].get(field))] if v is not None]
    return round(sum(parts), 2) if parts else None


def _to_usd_mwh(raw: str | None) -> float | None:
    """Source CSV stores rates as $/kWh (e.g. 0.2042) — convert to $/MWh (204.2)."""
    if raw is None or raw == "":
        return None
    try:
        val = float(raw)
    except ValueError:
        return None
    return round(val * 1000, 2) if val > 0 else None


_ZIP_TABLE: dict | None = None
_META: dict | None = None


def _table() -> dict:
    global _ZIP_TABLE
    if _ZIP_TABLE is None:
        _ZIP_TABLE = _load_zip_table()
        log.info("zip_lookup: loaded %d zip codes", len(_ZIP_TABLE))
    return _ZIP_TABLE


def _meta() -> dict:
    global _META
    if _META is None:
        meta_path = _DATA_DIR / "zip_utility_rates.meta.json"
        try:
            _META = json.loads(meta_path.read_text())
        except Exception:
            _META = {}
    return _META


def lookup_zip(zip_code: str) -> dict:
    """
    Look up the utilities serving a zip code, their ISO membership (if
    known), and average rates.

    Returns
    -------
    dict with:
        zip, found (bool), utilities (list of {name, state, ownership,
        res_rate_usd_mwh, comm_rate_usd_mwh, ind_rate_usd_mwh,
        iso, iso_status}), iso (the ISO if every listed utility agrees,
        else None), multi_utility (bool), data_vintage_year

    All rates are $/MWh (source data is $/kWh — multiplied by 1000 for
    consistency with the wholesale price figures elsewhere in this tool).

    iso_status per utility is one of:
        "mapped"      — a real ISO membership (pjm/caiso/ercot/miso/
                          nyiso/isone/spp)
        "non_rto"     — confirmed NOT a member of any of the 7 ISOs
                          (a legitimate, common answer, not a gap)
        "unmapped"    — this utility hasn't been researched yet; honestly
                          reported as unknown rather than guessed

    Raises ValueError for a malformed zip code (not 5 digits).
    """
    zip_code = (zip_code or "").strip()
    if not zip_code.isdigit() or len(zip_code) != 5:
        raise ValueError(f"zip_code must be a 5-digit US zip code, got {zip_code!r}")

    rows = _table().get(zip_code)
    if not rows:
        return {
            "zip": zip_code, "found": False, "utilities": [], "iso": None,
            "multi_utility": False, "data_vintage_year": _meta().get("vintage_year"),
        }

    utilities = []
    isos_seen = set()
    for r in rows:
        name = r["name"]
        if name in _UTILITY_TO_ISO:
            iso = _UTILITY_TO_ISO[name]
            status = "mapped" if iso else "non_rto"
        else:
            iso = None
            status = "unmapped"
        entry = {**r, "iso": iso, "iso_status": status}
        if status == "non_rto" and name in _NON_RTO_NOTES:
            entry["note"] = _NON_RTO_NOTES[name]
        utilities.append(entry)
        if status == "mapped":
            isos_seen.add(iso)

    # Only report a single top-level `iso` when every mapped utility for
    # this zip agrees — a genuinely mixed/boundary zip should surface as
    # ambiguous, not silently pick one.
    iso = next(iter(isos_seen)) if len(isos_seen) == 1 else None

    return {
        "zip": zip_code,
        "found": True,
        "utilities": utilities,
        "iso": iso,
        "multi_utility": len(rows) > 1,
        "data_vintage_year": _meta().get("vintage_year"),
    }


def fetch_latest_state_residential_rate(state: str, session: requests.Session | None = None) -> dict | None:
    """
    Latest available monthly average residential rate for a state, from
    EIA's electricity/retail-sales series — typically only 1-2 months
    behind "today", far fresher than the NREL/OpenEI dataset lookup_zip
    uses (an annual snapshot). Coarser (state-wide, not per-utility) in
    exchange for being current — shown alongside the per-utility rate,
    not instead of it, so the tool is honest about the precision/freshness
    tradeoff rather than picking one silently.

    Returns {state, period (YYYY-MM), res_rate_usd_mwh} or None if the key
    is missing, the state has no data, or the request fails — this is
    always a best-effort supplement, never required for the rest of the
    tool to work.
    """
    api_key = os.environ.get("EIA_API_KEY", "")
    if not api_key:
        return None

    state = (state or "").strip().upper()
    if not state:
        return None

    cached_ts, cached_val = _STATE_RATE_CACHE.get(state, (0, None))
    if time.time() - cached_ts < _STATE_RATE_CACHE_TTL_S:
        return cached_val

    sess = session or requests.Session()
    try:
        r = sess.get(_EIA_RETAIL_SALES_URL, params={
            "api_key": api_key,
            "frequency": "monthly",
            "data[0]": "price",
            "facets[stateid][]": state,
            "facets[sectorid][]": "RES",
            "sort[0][column]": "period",
            "sort[0][direction]": "desc",
            "length": 1,
        }, timeout=20)
        r.raise_for_status()
        rows = r.json().get("response", {}).get("data", [])
    except Exception as exc:
        log.warning("EIA retail-sales fetch failed for %s: %s", state, exc)
        return None

    if not rows or rows[0].get("price") is None:
        _STATE_RATE_CACHE[state] = (time.time(), None)
        return None

    try:
        cents_per_kwh = float(rows[0]["price"])
    except (TypeError, ValueError):
        return None

    result = {
        "state": state,
        "period": rows[0]["period"],
        "res_rate_usd_mwh": round(cents_per_kwh * 10, 2),   # ¢/kWh -> $/MWh
    }
    _STATE_RATE_CACHE[state] = (time.time(), result)
    return result
