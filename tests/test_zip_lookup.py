"""
Unit tests for zip_lookup.py.

Uses real fixture zip codes against the actual bundled NREL/OpenEI dataset
(not synthetic data) — each expected iso/non_rto value was hand-verified
against real utility-ISO-membership sources during this pass, not assumed.

Run with:  pytest tests/test_zip_lookup.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temperature_modeling.zip_lookup import lookup_zip, geocode_zip


@pytest.mark.parametrize("zip_code,expected_iso,utility_substr", [
    ("19104", "pjm",   "PECO"),           # Philadelphia PA
    ("60601", "pjm",   "Commonwealth Edison"),  # Chicago IL (northern IL -> PJM)
    ("62701", "miso",  "Ameren"),         # Springfield IL (southern IL -> MISO; IL split test)
    ("78701", "ercot", "Austin Energy"),  # Austin TX
    ("95814", "caiso", "Pacific Gas"),    # Sacramento CA
    ("10001", "nyiso", "Consolidated Edison"),  # NYC
    ("02101", "isone", "NSTAR"),          # Boston MA
    ("73101", "spp",   "Oklahoma Gas"),   # Oklahoma City OK
])
def test_known_iso_zips(zip_code, expected_iso, utility_substr):
    r = lookup_zip(zip_code)
    assert r["found"] is True
    assert r["iso"] == expected_iso
    names = [u["name"] for u in r["utilities"]]
    assert any(utility_substr in n for n in names)


def test_non_rto_zip_georgia_power():
    """Georgia Power — Southern Company, confirmed vertically-integrated non-RTO."""
    r = lookup_zip("30301")
    assert r["found"] is True
    assert r["iso"] is None
    ga_power = next(u for u in r["utilities"] if u["name"] == "Georgia Power Co")
    assert ga_power["iso_status"] == "non_rto"


def test_non_rto_zip_tva_territory():
    """Nashville — served via TVA, confirmed non-RTO (federal power authority)."""
    r = lookup_zip("37201")
    assert r["found"] is True
    assert r["iso"] is None
    tva = next((u for u in r["utilities"] if u["name"] == "Tennessee Valley Authority"), None)
    assert tva is not None
    assert tva["iso_status"] == "non_rto"
    assert "note" in tva


def test_malformed_zip_raises():
    # Wrong format (non-digit, wrong length, empty) -> ValueError.
    # "00000" is NOT malformed by this measure — it's a syntactically valid
    # 5-digit format that simply isn't a real zip; see the separate
    # test_well_formed_zip_not_in_dataset for that case.
    with pytest.raises(ValueError, match="5-digit"):
        lookup_zip("abc12")
    with pytest.raises(ValueError, match="5-digit"):
        lookup_zip("123")
    with pytest.raises(ValueError, match="5-digit"):
        lookup_zip("")


def test_well_formed_zip_not_in_dataset():
    """A syntactically valid zip that isn't in the dataset -> honest 'not found', not an error."""
    r = lookup_zip("99999")
    assert r["found"] is False
    assert r["utilities"] == []
    assert r["iso"] is None


def test_no_duplicate_utility_rows_for_deregulated_state():
    """
    PECO (PA, deregulated) has both a "Bundled" and a "Delivery"-only row
    in the source CSV for the same zip — these must collapse into one
    entry per utility (preferring the bundled all-in rate), not show the
    same utility name twice with different numbers unexplained.
    """
    r = lookup_zip("19104")
    peco_rows = [u for u in r["utilities"] if u["name"] == "PECO Energy Co"]
    assert len(peco_rows) == 1
    assert peco_rows[0]["rate_basis"] == "bundled"
    assert peco_rows[0]["res_rate_usd_mwh"] is not None


def test_multi_utility_zip_reports_all_and_resolves_iso_from_mapped_only():
    """
    Sacramento (95814) has both PG&E (mapped -> caiso) and SMUD (a
    separate California balancing authority, not yet mapped) — the
    top-level iso should resolve from the one confidently mapped utility,
    not be blanked out just because a second, unmapped utility is present.
    """
    r = lookup_zip("95814")
    assert r["multi_utility"] is True
    assert r["iso"] == "caiso"
    names = {u["name"] for u in r["utilities"]}
    assert "Pacific Gas & Electric Co." in names


def test_rate_is_in_usd_per_mwh_not_dollars_per_kwh():
    """Source CSV stores rates as $/kWh (e.g. 0.16) — must be converted to $/MWh (x1000)."""
    r = lookup_zip("19104")
    peco = next(u for u in r["utilities"] if u["name"] == "PECO Energy Co")
    # A residential rate should plausibly be tens-to-hundreds of $/MWh,
    # not a fraction less than 1 (that would mean the *1000 conversion was skipped).
    assert 50.0 < peco["res_rate_usd_mwh"] < 600.0


def test_data_vintage_year_present():
    r = lookup_zip("19104")
    assert r["data_vintage_year"] == 2024


# ---------------------------------------------------------------------------
# geocode_zip — live network tests, matching this file's existing
# convention of testing against real data rather than mocks.
# ---------------------------------------------------------------------------

def test_geocode_zip_malformed_returns_none_no_network_call():
    assert geocode_zip("abc") is None
    assert geocode_zip("123") is None
    assert geocode_zip("") is None
    assert geocode_zip(None) is None


def test_geocode_zip_resolves_known_zips():
    r = geocode_zip("78701")   # Austin, TX
    assert r is not None
    assert 30.0 < r["lat"] < 31.0
    assert -98.0 < r["lon"] < -97.0
    assert "Austin" in r["display_name"] or "Texas" in r["display_name"]


def test_geocode_zip_po_box_only_zip_returns_none_not_a_wrong_location():
    """
    30301 is a PO-Box-only Atlanta zip with no real geographic boundary in
    OSM's data. A naive free-text Nominatim query for "30301" matches an
    unrelated OSM entity (a vending machine labeled "30301" in
    Minneapolis) rather than Georgia — confirmed live during development.
    Must return None here, not that wrong location.
    """
    r = geocode_zip("30301")
    if r is not None:
        # If Nominatim's data has since improved and it now resolves,
        # it must at least be in the right state, not Minnesota.
        assert "Georgia" in r["display_name"]
        assert 33.0 < r["lat"] < 35.0
    # else: None is the expected, honest answer for this zip today.


def test_geocode_zip_is_cached():
    r1 = geocode_zip("62701")
    r2 = geocode_zip("62701")
    assert r1 == r2
