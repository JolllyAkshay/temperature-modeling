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

from temperature_modeling.zip_lookup import lookup_zip


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
    assert peco_rows[0]["res_rate_cents_kwh"] is not None


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


def test_rate_is_in_cents_not_dollars():
    """Source CSV stores rates as a dollar fraction (e.g. 0.16) — must be converted to cents/kWh."""
    r = lookup_zip("19104")
    peco = next(u for u in r["utilities"] if u["name"] == "PECO Energy Co")
    # A residential rate should plausibly be single-to-double-digit cents/kWh,
    # not a fraction less than 1 (that would mean the *100 conversion was skipped).
    assert 5.0 < peco["res_rate_cents_kwh"] < 60.0


def test_data_vintage_year_present():
    r = lookup_zip("19104")
    assert r["data_vintage_year"] == 2024
