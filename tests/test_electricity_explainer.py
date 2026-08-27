"""
Unit tests for electricity_explainer.py's assembly logic.

Mocks the four underlying modules (zip_lookup, carbon_intensity,
capacity_market, market_competitiveness, forward_curve) — no live network
calls in this test file. Live end-to-end behavior against real data was
verified manually during development; see the module docstrings for what
was checked.

Run with:  pytest tests/test_electricity_explainer.py -v
"""

import sys
from pathlib import Path
from unittest import mock

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import temperature_modeling.electricity_explainer as ee


def _fake_zip_result(iso="pjm", found=True):
    return {
        "zip": "07029", "found": found,
        "utilities": [{"name": "Public Service Elec & Gas Co", "iso": iso, "iso_status": "mapped"}] if found else [],
        "iso": iso, "multi_utility": False, "data_vintage_year": 2024,
    }


def _fake_curve():
    return {
        "model_source": "ols-log-linear",
        "curve": [{
            "month": "2026-09",
            "scenarios": {"base": {"monthly_avg": 24.39, "on_peak": 26.51, "off_peak": 20.15}},
        }],
    }


@mock.patch("temperature_modeling.forward_curve.build_forward_curve")
@mock.patch.object(ee, "get_market_competitiveness")
@mock.patch.object(ee, "get_capacity_market_data")
@mock.patch.object(ee, "fetch_carbon_intensity")
@mock.patch.object(ee, "lookup_zip")
def test_full_assembly_for_rto_zip(mock_lookup, mock_fuel, mock_cap, mock_comp, mock_curve):
    mock_lookup.return_value = _fake_zip_result(iso="pjm")
    mock_fuel.return_value = {"lbs_co2_per_mwh": 700.0, "fuel_mix": {"Natural Gas": 30000}, "clean_pct": 40.0}
    mock_cap.return_value = {"mechanism": "RPM", "clearing_price_mw_year": 121706}
    mock_comp.return_value = {"headline_metric": "Energy market HHI", "headline_value": "714"}
    mock_curve.return_value = _fake_curve()

    r = ee.explain_electricity("07029")

    assert r["found"] is True
    assert r["iso"] == "pjm"
    assert r["non_rto"] is False
    assert r["fuel_mix"]["available"] is True
    assert r["fuel_mix"]["data"]["clean_pct"] == 40.0
    assert r["capacity_auctions"]["available"] is True
    assert r["market_competitiveness"]["available"] is True
    assert r["wholesale_price_context"]["available"] is True
    assert r["wholesale_price_context"]["data"]["monthly_avg_usd_mwh"] == 24.39
    assert "disclaimer" in r["wholesale_price_context"]
    assert "not your electric bill" in r["wholesale_price_context"]["disclaimer"] or \
           "your electric bill" in r["wholesale_price_context"]["disclaimer"]


@mock.patch.object(ee, "get_market_competitiveness")
@mock.patch.object(ee, "get_capacity_market_data")
@mock.patch.object(ee, "fetch_carbon_intensity")
@mock.patch.object(ee, "lookup_zip")
def test_non_rto_zip_degrades_gracefully(mock_lookup, mock_fuel, mock_cap, mock_comp):
    mock_lookup.return_value = _fake_zip_result(iso=None)

    r = ee.explain_electricity("30301")

    assert r["found"] is True
    assert r["iso"] is None
    assert r["non_rto"] is True
    for key in ("fuel_mix", "capacity_auctions", "market_competitiveness", "wholesale_price_context"):
        assert r[key]["available"] is False
        assert "reason" in r[key]
    # None of the underlying data-fetching modules should even be called
    # for a non-RTO zip — nothing to fetch.
    mock_fuel.assert_not_called()
    mock_cap.assert_not_called()
    mock_comp.assert_not_called()


@mock.patch.object(ee, "lookup_zip")
def test_zip_not_found_degrades_gracefully(mock_lookup):
    mock_lookup.return_value = {
        "zip": "99999", "found": False, "utilities": [], "iso": None,
        "multi_utility": False, "data_vintage_year": 2024,
    }
    r = ee.explain_electricity("99999")
    assert r["found"] is False
    for key in ("fuel_mix", "capacity_auctions", "market_competitiveness", "wholesale_price_context"):
        assert r[key]["available"] is False


@mock.patch.object(ee, "lookup_zip")
def test_malformed_zip_propagates_value_error(mock_lookup):
    mock_lookup.side_effect = ValueError("zip_code must be a 5-digit US zip code, got 'abc'")
    try:
        ee.explain_electricity("abc")
        assert False, "expected ValueError"
    except ValueError:
        pass


@mock.patch("temperature_modeling.forward_curve.build_forward_curve")
@mock.patch.object(ee, "get_market_competitiveness")
@mock.patch.object(ee, "get_capacity_market_data")
@mock.patch.object(ee, "fetch_carbon_intensity")
@mock.patch.object(ee, "lookup_zip")
def test_one_section_failing_does_not_break_the_others(mock_lookup, mock_fuel, mock_cap, mock_comp, mock_curve):
    """A single upstream module raising shouldn't take down the whole response."""
    mock_lookup.return_value = _fake_zip_result(iso="pjm")
    mock_fuel.side_effect = Exception("EIA is down")
    mock_cap.return_value = {"mechanism": "RPM"}
    mock_comp.return_value = {"headline_metric": "HHI", "headline_value": "714"}
    mock_curve.return_value = _fake_curve()

    r = ee.explain_electricity("07029")

    assert r["fuel_mix"]["available"] is False
    assert r["capacity_auctions"]["available"] is True
    assert r["market_competitiveness"]["available"] is True
    assert r["wholesale_price_context"]["available"] is True
