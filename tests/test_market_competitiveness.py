"""
Sanity tests for market_competitiveness.py's curated data.

Run with:  pytest tests/test_market_competitiveness.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temperature_modeling.market_competitiveness import get_market_competitiveness

_ALL_ISOS = ["pjm", "caiso", "ercot", "miso", "nyiso", "isone", "spp"]


@pytest.mark.parametrize("iso", _ALL_ISOS)
def test_every_iso_has_curated_data(iso):
    r = get_market_competitiveness(iso)
    assert "error" not in r
    assert r["iso"] == iso.upper()
    assert r["source"]                      # every entry must be sourced
    assert r["assessment"]
    assert r["year"] == 2024


def test_unknown_iso_returns_error():
    r = get_market_competitiveness("not_a_real_iso")
    assert "error" in r


def test_case_insensitive():
    assert get_market_competitiveness("PJM")["iso"] == "PJM"
    assert get_market_competitiveness("pjm")["iso"] == "PJM"


@pytest.mark.parametrize("iso", ["pjm", "caiso", "ercot", "miso", "isone", "spp"])
def test_most_isos_have_a_headline_metric(iso):
    """
    NYISO's 2024 report is the one confirmed exception (qualitative
    assessment only, no single structural number) — every other ISO's
    monitor reports a specific headline metric.
    """
    r = get_market_competitiveness(iso)
    assert r["headline_metric"] is not None
    assert r["headline_value"] is not None


def test_nyiso_is_the_documented_qualitative_exception():
    r = get_market_competitiveness("nyiso")
    assert r["headline_metric"] is None
    assert r["headline_value"] is None
    assert "competitively" in r["assessment"].lower()


def test_no_metric_falsely_claims_to_be_bid_data():
    """
    The whole point of this module is to NOT pretend to show individual
    trader bids (which are never public) — guard against that framing
    creeping back in.
    """
    for iso in _ALL_ISOS:
        r = get_market_competitiveness(iso)
        text = f"{r.get('headline_metric')} {r.get('assessment')}".lower()
        assert "trader" not in text
        assert " bid " not in f" {text} "
