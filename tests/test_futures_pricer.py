"""
Unit tests for the PJM futures contract pricer and its supporting pieces
(NERC peak-hour classification, the price-model collinearity fix that made
separate peak/off-peak models possible).

Run with:  pytest tests/test_futures_pricer.py -v
"""

import math
import sys
from datetime import date, datetime
from pathlib import Path
from unittest import mock

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temperature_modeling._nerc_calendar import is_nerc_holiday, is_peak_hour, nerc_holidays
from temperature_modeling.price_forecast import _PJM_WESTERN_HUB_ID, _fit_price_model
import temperature_modeling.futures_pricer as fp


# ---------------------------------------------------------------------------
# NERC calendar
# ---------------------------------------------------------------------------

def test_pjm_western_hub_id_is_correct():
    """Regression guard: pnode_id=1 is PJM-RTO (wrong), Western Hub is 51288."""
    assert _PJM_WESTERN_HUB_ID == 51288


def test_peak_hour_window_boundaries():
    # 2026-08-25 is a Tuesday (confirmed via date.weekday())
    tue = date(2026, 8, 25)
    assert tue.weekday() == 1
    assert is_peak_hour(datetime(2026, 8, 25, 6, 0)) is False   # before HE0800
    assert is_peak_hour(datetime(2026, 8, 25, 7, 0)) is True    # HE0800 starts here
    assert is_peak_hour(datetime(2026, 8, 25, 22, 0)) is True   # HE2300 starts here
    assert is_peak_hour(datetime(2026, 8, 25, 23, 0)) is False  # past HE2300


def test_saturday_and_sunday_fully_off_peak():
    sat = date(2026, 8, 22)
    sun = date(2026, 8, 23)
    assert sat.weekday() == 5 and sun.weekday() == 6
    for h in range(24):
        assert is_peak_hour(datetime(2026, 8, 22, h)) is False
        assert is_peak_hour(datetime(2026, 8, 23, h)) is False


def test_nerc_holiday_on_a_weekday_is_off_peak():
    # Christmas 2026 falls on a Friday — would otherwise be a normal peak day
    xmas = date(2026, 12, 25)
    assert xmas.weekday() == 4
    assert is_nerc_holiday(xmas) is True
    assert is_peak_hour(datetime(2026, 12, 25, 12, 0)) is False
    # Control: the day before is a normal Thursday, same hour is peak
    assert is_peak_hour(datetime(2026, 12, 24, 12, 0)) is True


def test_nerc_holidays_six_per_year():
    holidays = nerc_holidays(2026)
    assert len(holidays) == 6
    assert date(2026, 1, 1) in holidays    # New Year's
    assert date(2026, 7, 4) in holidays    # Independence Day
    assert date(2026, 12, 25) in holidays  # Christmas
    # Memorial Day: last Monday in May
    memorial = next(d for d in holidays if d.month == 5)
    assert memorial.weekday() == 0
    assert (memorial + __import__("datetime").timedelta(days=7)).month == 6
    # Labor Day: first Monday in September
    labor = next(d for d in holidays if d.month == 9)
    assert labor.weekday() == 0 and labor.day <= 7
    # Thanksgiving: 4th Thursday in November
    thanksgiving = next(d for d in holidays if d.month == 11)
    assert thanksgiving.weekday() == 3


# ---------------------------------------------------------------------------
# _fit_price_model collinearity fix
# ---------------------------------------------------------------------------

def _synthetic_history(n=60, constant_weekday=None, start_year=2025, start_month=1):
    """
    Build a minimal synthetic history spanning several months. If
    constant_weekday is 0.0 or 1.0, every row's date is forced to a weekday
    or weekend to make the weekday_frac feature degenerate — reproducing
    the exact condition that broke the on-peak-only fit before the fix.
    """
    rows = []
    d = date(start_year, start_month, 1)
    made = 0
    while made < n:
        is_wd = d.weekday() < 5
        if constant_weekday == 1.0 and not is_wd:
            d = date.fromordinal(d.toordinal() + 1)
            continue
        if constant_weekday == 0.0 and is_wd:
            d = date.fromordinal(d.toordinal() + 1)
            continue
        price = 30.0 + 5.0 * math.sin(2 * math.pi * d.month / 12) + (made % 7)
        rows.append({
            "date": d.isoformat(),
            "load_mw": 90000.0 + 500 * (made % 5),
            "price_usd_mwh": round(price, 2),
        })
        made += 1
        d = date.fromordinal(d.toordinal() + 1)
    return rows


def test_fit_price_model_drops_degenerate_constant_feature():
    """
    Before the fix: fitting on rows that are ALL weekdays (as PJM's on-peak
    subset always is, since peak hours only exist Mon-Fri) made weekday_frac
    collinear with the intercept, producing an unstable fit that blew up at
    prediction time. Now that column should be detected and dropped.
    """
    history = _synthetic_history(n=80, constant_weekday=1.0)
    model = _fit_price_model(history)
    assert model is not None
    assert "weekday_frac" not in model["feature_names"]
    assert "intercept" in model["feature_names"]
    assert len(model["coeffs"]) == len(model["feature_names"])
    assert all(math.isfinite(c) for c in model["coeffs"])


def test_fit_price_model_keeps_varying_weekday_feature():
    """A normal mixed weekday/weekend history should keep weekday_frac."""
    history = _synthetic_history(n=80, constant_weekday=None)
    model = _fit_price_model(history)
    assert model is not None
    assert "weekday_frac" in model["feature_names"]


# ---------------------------------------------------------------------------
# price_contract — validation (no mocking needed, fails before build_forward_curve)
# ---------------------------------------------------------------------------

@pytest.mark.parametrize("kwargs,match", [
    (dict(iso="caiso", delivery_month="2026-09", peak_type="monthly_avg", quoted_price=30), "scoped to PJM"),
    (dict(iso="pjm", delivery_month="2026-09", peak_type="nonsense", quoted_price=30), "peak_type must be"),
    (dict(iso="pjm", delivery_month="2026-09", peak_type="monthly_avg", quoted_price=-5), "positive number"),
    (dict(iso="pjm", delivery_month="2026-09", peak_type="monthly_avg", quoted_price=0), "positive number"),
    (dict(iso="pjm", delivery_month="September 2026", peak_type="monthly_avg", quoted_price=30), "YYYY-MM"),
    (dict(iso="pjm", delivery_month="2026-13", peak_type="monthly_avg", quoted_price=30), "YYYY-MM"),
    (dict(iso="pjm", delivery_month="2020-01", peak_type="monthly_avg", quoted_price=30), "not in the future"),
    (dict(iso="pjm", delivery_month="2099-01", peak_type="monthly_avg", quoted_price=30), "months out"),
])
def test_price_contract_validation_errors(kwargs, match):
    with pytest.raises(ValueError, match=match):
        fp.price_contract(**kwargs)


def test_price_contract_rejects_bad_scenario():
    with pytest.raises(ValueError, match="scenario must be"):
        fp.price_contract(iso="pjm", delivery_month="2026-09", peak_type="monthly_avg",
                           quoted_price=30, scenario="mild")


# ---------------------------------------------------------------------------
# price_contract — spread/signal/confidence logic, via a controlled curve
# ---------------------------------------------------------------------------

def _fake_curve(month="2026-09", monthly_avg=30.0, low=20.0, high=40.0,
                 on_peak=35.0, on_peak_low=25.0, on_peak_high=45.0,
                 peak_split_method="empirical_hourly", model_source="ols-log-linear"):
    return {
        "model_source": model_source,
        "curve": [{
            "month": month,
            "scenarios": {
                "base": {
                    "monthly_avg": monthly_avg, "low_usd_mwh": low, "high_usd_mwh": high,
                    "on_peak": on_peak, "on_peak_low_usd_mwh": on_peak_low, "on_peak_high_usd_mwh": on_peak_high,
                    "off_peak": 20.0, "off_peak_low_usd_mwh": None, "off_peak_high_usd_mwh": None,
                    "peak_split_method": peak_split_method,
                },
            },
        }],
    }


@mock.patch.object(fp, "fetch_hh_futures", return_value={"2026-09": 3.0})
@mock.patch.object(fp, "build_forward_curve")
def test_signal_within_band(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve(monthly_avg=30.0, low=20.0, high=40.0)
    r = fp.price_contract("pjm", "2026-09", "monthly_avg", 35.0)
    assert r["signal"] == "within_band"
    assert r["model_price"] == 30.0
    assert r["spread_usd_mwh"] == 5.0
    assert r["spread_pct"] == pytest.approx(16.7, abs=0.1)
    assert r["confidence"] == "normal"


@mock.patch.object(fp, "fetch_hh_futures", return_value={"2026-09": 3.0})
@mock.patch.object(fp, "build_forward_curve")
def test_signal_above_band(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve(monthly_avg=30.0, low=20.0, high=40.0)
    r = fp.price_contract("pjm", "2026-09", "monthly_avg", 100.0)
    assert r["signal"] == "above_band"


@mock.patch.object(fp, "fetch_hh_futures", return_value={"2026-09": 3.0})
@mock.patch.object(fp, "build_forward_curve")
def test_signal_below_band(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve(monthly_avg=30.0, low=20.0, high=40.0)
    r = fp.price_contract("pjm", "2026-09", "monthly_avg", 5.0)
    assert r["signal"] == "below_band"


@mock.patch.object(fp, "fetch_hh_futures", return_value={"2026-09": 3.0})
@mock.patch.object(fp, "build_forward_curve")
def test_no_band_available_when_fallback_heuristic(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve(low=None, high=None, model_source="fallback-heuristic")
    r = fp.price_contract("pjm", "2026-09", "monthly_avg", 35.0)
    assert r["band_source"] == "none"
    assert r["signal"] == "no_band_available"
    assert r["confidence"] == "reduced"
    assert any("fallback heuristic" in n for n in r["confidence_notes"])


@mock.patch.object(fp, "fetch_hh_futures", return_value={"2026-09": 3.0})
@mock.patch.object(fp, "build_forward_curve")
def test_on_peak_uses_its_own_band_not_the_blended_one(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve(monthly_avg=30.0, low=20.0, high=40.0,
                                           on_peak=35.0, on_peak_low=25.0, on_peak_high=45.0)
    r = fp.price_contract("pjm", "2026-09", "on_peak", 42.0)
    assert r["model_price"] == 35.0
    assert r["band_low"] == 25.0 and r["band_high"] == 45.0
    assert r["signal"] == "within_band"   # 42 is within [25,45] but outside the blended [20,40]


@mock.patch.object(fp, "fetch_hh_futures", return_value={})   # month not in STEO's real coverage
@mock.patch.object(fp, "build_forward_curve")
def test_confidence_flags_missing_gas_curve_month(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve()
    r = fp.price_contract("pjm", "2026-09", "monthly_avg", 30.0)
    assert r["gas_curve_covers_month"] is False
    assert r["confidence"] == "reduced"


@mock.patch.object(fp, "fetch_hh_futures", return_value={})
@mock.patch.object(fp, "build_forward_curve")
def test_confidence_flags_long_lead_time(mock_curve, mock_gas):
    # 24 months out from "today", computed dynamically so this doesn't rot
    # into a false negative once real dates move past a hardcoded year.
    today = date.today()
    y, m = today.year + (today.month + 24 - 1) // 12, (today.month + 24 - 1) % 12 + 1
    far_month = f"{y:04d}-{m:02d}"
    mock_curve.return_value = _fake_curve(month=far_month)
    r = fp.price_contract("pjm", far_month, "monthly_avg", 30.0)
    assert r["lead_months"] > 18
    assert r["confidence"] == "reduced"
    assert any("STEO" in n for n in r["confidence_notes"])


@mock.patch.object(fp, "fetch_hh_futures", return_value={"2026-09": 3.0})
@mock.patch.object(fp, "build_forward_curve")
def test_confidence_flags_synthetic_peak_split_for_peak_types_only(mock_curve, mock_gas):
    mock_curve.return_value = _fake_curve(peak_split_method="synthetic_ratio")

    r_peak = fp.price_contract("pjm", "2026-09", "on_peak", 30.0)
    assert r_peak["confidence"] == "reduced"
    assert any("synthetic" in n for n in r_peak["confidence_notes"])

    # monthly_avg doesn't go through the peak split at all — shouldn't be flagged for it
    r_avg = fp.price_contract("pjm", "2026-09", "monthly_avg", 30.0)
    assert not any("synthetic" in n for n in r_avg["confidence_notes"])
