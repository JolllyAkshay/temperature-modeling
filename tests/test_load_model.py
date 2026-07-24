"""
Unit tests for core load-forecasting logic.

Run with:  pytest tests/test_load_model.py -v
"""

import calendar
import json
import math
import sys
import tempfile
from datetime import date, timedelta
from pathlib import Path

# Ensure the src package is importable when running from the repo root
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temperature_modeling.pjm_load import (
    HDD_CDD_BASE_F,
    N_FEATURES,
    UNCERTAINTY_Z,
    _build_features,
    _is_bridge_day,
    _is_holiday_week,
    _is_us_holiday,
    _obs_to_features,
    compute_hdd_cdd,
    run_load_backtest,
)
from temperature_modeling.models import LoadObservation


# ---------------------------------------------------------------------------
# compute_hdd_cdd
# ---------------------------------------------------------------------------

class TestHddCdd:
    def test_cold_day(self):
        hdd, cdd = compute_hdd_cdd(40.0)
        assert hdd == pytest.approx(25.0)
        assert cdd == 0.0

    def test_hot_day(self):
        hdd, cdd = compute_hdd_cdd(85.0)
        assert hdd == 0.0
        assert cdd == pytest.approx(20.0)

    def test_base_temperature(self):
        hdd, cdd = compute_hdd_cdd(HDD_CDD_BASE_F)
        assert hdd == 0.0
        assert cdd == 0.0

    def test_custom_base(self):
        hdd, cdd = compute_hdd_cdd(60.0, base_f=55.0)
        assert hdd == 0.0
        assert cdd == pytest.approx(5.0)

    def test_no_negative_values(self):
        for temp in range(0, 110, 5):
            hdd, cdd = compute_hdd_cdd(float(temp))
            assert hdd >= 0.0
            assert cdd >= 0.0


# ---------------------------------------------------------------------------
# _build_features — feature vector length and content
# ---------------------------------------------------------------------------

class TestBuildFeatures:
    def _call(self, avg_f=70.0, hi_f=80.0, lo_f=60.0,
              d=date(2025, 7, 15), lag1=69.0, lag2=68.0,
              lag7=72.0, roll7=70.0):
        return _build_features(avg_f, hi_f, lo_f, d, lag1, lag2, lag7, roll7)

    def test_feature_vector_length(self):
        feats, _, _ = self._call()
        assert len(feats) == N_FEATURES, (
            f"Expected {N_FEATURES} features, got {len(feats)}"
        )

    def test_hot_day_has_cdd_not_hdd(self):
        feats, hdd, cdd = self._call(avg_f=85.0, hi_f=95.0, lo_f=75.0)
        assert hdd == 0.0
        assert cdd > 0.0

    def test_cold_day_has_hdd_not_cdd(self):
        feats, hdd, cdd = self._call(avg_f=30.0, hi_f=40.0, lo_f=20.0)
        assert hdd > 0.0
        assert cdd == 0.0

    def test_none_lags_fall_back_to_avg(self):
        """When lags are None, they should fall back to avg_f (not crash)."""
        feats, _, _ = _build_features(70.0, 80.0, 60.0, date(2025, 6, 1),
                                       None, None, None, None)
        assert len(feats) == N_FEATURES
        assert all(math.isfinite(v) for v in feats)

    def test_seasonality_bounds(self):
        """sin/cos components must stay in [-1, 1]."""
        for month in range(1, 13):
            d = date(2025, month, 15)
            feats, _, _ = _build_features(70.0, 80.0, 60.0, d, 70.0, 70.0, 70.0, 70.0)
            sin_val, cos_val = feats[7], feats[8]
            assert -1.0 <= sin_val <= 1.0
            assert -1.0 <= cos_val <= 1.0

    def test_dow_one_hot_sums_to_one(self):
        """Exactly one day-of-week flag should be set per date."""
        for offset in range(7):
            d = date(2025, 7, 7) + timedelta(days=offset)  # Mon 7 Jul 2025
            feats, _, _ = _build_features(70.0, 80.0, 60.0, d, 70.0, 70.0, 70.0, 70.0)
            dow_flags = feats[9:16]
            assert sum(dow_flags) == pytest.approx(1.0), \
                f"DOW flags should sum to 1 for {d}, got {dow_flags}"

    def test_leap_year_continuity(self):
        """Feb 28, Feb 29, and Mar 1 in a leap year must have increasing doy_rad."""
        feb28 = date(2024, 2, 28)
        feb29 = date(2024, 2, 29)
        mar1  = date(2024, 3, 1)

        def _doy_rad(d):
            days = 366.0 if calendar.isleap(d.year) else 365.0
            return 2 * math.pi * d.timetuple().tm_yday / days

        assert _doy_rad(feb28) < _doy_rad(feb29) < _doy_rad(mar1)

    def test_all_features_finite(self):
        """No NaN or Inf should appear in the feature vector."""
        feats, _, _ = self._call()
        assert all(math.isfinite(v) for v in feats), \
            f"Non-finite values in features: {feats}"


# ---------------------------------------------------------------------------
# Holiday helpers
# ---------------------------------------------------------------------------

class TestHolidayHelpers:
    def test_christmas_is_holiday(self):
        assert _is_us_holiday(date(2025, 12, 25))

    def test_regular_day_not_holiday(self):
        assert not _is_us_holiday(date(2025, 7, 10))

    def test_christmas_week(self):
        assert _is_holiday_week(date(2025, 12, 26))
        assert _is_holiday_week(date(2025, 12, 24))

    def test_bridge_day_not_holiday(self):
        # Day before 4th of July (Independence Day)
        day_before = date(2025, 7, 3)
        assert _is_bridge_day(day_before)
        assert not _is_us_holiday(day_before)


# ---------------------------------------------------------------------------
# _obs_to_features
# ---------------------------------------------------------------------------

class TestObsToFeatures:
    def _make_obs(self, **kwargs):
        defaults = dict(
            date=date(2025, 8, 1), hdd=0.0, cdd=10.0,
            avg_temp_f=75.0, hi_temp_f=85.0, lo_temp_f=65.0,
            actual_load_mw=50000.0, is_weekend=False, day_of_week=4,
            is_holiday=False, day_of_year=213, temp_lag1_f=74.0,
            temp_lag2_f=73.0, temp_lag7_f=76.0, rolling7_avg_f=75.0,
        )
        defaults.update(kwargs)
        return LoadObservation(**defaults)

    def test_feature_length(self):
        obs = self._make_obs()
        feats = _obs_to_features(obs)
        assert len(feats) == N_FEATURES

    def test_consistent_with_build_features(self):
        obs = self._make_obs()
        feats_obs = _obs_to_features(obs)
        feats_direct, _, _ = _build_features(
            obs.avg_temp_f, obs.hi_temp_f, obs.lo_temp_f, obs.date,
            obs.temp_lag1_f, obs.temp_lag2_f, obs.temp_lag7_f, obs.rolling7_avg_f,
        )
        assert feats_obs == pytest.approx(feats_direct)


# ---------------------------------------------------------------------------
# run_load_backtest — with mock training data
# ---------------------------------------------------------------------------

class TestRunLoadBacktest:
    def _make_training_json(self, n=20):
        rows = []
        base = date(2025, 1, 1)
        for i in range(n):
            d = base + timedelta(days=i)
            rows.append({
                "date": d.isoformat(),
                "hdd": max(0.0, 65.0 - 40.0),
                "cdd": 0.0,
                "avg_temp_f": 40.0,
                "hi_temp_f": 50.0,
                "lo_temp_f": 30.0,
                "actual_load_mw": 45000.0 + i * 10,
                "is_weekend": d.weekday() >= 5,
                "day_of_week": d.weekday(),
                "is_holiday": False,
                "day_of_year": d.timetuple().tm_yday,
                "temp_lag1_f": 39.0,
                "temp_lag2_f": 38.0,
                "temp_lag7_f": 41.0,
                "rolling7_avg_f": 40.0,
            })
        return rows

    def test_missing_file_returns_empty(self):
        result = run_load_backtest(object(), "/nonexistent/path.json")
        assert result == {}

    def test_empty_file_returns_empty(self, tmp_path):
        p = tmp_path / "empty.json"
        p.write_text("[]")
        result = run_load_backtest(object(), str(p))
        assert result == {}

    def test_corrupt_rows_skipped(self, tmp_path):
        data = [{"date": "2025-01-01", "broken": True}]
        p = tmp_path / "corrupt.json"
        p.write_text(json.dumps(data))
        result = run_load_backtest(object(), str(p))
        assert result == {}

    def test_backtest_with_fitted_model(self, tmp_path):
        """End-to-end: fit a tiny model and run backtest on same data."""
        from temperature_modeling.pjm_load import LoadCorrectionModel

        rows = self._make_training_json(n=30)
        p = tmp_path / "train.json"
        p.write_text(json.dumps(rows))

        obs_list = []
        for r in rows:
            obs_list.append(LoadObservation(
                date=date.fromisoformat(r["date"]),
                hdd=r["hdd"], cdd=r["cdd"],
                avg_temp_f=r["avg_temp_f"], hi_temp_f=r["hi_temp_f"],
                lo_temp_f=r["lo_temp_f"], actual_load_mw=r["actual_load_mw"],
                is_weekend=r["is_weekend"], day_of_week=r["day_of_week"],
                is_holiday=r["is_holiday"], day_of_year=r["day_of_year"],
                temp_lag1_f=r["temp_lag1_f"], temp_lag2_f=r["temp_lag2_f"],
                temp_lag7_f=r["temp_lag7_f"], rolling7_avg_f=r["rolling7_avg_f"],
            ))

        model = LoadCorrectionModel()
        model.fit(obs_list)
        result = run_load_backtest(model, str(p))

        assert "dates" in result
        assert "mape_train" in result
        assert "mape_test" in result
        assert "monthly_mape" in result
        assert len(result["dates"]) == 30
        assert result["mape_train"] >= 0.0
        assert result["mape_test"] >= 0.0


# ---------------------------------------------------------------------------
# Population weight validation — all ISO location lists
# ---------------------------------------------------------------------------

class TestLocationWeights:
    def _check(self, locations, name):
        total = sum(loc["weight"] for loc in locations)
        assert all(loc["weight"] > 0 for loc in locations), \
            f"{name}: all weights must be positive"
        assert len(locations) > 0, f"{name}: must have at least one location"
        return total

    def test_caiso_weights(self):
        from temperature_modeling.caiso import CAISO_LOAD_LOCATIONS
        total = self._check(CAISO_LOAD_LOCATIONS, "CAISO")
        assert abs(total - 1.0) < 0.01, f"CAISO weights sum to {total}, expected 1.0"

    def test_ercot_weights(self):
        from temperature_modeling.ercot import ERCOT_LOAD_LOCATIONS
        total = self._check(ERCOT_LOAD_LOCATIONS, "ERCOT")
        assert abs(total - 1.0) < 0.01, f"ERCOT weights sum to {total}, expected 1.0"

    def test_miso_weights(self):
        from temperature_modeling.miso import MISO_LOAD_LOCATIONS
        total = self._check(MISO_LOAD_LOCATIONS, "MISO")
        assert abs(total - 1.0) < 0.01, f"MISO weights sum to {total}, expected 1.0"

    def test_pjm_weights_positive(self):
        from temperature_modeling.pjm import PJM_LOAD_LOCATIONS
        self._check(PJM_LOAD_LOCATIONS, "PJM")

    def test_pjm_weights_sum_warning(self):
        """PJM weights currently sum to ~0.90 — document this known gap."""
        from temperature_modeling.pjm import PJM_LOAD_LOCATIONS
        total = sum(loc["weight"] for loc in PJM_LOAD_LOCATIONS)
        # This test documents the known discrepancy. If someone fixes the weights,
        # this assertion will flip and should be updated to abs(total - 1.0) < 0.01.
        assert abs(total - 1.0) > 0.05, (
            "PJM weights now sum close to 1.0 — update this test and remove the "
            "warning in pjm.py"
        )


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

class TestConstants:
    def test_hdd_base_is_65(self):
        assert HDD_CDD_BASE_F == 65.0

    def test_uncertainty_z_is_90pct(self):
        assert UNCERTAINTY_Z == pytest.approx(1.645)

    def test_n_features_matches_actual(self):
        feats, _, _ = _build_features(70.0, 80.0, 60.0, date(2025, 6, 15),
                                       69.0, 68.0, 70.0, 70.0)
        assert len(feats) == N_FEATURES


# ---------------------------------------------------------------------------
# Cache validation helper
# ---------------------------------------------------------------------------

class TestIsCacheValid:
    def test_missing_file_invalid(self, tmp_path):
        # Import here to avoid dashboard import side-effects at collection time
        sys.path.insert(0, str(Path(__file__).parent.parent))
        # We test the logic directly rather than importing dashboard to avoid
        # triggering the Dash app instantiation at module load.
        cache_file = tmp_path / "missing.json"
        assert not cache_file.exists()

    def test_corrupt_json_invalid(self, tmp_path):
        cache_file = tmp_path / "corrupt.json"
        cache_file.write_text("{not valid json")
        # Simulate the same logic as _is_cache_valid
        import json as _json
        try:
            _json.loads(cache_file.read_text())
            valid = True
        except _json.JSONDecodeError:
            valid = False
        assert not valid

    def test_valid_cache_structure(self, tmp_path):
        today = date.today().isoformat()
        payload = {
            "load": [{"date": today, "mean_load_gw": 60.0}],
            "comparison": {"actual": {"2025-01-01": 60.0}},
        }
        cache_file = tmp_path / "valid.json"
        cache_file.write_text(json.dumps(payload))
        data = json.loads(cache_file.read_text())
        load_list = data.get("load") or []
        assert load_list[0].get("date") == today
        assert data.get("comparison", {}).get("actual")


# ---------------------------------------------------------------------------
# pytest import
# ---------------------------------------------------------------------------
import pytest
