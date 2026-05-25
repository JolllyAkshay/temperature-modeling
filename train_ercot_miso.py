"""
Train XGBoost load models for ERCOT and MISO.

Fetches 2 years of EIA hourly load data and ERA5 daily temperatures,
builds training observations, trains an XGBoost model for each ISO,
evaluates on a chronological 80/20 split, and saves the models to pkl.

Usage
-----
    python train_ercot_miso.py

Prerequisites
-------------
    pip install xgboost requests
    EIA_API_KEY environment variable (or use DEMO_KEY for limited access)

Runtime
-------
    ~10-20 minutes (ERA5 fetches for 12 locations × 2 years each ISO,
    most will be served from api_cache/ on subsequent runs).
"""

import json
import os
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta

import requests

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "src"))

from temperature_modeling._era5 import fetch_era5_daily
from temperature_modeling.ercot import ERCOT_LOAD_LOCATIONS
from temperature_modeling.ercot_load import (
    ERCOTLoadModel,
    build_ercot_training_data,
    fetch_ercot_load_daily,
    fetch_era5_daily_hi_lo,
    save_ercot_model,
    evaluate_load_model,
)
from temperature_modeling.miso import MISO_LOAD_LOCATIONS
from temperature_modeling.miso_load import (
    MISOLoadModel,
    build_miso_training_data,
    fetch_miso_load_daily,
    save_miso_model,
)
from temperature_modeling.models import Coordinates

# ---------------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------------

# 2-year training window; ERA5 has a ~5-day lag so end at least 7 days ago
TRAIN_END   = date.today() - timedelta(days=7)
TRAIN_START = TRAIN_END - timedelta(days=730)

TRAINING_DATA_DIR = os.path.join(os.path.dirname(__file__), "api_cache")

ERCOT_TRAINING_JSON = os.path.join(TRAINING_DATA_DIR, "ercot_load_training.json")
MISO_TRAINING_JSON  = os.path.join(TRAINING_DATA_DIR, "miso_load_training.json")


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _fetch_era5_for_locations(locations, start, end, session):
    """Fetch ERA5 avg-temp and hi/lo for all locations in parallel."""
    avg_by_label = {}
    hi_lo_by_label = {}

    def _avg(loc):
        coords = Coordinates(loc["lat"], loc["lon"])
        return loc["label"], fetch_era5_daily(coords, start, end, session)

    def _hilo(loc):
        coords = Coordinates(loc["lat"], loc["lon"])
        return loc["label"], fetch_era5_daily_hi_lo(coords, start, end, session)

    with ThreadPoolExecutor(max_workers=6) as pool:
        avg_futs  = {pool.submit(_avg,  loc): loc for loc in locations}
        hilo_futs = {pool.submit(_hilo, loc): loc for loc in locations}
        for fut in as_completed(avg_futs):
            try:
                label, data = fut.result()
                avg_by_label[label] = data
            except Exception as exc:
                print(f"  ERA5 avg fetch error: {exc}")
        for fut in as_completed(hilo_futs):
            try:
                label, data = fut.result()
                hi_lo_by_label[label] = data
            except Exception as exc:
                print(f"  ERA5 hi/lo fetch error: {exc}")

    return avg_by_label, hi_lo_by_label


def _obs_to_json(obs_list):
    return [
        {
            "date":          o.date.isoformat(),
            "hdd":           o.hdd,
            "cdd":           o.cdd,
            "avg_temp_f":    o.avg_temp_f,
            "hi_temp_f":     o.hi_temp_f,
            "lo_temp_f":     o.lo_temp_f,
            "actual_load_mw": o.actual_load_mw,
            "is_weekend":    o.is_weekend,
            "day_of_week":   o.day_of_week,
            "is_holiday":    o.is_holiday,
            "day_of_year":   o.day_of_year,
            "temp_lag1_f":   o.temp_lag1_f,
            "temp_lag2_f":   o.temp_lag2_f,
            "temp_lag7_f":   o.temp_lag7_f,
            "rolling7_avg_f": o.rolling7_avg_f,
        }
        for o in obs_list
    ]


def _train_iso(
    iso_name, locations, fetch_load_fn, build_training_fn,
    ModelClass, save_fn, training_json_path, session,
):
    print(f"\n{'='*60}")
    print(f"  {iso_name}  |  {TRAIN_START} to {TRAIN_END}")
    print(f"{'='*60}")

    # ── Load data ────────────────────────────────────────────────────────────
    if os.path.exists(training_json_path):
        print(f"  Loading cached training data from {training_json_path} ...")
        from temperature_modeling.models import LoadObservation
        raw = json.loads(open(training_json_path).read())
        obs = []
        for r in raw:
            try:
                obs.append(LoadObservation(
                    date=date.fromisoformat(r["date"]),
                    hdd=r["hdd"], cdd=r["cdd"],
                    avg_temp_f=r["avg_temp_f"],
                    hi_temp_f=r["hi_temp_f"],
                    lo_temp_f=r["lo_temp_f"],
                    actual_load_mw=r["actual_load_mw"],
                    is_weekend=r["is_weekend"],
                    day_of_week=r["day_of_week"],
                    is_holiday=r["is_holiday"],
                    day_of_year=r["day_of_year"],
                    temp_lag1_f=r.get("temp_lag1_f"),
                    temp_lag2_f=r.get("temp_lag2_f"),
                    temp_lag7_f=r.get("temp_lag7_f"),
                    rolling7_avg_f=r.get("rolling7_avg_f"),
                ))
            except (KeyError, ValueError):
                continue
        print(f"  Loaded {len(obs)} cached observations.")
    else:
        print(f"  Fetching EIA load data ({TRAIN_START} -> {TRAIN_END}) ...")
        load_daily = fetch_load_fn(TRAIN_START, TRAIN_END, session)
        print(f"  Got {len(load_daily)} daily load values.")

        print(f"  Fetching ERA5 temperatures for {len(locations)} locations ...")
        avg_by_label, hi_lo_by_label = _fetch_era5_for_locations(
            locations, TRAIN_START, TRAIN_END, session
        )

        print(f"  Building training observations ...")
        obs = build_training_fn(load_daily, avg_by_label, hi_lo_by_label)
        print(f"  Built {len(obs)} observations.")

        os.makedirs(TRAINING_DATA_DIR, exist_ok=True)
        open(training_json_path, "w").write(json.dumps(_obs_to_json(obs), indent=2))
        print(f"  Saved training data -> {training_json_path}")

    if len(obs) < 100:
        print(f"  ERROR: Too few observations ({len(obs)}) to train reliably.")
        return None

    # ── Train ────────────────────────────────────────────────────────────────
    print(f"  Training {ModelClass.__name__} ...")
    model = ModelClass()
    model.fit(obs)

    # ── Evaluate ─────────────────────────────────────────────────────────────
    ev = evaluate_load_model(model, obs, test_fraction=0.2)
    print(f"  Train n : {ev['n_train']}")
    print(f"  Test  n : {ev['n_test']}")
    print(f"  Test RMSE  : {ev['test_rmse_mw']:.0f} MW")
    print(f"  Test MAE   : {ev['test_mae_mw']:.0f} MW")
    print(f"  Test MAPE  : {ev['test_mape_pct']:.2f}%")

    # ── Save ─────────────────────────────────────────────────────────────────
    save_fn(model)
    print(f"  Model saved.")

    return model


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    session = requests.Session()
    session.headers["User-Agent"] = "grid-load-training/1.0"

    ercot_model = _train_iso(
        iso_name="ERCOT",
        locations=ERCOT_LOAD_LOCATIONS,
        fetch_load_fn=fetch_ercot_load_daily,
        build_training_fn=build_ercot_training_data,
        ModelClass=ERCOTLoadModel,
        save_fn=save_ercot_model,
        training_json_path=ERCOT_TRAINING_JSON,
        session=session,
    )

    miso_model = _train_iso(
        iso_name="MISO",
        locations=MISO_LOAD_LOCATIONS,
        fetch_load_fn=fetch_miso_load_daily,
        build_training_fn=build_miso_training_data,
        ModelClass=MISOLoadModel,
        save_fn=save_miso_model,
        training_json_path=MISO_TRAINING_JSON,
        session=session,
    )

    print("\n" + "="*60)
    print("  Done.")
    if ercot_model:
        print("  ERCOT model: api_cache/ercot_load_model.pkl")
    if miso_model:
        print("  MISO  model: api_cache/miso_load_model.pkl")
    print("  Run dashboard.py to see both ISOs in the UI.")
    print("="*60)


if __name__ == "__main__":
    main()
