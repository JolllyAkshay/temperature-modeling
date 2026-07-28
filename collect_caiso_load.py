"""
Fetch historical CAISO load + ERA5 temperatures, train and save the CAISO
load correction model.

Usage:
    python collect_caiso_load.py           # fetch data + train
    python collect_caiso_load.py --retrain # force retrain even if model exists
"""

import json
import os
import sys
from datetime import date, timedelta
from pathlib import Path

import requests
from dotenv import load_dotenv
load_dotenv()

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from temperature_modeling._era5 import fetch_era5_daily
from temperature_modeling.models import Coordinates
from temperature_modeling.caiso import CAISO_LOAD_LOCATIONS
from temperature_modeling.caiso_load import (
    CAISOLoadModel,
    build_caiso_training_data,
    evaluate_load_model,
    fetch_caiso_load_daily,
    fetch_era5_daily_hi_lo,
    save_caiso_model,
    _CAISO_MODEL_PATH,
)

TRAINING_DATA_PATH = _HERE / "api_cache" / "caiso_load_training.json"


def main():
    retrain = "--retrain" in sys.argv

    session = requests.Session()
    session.headers.update({"User-Agent": "temperature-modeling/1.0"})

    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=1825)

    # ------------------------------------------------------------------
    # 1. CAISO RTO load
    # ------------------------------------------------------------------
    print(f"\n[1/4] Fetching CAISO (CISO) daily load  {start} to {end} ...")
    load_daily = fetch_caiso_load_daily(start, end, session)
    print(f"      {len(load_daily)} days fetched")

    # ------------------------------------------------------------------
    # 2. ERA5 avg + hi/lo for all 12 CAISO locations
    # ------------------------------------------------------------------
    print(f"\n[2/4] Fetching ERA5 temperatures for {len(CAISO_LOAD_LOCATIONS)} CAISO locations ...")
    era5_avg: dict = {}
    era5_hilo: dict = {}

    for i, loc in enumerate(CAISO_LOAD_LOCATIONS, 1):
        label  = loc["label"]
        coords = Coordinates(lat=loc["lat"], lon=loc["lon"])
        print(f"      [{i}/{len(CAISO_LOAD_LOCATIONS)}] {label} ...", end=" ", flush=True)

        avg_temps = fetch_era5_daily(coords, start, end, session, variable="temperature_2m")
        hi_lo     = fetch_era5_daily_hi_lo(coords, start, end, session)

        era5_avg[label]  = avg_temps
        era5_hilo[label] = hi_lo
        print(f"{len(avg_temps)} days avg, {len(hi_lo)} days hi/lo")

    # ------------------------------------------------------------------
    # 3. Build training observations
    # ------------------------------------------------------------------
    print("\n[3/4] Building training observations ...")
    observations = build_caiso_training_data(load_daily, era5_avg, era5_hilo)
    print(f"      {len(observations)} observations built")

    n_holiday  = sum(1 for o in observations if o.is_holiday)
    n_weekend  = sum(1 for o in observations if o.is_weekend)
    n_with_lag = sum(1 for o in observations if o.temp_lag1_f is not None)
    print(f"      Holidays: {n_holiday}  Weekends: {n_weekend}  With lag: {n_with_lag}")

    os.makedirs(TRAINING_DATA_PATH.parent, exist_ok=True)
    with open(TRAINING_DATA_PATH, "w", encoding="utf-8") as f:
        json.dump(
            [
                {
                    "date":           str(o.date),
                    "hdd":            o.hdd,
                    "cdd":            o.cdd,
                    "avg_temp_f":     o.avg_temp_f,
                    "hi_temp_f":      o.hi_temp_f,
                    "lo_temp_f":      o.lo_temp_f,
                    "actual_load_mw": o.actual_load_mw,
                    "is_weekend":     o.is_weekend,
                    "day_of_week":    o.day_of_week,
                    "is_holiday":     o.is_holiday,
                    "day_of_year":    o.day_of_year,
                    "temp_lag1_f":    o.temp_lag1_f,
                    "temp_lag2_f":    o.temp_lag2_f,
                    "temp_lag7_f":    o.temp_lag7_f,
                    "rolling7_avg_f": o.rolling7_avg_f,
                    "apparent_hi_f":  o.apparent_hi_f,
                    "dewpoint_hi_f":  o.dewpoint_hi_f,
                    "wind_speed_mph": o.wind_speed_mph,
                }
                for o in observations
            ],
            f, indent=2,
        )
    print(f"      Saved: {TRAINING_DATA_PATH}")

    # ------------------------------------------------------------------
    # 4. Train model
    # ------------------------------------------------------------------
    if os.path.exists(_CAISO_MODEL_PATH) and not retrain:
        print(f"\n[4/4] Model already exists at {_CAISO_MODEL_PATH}. Use --retrain to overwrite.")
        return

    print(f"\n[4/4] Training XGBoost CAISO load model on {len(observations)} observations ...")
    n_test    = max(1, int(len(observations) * 0.2))
    train_obs = observations[:-n_test]

    model = CAISOLoadModel()
    model.fit(train_obs)

    metrics = evaluate_load_model(model, observations)
    print(f"      Train samples : {metrics['n_train']}")
    print(f"      Test  samples : {metrics['n_test']}")
    print(f"      Test  RMSE    : {metrics['test_rmse_mw']:,.0f} MW")
    print(f"      Test  MAE     : {metrics['test_mae_mw']:,.0f} MW")
    print(f"      Test  MAPE    : {metrics['test_mape_pct']:.1f}%")

    save_caiso_model(model)
    print("\nDone.")


if __name__ == "__main__":
    main()
