"""
Fetch historical NYISO load + ERA5 temperatures, train and save the NYISO
load correction model.

Usage:
    python collect_nyiso_load.py           # fetch data + train
    python collect_nyiso_load.py --retrain # force retrain even if model exists
"""

import json
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
from temperature_modeling.nyiso import NYISO_LOAD_LOCATIONS
from temperature_modeling.nyiso_load import (
    _NYISO_MODEL_PATH,
    build_nyiso_training_data,
    fetch_nyiso_load_daily,
    weighted_avg_temp_f_nyiso,
)
from temperature_modeling.pjm_load import (
    LoadCorrectionModel,
    evaluate_load_model,
    fetch_era5_daily_hi_lo,
    save_load_model,
)

TRAINING_DATA_PATH = _HERE / "api_cache" / "nyiso_load_training.json"


def main():
    retrain = "--retrain" in sys.argv
    session = requests.Session()
    session.headers["User-Agent"] = "temperature-modeling/1.0"

    end   = date.today() - timedelta(days=1)
    start = end - timedelta(days=730)

    print(f"\n[1/4] Fetching NYISO (NYIS) daily load  {start} to {end} ...")
    load_daily = fetch_nyiso_load_daily(start, end, session)
    print(f"      {len(load_daily)} days fetched")

    print(f"\n[2/4] Fetching ERA5 for {len(NYISO_LOAD_LOCATIONS)} NYISO locations ...")
    era5_avg, era5_hilo = {}, {}
    for i, loc in enumerate(NYISO_LOAD_LOCATIONS, 1):
        label = loc["label"]
        coords = Coordinates(lat=loc["lat"], lon=loc["lon"])
        print(f"      [{i}/{len(NYISO_LOAD_LOCATIONS)}] {label} ...", end=" ", flush=True)
        era5_avg[label]  = fetch_era5_daily(coords, start, end, session)
        era5_hilo[label] = fetch_era5_daily_hi_lo(coords, start, end, session)
        print(f"{len(era5_avg[label])} days")

    print("\n[3/4] Building training observations ...")
    observations = build_nyiso_training_data(load_daily, era5_avg, era5_hilo)
    print(f"      {len(observations)} observations")

    TRAINING_DATA_PATH.parent.mkdir(parents=True, exist_ok=True)
    with TRAINING_DATA_PATH.open("w") as f:
        json.dump([{
            "date": str(o.date), "hdd": o.hdd, "cdd": o.cdd,
            "avg_temp_f": o.avg_temp_f, "hi_temp_f": o.hi_temp_f,
            "lo_temp_f": o.lo_temp_f, "actual_load_mw": o.actual_load_mw,
            "is_weekend": o.is_weekend, "day_of_week": o.day_of_week,
            "is_holiday": o.is_holiday, "day_of_year": o.day_of_year,
            "temp_lag1_f": o.temp_lag1_f, "temp_lag2_f": o.temp_lag2_f,
            "temp_lag7_f": o.temp_lag7_f, "rolling7_avg_f": o.rolling7_avg_f,
            "apparent_hi_f": o.apparent_hi_f,
        } for o in observations], f, indent=2)

    if Path(_NYISO_MODEL_PATH).exists() and not retrain:
        print(f"\n[4/4] Model exists at {_NYISO_MODEL_PATH}. Pass --retrain to overwrite.")
        return

    print(f"\n[4/4] Training XGBoost NYISO model on {len(observations)} observations ...")
    n_test  = max(1, int(len(observations) * 0.2))
    model   = LoadCorrectionModel()
    model.fit(observations[:-n_test])
    metrics = evaluate_load_model(model, observations)
    print(f"      Test MAPE: {metrics['test_mape_pct']:.1f}%  RMSE: {metrics['test_rmse_mw']:,.0f} MW")
    save_load_model(model, _NYISO_MODEL_PATH)
    print(f"      Saved: {_NYISO_MODEL_PATH}")
    print("\nDone.")


if __name__ == "__main__":
    main()
