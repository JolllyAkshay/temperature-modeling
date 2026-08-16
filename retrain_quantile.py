"""
Retrain all 7 ISO models with quantile regression (p10/p90) added.
Reads existing training JSONs, overwrites PKLs, then uploads to HuggingFace.

Usage:  python retrain_quantile.py
"""
import json
import os
import sys
import time
from datetime import date
from pathlib import Path

from dotenv import load_dotenv
load_dotenv()

sys.stdout.reconfigure(encoding="utf-8")

sys.path.insert(0, str(Path(__file__).parent / "src"))

from temperature_modeling.pjm_load import (
    LoadCorrectionModel, save_load_model, load_load_model,
    _MODEL_PATH,
)
from temperature_modeling.models import LoadObservation

_HERE = Path(__file__).parent
_CACHE = _HERE / "api_cache"

_ISO_TRAINING: dict = {
    "pjm":   str(_CACHE / "pjm_load_training.json"),
    "caiso": str(_CACHE / "caiso_load_training.json"),
    "ercot": str(_CACHE / "ercot_load_training.json"),
    "miso":  str(_CACHE / "miso_load_training.json"),
    "nyiso": str(_CACHE / "nyiso_load_training.json"),
    "isone": str(_CACHE / "isone_load_training.json"),
    "spp":   str(_CACHE / "spp_load_training.json"),
}

_ISO_PKL: dict = {
    "pjm":   str(_CACHE / "pjm_load_model.pkl"),
    "caiso": str(_CACHE / "caiso_load_model.pkl"),
    "ercot": str(_CACHE / "ercot_load_model.pkl"),
    "miso":  str(_CACHE / "miso_load_model.pkl"),
    "nyiso": str(_CACHE / "nyiso_load_model.pkl"),
    "isone": str(_CACHE / "isone_load_model.pkl"),
    "spp":   str(_CACHE / "spp_load_model.pkl"),
}


def _load_observations(path: str) -> list:
    from datetime import timedelta
    with open(path, encoding="utf-8") as f:
        raw = json.load(f)
    obs = []
    for r in raw:
        try:
            obs.append(LoadObservation(
                date=date.fromisoformat(r["date"]),
                hdd=r["hdd"], cdd=r["cdd"],
                avg_temp_f=r["avg_temp_f"],
                hi_temp_f=r.get("hi_temp_f", r["avg_temp_f"] + 5),
                lo_temp_f=r.get("lo_temp_f", r["avg_temp_f"] - 5),
                actual_load_mw=r["actual_load_mw"],
                is_weekend=r.get("is_weekend", False),
                day_of_week=r.get("day_of_week", 0),
                is_holiday=r.get("is_holiday", False),
                day_of_year=r.get("day_of_year", 1),
                temp_lag1_f=r.get("temp_lag1_f"),
                temp_lag2_f=r.get("temp_lag2_f"),
                temp_lag7_f=r.get("temp_lag7_f"),
                rolling7_avg_f=r.get("rolling7_avg_f"),
                apparent_hi_f=r.get("apparent_hi_f"),
                dewpoint_hi_f=r.get("dewpoint_hi_f"),
                wind_speed_mph=r.get("wind_speed_mph"),
                load_lag1_mw=r.get("load_lag1_mw"),
                load_lag7_mw=r.get("load_lag7_mw"),
                rolling7_load_mw=r.get("rolling7_load_mw"),
            ))
        except Exception as e:
            print(f"  Skipping row: {e}")

    # Compute load autocorrelation lags from actual_load_mw if not stored in JSON
    obs.sort(key=lambda o: o.date)
    if obs and obs[0].load_lag1_mw is None:
        load_by_date = {o.date: o.actual_load_mw for o in obs}
        for o in obs:
            o.load_lag1_mw = load_by_date.get(o.date - timedelta(days=1))
            o.load_lag7_mw = load_by_date.get(o.date - timedelta(days=7))
            vals = [load_by_date.get(o.date - timedelta(days=k)) for k in range(1, 8)]
            vals = [v for v in vals if v is not None]
            o.rolling7_load_mw = sum(vals) / len(vals) if vals else None

    return obs


def retrain_iso(iso: str) -> bool:
    training_path = _ISO_TRAINING[iso]
    pkl_path      = _ISO_PKL[iso]

    if not Path(training_path).exists():
        print(f"  [{iso.upper()}] Training data not found — skipping")
        return False

    print(f"  [{iso.upper()}] Loading training data…", end=" ", flush=True)
    obs = _load_observations(training_path)
    print(f"{len(obs)} observations")

    print(f"  [{iso.upper()}] Fitting p50 + p10 + p90 models…", end=" ", flush=True)
    t0 = time.time()
    model = LoadCorrectionModel()
    model.fit(obs)
    elapsed = time.time() - t0
    print(f"done in {elapsed:.0f}s")

    has_quantiles = model._model_p10 is not None
    print(f"  [{iso.upper()}] Quantile models: {'✓ p10/p90 trained' if has_quantiles else '✗ fallback only'}")

    save_load_model(model, pkl_path)
    print(f"  [{iso.upper()}] Saved → {pkl_path}")
    return True


def upload_pkls():
    from huggingface_hub import HfApi
    token = os.environ.get("HF_TOKEN")
    if not token:
        print("HF_TOKEN not set — skipping upload")
        return
    api = HfApi()
    for iso, pkl_path in _ISO_PKL.items():
        if not Path(pkl_path).exists():
            continue
        remote = f"api_cache/{Path(pkl_path).name}"
        print(f"  Uploading {remote}…", end=" ", flush=True)
        api.upload_file(
            path_or_fileobj=pkl_path,
            path_in_repo=remote,
            repo_id="JollyAkshay/grid-dashboard",
            repo_type="space",
            token=token,
        )
        print("✓")


if __name__ == "__main__":
    print("=== Quantile model retraining ===\n")
    trained = []
    for iso in _ISO_TRAINING:
        print(f"\n[{iso.upper()}]")
        if retrain_iso(iso):
            trained.append(iso)

    print(f"\n=== Retraining complete: {len(trained)}/{len(_ISO_TRAINING)} ISOs ===")
    print(f"ISOs trained: {', '.join(trained)}\n")

    print("=== Uploading PKLs to HuggingFace ===")
    upload_pkls()

    print("\nDone. HuggingFace Space will redeploy automatically.")
