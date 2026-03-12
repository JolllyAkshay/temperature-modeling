"""
Collect a real training dataset from the Open-Meteo API.

Fetches GraphCast historical forecasts (gfs_graphcast025) paired with ERA5
reanalysis observations for a configurable date range and location, then
trains all five correction models and saves the results.

Usage
-----
    python collect_training_data.py

Adjust COORDS, START_DATE, END_DATE, and OUTPUT_FILE at the top as needed.

Data availability
-----------------
GraphCast archive starts 2024-02-05.  ERA5 has a ~5-day lag, so END_DATE
should be at least 16 days before today to ensure all valid dates are
available in the archive.

Rate limits
-----------
Open-Meteo's free tier allows ~10,000 calls/day and has no hard rate limit,
but the script adds a small delay between requests to be polite.  For a full
year of init dates (~365 calls × 2 APIs) expect ~5–10 minutes.
"""

import json
import time
from datetime import date, timedelta
from pathlib import Path

import requests

from temperature_modeling import (
    build_climate_normals,
    collect_verification_records,
    extract_features,
    score_by_lead,
    train_and_evaluate,
)
from temperature_modeling.models import Coordinates

# ── Configuration ─────────────────────────────────────────────────────────────

# Target location (Columbus, OH — change to your area of interest)
COORDS = Coordinates(lat=39.96, lon=-82.99)

# GraphCast archive starts 2024-02-05; use a window ending ~20 days ago
# so all 16-day valid dates are in the ERA5 archive.
START_DATE = date(2024, 3, 1)    # first init date to verify
END_DATE   = date(2025, 12, 31)  # last init date to verify

# Output file for the raw verification records (JSON)
OUTPUT_FILE = Path("verification_records.json")

# Model types to train
MODEL_TYPES = ["mean_bias", "linear", "ridge", "random_forest", "xgboost"]

# Seconds to sleep between each init-date API pair (be polite to Open-Meteo)
REQUEST_DELAY = 0.5


# ── Helpers ───────────────────────────────────────────────────────────────────

def _date_range(start: date, end: date):
    d = start
    while d <= end:
        yield d
        d += timedelta(days=1)


def save_records(records, path: Path):
    data = [
        {
            "init_date":        r.init_date.isoformat(),
            "valid_date":       r.valid_date.isoformat(),
            "lead_days":        r.lead_days,
            "forecast_temp_c":  r.forecast_temp_c,
            "observed_temp_c":  r.observed_temp_c,
            "error_c":          r.error_c,
        }
        for r in records
    ]
    path.write_text(json.dumps(data, indent=2))
    print(f"  Saved {len(data)} records → {path}")


def load_records(path: Path):
    from temperature_modeling.models import ForecastSample
    data = json.loads(path.read_text())
    return [
        ForecastSample(
            init_date=date.fromisoformat(d["init_date"]),
            valid_date=date.fromisoformat(d["valid_date"]),
            lead_days=d["lead_days"],
            forecast_temp_c=d["forecast_temp_c"],
            observed_temp_c=d["observed_temp_c"],
            error_c=d["error_c"],
        )
        for d in data
    ]


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    session = requests.Session()
    session.headers["User-Agent"] = "temperature-modeling-research/0.1"

    # ── Step 1: collect verification records ──────────────────────────────────
    if OUTPUT_FILE.exists():
        print(f"Loading cached records from {OUTPUT_FILE} ...")
        records = load_records(OUTPUT_FILE)
    else:
        all_dates = list(_date_range(START_DATE, END_DATE))
        print(f"Collecting verification records for {len(all_dates)} init dates ...")
        print(f"  Location : {COORDS.lat}°N, {COORDS.lon}°E")
        print(f"  Period   : {START_DATE} → {END_DATE}")
        print()

        records = []
        for i, init_date in enumerate(all_dates, 1):
            try:
                batch = collect_verification_records(
                    COORDS, [init_date], session, max_lead_days=16
                )
                records.extend(batch)
                status = f"  [{i:>4}/{len(all_dates)}] {init_date}  +{len(batch):>2} samples  total={len(records)}"
                print(status, end="\r")
                time.sleep(REQUEST_DELAY)
            except Exception as exc:
                print(f"\n  [{i:>4}/{len(all_dates)}] {init_date}  ERROR: {exc}")

        print()  # newline after \r progress
        save_records(records, OUTPUT_FILE)

    print(f"\nLoaded {len(records)} forecast–observation pairs.")

    # ── Step 2: raw skill by lead time ────────────────────────────────────────
    skill = score_by_lead(records)
    print(f"\n{'─'*60}")
    print(f"  RAW GRAPHCAST SKILL BY LEAD TIME")
    print(f"{'─'*60}")
    print(f"  {'Lead':>4}  {'RMSE':>7}  {'MAE':>7}  {'Bias':>7}  {'n':>5}")
    print(f"  {'────':>4}  {'───────':>7}  {'───────':>7}  {'───────':>7}  {'─────':>5}")
    for s in skill:
        marker = " ◄" if 10 <= s.lead_days <= 15 else ""
        print(f"  {s.lead_days:>4}  {s.rmse:>7.3f}  {s.mae:>7.3f}  {s.bias:>+7.3f}  {s.n:>5}{marker}")
    print(f"\n  ◄ = target 10–15 day window")

    # ── Step 3: feature extraction ────────────────────────────────────────────
    # Anchor climatology to the start of the verification window to prevent
    # data leakage (ERA5 years before START_DATE only).
    print(f"\nBuilding ERA5 climatological normals (5 years before {START_DATE}) ...")
    normals = build_climate_normals(COORDS, START_DATE, session)
    vectors = extract_features(records, normals)
    print(f"Extracted {len(vectors)} feature vectors.")

    # ── Step 4: train and evaluate correction models ──────────────────────────
    print(f"\n{'─'*60}")
    print(f"  CORRECTION MODEL COMPARISON  (chronological 80/20 split)")
    print(f"{'─'*60}")
    print(
        f"  {'Model':<16} {'Train n':>7} {'Test n':>7} "
        f"{'Raw RMSE':>9} {'Corr RMSE':>10} {'Skill':>8}"
    )
    print(
        f"  {'─'*16} {'───────':>7} {'───────':>7} "
        f"{'─────────':>9} {'──────────':>10} {'────────':>8}"
    )

    best_model = None
    best_eval = None
    for mtype in MODEL_TYPES:
        try:
            model, ev = train_and_evaluate(vectors, model_type=mtype)
            skill_pct = f"{ev.window_skill_score:+.1%}"
            print(
                f"  {ev.model_type:<16} {ev.n_train:>7} {ev.n_test:>7} "
                f"{ev.window_raw_rmse:>9.3f} {ev.window_corrected_rmse:>10.3f} {skill_pct:>8}"
            )
            if best_eval is None or ev.window_corrected_rmse < best_eval.window_corrected_rmse:
                best_model = model
                best_eval = ev
        except Exception as exc:
            print(f"  {mtype:<16}  [skipped: {exc}]")

    # ── Step 5: per-lead breakdown for best model ─────────────────────────────
    if best_eval is not None:
        print(f"\n{'─'*60}")
        print(f"  PER-LEAD BREAKDOWN — {best_eval.model_type}")
        print(f"{'─'*60}")
        print(f"  {'Lead':>4}  {'Raw RMSE':>9}  {'Corr RMSE':>10}  {'Δ RMSE':>8}  {'Skill':>8}")
        print(f"  {'────':>4}  {'─────────':>9}  {'──────────':>10}  {'────────':>8}  {'────────':>8}")
        all_leads = sorted(
            set(best_eval.per_lead_raw_rmse) | set(best_eval.per_lead_corrected_rmse)
        )
        for ld in all_leads:
            raw  = best_eval.per_lead_raw_rmse.get(ld, float("nan"))
            corr = best_eval.per_lead_corrected_rmse.get(ld, float("nan"))
            delta = corr - raw
            skill_ld = (1.0 - corr / raw) if raw else float("nan")
            marker = " ◄" if 10 <= ld <= 15 else ""
            print(
                f"  {ld:>4}  {raw:>9.3f}  {corr:>10.3f}  {delta:>+8.3f}  {skill_ld:>+8.1%}{marker}"
            )

    print()


if __name__ == "__main__":
    main()
