"""
Real-data backtest: GraphCast-only vs GraphCast + ERA5 synoptic features.

Fetches live data from Open-Meteo (no mocks).  Runs two experiments from
the same collected records:

  A) GC only   -- temporal + climatological features (FEATURE_FIELDS)
  C) GC + ERA5 -- adds z500, t850, soil moisture, snow depth (FEATURE_FIELDS_ERA5)

Usage:
    py backtest_real.py
"""

import sys
import requests
from datetime import date, timedelta

from temperature_modeling import (
    build_climate_normals,
    collect_verification_records,
    extract_features,
    score_by_lead,
    train_and_evaluate,
    FEATURE_FIELDS_ERA5,
)
from temperature_modeling.models import Coordinates

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

COORDS = Coordinates(lat=39.9612, lon=-82.9988)   # Columbus, OH

# 60 init dates inside the GraphCast archive (available from 2024-02-05).
# Using Jan–Mar 2025 so ERA5 reanalysis is fully settled.
N_DAYS = 365
END_DATE   = date(2025, 12, 31)
START_DATE = date(2024,  2,  5)   # GraphCast archive start

MAX_LEAD = 15
MODELS   = ["xgboost", "ridge"]

# ---------------------------------------------------------------------------

def _pct(skill: float) -> str:
    return f"{skill:+.1%}" if skill == skill else "  n/a "


def main() -> None:
    session = requests.Session()
    session.headers["User-Agent"] = "temperature-modeling-backtest/1.0"

    init_dates = [
        START_DATE + timedelta(days=i)
        for i in range((END_DATE - START_DATE).days + 1)
    ]

    n_days = (END_DATE - START_DATE).days + 1

    print("=" * 65)
    print(f"  Real-data backtest  |  Columbus OH  ({COORDS.lat}, {COORDS.lon})")
    print(f"  Init dates : {START_DATE} to {END_DATE}  ({n_days} dates)")
    print(f"  Lead days  : 1 - {MAX_LEAD}")
    print(f"  Models     : {', '.join(MODELS)}")
    print("=" * 65)

    # ------------------------------------------------------------------
    # 1. Climate normals  (one API call for 5-year ERA5 period)
    # ------------------------------------------------------------------
    print("\n[1/3] Building climate normals (5-year ERA5)...", flush=True)
    normals = build_climate_normals(COORDS, START_DATE, session)
    print(f"      Done. {len(normals.doy_mean)} DOY entries.")

    # ------------------------------------------------------------------
    # 2. Collect records once with ERA5 extra variables
    #    Each init date => 3 API calls:
    #      - GraphCast historical forecast
    #      - ERA5 temperature_2m (truth)
    #      - ERA5 z500 / t850 / soil / snow (init-state synoptic)
    # ------------------------------------------------------------------
    n_dates = len(init_dates)
    print(f"\n[2/3] Collecting records with ERA5 extra features"
          f" ({n_dates} init dates, ~{n_dates * 3} API calls)...", flush=True)
    records = collect_verification_records(
        COORDS,
        init_dates,
        session,
        max_lead_days=MAX_LEAD,
        include_satellite_features=False,
        include_era5_extra=True,
    )
    print(f"      Done. {len(records)} forecast-observation pairs.")

    if len(records) < 10:
        print("ERROR: Too few records to train. Check network / date range.")
        sys.exit(1)

    # ------------------------------------------------------------------
    # 3. Feature extraction — two feature sets, same records
    # ------------------------------------------------------------------
    vectors_base = extract_features(records, normals, include_era5_extra=False)
    vectors_era5 = extract_features(records, normals, include_era5_extra=True)

    # ------------------------------------------------------------------
    # Raw GraphCast skill by lead time
    # ------------------------------------------------------------------
    skill = score_by_lead(records)

    print(f"\n{'-' * 65}")
    print(f"  RAW GRAPHCAST SKILL  ({len(records)} samples)")
    print(f"{'-' * 65}")
    print(f"  {'Lead':>4}  {'RMSE (C)':>9}  {'MAE (C)':>8}  {'Bias (C)':>9}  {'n':>4}")
    print(f"  {'----':>4}  {'---------':>9}  {'--------':>8}  {'---------':>9}  {'----':>4}")
    for s in skill:
        marker = " <" if 10 <= s.lead_days <= MAX_LEAD else ""
        print(f"  {s.lead_days:>4}  {s.rmse:>9.3f}  {s.mae:>8.3f}  {s.bias:>+9.3f}  {s.n:>4}{marker}")
    print(f"\n  < = target 10-{MAX_LEAD} day window")

    # ------------------------------------------------------------------
    # Train and evaluate: both experiments × both models
    # ------------------------------------------------------------------
    print(f"\n[3/3] Training and evaluating models...", flush=True)
    results = {}   # (model_type, feature_set) -> ModelEvaluation
    for m in MODELS:
        _, ev_b = train_and_evaluate(vectors_base, model_type=m)
        _, ev_e = train_and_evaluate(
            vectors_era5, model_type=m, feature_fields=FEATURE_FIELDS_ERA5
        )
        results[(m, "base")] = ev_b
        results[(m, "era5")] = ev_e
        print(f"      {m:<14} GC-only skill: {_pct(ev_b.window_skill_score)}  "
              f"GC+ERA5 skill: {_pct(ev_e.window_skill_score)}")
    print("      Done.")

    # ------------------------------------------------------------------
    # Summary comparison table
    # ------------------------------------------------------------------
    print(f"\n{'-' * 70}")
    print(f"  MODEL COMPARISON  (window: days 10-{MAX_LEAD})")
    print(f"{'-' * 70}")
    print(
        f"  {'Model':<10} {'Features':<14} {'Train':>6} {'Test':>6} "
        f"{'Raw RMSE':>9} {'Corr RMSE':>10} {'Skill':>8}"
    )
    print(
        f"  {'-'*10} {'-'*14} {'------':>6} {'------':>6} "
        f"{'---------':>9} {'----------':>10} {'--------':>8}"
    )
    for m in MODELS:
        for fs, label in [("base", "GC only"), ("era5", "GC + ERA5")]:
            ev = results[(m, fs)]
            print(
                f"  {m:<10} {label:<14} {ev.n_train:>6} {ev.n_test:>6} "
                f"{ev.window_raw_rmse:>9.3f} {ev.window_corrected_rmse:>10.3f} "
                f"{_pct(ev.window_skill_score):>8}"
            )

    # ------------------------------------------------------------------
    # Per-lead breakdown for best-performing model
    # ------------------------------------------------------------------
    best_model = max(
        MODELS,
        key=lambda m: results[(m, "era5")].window_skill_score
    )
    ev_base = results[(best_model, "base")]
    ev_era5 = results[(best_model, "era5")]

    print(f"\n{'-' * 70}")
    print(f"  PER-LEAD BREAKDOWN  (model: {best_model})")
    print(f"{'-' * 70}")
    print(
        f"  {'Lead':>4}  {'Raw':>7}  {'GC-only':>8}  {'GC+ERA5':>8}  "
        f"{'SkillA':>7}  {'SkillC':>7}  {'Delta':>7}"
    )
    print(
        f"  {'----':>4}  {'-------':>7}  {'--------':>8}  {'--------':>8}  "
        f"{'-------':>7}  {'-------':>7}  {'-------':>7}"
    )

    all_leads = sorted(
        set(ev_base.per_lead_raw_rmse) | set(ev_era5.per_lead_raw_rmse)
    )
    for ld in all_leads:
        raw = ev_base.per_lead_raw_rmse.get(ld, float("nan"))
        ca  = ev_base.per_lead_corrected_rmse.get(ld, float("nan"))
        cc  = ev_era5.per_lead_corrected_rmse.get(ld, float("nan"))
        ska = (1.0 - ca / raw) if raw else float("nan")
        skc = (1.0 - cc / raw) if raw else float("nan")
        delta = cc - ca   # negative means ERA5 features reduce RMSE
        marker = " <" if 10 <= ld <= MAX_LEAD else ""
        print(
            f"  {ld:>4}  {raw:>7.3f}  {ca:>8.3f}  {cc:>8.3f}  "
            f"{_pct(ska):>7}  {_pct(skc):>7}  {delta:>+7.3f}{marker}"
        )

    print(f"\n  < = target window")
    print(f"  Delta = GC+ERA5 corrected RMSE minus GC-only  (negative = ERA5 helps)")
    print()


if __name__ == "__main__":
    main()
