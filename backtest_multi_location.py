"""
Single-location backtest: GC-only vs GC + ERA5 synoptic features.

Each run picks ONE location (from structured grid or random), runs the full
backtest, appends results to backtest_results.jsonl, then prints an updated
summary of all locations collected so far.

API responses are cached locally in api_cache/ — repeat runs reuse disk data.

Usage:
    py backtest_multi_location.py           # random location
    py backtest_multi_location.py 42        # fixed seed (reproducible)
    py backtest_multi_location.py grid      # next uncollected grid point
    py backtest_multi_location.py summary   # print accumulated results only
    py backtest_multi_location.py pooled    # train pooled model
    py backtest_multi_location.py progress     # show grid coverage map
    py backtest_multi_location.py retrain-all  # retrain all cached locations in parallel
"""

import json
import os
import random
import sys
import time
import requests
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

from temperature_modeling import (
    build_climate_normals,
    collect_verification_records,
    extract_features,
    score_by_lead,
    train_and_evaluate_banded,
    FEATURE_FIELDS_ERA5,
    FEATURE_FIELDS_SAT,
    FEATURE_FIELDS_ENS,
)
from temperature_modeling.models import Coordinates

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------

# Named land sub-regions inside CONUS.  Random locations are drawn uniformly
# from one of these boxes, guaranteeing we stay over land.
LAND_REGIONS = [
    ("Pacific NW",    (42.0, 49.0), (-124.0, -116.0)),
    ("California",    (33.0, 42.0), (-124.0, -114.0)),
    ("Mountain West", (36.0, 49.0), (-116.0, -104.0)),
    ("Southwest",     (31.0, 36.0), (-114.0, -103.0)),
    ("Great Plains",  (36.0, 49.0), (-104.0,  -96.0)),
    ("Texas",         (26.0, 36.0), (-103.0,  -94.0)),
    ("Midwest",       (36.0, 49.0), ( -96.0,  -82.0)),
    ("Deep South",    (29.0, 36.0), ( -94.0,  -77.0)),
    ("Northeast",     (40.0, 47.0), ( -82.0,  -67.0)),
    ("Mid-Atlantic",  (36.0, 40.0), ( -83.0,  -74.0)),
    ("Southeast",     (29.0, 36.0), ( -85.0,  -75.0)),
]

# ---------------------------------------------------------------------------
# Structured CONUS grid  (~90 land points, stratified by climate regime)
# Each entry: (label, lat, lon)
# ---------------------------------------------------------------------------
CONUS_GRID = [
    # Pacific Coast (marine influence)
    ("Pacific NW coast",        48.5, -124.5),
    ("Oregon coast",            46.5, -124.0),
    ("N California coast",      41.5, -124.0),
    ("San Francisco Bay",       38.5, -122.5),
    ("Central CA coast",        36.0, -121.5),
    ("S California coast",      34.0, -118.5),
    # Cascades / Sierra Nevada
    ("N Cascades WA",           48.0, -121.5),
    ("S Cascades OR",           44.0, -121.5),
    ("N Sierra Nevada",         40.5, -122.5),
    ("Central Sierra Nevada",   38.0, -119.5),
    ("S Sierra Nevada",         36.5, -118.5),
    # Great Basin / Intermountain
    ("Columbia Basin WA",       46.5, -119.5),
    ("Snake River Plain ID",    43.5, -116.5),
    ("Central Nevada",          39.5, -116.5),
    ("S Nevada",                37.0, -115.0),
    ("SW Utah",                 37.5, -113.5),
    ("NW Arizona",              36.0, -113.0),
    # Rocky Mountains
    ("Glacier MT",              48.5, -114.0),
    ("Helena MT",               46.5, -112.0),
    ("Yellowstone WY",          44.5, -110.5),
    ("Central WY",              42.5, -107.5),
    ("Fort Collins CO",         40.5, -105.5),
    ("Central CO",              39.0, -106.5),
    ("SW Colorado",             37.5, -107.5),
    ("N New Mexico",            36.5, -106.0),
    # ── PJM priority ─────────────────────────────────────────────────────────
    # Mid-Atlantic core
    ("Philadelphia PA",         40.0,  -75.5),
    ("Frederick MD",            39.5,  -77.5),
    ("NE Pennsylvania",         41.0,  -75.5),
    # Virginia / Appalachians
    ("Shenandoah VA",           38.5,  -78.5),
    ("Roanoke VA",              37.5,  -80.0),
    ("SW Virginia",             36.5,  -82.0),
    ("E Kentucky",              37.5,  -84.5),
    # Ohio / Indiana
    ("Columbus OH",             40.0,  -83.0),
    ("Toledo OH",               41.5,  -83.5),
    ("Indianapolis IN",         39.5,  -86.5),
    # Carolinas (Duke Energy Progress in PJM)
    ("Charlotte NC",            35.0,  -80.5),
    # ── Southwest Desert ─────────────────────────────────────────────────────
    ("Phoenix AZ",              33.5, -112.0),
    ("Tucson AZ",               32.0, -110.5),
    ("El Paso TX",              31.5, -106.5),
    ("SE New Mexico",           33.0, -104.5),
    ("W Texas high plains",     32.5, -101.5),
    # Great Plains (N–S gradient)
    ("N Dakota",                48.0, -102.5),
    ("S Dakota",                46.0, -100.0),
    ("Central SD",              44.0,  -98.5),
    ("W Nebraska",              41.5, -101.5),
    ("Central Nebraska",        41.0,  -98.5),
    ("N Kansas",                39.5,  -98.5),
    ("Central Kansas",          38.0,  -98.5),
    ("SW Kansas",               37.0, -100.5),
    ("N Oklahoma",              36.5,  -97.5),
    ("W Oklahoma",              35.0,  -99.0),
    # Texas
    ("Texas Panhandle",         35.5, -101.5),
    ("DFW area TX",             33.0,  -97.5),
    ("Austin TX",               30.5,  -97.5),
    ("Houston TX",              29.5,  -95.5),
    ("S Texas",                 27.5,  -99.5),
    # Midwest / Great Lakes
    ("N Minnesota",             47.5,  -93.5),
    ("Central MN",              45.5,  -94.5),
    ("Madison WI",              43.5,  -89.5),
    ("Chicago IL",              42.0,  -88.0),
    ("N Michigan",              45.5,  -84.5),
    ("Upper Peninsula MI",      46.5,  -87.0),
    # Southeast
    ("Nashville TN",            36.0,  -86.5),
    ("Memphis TN",              35.0,  -90.0),
    ("Knoxville TN",            36.0,  -84.0),
    ("Atlanta GA",              33.5,  -84.5),
    ("Central MS",              32.5,  -90.0),
    ("Savannah GA",             32.0,  -81.5),
    ("Mobile AL",               30.5,  -88.0),
    ("Tallahassee FL",          30.5,  -84.5),
    ("Central Florida",         28.5,  -81.5),
    ("Central AR",              34.5,  -92.5),
    # Deep South / Gulf Coast
    ("New Orleans LA",          30.0,  -90.0),
    ("Baton Rouge LA",          30.5,  -91.5),
    ("Houston coast TX",        29.5,  -94.5),
    ("Mississippi coast",       30.5,  -88.5),
    ("N Florida Gulf",          29.5,  -83.5),
    # Mid-Atlantic remainder (Pittsburgh already collected)
    ("Pittsburgh PA",           40.5,  -80.0),
    # Northeast / New England
    ("N Maine",                 47.0,  -68.5),
    ("Central Maine",           44.5,  -70.5),
    ("Vermont",                 44.0,  -72.5),
    ("Boston MA",               42.5,  -71.5),
    ("Hartford CT",             41.5,  -72.5),
    ("Albany NY",               42.5,  -74.0),
    ("Upstate NY",              43.0,  -76.5),
    ("Adirondacks NY",          44.5,  -74.5),
    ("Long Island NY",          40.5,  -73.5),
]

# Minimum separation (degrees) to consider a grid point already collected
_GRID_SKIP_RADIUS = 0.75


def _next_grid_point(results: list):
    """Return the next (label, lat, lon) from CONUS_GRID not yet collected."""
    collected = [(r["lat"], r["lon"]) for r in results]

    def _already_done(lat, lon):
        for clat, clon in collected:
            if abs(clat - lat) < _GRID_SKIP_RADIUS and abs(clon - lon) < _GRID_SKIP_RADIUS:
                return True
        return False

    for label, lat, lon in CONUS_GRID:
        if not _already_done(lat, lon):
            return label, lat, lon
    return None


def _print_progress(results: list) -> None:
    """Print grid coverage status."""
    collected = [(r["lat"], r["lon"]) for r in results]

    def _already_done(lat, lon):
        for clat, clon in collected:
            if abs(clat - lat) < _GRID_SKIP_RADIUS and abs(clon - lon) < _GRID_SKIP_RADIUS:
                return True
        return False

    done = [(lbl, lat, lon) for lbl, lat, lon in CONUS_GRID if _already_done(lat, lon)]
    todo = [(lbl, lat, lon) for lbl, lat, lon in CONUS_GRID if not _already_done(lat, lon)]

    print(f"\n  GRID PROGRESS: {len(done)}/{len(CONUS_GRID)} collected ({len(done)/len(CONUS_GRID):.0%})")
    print(f"\n  Remaining ({len(todo)}):")
    for lbl, lat, lon in todo:
        print(f"    {lat:.1f}N {abs(lon):.1f}W  {lbl}")

START_DATE  = date(2024, 2, 5)    # GraphCast archive start
END_DATE    = date(2025, 12, 31)
MAX_LEAD    = 15
MODEL       = "xgboost"

PROJECT_DIR  = Path(__file__).parent
RESULTS_FILE = PROJECT_DIR / "backtest_results.jsonl"

# ---------------------------------------------------------------------------

def _pct(v: float) -> str:
    return f"{v:+.1%}" if v == v else "   n/a"


def _run_pooled() -> None:
    """Re-collect records for all saved locations and train one pooled model."""
    results = _load_results()
    if not results:
        print("  No saved results to pool. Run some locations first.")
        return

    session = requests.Session()
    session.headers["User-Agent"] = "temperature-modeling-backtest/1.0"

    all_base, all_era5, all_sat = [], [], []
    print(f"\n  Pooling {len(results)} locations from cache...")

    for r in results:
        coords = Coordinates(lat=r["lat"], lon=r["lon"])
        init_dates = [START_DATE + timedelta(days=i)
                      for i in range((END_DATE - START_DATE).days + 1)]
        normals = build_climate_normals(coords, START_DATE, session)
        records = collect_verification_records(
            coords, init_dates, session,
            max_lead_days=MAX_LEAD,
            include_era5_extra=True,
            include_nasa_power=True,
        )
        all_base += extract_features(records, normals,
                                     lat=coords.lat, lon=coords.lon)
        all_era5 += extract_features(records, normals, include_era5_extra=True,
                                     lat=coords.lat, lon=coords.lon)
        all_sat  += extract_features(records, normals, include_era5_extra=True,
                                     include_nasa_power=True,
                                     lat=coords.lat, lon=coords.lon)
        print(f"    {r['label']}: {len(records)} records", flush=True)

    print(f"\n  Total: {len(all_base)} samples across {len(results)} locations")
    print(f"  Training pooled xgboost per-band...", flush=True)

    _, ev_base = train_and_evaluate_banded(all_base, model_type=MODEL)
    _, ev_era5 = train_and_evaluate_banded(all_era5, model_type=MODEL,
                                            feature_fields=FEATURE_FIELDS_ERA5)
    _, ev_sat  = train_and_evaluate_banded(all_sat,  model_type=MODEL,
                                            feature_fields=FEATURE_FIELDS_SAT)

    print(f"\n  POOLED MODEL RESULTS  ({len(results)} locations, {len(all_base)} samples)")
    print(f"  {'Model':<16}  {'Window skill':>12}  {'Window RMSE':>11}")
    print(f"  {'-'*16}  {'-'*12}  {'-'*11}")
    for label, ev in [("GC-only", ev_base), ("GC+ERA5", ev_era5), ("GC+ERA5+SAT", ev_sat)]:
        print(f"  {label:<16}  {_pct(ev.window_skill_score):>12}  {ev.window_corrected_rmse:>11.3f}")

    print(f"\n  PER-LEAD (pooled):")
    print(f"  {'Lead':>4}  {'Raw':>6}  {'GC-only':>8}  {'GC+ERA5':>8}  {'GC+E+SAT':>9}")
    all_leads = sorted(ev_base.per_lead_raw_rmse)
    for ld in all_leads:
        marker = " <" if 10 <= ld <= MAX_LEAD else ""
        print(
            f"  {ld:>4}  {ev_base.per_lead_raw_rmse[ld]:>6.3f}"
            f"  {ev_base.per_lead_corrected_rmse[ld]:>8.3f}"
            f"  {ev_era5.per_lead_corrected_rmse[ld]:>8.3f}"
            f"  {ev_sat.per_lead_corrected_rmse[ld]:>9.3f}{marker}"
        )
    print()


def _run_location(coords: Coordinates, label: str) -> dict:
    session = requests.Session()
    session.headers["User-Agent"] = "temperature-modeling-backtest/1.0"

    init_dates = [
        START_DATE + timedelta(days=i)
        for i in range((END_DATE - START_DATE).days + 1)
    ]

    print(f"  [1/3] Climate normals...", flush=True)
    normals = build_climate_normals(coords, START_DATE, session)

    print(f"  [2/3] Collecting records ({len(init_dates)} init dates)...", flush=True)
    t0 = time.time()
    records = collect_verification_records(
        coords,
        init_dates,
        session,
        max_lead_days=MAX_LEAD,
        include_satellite_features=False,
        include_era5_extra=True,
        include_nasa_power=True,
        include_ecmwf_ens=True,
    )
    elapsed = time.time() - t0
    ens_count = sum(1 for r in records if r.ens_spread_c is not None)
    print(f"        {len(records)} samples in {elapsed:.0f}s  (ENS: {ens_count}/{len(records)})", flush=True)

    if len(records) < 10:
        raise ValueError(f"Too few records: {len(records)}")

    vectors_base = extract_features(records, normals, include_era5_extra=False,
                                    lat=coords.lat, lon=coords.lon)
    vectors_era5 = extract_features(records, normals, include_era5_extra=True,
                                    lat=coords.lat, lon=coords.lon)
    vectors_sat  = extract_features(records, normals, include_era5_extra=True,
                                    include_nasa_power=True,
                                    lat=coords.lat, lon=coords.lon)
    vectors_ens  = extract_features(records, normals, include_era5_extra=True,
                                    include_nasa_power=True,
                                    include_ecmwf_ens=True,
                                    lat=coords.lat, lon=coords.lon)

    print(f"  [3/3] Training {MODEL} per-band (GC-only | GC+ERA5 | GC+ERA5+SAT | GC+ENS)...", flush=True)
    _, ev_base = train_and_evaluate_banded(vectors_base, model_type=MODEL)
    _, ev_era5 = train_and_evaluate_banded(
        vectors_era5, model_type=MODEL, feature_fields=FEATURE_FIELDS_ERA5
    )
    _, ev_sat = train_and_evaluate_banded(
        vectors_sat, model_type=MODEL, feature_fields=FEATURE_FIELDS_SAT
    )
    _, ev_ens = train_and_evaluate_banded(
        vectors_ens, model_type=MODEL, feature_fields=FEATURE_FIELDS_ENS
    )

    # Raw skill by lead (from GC-only records — same obs either way)
    skill = score_by_lead(records)
    raw_bias_window = sum(
        s.bias for s in skill if 10 <= s.lead_days <= MAX_LEAD
    ) / sum(1 for s in skill if 10 <= s.lead_days <= MAX_LEAD)

    return {
        "label":          label,
        "lat":            coords.lat,
        "lon":            coords.lon,
        "n_records":      len(records),
        "raw_rmse":       ev_base.window_raw_rmse,
        "raw_bias":       round(raw_bias_window, 3),
        "gc_skill":       ev_base.window_skill_score,
        "era5_skill":     ev_era5.window_skill_score,
        "sat_skill":      ev_sat.window_skill_score,
        "ens_skill":      ev_ens.window_skill_score,
        "gc_corr_rmse":   ev_base.window_corrected_rmse,
        "era5_corr_rmse": ev_era5.window_corrected_rmse,
        "sat_corr_rmse":  ev_sat.window_corrected_rmse,
        "ens_corr_rmse":  ev_ens.window_corrected_rmse,
        "per_lead_raw":   ev_base.per_lead_raw_rmse,
        "per_lead_gc":    ev_base.per_lead_corrected_rmse,
        "per_lead_era5":  ev_era5.per_lead_corrected_rmse,
        "per_lead_sat":   ev_sat.per_lead_corrected_rmse,
        "per_lead_ens":   ev_ens.per_lead_corrected_rmse,
    }


def _load_results() -> list:
    if not RESULTS_FILE.exists():
        return []
    results = []
    with open(RESULTS_FILE, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                results.append(json.loads(line))
    return results


def _print_summary(results: list) -> None:
    if not results:
        print("  No results yet.")
        return

    gains_ens = [r.get("ens_skill", r.get("sat_skill", r["era5_skill"])) - r["gc_skill"] for r in results]
    ranked = sorted(zip(results, gains_ens), key=lambda x: -x[1])

    print(f"\n{'=' * 96}")
    print(f"  ACCUMULATED RESULTS  ({len(results)} locations  |  window days 10-{MAX_LEAD}  |  {MODEL})")
    print(f"{'=' * 96}")
    print(
        f"  {'Location':<28} {'n':>5}  {'Raw':>6}  {'Bias':>6}  "
        f"{'GC-only':>8}  {'GC+ERA5':>8}  {'GC+SAT':>8}  {'GC+ENS':>8}  {'Gain':>7}"
    )
    print(
        f"  {'-'*28} {'-----':>5}  {'------':>6}  {'------':>6}  "
        f"{'--------':>8}  {'--------':>8}  {'--------':>8}  {'--------':>8}  {'-------':>7}"
    )

    for r, g_ens in ranked:
        marker = " +" if g_ens > 0 else ""
        sat_skill = r.get("sat_skill", r["era5_skill"])
        ens_skill = r.get("ens_skill", sat_skill)
        print(
            f"  {r['label']:<28} {r['n_records']:>5}  {r['raw_rmse']:>6.3f}  "
            f"{r['raw_bias']:>+6.3f}  {_pct(r['gc_skill']):>8}  "
            f"{_pct(r['era5_skill']):>8}  {_pct(sat_skill):>8}  {_pct(ens_skill):>8}  {g_ens:>+7.1%}{marker}"
        )

    n = len(results)
    avg_raw  = sum(r["raw_rmse"]   for r in results) / n
    avg_bias = sum(r["raw_bias"]   for r in results) / n
    avg_gc   = sum(r["gc_skill"]   for r in results) / n
    avg_era5 = sum(r["era5_skill"] for r in results) / n
    avg_sat  = sum(r.get("sat_skill", r["era5_skill"]) for r in results) / n
    avg_ens  = sum(r.get("ens_skill", r.get("sat_skill", r["era5_skill"])) for r in results) / n
    avg_gain = sum(gains_ens) / n
    n_improved = sum(1 for g in gains_ens if g > 0)

    print(
        f"  {'-'*28} {'-----':>5}  {'------':>6}  {'------':>6}  "
        f"{'--------':>8}  {'--------':>8}  {'--------':>9}  {'-------':>7}"
    )
    print(
        f"  {'MEAN  (' + str(n) + ' locs)':<28} {'':>5}  {avg_raw:>6.3f}  "
        f"{avg_bias:>+6.3f}  {_pct(avg_gc):>8}  {_pct(avg_era5):>8}  "
        f"{_pct(avg_sat):>8}  {_pct(avg_ens):>8}  {avg_gain:>+7.1%}"
    )
    print(f"\n  ENS improved: {n_improved}/{n} locations")

    # Per-lead aggregate
    all_lead_sets = [set(int(k) for k in r["per_lead_raw"]) for r in results]
    all_leads = sorted(set().union(*all_lead_sets))
    print(f"\n  PER-LEAD AGGREGATE  (mean corrected RMSE across {n} locations)")
    print(
        f"  {'Lead':>4}  {'Raw':>6}  {'GC-only':>8}  {'GC+ERA5':>8}  {'GC+E+SAT':>9}  "
        f"{'SkillA':>7}  {'SkillE':>7}  {'SkillS':>7}"
    )
    print(
        f"  {'----':>4}  {'------':>6}  {'--------':>8}  {'--------':>8}  {'--------':>8}  "
        f"{'-------':>7}  {'-------':>7}  {'-------':>7}"
    )
    for ld in all_leads:
        ld_s = str(ld)
        raws  = [r["per_lead_raw"].get(ld_s)  for r in results if r["per_lead_raw"].get(ld_s)  is not None]
        gcs   = [r["per_lead_gc"].get(ld_s)   for r in results if r["per_lead_gc"].get(ld_s)   is not None]
        era5s = [r["per_lead_era5"].get(ld_s) for r in results if r["per_lead_era5"].get(ld_s) is not None]
        sats  = [r.get("per_lead_sat", r["per_lead_era5"]).get(ld_s)
                 for r in results
                 if r.get("per_lead_sat", r["per_lead_era5"]).get(ld_s) is not None]
        if not raws:
            continue
        raw = sum(raws)  / len(raws)
        ca  = sum(gcs)   / len(gcs)
        ce  = sum(era5s) / len(era5s)
        cs  = sum(sats)  / len(sats) if sats else ce
        ska = (1.0 - ca / raw) if raw else float("nan")
        ske = (1.0 - ce / raw) if raw else float("nan")
        sks = (1.0 - cs / raw) if raw else float("nan")
        marker = " <" if 10 <= ld <= MAX_LEAD else ""
        print(
            f"  {ld:>4}  {raw:>6.3f}  {ca:>8.3f}  {ce:>8.3f}  {cs:>9.3f}  "
            f"{_pct(ska):>7}  {_pct(ske):>7}  {_pct(sks):>7}{marker}"
        )
    print(f"\n  < = target window  |  + = SAT helped that location")
    print()


def _retrain_one(lat: float, lon: float, label: str) -> dict:
    """Run _run_location for one cached location; returns result dict."""
    coords = Coordinates(lat=lat, lon=lon)
    return _run_location(coords, label)


def _run_retrain_all() -> None:
    """Re-train all already-collected locations in parallel using cached data."""
    existing = _load_results()
    if not existing:
        print("  No collected results to retrain.")
        return

    locs = [(r["lat"], r["lon"], r["label"]) for r in existing]
    print(f"  Retraining {len(locs)} locations in parallel (using cache)...\n")

    new_results = []
    workers = min(len(locs), os.cpu_count() or 4)
    with ProcessPoolExecutor(max_workers=workers) as pool:
        futures = {pool.submit(_retrain_one, lat, lon, label): label
                   for lat, lon, label in locs}
        for fut in as_completed(futures):
            label = futures[fut]
            try:
                result = fut.result()
                new_results.append(result)
                gain = result["ens_skill"] - result["gc_skill"]
                print(
                    f"  {label:<44}  "
                    f"GC-only {_pct(result['gc_skill'])}  "
                    f"GC+ERA5 {_pct(result['era5_skill'])}  "
                    f"GC+SAT {_pct(result['sat_skill'])}  "
                    f"gain {gain:>+.1%}"
                )
            except Exception as exc:
                print(f"  FAILED {label}: {exc}")

    # Overwrite results file with fresh results
    with open(RESULTS_FILE, "w", encoding="utf-8") as f:
        for r in new_results:
            f.write(json.dumps(r) + "\n")
    print(f"\n  Written {len(new_results)} results to {RESULTS_FILE.name}")
    _print_summary(_load_results())


def main() -> None:
    arg = sys.argv[1] if len(sys.argv) > 1 else None

    # Summary-only mode
    if arg == "summary":
        _print_summary(_load_results())
        return

    # Pooled multi-location model
    if arg == "pooled":
        _run_pooled()
        return

    # Retrain all collected locations in parallel from cache
    if arg == "retrain-all":
        _run_retrain_all()
        return

    # Grid progress report
    if arg == "progress":
        _print_progress(_load_results())
        return

    # Structured grid mode — next uncollected point
    if arg == "grid":
        results = _load_results()
        nxt = _next_grid_point(results)
        if nxt is None:
            print("  All grid points collected!")
            _print_progress(results)
            return
        region_name, lat, lon = nxt
        coords = Coordinates(lat=lat, lon=lon)
        label  = f"{lat:.2f}N {abs(lon):.2f}W ({region_name})"
    else:
        # Pick random seed
        seed = int(arg) if arg else random.randint(0, 2**31)
        rng  = random.Random(seed)

        region_name, lat_range, lon_range = rng.choice(LAND_REGIONS)
        lat   = round(rng.uniform(*lat_range), 3)
        lon   = round(rng.uniform(*lon_range), 3)
        coords = Coordinates(lat=lat, lon=lon)
        label  = f"{lat:.2f}N {abs(lon):.2f}W ({region_name})"

    seed_str = f"  (seed {seed})" if arg != "grid" else "  [grid]"
    print("=" * 60)
    print(f"  Backtest  |  {label}{seed_str}")
    print(f"  Period    : {START_DATE} to {END_DATE}")
    print(f"  API cache : {(PROJECT_DIR / 'api_cache').resolve()}")
    print("=" * 60)

    try:
        result = _run_location(coords, label)
    except Exception as exc:
        print(f"\n  FAILED: {exc}")
        return

    gain = result["ens_skill"] - result["gc_skill"]
    print(
        f"\n  Result  GC-only {_pct(result['gc_skill'])}  "
        f"GC+ERA5 {_pct(result['era5_skill'])}  "
        f"GC+ERA5+SAT {_pct(result['sat_skill'])}  "
        f"GC+ENS {_pct(result['ens_skill'])}  "
        f"gain {gain:>+.1%}"
    )

    # Append to results file
    with open(RESULTS_FILE, "a", encoding="utf-8") as f:
        f.write(json.dumps(result) + "\n")
    print(f"  Saved to {RESULTS_FILE.name}")

    _print_summary(_load_results())


if __name__ == "__main__":
    main()
