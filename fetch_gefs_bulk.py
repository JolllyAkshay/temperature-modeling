"""
Bulk precomputation of GEFS ensemble spread for all cached locations.

Downloads GRIB2 spread/mean files from NOAA S3, extracts all locations at once
per file (one GRIB2 decode → all grid points), and caches results in
api_cache/gefs_spread/{YYYYMMDD}.json.

Usage:
    py fetch_gefs_bulk.py                  # all dates, all cached locations
    py fetch_gefs_bulk.py --workers 4      # parallel workers (default: 4)
    py fetch_gefs_bulk.py --resume         # skip dates that are fully cached
    py fetch_gefs_bulk.py --date 2024-03-01  # single date (debug)
"""

import argparse
import json
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

# ---------------------------------------------------------------------------
# Setup path so we can import from src/
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from temperature_modeling._era5 import (
    _gefs_cache_load,
    _gefs_cache_path,
    _loc_key,
    _GEFS_SPREAD_CACHE_DIR,
    fetch_gefs_spread_bulk,
)
from temperature_modeling.models import Coordinates

# Full 87-point CONUS grid (imported lazily to avoid circular imports)
def _load_full_grid() -> list[Coordinates]:
    """Load all 87 CONUS grid points from backtest_multi_location.py."""
    import importlib.util, pathlib
    spec = importlib.util.spec_from_file_location(
        "_bml", str(_HERE / "backtest_multi_location.py")
    )
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return [Coordinates(lat=lat, lon=lon) for _, lat, lon in mod.CONUS_GRID]

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
RESULTS_FILE = _HERE / "backtest_results.jsonl"
MAX_LEAD_DAYS = 16  # days 1-16

# Date range matching backtest period
START_DATE = date(2024, 2, 5)
END_DATE = date(2025, 12, 31)


def load_locations() -> list[Coordinates]:
    """Load unique lat/lon from backtest_results.jsonl."""
    seen: set[tuple] = set()
    locs: list[Coordinates] = []
    with open(RESULTS_FILE) as f:
        for line in f:
            r = json.loads(line.strip())
            key = (r["lat"], r["lon"])
            if key not in seen:
                seen.add(key)
                locs.append(Coordinates(lat=r["lat"], lon=r["lon"]))
    return locs


def all_init_dates() -> list[date]:
    d = START_DATE
    dates = []
    while d <= END_DATE:
        dates.append(d)
        d += timedelta(days=1)
    return dates


def is_cached(init_date: date, loc_keys: set[str]) -> bool:
    """Return True if all locations are already in this date's cache."""
    if _GEFS_SPREAD_CACHE_DIR is None:
        return False
    cache = _gefs_cache_load(init_date)
    if cache is None:
        return False
    return loc_keys.issubset(cache.keys())


def process_date(init_date: date, locs: list[Coordinates]) -> tuple[date, int, float]:
    """Download GEFS spread for one init_date across all locations. Returns (date, n_ok, elapsed)."""
    session = requests.Session()
    session.headers["User-Agent"] = "gefs-spread-prefetch/1.0"
    t0 = time.time()
    try:
        result = fetch_gefs_spread_bulk(locs, init_date, MAX_LEAD_DAYS, session)
        n_ok = sum(1 for v in result.values() if v)
    except Exception as e:
        print(f"  ERROR {init_date}: {e}", flush=True)
        n_ok = 0
    elapsed = time.time() - t0
    return init_date, n_ok, elapsed


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--resume", action="store_true")
    parser.add_argument("--date", type=str, default=None, help="Single date YYYY-MM-DD")
    parser.add_argument("--all-grid", action="store_true", help="Use all 87 CONUS grid points instead of collected-only")
    args = parser.parse_args()

    locs = _load_full_grid() if args.all_grid else load_locations()
    loc_keys = {_loc_key(c.lat, c.lon) for c in locs}
    print(f"Locations: {len(locs)}")

    if args.date:
        dates = [date.fromisoformat(args.date)]
    else:
        dates = all_init_dates()

    if args.resume:
        dates = [d for d in dates if not is_cached(d, loc_keys)]
        print(f"Resuming: {len(dates)} dates need processing")
    else:
        print(f"Total dates: {len(dates)}")

    if not dates:
        print("All dates cached. Done.")
        return

    print(f"Workers: {args.workers}")
    print(f"Estimated time: {len(dates) * 50 / args.workers / 3600:.1f} hours")
    print()

    done = 0
    failed = 0
    t_start = time.time()

    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {pool.submit(process_date, d, locs): d for d in dates}
        for fut in as_completed(futures):
            d, n_ok, elapsed = fut.result()
            done += 1
            if n_ok == 0:
                failed += 1
            eta_s = (time.time() - t_start) / done * (len(dates) - done)
            print(
                f"  [{done}/{len(dates)}] {d}  locs={n_ok}/{len(locs)}  {elapsed:.0f}s"
                f"  ETA {eta_s/3600:.1f}h",
                flush=True,
            )

    print()
    print(f"Done: {done} dates, {failed} failures.")


if __name__ == "__main__":
    main()
