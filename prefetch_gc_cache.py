"""
Pre-warm the GraphCast API cache for specified locations.
Run this in parallel with the watchdog to speed up grid collection.

Usage:  python prefetch_gc_cache.py
"""

import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, timedelta
from pathlib import Path

import requests

_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from temperature_modeling.models import Coordinates
from temperature_modeling.verification import _fetch_gc_daily, _GC_ARCHIVE_START

START_DATE    = _GC_ARCHIVE_START
END_DATE      = date(2025, 12, 31)
MAX_LEAD_DAYS = 16

# Remaining PJM locations not yet collected
PJM_REMAINING = [
    ("E Kentucky",    37.5,  -84.5),
    ("Columbus OH",   40.0,  -83.0),
    ("Toledo OH",     41.5,  -83.5),
    ("Indianapolis IN", 39.5, -86.5),
    ("Charlotte NC",  35.0,  -80.5),
]

def all_init_dates():
    d, dates = START_DATE, []
    while d <= END_DATE:
        dates.append(d)
        d += timedelta(days=1)
    return dates

def prefetch_location(label, lat, lon):
    coords  = Coordinates(lat=lat, lon=lon)
    session = requests.Session()
    session.headers["User-Agent"] = "gc-cache-prefetch/1.0"
    dates   = all_init_dates()
    done, skipped, errors = 0, 0, 0

    for init_date in dates:
        try:
            result = _fetch_gc_daily(coords, init_date, MAX_LEAD_DAYS, session)
            if result:
                done += 1
            else:
                errors += 1
            time.sleep(0.15)
        except Exception:
            errors += 1
            time.sleep(1.0)

    return label, done, errors

def main():
    print(f"Pre-warming GC cache for {len(PJM_REMAINING)} PJM locations "
          f"({len(all_init_dates())} dates each)...", flush=True)

    with ThreadPoolExecutor(max_workers=len(PJM_REMAINING)) as pool:
        futures = {
            pool.submit(prefetch_location, lbl, lat, lon): lbl
            for lbl, lat, lon in PJM_REMAINING
        }
        for fut in as_completed(futures):
            label, done, errors = fut.result()
            print(f"  [{label}] done={done}  errors={errors}", flush=True)

    print("Pre-fetch complete.", flush=True)

if __name__ == "__main__":
    main()
