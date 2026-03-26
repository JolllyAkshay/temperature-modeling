"""
Grid collection watchdog.
Runs backtest_multi_location.py grid in a continuous loop until all points
are collected. Logs progress every 30 minutes and restarts on failure.

Usage:  python grid_watchdog.py
"""

import json
import os
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE = Path(__file__).parent
RESULTS_FILE = _HERE / "backtest_results.jsonl"
LOG_FILE     = _HERE / "logs" / "grid_watchdog.log"
GRID_SCRIPT  = [sys.executable, str(_HERE / "backtest_multi_location.py"), "grid"]

PROGRESS_INTERVAL_S = 30 * 60   # log progress every 30 min
FAIL_RETRY_DELAY_S  = 60        # wait after a failed run before retrying
TOTAL_GRID_POINTS   = 87


def log(msg: str):
    ts  = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_collected() -> int:
    if not RESULTS_FILE.exists():
        return 0
    # Unique locations within 0.75° radius (mirrors _already_done logic)
    rows = []
    with open(RESULTS_FILE, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    unique = []
    for r in rows:
        lat, lon = r["lat"], r["lon"]
        if not any(abs(u[0] - lat) < 0.75 and abs(u[1] - lon) < 0.75 for u in unique):
            unique.append((lat, lon))
    return len(unique)


def log_progress():
    n = count_collected()
    pct = n / TOTAL_GRID_POINTS * 100
    log(f"PROGRESS: {n}/{TOTAL_GRID_POINTS} collected ({pct:.0f}%)")


def run():
    LOG_FILE.parent.mkdir(exist_ok=True)
    log("=" * 60)
    log("Grid watchdog started")
    log_progress()

    last_progress_log = time.time()
    consecutive_failures = 0

    while True:
        # Check if all done
        n = count_collected()
        if n >= TOTAL_GRID_POINTS:
            log(f"All {TOTAL_GRID_POINTS} grid points collected. Done.")
            break

        log(f"Starting grid collection ({n}/{TOTAL_GRID_POINTS} done)...")

        proc = subprocess.Popen(
            GRID_SCRIPT,
            cwd=str(_HERE),
            stdout=open(_HERE / "logs" / "grid_run.log", "a", encoding="utf-8"),
            stderr=subprocess.STDOUT,
        )

        # Monitor: wait for completion, checking progress timer
        while proc.poll() is None:
            time.sleep(10)
            if time.time() - last_progress_log >= PROGRESS_INTERVAL_S:
                log_progress()
                last_progress_log = time.time()

        exit_code = proc.returncode
        n_after = count_collected()

        if exit_code == 0:
            consecutive_failures = 0
            log(f"Run completed OK (exit 0). Now {n_after}/{TOTAL_GRID_POINTS} collected.")
        else:
            consecutive_failures += 1
            log(f"Run failed (exit {exit_code}). Failures in a row: {consecutive_failures}. Retrying in {FAIL_RETRY_DELAY_S}s...")
            if consecutive_failures >= 5:
                log("5 consecutive failures — pausing 10 min before continuing.")
                time.sleep(600)
                consecutive_failures = 0
            else:
                time.sleep(FAIL_RETRY_DELAY_S)

        # Log progress periodically regardless
        if time.time() - last_progress_log >= PROGRESS_INTERVAL_S:
            log_progress()
            last_progress_log = time.time()

    log("Watchdog exiting.")


if __name__ == "__main__":
    run()
