"""
Training watchdog for backtest_multi_location.py retrain-all.

Runs retrain-all in a subprocess, monitors progress every 5 minutes,
and restarts automatically on failure.

Usage:  python retrain_watchdog.py
"""

import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

_HERE     = Path(__file__).parent
LOG_FILE  = _HERE / "logs" / "retrain_watchdog.log"
RUN_LOG   = _HERE / "logs" / "retrain_run.log"
SCRIPT    = [sys.executable, str(_HERE / "backtest_multi_location.py"), "retrain-all"]
TOTAL     = 87
PROGRESS_INTERVAL_S = 5 * 60   # log progress every 5 min
FAIL_RETRY_DELAY_S  = 60


def log(msg: str):
    ts   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{ts}] {msg}"
    print(line, flush=True)
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(line + "\n")


def count_completed() -> int:
    """Count completed locations by counting result lines in the run log."""
    if not RUN_LOG.exists():
        return 0
    count = 0
    try:
        with open(RUN_LOG, encoding="utf-8", errors="ignore") as f:
            for line in f:
                # Each completed location prints a line with "GC-only" in it
                if "GC-only" in line or "gain" in line:
                    count += 1
    except OSError:
        pass
    return count


def run():
    LOG_FILE.parent.mkdir(exist_ok=True)
    log("=" * 60)
    log("Retrain watchdog started")

    consecutive_failures = 0
    last_progress_log    = time.time()

    while True:
        n = count_completed()
        log(f"Starting retrain-all ({n}/{TOTAL} previously completed lines in log)...")

        # Clear run log for fresh run so counts are accurate
        RUN_LOG.write_text("", encoding="utf-8")

        with open(RUN_LOG, "a", encoding="utf-8") as run_log_fh:
            proc = subprocess.Popen(
                SCRIPT,
                cwd=str(_HERE),
                stdout=run_log_fh,
                stderr=subprocess.STDOUT,
            )

        # Monitor while running
        while proc.poll() is None:
            time.sleep(15)
            if time.time() - last_progress_log >= PROGRESS_INTERVAL_S:
                n = count_completed()
                pct = n / TOTAL * 100
                log(f"PROGRESS: {n}/{TOTAL} locations retrained ({pct:.0f}%)")
                last_progress_log = time.time()

        exit_code = proc.returncode
        n_done    = count_completed()

        if exit_code == 0:
            consecutive_failures = 0
            log(f"Retrain completed OK (exit 0). {n_done}/{TOTAL} locations in log.")
            log("Retrain watchdog exiting.")
            break
        else:
            consecutive_failures += 1
            log(f"Retrain failed (exit {exit_code}). Failures: {consecutive_failures}. "
                f"Retrying in {FAIL_RETRY_DELAY_S}s...")
            if consecutive_failures >= 3:
                log("3 consecutive failures — pausing 5 min before retrying.")
                time.sleep(300)
                consecutive_failures = 0
            else:
                time.sleep(FAIL_RETRY_DELAY_S)

        if time.time() - last_progress_log >= PROGRESS_INTERVAL_S:
            n = count_completed()
            log(f"PROGRESS: {n}/{TOTAL} locations retrained ({n/TOTAL*100:.0f}%)")
            last_progress_log = time.time()


if __name__ == "__main__":
    run()
