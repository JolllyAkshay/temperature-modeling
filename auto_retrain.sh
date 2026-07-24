#!/bin/bash
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Waiting for 87/87 grid points..."
while true; do
    last=$(tail -1 /c/Users/nehaa/temperature-modeling/logs/grid_watchdog.log)
    if echo "$last" | grep -q "All 87 grid points collected"; then
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Grid complete! Starting retrain..."
        cd /c/Users/nehaa/temperature-modeling
        python backtest_multi_location.py retrain-all 2>&1
        echo "[$(date '+%Y-%m-%d %H:%M:%S')] Retrain done."
        break
    fi
    sleep 30
done
