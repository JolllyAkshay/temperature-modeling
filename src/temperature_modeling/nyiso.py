"""
NYISO (New York ISO) population-weighted load locations.
Covers the NY Control Area — roughly 20M people across 11 load zones (A-K).
Weights proportional to zone population and cooling/heating load sensitivity.
"""

import logging as _logging

_log = _logging.getLogger(__name__)

NYISO_LOAD_LOCATIONS: list[dict] = [
    {"label": "New York City NY",  "lat": 40.71, "lon": -74.01, "weight": 0.40},
    {"label": "Long Island NY",    "lat": 40.79, "lon": -73.13, "weight": 0.14},
    {"label": "Westchester NY",    "lat": 41.10, "lon": -73.79, "weight": 0.07},
    {"label": "Albany NY",         "lat": 42.65, "lon": -73.75, "weight": 0.06},
    {"label": "Buffalo NY",        "lat": 42.89, "lon": -78.87, "weight": 0.08},
    {"label": "Rochester NY",      "lat": 43.16, "lon": -77.61, "weight": 0.06},
    {"label": "Syracuse NY",       "lat": 43.05, "lon": -76.15, "weight": 0.05},
    {"label": "Binghamton NY",     "lat": 42.10, "lon": -75.91, "weight": 0.04},
    {"label": "Plattsburgh NY",    "lat": 44.70, "lon": -73.46, "weight": 0.03},
    {"label": "Utica NY",          "lat": 43.10, "lon": -75.23, "weight": 0.04},
    {"label": "Poughkeepsie NY",   "lat": 41.70, "lon": -73.93, "weight": 0.03},
]

_nyiso_weight_sum = sum(loc["weight"] for loc in NYISO_LOAD_LOCATIONS)
if abs(_nyiso_weight_sum - 1.0) > 0.01:
    _log.warning(
        "NYISO_LOAD_LOCATIONS weights sum to %.4f (expected 1.0); "
        "weighted average normalises, so results are correct, but consider updating weights.",
        _nyiso_weight_sum,
    )
