"""
ISO-NE (New England ISO) population-weighted load locations.
Covers Connecticut, Maine, Massachusetts, New Hampshire, Rhode Island, Vermont.
Weights proportional to state population and electricity demand intensity.
"""

import logging as _logging

_log = _logging.getLogger(__name__)

ISONE_LOAD_LOCATIONS: list[dict] = [
    {"label": "Boston MA",        "lat": 42.36, "lon": -71.06, "weight": 0.28},
    {"label": "Worcester MA",     "lat": 42.26, "lon": -71.80, "weight": 0.08},
    {"label": "Springfield MA",   "lat": 42.10, "lon": -72.59, "weight": 0.06},
    {"label": "Hartford CT",      "lat": 41.76, "lon": -72.68, "weight": 0.12},
    {"label": "Bridgeport CT",    "lat": 41.18, "lon": -73.19, "weight": 0.08},
    {"label": "Providence RI",    "lat": 41.82, "lon": -71.42, "weight": 0.09},
    {"label": "Manchester NH",    "lat": 42.99, "lon": -71.46, "weight": 0.06},
    {"label": "Burlington VT",    "lat": 44.48, "lon": -73.21, "weight": 0.05},
    {"label": "Portland ME",      "lat": 43.66, "lon": -70.26, "weight": 0.07},
    {"label": "Bangor ME",        "lat": 44.80, "lon": -68.78, "weight": 0.04},
    {"label": "Concord NH",       "lat": 43.21, "lon": -71.54, "weight": 0.04},
    {"label": "New Haven CT",     "lat": 41.31, "lon": -72.93, "weight": 0.03},
]

_isone_weight_sum = sum(loc["weight"] for loc in ISONE_LOAD_LOCATIONS)
if abs(_isone_weight_sum - 1.0) > 0.01:
    _log.warning(
        "ISONE_LOAD_LOCATIONS weights sum to %.4f (expected 1.0); "
        "weighted average normalises, so results are correct, but consider updating weights.",
        _isone_weight_sum,
    )
