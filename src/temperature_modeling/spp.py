"""
SPP (Southwest Power Pool) population-weighted load locations.
Covers Kansas, Nebraska, Oklahoma, most of Arkansas, parts of Missouri, SD, ND, MT, WY, NM.
Weights proportional to load zone population and thermal load sensitivity.
"""

import logging as _logging

_log = _logging.getLogger(__name__)

SPP_LOAD_LOCATIONS: list[dict] = [
    {"label": "Oklahoma City OK", "lat": 35.47, "lon": -97.52, "weight": 0.14},
    {"label": "Tulsa OK",         "lat": 36.15, "lon": -95.99, "weight": 0.11},
    {"label": "Wichita KS",       "lat": 37.69, "lon": -97.34, "weight": 0.10},
    {"label": "Kansas City KS",   "lat": 39.12, "lon": -94.63, "weight": 0.09},
    {"label": "Omaha NE",         "lat": 41.26, "lon": -96.00, "weight": 0.09},
    {"label": "Little Rock AR",   "lat": 34.75, "lon": -92.29, "weight": 0.08},
    {"label": "Springfield MO",   "lat": 37.21, "lon": -93.29, "weight": 0.06},
    {"label": "Lincoln NE",       "lat": 40.81, "lon": -96.70, "weight": 0.07},
    {"label": "Amarillo TX",      "lat": 35.22, "lon": -101.83, "weight": 0.06},
    {"label": "Lubbock TX",       "lat": 33.58, "lon": -101.86, "weight": 0.05},
    {"label": "Sioux Falls SD",   "lat": 43.54, "lon": -96.73, "weight": 0.05},
    {"label": "Fargo ND",         "lat": 46.88, "lon": -96.79, "weight": 0.04},
    {"label": "Billings MT",      "lat": 45.78, "lon": -108.50, "weight": 0.03},
    {"label": "Albuquerque NM",   "lat": 35.08, "lon": -106.65, "weight": 0.03},
]

_spp_weight_sum = sum(loc["weight"] for loc in SPP_LOAD_LOCATIONS)
if abs(_spp_weight_sum - 1.0) > 0.01:
    _log.warning(
        "SPP_LOAD_LOCATIONS weights sum to %.4f (expected 1.0); "
        "weighted average normalises, so results are correct, but consider updating weights.",
        _spp_weight_sum,
    )
