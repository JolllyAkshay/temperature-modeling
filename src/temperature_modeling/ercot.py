"""
ERCOT (Electric Reliability Council of Texas) population-weighted load locations.
Covers the ERCOT EIA respondent footprint (~27M people in Texas).
Weights sum to 1.0 and are proportional to metro population within ERCOT territory.
Note: El Paso is in WECC, not ERCOT.
"""

ERCOT_LOAD_LOCATIONS: list[dict] = [
    {"label": "Houston TX",        "lat": 29.76, "lon": -95.37,  "weight": 0.28},
    {"label": "Dallas TX",         "lat": 32.78, "lon": -96.80,  "weight": 0.22},
    {"label": "Austin TX",         "lat": 30.27, "lon": -97.74,  "weight": 0.10},
    {"label": "San Antonio TX",    "lat": 29.42, "lon": -98.49,  "weight": 0.09},
    {"label": "Fort Worth TX",     "lat": 32.75, "lon": -97.33,  "weight": 0.07},
    {"label": "Waco TX",           "lat": 31.55, "lon": -97.15,  "weight": 0.04},
    {"label": "Corpus Christi TX", "lat": 27.80, "lon": -97.40,  "weight": 0.04},
    {"label": "Lubbock TX",        "lat": 33.58, "lon": -101.86, "weight": 0.04},
    {"label": "Beaumont TX",       "lat": 30.08, "lon": -94.10,  "weight": 0.04},
    {"label": "Midland TX",        "lat": 31.99, "lon": -102.08, "weight": 0.03},
    {"label": "Amarillo TX",       "lat": 35.22, "lon": -101.83, "weight": 0.03},
    {"label": "Laredo TX",         "lat": 27.51, "lon": -99.51,  "weight": 0.02},
]
