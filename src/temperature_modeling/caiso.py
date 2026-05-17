"""
CAISO (California ISO) population-weighted load locations.
Covers the CISO EIA respondent footprint (~30M people in California).
Weights sum to 1.0 and are proportional to metro population within CISO territory.
"""

CAISO_LOAD_LOCATIONS: list[dict] = [
    {"label": "Los Angeles CA",   "lat": 34.05, "lon": -118.25, "weight": 0.35},
    {"label": "Riverside CA",     "lat": 33.95, "lon": -117.40, "weight": 0.12},
    {"label": "San Francisco CA", "lat": 37.77, "lon": -122.42, "weight": 0.10},
    {"label": "San Diego CA",     "lat": 32.72, "lon": -117.16, "weight": 0.09},
    {"label": "Sacramento CA",    "lat": 38.58, "lon": -121.49, "weight": 0.07},
    {"label": "San Jose CA",      "lat": 37.34, "lon": -121.89, "weight": 0.05},
    {"label": "Fresno CA",        "lat": 36.74, "lon": -119.79, "weight": 0.05},
    {"label": "Bakersfield CA",   "lat": 35.37, "lon": -119.02, "weight": 0.04},
    {"label": "Ventura CA",       "lat": 34.20, "lon": -119.17, "weight": 0.04},
    {"label": "Stockton CA",      "lat": 37.96, "lon": -121.29, "weight": 0.03},
    {"label": "Palm Springs CA",  "lat": 33.83, "lon": -116.54, "weight": 0.03},
    {"label": "Santa Barbara CA", "lat": 34.42, "lon": -119.70, "weight": 0.03},
]
