"""
MISO (Midcontinent Independent System Operator) population-weighted load locations.
Covers MISO North (MN, WI, MI, IA, ND, SD, Manitoba) + Central (IL, MO) +
South (AR, LA, MS, TX panhandle) footprint (~45M people).
Weights sum to 1.0 and are proportional to metro population within MISO territory.
"""

MISO_LOAD_LOCATIONS: list[dict] = [
    {"label": "Chicago IL",      "lat": 41.85, "lon": -87.65, "weight": 0.18},
    {"label": "Detroit MI",      "lat": 42.33, "lon": -83.05, "weight": 0.11},
    {"label": "Minneapolis MN",  "lat": 44.98, "lon": -93.27, "weight": 0.10},
    {"label": "Milwaukee WI",    "lat": 43.04, "lon": -87.91, "weight": 0.07},
    {"label": "St. Louis MO",    "lat": 38.63, "lon": -90.20, "weight": 0.08},
    {"label": "New Orleans LA",  "lat": 29.95, "lon": -90.07, "weight": 0.08},
    {"label": "Memphis TN",      "lat": 35.15, "lon": -90.05, "weight": 0.07},
    {"label": "Kansas City MO",  "lat": 39.10, "lon": -94.58, "weight": 0.07},
    {"label": "Des Moines IA",   "lat": 41.60, "lon": -93.61, "weight": 0.06},
    {"label": "Grand Rapids MI", "lat": 42.96, "lon": -85.67, "weight": 0.06},
    {"label": "Little Rock AR",  "lat": 34.75, "lon": -92.29, "weight": 0.06},
    {"label": "Fargo ND",        "lat": 46.88, "lon": -96.79, "weight": 0.06},
]
