from .exceptions import LocationNotInPJMError

PJM_STATES: frozenset = frozenset({
    "DE",  # Delaware
    "IL",  # Illinois
    "IN",  # Indiana
    "KY",  # Kentucky
    "MD",  # Maryland
    "MI",  # Michigan
    "NJ",  # New Jersey
    "NC",  # North Carolina
    "OH",  # Ohio
    "PA",  # Pennsylvania
    "TN",  # Tennessee
    "VA",  # Virginia
    "WV",  # West Virginia
    "DC",  # District of Columbia
})

# Full state name → two-letter abbreviation (used by geocoding)
STATE_NAME_TO_ABBR: dict = {
    "Delaware": "DE",
    "Illinois": "IL",
    "Indiana": "IN",
    "Kentucky": "KY",
    "Maryland": "MD",
    "Michigan": "MI",
    "New Jersey": "NJ",
    "North Carolina": "NC",
    "Ohio": "OH",
    "Pennsylvania": "PA",
    "Tennessee": "TN",
    "Virginia": "VA",
    "West Virginia": "WV",
    "District of Columbia": "DC",
    # Also handle common abbreviations passed directly
    "DE": "DE", "IL": "IL", "IN": "IN", "KY": "KY", "MD": "MD",
    "MI": "MI", "NJ": "NJ", "NC": "NC", "OH": "OH", "PA": "PA",
    "TN": "TN", "VA": "VA", "WV": "WV", "DC": "DC",
}


# PJM grid monitoring locations with population-based load weights.
# Weights are proportional to the metro-area population within each grid cell
# and sum to 1.0. Used to compute a single population-weighted temperature
# representative of the PJM footprint.
PJM_LOAD_LOCATIONS: list[dict] = [
    {"label": "Philadelphia PA", "lat": 40.0, "lon": -75.5, "weight": 0.14},
    {"label": "Frederick MD",    "lat": 39.5, "lon": -77.5, "weight": 0.08},
    {"label": "NE Pennsylvania", "lat": 41.0, "lon": -75.5, "weight": 0.06},
    {"label": "Shenandoah VA",   "lat": 38.5, "lon": -78.5, "weight": 0.04},
    {"label": "Roanoke VA",      "lat": 37.5, "lon": -80.0, "weight": 0.05},
    {"label": "SW Virginia",     "lat": 36.5, "lon": -82.0, "weight": 0.03},
    {"label": "E Kentucky",      "lat": 37.5, "lon": -84.5, "weight": 0.04},
    {"label": "Columbus OH",     "lat": 40.0, "lon": -83.0, "weight": 0.12},
    {"label": "Toledo OH",       "lat": 41.5, "lon": -83.5, "weight": 0.06},
    {"label": "Indianapolis IN", "lat": 39.5, "lon": -86.5, "weight": 0.10},
    {"label": "Charlotte NC",    "lat": 35.0, "lon": -80.5, "weight": 0.09},
    {"label": "Pittsburgh PA",   "lat": 40.5, "lon": -80.0, "weight": 0.09},
]


def validate_pjm_state(state: str) -> None:
    """
    Raise LocationNotInPJMError if state is not in PJM territory.

    Parameters
    ----------
    state:
        Two-letter US state abbreviation or "DC", case-insensitive.

    Raises
    ------
    LocationNotInPJMError
    """
    normalized = state.upper().strip()
    if normalized not in PJM_STATES:
        raise LocationNotInPJMError(normalized)
