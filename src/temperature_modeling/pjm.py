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
