import requests

from .exceptions import GeocodingError
from .models import Coordinates, Location
from .pjm import STATE_NAME_TO_ABBR, validate_pjm_state

NOMINATIM_URL = "https://nominatim.openstreetmap.org/search"


def geocode_location(location: str, session: requests.Session) -> Location:
    """
    Convert a city/state string to a Location using Nominatim.

    Parameters
    ----------
    location:
        Free-text location, e.g. "Philadelphia, PA" or "Columbus, Ohio".
    session:
        A requests.Session with User-Agent header already set.

    Returns
    -------
    Location with coordinates and resolved state abbreviation.

    Raises
    ------
    GeocodingError
        If Nominatim returns no results or the HTTP request fails.
    LocationNotInPJMError
        If the resolved state is outside PJM territory.
    """
    try:
        response = session.get(
            NOMINATIM_URL,
            params={
                "q": location,
                "format": "json",
                "addressdetails": 1,
                "limit": 1,
                "countrycodes": "us",
            },
        )
        response.raise_for_status()
    except requests.RequestException as exc:
        raise GeocodingError(f"Geocoding request failed for '{location}': {exc}") from exc

    results = response.json()
    if not results:
        raise GeocodingError(f"No results found for location: '{location}'")

    result = results[0]
    address = result.get("address", {})

    # Nominatim returns full state name; map to abbreviation
    state_full = address.get("state", "")
    state_abbr = STATE_NAME_TO_ABBR.get(state_full)
    if not state_abbr:
        raise GeocodingError(
            f"Could not determine US state for '{location}' "
            f"(Nominatim returned state='{state_full}')"
        )

    validate_pjm_state(state_abbr)

    coords = Coordinates(lat=float(result["lat"]), lon=float(result["lon"]))
    display_name = result.get("display_name", location)

    return Location(coordinates=coords, state=state_abbr, display_name=display_name)
