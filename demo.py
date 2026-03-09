"""
Demo script: runs WeatherClient with mocked HTTP responses.
No real network access needed — all API calls are intercepted
and replied to with realistic sample data.

Usage:
    python demo.py
"""

import json
from datetime import date
from unittest.mock import MagicMock, patch

# ── Realistic mock payloads ──────────────────────────────────────────────────

NOMINATIM_RESPONSE = [
    {
        "lat": "39.9611755",
        "lon": "-82.9987942",
        "display_name": "Columbus, Franklin County, Ohio, United States",
        "address": {"state": "Ohio", "country": "United States"},
    }
]

NWS_POINTS_RESPONSE = {
    "properties": {
        "forecast": "https://api.weather.gov/gridpoints/ILN/83,71/forecast",
        "forecastHourly": "https://api.weather.gov/gridpoints/ILN/83,71/forecast/hourly",
        "relativeLocation": {
            "properties": {"city": "Columbus", "state": "Ohio"}
        },
    }
}

NWS_FORECAST_RESPONSE = {
    "properties": {
        "periods": [
            {
                "name": "Tonight",
                "startTime": "2026-03-09T18:00:00-05:00",
                "endTime": "2026-03-10T06:00:00-05:00",
                "isDaytime": False,
                "temperature": 38,
                "shortForecast": "Partly Cloudy",
            },
            {
                "name": "Monday",
                "startTime": "2026-03-10T06:00:00-05:00",
                "endTime": "2026-03-10T18:00:00-05:00",
                "isDaytime": True,
                "temperature": 52,
                "shortForecast": "Mostly Sunny",
            },
            {
                "name": "Monday Night",
                "startTime": "2026-03-10T18:00:00-05:00",
                "endTime": "2026-03-11T06:00:00-05:00",
                "isDaytime": False,
                "temperature": 35,
                "shortForecast": "Clear",
            },
            {
                "name": "Tuesday",
                "startTime": "2026-03-11T06:00:00-05:00",
                "endTime": "2026-03-11T18:00:00-05:00",
                "isDaytime": True,
                "temperature": 57,
                "shortForecast": "Partly Sunny",
            },
            {
                "name": "Tuesday Night",
                "startTime": "2026-03-11T18:00:00-05:00",
                "endTime": "2026-03-12T06:00:00-05:00",
                "isDaytime": False,
                "temperature": 41,
                "shortForecast": "Chance Rain Showers",
            },
            {
                "name": "Wednesday",
                "startTime": "2026-03-12T06:00:00-05:00",
                "endTime": "2026-03-12T18:00:00-05:00",
                "isDaytime": True,
                "temperature": 55,
                "shortForecast": "Rain Showers Likely",
            },
            {
                "name": "Wednesday Night",
                "startTime": "2026-03-12T18:00:00-05:00",
                "endTime": "2026-03-13T06:00:00-05:00",
                "isDaytime": False,
                "temperature": 39,
                "shortForecast": "Mostly Cloudy",
            },
        ]
    }
}

NWS_HOURLY_RESPONSE = {
    "properties": {
        "periods": [
            {
                "startTime": f"2026-03-09T{h:02d}:00:00-05:00",
                "temperature": temp,
                "shortForecast": cond,
            }
            for h, temp, cond in [
                (18, 45, "Partly Cloudy"),
                (19, 43, "Partly Cloudy"),
                (20, 41, "Mostly Clear"),
                (21, 40, "Clear"),
                (22, 39, "Clear"),
                (23, 38, "Clear"),
            ]
        ]
    }
}

NCEI_STATIONS_RESPONSE = {
    "results": [
        {
            "id": "GHCND:USW00014821",
            "name": "COLUMBUS INTERNATIONAL AIRPORT",
            "latitude": 39.9981,
            "longitude": -82.8919,
        }
    ]
}

# 28 days of Feb 2026 — TMAX and TMIN pairs (°C)
_TMAX = [5.6, 3.3, 1.1, -0.6, 2.2, 7.8, 9.4, 8.3, 6.7, 4.4,
          3.3, 0.6, -2.2, 1.7, 6.1, 8.9, 10.0, 7.8, 5.6, 3.3,
          2.2, 1.1, 4.4, 7.2, 9.4, 11.1, 8.9, 6.7]
_TMIN = [-2.2, -4.4, -6.7, -8.3, -5.6, -1.7, 2.2, 1.1, -0.6, -3.3,
          -5.0, -7.2, -10.0, -6.7, -2.2, 1.1, 3.3, 2.2, -0.6, -3.3,
          -4.4, -5.6, -2.2, 0.6, 2.8, 4.4, 2.2, 0.0]

NCEI_DATA_RESPONSE = {
    "results": [
        {
            "date": f"2026-02-{day:02d}T00:00:00",
            "datatype": dtype,
            "value": val,
        }
        for day in range(1, 29)
        for dtype, val in [("TMAX", _TMAX[day - 1]), ("TMIN", _TMIN[day - 1])]
    ]
}


# ── Mock HTTP session ────────────────────────────────────────────────────────

def _make_response(payload):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = payload
    r.raise_for_status.return_value = None
    return r


def mock_session_get(url, **kwargs):
    if "nominatim" in url:
        return _make_response(NOMINATIM_RESPONSE)
    if "/points/" in url:
        return _make_response(NWS_POINTS_RESPONSE)
    if "forecast/hourly" in url:
        return _make_response(NWS_HOURLY_RESPONSE)
    if "forecast" in url:
        return _make_response(NWS_FORECAST_RESPONSE)
    if "/stations" in url:
        return _make_response(NCEI_STATIONS_RESPONSE)
    if "/data" in url:
        return _make_response(NCEI_DATA_RESPONSE)
    raise ValueError(f"Unexpected URL in mock: {url}")


# ── Run demo ─────────────────────────────────────────────────────────────────

def main():
    import requests
    from temperature_modeling import WeatherClient

    client = WeatherClient(ncei_token="DEMO_TOKEN")

    # Patch the underlying session.get so no real network calls are made
    with patch.object(client._session, "get", side_effect=mock_session_get):
        result = client.get_weather(
            "Columbus, OH",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
        )

    # ── Print results ────────────────────────────────────────────────────────
    print("=" * 60)
    print(f"  Location : {result.location.display_name}")
    print(f"  State    : {result.location.state}")
    print(f"  Lat/Lon  : {result.location.coordinates.lat}, {result.location.coordinates.lon}")
    print("=" * 60)

    print(f"\n{'─'*60}")
    print(f"  HISTORICAL  (Feb 2026 — {len(result.historical)} records)")
    print(f"{'─'*60}")

    # Group by date: find TMAX and TMIN per day
    from collections import defaultdict
    by_day = defaultdict(dict)
    for obs in result.historical:
        day_str = obs.timestamp.strftime("%Y-%m-%d")
        # We interleaved TMAX/TMIN; use temp_f sign to differentiate (crude)
        # Better: track by index — TMAX comes first per day
        if "max" not in by_day[day_str]:
            by_day[day_str]["max"] = obs.temp_f
        else:
            by_day[day_str]["min"] = obs.temp_f

    print(f"  {'Date':<12} {'High (°F)':>10} {'Low (°F)':>10}")
    print(f"  {'-'*12} {'-'*10} {'-'*10}")
    for day, vals in sorted(by_day.items()):
        print(f"  {day:<12} {vals.get('max', '—'):>10.1f} {vals.get('min', '—'):>10.1f}")

    print(f"\n{'─'*60}")
    print(f"  7-DAY FORECAST  ({len(result.forecast_periods)} periods)")
    print(f"{'─'*60}")
    print(f"  {'Period':<20} {'Temp (°F)':>10}  Condition")
    print(f"  {'-'*20} {'-'*10}  {'-'*20}")
    for p in result.forecast_periods:
        icon = "☀" if p.is_daytime else "🌙"
        print(f"  {p.name:<20} {p.temp_f:>10}  {p.short_forecast}")

    print(f"\n{'─'*60}")
    print(f"  HOURLY FORECAST  (next {len(result.hourly_forecast)} hours)")
    print(f"{'─'*60}")
    print(f"  {'Hour':<8} {'Temp (°F)':>10}  Condition")
    print(f"  {'-'*8} {'-'*10}  {'-'*20}")
    for h in result.hourly_forecast:
        print(f"  {h.timestamp.strftime('%H:%M'):<8} {h.temp_f:>10}  {h.short_forecast}")

    print()


if __name__ == "__main__":
    main()
