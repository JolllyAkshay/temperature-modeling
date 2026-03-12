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

# Open-Meteo hourly surface skin temp (skin_temperature) — 28 days × 24 hours
_SKIN_TEMPS_C = [
    -1.2, -1.5, -1.8, -2.0, -2.1, -2.0, -1.4, 0.3, 2.1, 3.8, 4.9, 5.4,
     5.6,  5.3,  4.8,  4.1,  3.2,  2.3,  1.5, 0.9, 0.4, 0.0,-0.4,-0.8,
]  # one representative day repeated for simplicity

OPEN_METEO_RESPONSE = {
    "hourly": {
        "time": [
            f"2026-02-{day:02d}T{h:02d}:00"
            for day in range(1, 29)
            for h in range(24)
        ],
        "skin_temperature": [
            _SKIN_TEMPS_C[h % 24] + (day - 1) * 0.05
            for day in range(1, 29)
            for h in range(24)
        ],
    }
}

# GraphCast (gfs_graphcast025) hourly 2m air temp — 28 days × 24 hours
_GC_TEMPS_C = [
    -0.8, -1.1, -1.4, -1.6, -1.7, -1.6, -1.0, 0.6, 2.4, 4.0, 5.1, 5.7,
     5.9,  5.6,  5.0,  4.3,  3.4,  2.5,  1.7, 1.1, 0.6, 0.2, -0.2, -0.6,
]  # one representative day repeated for simplicity

GRAPHCAST_RESPONSE = {
    "hourly": {
        "time": [
            f"2026-02-{day:02d}T{h:02d}:00"
            for day in range(1, 29)
            for h in range(24)
        ],
        "temperature_2m": [
            _GC_TEMPS_C[h % 24] + (day - 1) * 0.06
            for day in range(1, 29)
            for h in range(24)
        ],
    }
}

# ERA5 reanalysis hourly temperature_2m — used as observational truth in
# verification.  Slightly cooler than GraphCast to produce a realistic warm bias.
_ERA5_T2M_C = [
    -1.0, -1.3, -1.6, -1.8, -1.9, -1.8, -1.2, 0.4, 2.2, 3.7, 4.8, 5.2,
     5.4,  5.1,  4.6,  3.9,  3.0,  2.1,  1.3, 0.7, 0.2,-0.2,-0.5,-0.9,
]  # one representative day repeated for simplicity

ERA5_T2M_RESPONSE = {
    "hourly": {
        "time": [
            f"2026-02-{day:02d}T{h:02d}:00"
            for day in range(1, 29)
            for h in range(24)
        ],
        "temperature_2m": [
            _ERA5_T2M_C[h % 24] + (day - 1) * 0.045
            for day in range(1, 29)
            for h in range(24)
        ],
    }
}

# NASA POWER daily Earth Skin Temp (TS) — 28 days
NASA_POWER_RESPONSE = {
    "properties": {
        "parameter": {
            "TS": {
                f"202602{day:02d}": round(-1.0 + day * 0.3, 2)
                for day in range(1, 29)
            }
        }
    }
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
    if "weather.gov" in url and "forecast/hourly" in url:
        return _make_response(NWS_HOURLY_RESPONSE)
    if "weather.gov" in url and "forecast" in url:
        return _make_response(NWS_FORECAST_RESPONSE)
    if "/stations" in url:
        return _make_response(NCEI_STATIONS_RESPONSE)
    if "ncei" in url or "noaa.gov/cdo-web" in url:
        return _make_response(NCEI_DATA_RESPONSE)
    if "historical-forecast-api.open-meteo" in url:
        return _make_response(GRAPHCAST_RESPONSE)
    if "archive-api.open-meteo" in url:
        params = kwargs.get("params", {})
        if params.get("hourly") == "temperature_2m":
            return _make_response(ERA5_T2M_RESPONSE)
        return _make_response(OPEN_METEO_RESPONSE)  # skin_temperature
    if "open-meteo" in url:
        # Live forecast endpoint — distinguish GraphCast from plain ERA5.
        params = kwargs.get("params", {})
        if params.get("models") == "gfs_graphcast025":
            return _make_response(GRAPHCAST_RESPONSE)
        return _make_response(OPEN_METEO_RESPONSE)
    if "power.larc.nasa.gov" in url:
        return _make_response(NASA_POWER_RESPONSE)
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
            include_satellite=True,
            satellite_source="open-meteo",
        )
        result_nasa = client.get_weather(
            "Columbus, OH",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
            include_forecast=False,
            include_historical=False,
            include_satellite=True,
            satellite_source="nasa-power",
        )
        result_gc = client.get_weather(
            "Columbus, OH",
            start_date=date(2026, 2, 1),
            end_date=date(2026, 2, 28),
            include_forecast=False,
            include_historical=False,
            include_satellite=True,
            satellite_source="graphcast",
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

    print(f"\n{'─'*60}")
    print(f"  SATELLITE SURFACE TEMP  ({len(result.satellite)} readings, source: {result.satellite[0].source if result.satellite else 'n/a'})")
    print(f"{'─'*60}")
    print(f"  {'Timestamp':<20} {'Skin Temp (°C)':>14} {'Skin Temp (°F)':>14}")
    print(f"  {'-'*20} {'-'*14} {'-'*14}")
    for obs in result.satellite[:12]:  # first 12 hours
        print(f"  {obs.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {obs.surface_temp_c:>14.2f} {obs.surface_temp_f:>14.2f}")
    if len(result.satellite) > 12:
        print(f"  ... ({len(result.satellite) - 12} more hourly readings)")

    print(f"\n{'─'*60}")
    print(f"  NASA POWER SURFACE TEMP  ({len(result_nasa.satellite)} daily readings)")
    print(f"{'─'*60}")
    print(f"  {'Date':<12} {'Skin Temp (°C)':>14} {'Skin Temp (°F)':>14}")
    print(f"  {'-'*12} {'-'*14} {'-'*14}")
    for obs in result_nasa.satellite:
        print(f"  {obs.timestamp.strftime('%Y-%m-%d'):<12} {obs.surface_temp_c:>14.2f} {obs.surface_temp_f:>14.2f}")

    print(f"\n{'─'*60}")
    src_gc = result_gc.satellite[0].source if result_gc.satellite else "n/a"
    print(f"  GRAPHCAST 2M AIR TEMP  ({len(result_gc.satellite)} readings, source: {src_gc})")
    print(f"{'─'*60}")
    print(f"  {'Timestamp':<20} {'Temp 2m (°C)':>13} {'Temp 2m (°F)':>13}")
    print(f"  {'-'*20} {'-'*13} {'-'*13}")
    for obs in result_gc.satellite[:12]:  # first 12 hours
        print(f"  {obs.timestamp.strftime('%Y-%m-%d %H:%M'):<20} {obs.surface_temp_c:>13.2f} {obs.surface_temp_f:>13.2f}")
    if len(result_gc.satellite) > 12:
        print(f"  ... ({len(result_gc.satellite) - 12} more hourly readings)")

    # ── Verification: forecast error by lead time ─────────────────────────────
    from datetime import timedelta
    from temperature_modeling import collect_verification_records, score_by_lead
    from temperature_modeling.models import Coordinates

    coords = Coordinates(lat=39.9612, lon=-82.9988)
    # Use 5 initialization dates (Feb 1–5) so the demo is fast but informative.
    init_dates = [date(2026, 2, 1) + timedelta(days=i) for i in range(5)]

    with patch.object(client._session, "get", side_effect=mock_session_get):
        records = collect_verification_records(coords, init_dates, client._session)

    skill = score_by_lead(records)

    print(f"\n{'─'*60}")
    print(f"  GRAPHCAST VERIFICATION  ({len(records)} samples, {len(init_dates)} init dates)")
    print(f"{'─'*60}")
    print(f"  {'Lead':>4}  {'RMSE (°C)':>10}  {'MAE (°C)':>9}  {'Bias (°C)':>9}  {'n':>4}")
    print(f"  {'────':>4}  {'──────────':>10}  {'─────────':>9}  {'─────────':>9}  {'──':>4}")
    for s in skill:
        marker = " ◄" if 10 <= s.lead_days <= 15 else ""
        print(
            f"  {s.lead_days:>4}  {s.rmse:>10.3f}  {s.mae:>9.3f}  {s.bias:>+9.3f}  {s.n:>4}{marker}"
        )
    print(f"\n  ◄ = target 10–15 day window")

    # ── Feature extraction ────────────────────────────────────────────────────
    from temperature_modeling import build_climate_normals, extract_features

    # Build 5-year ERA5 climatology anchored to the start of our eval period.
    anchor = date(2026, 2, 1)
    with patch.object(client._session, "get", side_effect=mock_session_get):
        normals = build_climate_normals(coords, anchor, client._session)

    vectors = extract_features(records, normals)

    print(f"\n{'─'*60}")
    print(f"  FEATURE EXTRACTION  ({len(vectors)} vectors, {len(set(v.lead_days for v in vectors))} lead days)")
    print(f"{'─'*60}")

    # Print one representative vector per target lead day (10 and 13).
    for target_lead in (10, 13):
        sample = next((v for v in vectors if int(v.lead_days) == target_lead), None)
        if sample is None:
            continue
        print(f"\n  Lead day {target_lead}:")
        print(f"    {'Feature':<24} {'Value':>10}")
        print(f"    {'-'*24} {'-'*10}")
        for name, val in [
            ("lead_days",          sample.lead_days),
            ("lead_days_sq",       sample.lead_days_sq),
            ("forecast_temp_c",    sample.forecast_temp_c),
            ("clim_mean_c",        sample.clim_mean_c),
            ("forecast_anomaly_c", sample.forecast_anomaly_c),
            ("clim_std_c",         sample.clim_std_c),
            ("valid_sin_doy",      sample.valid_sin_doy),
            ("valid_cos_doy",      sample.valid_cos_doy),
            ("init_sin_doy",       sample.init_sin_doy),
            ("init_cos_doy",       sample.init_cos_doy),
            ("error_c  [target]",  sample.error_c),
        ]:
            print(f"    {name:<24} {val:>10.4f}")

    # ── Correction models ─────────────────────────────────────────────────────
    from temperature_modeling import train_and_evaluate

    MODEL_TYPES = ["mean_bias", "linear", "ridge", "random_forest", "xgboost"]

    print(f"\n{'─'*60}")
    print(f"  CORRECTION MODEL COMPARISON  (train/test split, 80/20)")
    print(f"{'─'*60}")
    print(
        f"  {'Model':<16} {'Train n':>7} {'Test n':>7} "
        f"{'Raw RMSE':>9} {'Corr RMSE':>10} {'Skill':>7}"
    )
    print(
        f"  {'─'*16} {'───────':>7} {'───────':>7} "
        f"{'─────────':>9} {'──────────':>10} {'───────':>7}"
    )

    best_model = None
    best_eval = None

    for mtype in MODEL_TYPES:
        try:
            model, ev = train_and_evaluate(vectors, model_type=mtype)
            skill_pct = f"{ev.window_skill_score:+.1%}"
            print(
                f"  {ev.model_type:<16} {ev.n_train:>7} {ev.n_test:>7} "
                f"{ev.window_raw_rmse:>9.3f} {ev.window_corrected_rmse:>10.3f} {skill_pct:>7}"
            )
            if best_eval is None or ev.window_corrected_rmse < best_eval.window_corrected_rmse:
                best_model = model
                best_eval = ev
        except Exception as exc:
            print(f"  {mtype:<16}  [skipped: {exc}]")

    print(f"\n  ◄ Window = lead days 10–15 (target correction zone)")

    # Per-lead breakdown for the best model.
    if best_eval is not None:
        print(f"\n{'─'*60}")
        print(f"  PER-LEAD BREAKDOWN — best model: {best_eval.model_type}")
        print(f"{'─'*60}")
        print(f"  {'Lead':>4}  {'Raw RMSE':>9}  {'Corr RMSE':>10}  {'Δ RMSE':>8}  {'Skill':>7}")
        print(f"  {'────':>4}  {'─────────':>9}  {'──────────':>10}  {'────────':>8}  {'───────':>7}")
        all_leads = sorted(
            set(best_eval.per_lead_raw_rmse) | set(best_eval.per_lead_corrected_rmse)
        )
        for ld in all_leads:
            raw = best_eval.per_lead_raw_rmse.get(ld, float("nan"))
            corr = best_eval.per_lead_corrected_rmse.get(ld, float("nan"))
            delta = corr - raw
            skill_ld = (1.0 - corr / raw) if raw else float("nan")
            marker = " ◄" if 10 <= ld <= 15 else ""
            print(
                f"  {ld:>4}  {raw:>9.3f}  {corr:>10.3f}  {delta:>+8.3f}  {skill_ld:>+7.1%}{marker}"
            )

    print()


if __name__ == "__main__":
    main()
