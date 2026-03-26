"""
US Temperature Forecast & PJM Grid Load Dashboard
Run:  python dashboard.py
Open: http://127.0.0.1:8050
"""

import json
import math
import os
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import date, datetime, timedelta
from pathlib import Path

import requests
import dash
from dash import dcc, html, Input, Output, State, callback
import plotly.graph_objects as go

# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
_HERE = Path(__file__).parent
sys.path.insert(0, str(_HERE / "src"))

from temperature_modeling.pjm import PJM_LOAD_LOCATIONS
from temperature_modeling.pjm_load import LoadCorrectionModel, load_load_model

FORECAST_CACHE_TTL_HOURS = 3

MODELS = {
    "gfs":       {"label": "GFS",       "param": None,               "days": 15, "color": "#2563eb"},
    "graphcast": {"label": "GraphCast", "param": "gfs_graphcast025", "days": 10, "color": "#16a34a"},
    "ecmwf":     {"label": "ECMWF IFS", "param": "ecmwf_ifs025",    "days": 10, "color": "#9333ea"},
}
DEFAULT_MODEL = "gfs"

def _cache_file(model_key):
    return _HERE / "api_cache" / f"forecast_dashboard_{model_key}.json"

_LOAD_FORECAST_CACHE_FILE = _HERE / "api_cache" / "load_forecast_cache.json"

WMO_LABELS = {
    0: ("Clear", ""),  1: ("Mostly Clear", ""),  2: ("Partly Cloudy", ""),
    3: ("Overcast", ""), 45: ("Foggy", ""), 48: ("Icy Fog", ""),
    51: ("Light Drizzle", ""), 53: ("Drizzle", ""), 55: ("Heavy Drizzle", ""),
    61: ("Light Rain", ""), 63: ("Rain", ""), 65: ("Heavy Rain", ""),
    71: ("Light Snow", ""), 73: ("Snow", ""), 75: ("Heavy Snow", ""),
    77: ("Snow Grains", ""), 80: ("Showers", ""), 81: ("Showers", ""),
    82: ("Heavy Showers", ""), 85: ("Snow Showers", ""), 86: ("Heavy Snow", ""),
    95: ("Thunderstorm", ""), 96: ("Thunderstorm", ""), 99: ("Thunderstorm", ""),
}

# ---------------------------------------------------------------------------
# CONUS grid (87 points)
# ---------------------------------------------------------------------------
CONUS_GRID = [
    ("Pacific NW coast",      48.5, -124.5), ("Oregon coast",          46.5, -124.0),
    ("N California coast",    41.5, -124.0), ("San Francisco CA",      37.5, -122.5),
    ("Los Angeles CA",        34.0, -118.5), ("San Diego CA",          32.5, -117.0),
    ("Seattle WA",            47.5, -122.0), ("Portland OR",           45.5, -122.5),
    ("N Nevada",              40.5, -117.5), ("Boise ID",              43.5, -116.0),
    ("W Montana",             47.0, -114.0), ("Great Falls MT",        47.5, -111.0),
    ("N Idaho",               47.0, -116.5), ("Spokane WA",            47.5, -117.5),
    ("Las Vegas NV",          36.0, -115.0), ("Salt Lake City UT",     40.5, -112.0),
    ("S Nevada",              36.5, -114.5), ("Central Utah",          39.0, -111.5),
    ("SW Colorado",           37.5, -107.5), ("N New Mexico",          36.5, -106.0),
    ("Phoenix AZ",            33.5, -112.0), ("Tucson AZ",             32.0, -110.5),
    ("El Paso TX",            31.5, -106.5), ("SE New Mexico",         33.0, -104.5),
    ("Albuquerque NM",        35.0, -106.5), ("Denver CO",             39.5, -105.0),
    ("Colorado Springs CO",   38.5, -104.5), ("Pueblo CO",             38.0, -104.5),
    ("S Colorado",            37.0, -104.5), ("W Kansas",              38.5, -100.5),
    ("Wichita KS",            37.5,  -97.5), ("Oklahoma City OK",      35.5,  -97.5),
    ("N Texas Panhandle",     35.5, -101.5), ("Lubbock TX",            33.5, -101.5),
    ("Midland TX",            31.5, -102.5), ("San Antonio TX",        29.5,  -98.5),
    ("Austin TX",             30.5,  -97.5), ("Dallas TX",             32.5,  -97.0),
    ("Houston TX",            29.5,  -95.5), ("SE Texas",              30.0,  -94.0),
    ("Louisiana",             30.5,  -92.5), ("Mississippi coast",     30.5,  -88.5),
    ("Mobile AL",             30.5,  -88.0), ("N Florida Gulf",        29.5,  -83.5),
    ("Tampa FL",              27.5,  -82.5), ("Miami FL",              25.5,  -80.5),
    ("Orlando FL",            28.5,  -81.5), ("Jacksonville FL",       30.0,  -81.5),
    ("Atlanta GA",            33.5,  -84.5), ("Savannah GA",           32.0,  -81.0),
    ("Charlotte NC",          35.5,  -81.0), ("Raleigh NC",            35.5,  -78.5),
    ("Wilmington NC",         34.0,  -78.0), ("Columbia SC",           34.0,  -81.0),
    ("Memphis TN",            35.0,  -90.0), ("Nashville TN",          36.0,  -86.5),
    ("Knoxville TN",          36.0,  -84.0), ("Louisville KY",         38.0,  -85.5),
    ("E Kentucky",            37.5,  -84.5), ("SW Virginia",           36.5,  -82.0),
    ("Roanoke VA",            37.5,  -80.0), ("Shenandoah VA",         38.5,  -78.5),
    ("Richmond VA",           37.5,  -77.5), ("Norfolk VA",            36.5,  -76.5),
    ("Washington DC",         38.5,  -77.0), ("Frederick MD",          39.5,  -77.5),
    ("Philadelphia PA",       40.0,  -75.5), ("NE Pennsylvania",       41.0,  -75.5),
    ("Pittsburgh PA",         40.5,  -80.0), ("Cincinnati OH",         39.0,  -84.5),
    ("Columbus OH",           40.0,  -83.0), ("Cleveland OH",          41.5,  -81.5),
    ("Detroit MI",            42.5,  -83.5), ("Indianapolis IN",       39.5,  -86.0),
    ("Chicago IL",            41.5,  -88.0), ("Milwaukee WI",          43.0,  -88.0),
    ("Minneapolis MN",        44.5,  -93.5), ("St Louis MO",           38.5,  -90.5),
    ("Kansas City MO",        39.0,  -94.5), ("Omaha NE",              41.0,  -96.0),
    ("Sioux Falls SD",        43.5,  -96.5), ("Bismarck ND",           47.0, -100.5),
    ("Fargo ND",              46.5,  -96.5), ("Billings MT",           45.5, -108.5),
    ("Casper WY",             42.5, -106.5), ("Cheyenne WY",           41.0, -104.5),
    ("Long Island NY",        40.5,  -73.5), ("Albany NY",             42.5,  -74.0),
    ("Upstate NY",            43.0,  -76.5), ("Adirondacks NY",        44.5,  -74.5),
    ("Hartford CT",           41.5,  -72.5), ("Boston MA",             42.5,  -71.5),
    ("Vermont",               44.0,  -72.5), ("Central Maine",         44.5,  -70.5),
    ("N Maine",               47.0,  -68.5),
]

# ---------------------------------------------------------------------------
# Load model — load once at startup
# ---------------------------------------------------------------------------
_LOAD_MODEL: LoadCorrectionModel | None = None
try:
    _LOAD_MODEL = load_load_model()
    print("  Load model loaded OK")
except Exception as e:
    print(f"  Warning: could not load PJM load model: {e}")


# ---------------------------------------------------------------------------
# Forecast fetching (temperature)
# ---------------------------------------------------------------------------

def _fetch_one(label, lat, lon, session, model_param, forecast_days):
    url = "https://api.open-meteo.com/v1/forecast"
    params = {
        "latitude": lat, "longitude": lon,
        "daily": "temperature_2m_max,temperature_2m_min,precipitation_sum,weathercode",
        "forecast_days": forecast_days,
        "temperature_unit": "fahrenheit",
        "precipitation_unit": "inch",
        "timezone": "auto",
    }
    if model_param:
        params["models"] = model_param
    try:
        r = session.get(url, params=params, timeout=20)
        r.raise_for_status()
        d = r.json()["daily"]
        return {
            "label": label, "lat": lat, "lon": lon,
            "dates": d["time"],
            "hi":    d["temperature_2m_max"],
            "lo":    d["temperature_2m_min"],
            "precip": d["precipitation_sum"],
            "wmo":   d["weathercode"],
        }
    except Exception:
        return None


def fetch_forecasts(model_key=DEFAULT_MODEL, force=False):
    cfg = MODELS[model_key]
    cache_file = _cache_file(model_key)

    if not force and cache_file.exists():
        age_h = (time.time() - cache_file.stat().st_mtime) / 3600
        if age_h < FORECAST_CACHE_TTL_HOURS:
            try:
                data = json.loads(cache_file.read_text())
                if data and data[0]["dates"][0] == date.today().isoformat():
                    print(f"  [{cfg['label']}] Using cached forecasts ({age_h:.1f}h old)")
                    return data
            except Exception:
                pass

    print(f"  [{cfg['label']}] Fetching {len(CONUS_GRID)} locations...", flush=True)
    results = []
    session = requests.Session()
    session.headers["User-Agent"] = "temp-forecast-dashboard/1.0"

    with ThreadPoolExecutor(max_workers=10) as pool:
        futures = {
            pool.submit(_fetch_one, lbl, lat, lon, session, cfg["param"], cfg["days"]): (lbl, lat, lon)
            for lbl, lat, lon in CONUS_GRID
        }
        for fut in as_completed(futures):
            res = fut.result()
            if res:
                results.append(res)

    results.sort(key=lambda r: (-r["lat"], r["lon"]))
    cache_file.parent.mkdir(parents=True, exist_ok=True)
    cache_file.write_text(json.dumps(results))
    print(f"  [{cfg['label']}] Done - {len(results)} locations.")
    return results


# ---------------------------------------------------------------------------
# PJM load forecast fetching
# ---------------------------------------------------------------------------

def fetch_pjm_load_forecast(force=False):
    """
    Fetch 15-day temperature forecasts for the 12 PJM load locations (GFS),
    apply the trained load model, and return a list of day dicts.
    """
    if _LOAD_MODEL is None:
        return []

    # Check cache (3-hour TTL)
    if not force and _LOAD_FORECAST_CACHE_FILE.exists():
        age_h = (time.time() - _LOAD_FORECAST_CACHE_FILE.stat().st_mtime) / 3600
        if age_h < FORECAST_CACHE_TTL_HOURS:
            try:
                data = json.loads(_LOAD_FORECAST_CACHE_FILE.read_text())
                if data and data[0]["date"] == date.today().isoformat():
                    return data
            except Exception:
                pass

    session = requests.Session()
    session.headers["User-Agent"] = "temp-forecast-dashboard/1.0"

    # Fetch 15-day GFS forecasts for each PJM location
    pjm_avg_c:  dict = {}
    pjm_hi_c:   dict = {}
    pjm_lo_c:   dict = {}
    forecast_dates_strs = None

    for loc in PJM_LOAD_LOCATIONS:
        result = _fetch_one(loc["label"], loc["lat"], loc["lon"], session, None, 15)
        if not result:
            continue
        label = loc["label"]
        # Open-Meteo returns °F (temperature_unit=fahrenheit); convert to °C for model
        def f_list_to_c(lst):
            return [(v - 32) * 5 / 9 if v is not None else None for v in lst]
        hi_c  = f_list_to_c(result["hi"])
        lo_c  = f_list_to_c(result["lo"])
        avg_c = [(h + l) / 2 if h is not None and l is not None else None
                 for h, l in zip(hi_c, lo_c)]
        pjm_avg_c[label] = avg_c
        pjm_hi_c[label]  = hi_c
        pjm_lo_c[label]  = lo_c
        if forecast_dates_strs is None:
            forecast_dates_strs = result["dates"][:15]

    if not pjm_avg_c or not forecast_dates_strs:
        return []

    forecast_dates_list = [date.fromisoformat(d) for d in forecast_dates_strs]

    # Fetch last 2 days of actual ERA5 avg temp for lag initialisation
    from temperature_modeling._era5 import fetch_era5_daily
    from temperature_modeling.models import Coordinates as _Coords
    from temperature_modeling.pjm_load import weighted_avg_temp_f
    from datetime import timedelta
    era5_session = requests.Session()
    era5_session.headers["User-Agent"] = "temp-forecast-dashboard/1.0"
    today = date.today()
    lag_start = today - timedelta(days=3)
    lag_end   = today - timedelta(days=1)
    recent_avg_f = []
    try:
        per_label = {}
        for loc in PJM_LOAD_LOCATIONS:
            temps = fetch_era5_daily(
                _Coords(loc["lat"], loc["lon"]), lag_start, lag_end, era5_session
            )
            per_label[loc["label"]] = temps
        for lag_d in [today - timedelta(days=2), today - timedelta(days=1)]:
            c_map = {
                loc["label"]: per_label[loc["label"]][lag_d]
                for loc in PJM_LOAD_LOCATIONS
                if lag_d in per_label.get(loc["label"], {})
            }
            if c_map:
                recent_avg_f.append(weighted_avg_temp_f(c_map))
    except Exception:
        recent_avg_f = []

    load_forecasts = _LOAD_MODEL.predict_with_uncertainty(
        forecast_temps_c=pjm_avg_c,
        forecast_hi_c=pjm_hi_c,
        forecast_lo_c=pjm_lo_c,
        gefs_spread_c={},
        forecast_dates=forecast_dates_list,
        recent_avg_temps_f=recent_avg_f if len(recent_avg_f) == 2 else None,
    )

    result_data = []
    for lf in load_forecasts:
        result_data.append({
            "date":           lf.valid_date.isoformat(),
            "mean_load_mw":   round(lf.mean_load_mw),
            "low_load_mw":    round(lf.low_load_mw),
            "high_load_mw":   round(lf.high_load_mw),
            "hdd":            round(lf.hdd, 1),
            "cdd":            round(lf.cdd, 1),
            "avg_temp_f":     round(lf.avg_temp_f, 1),
        })

    _LOAD_FORECAST_CACHE_FILE.parent.mkdir(parents=True, exist_ok=True)
    _LOAD_FORECAST_CACHE_FILE.write_text(json.dumps(result_data))
    return result_data


# ---------------------------------------------------------------------------
# Startup
# ---------------------------------------------------------------------------
print("\nLoading forecasts...")
FORECAST_CACHE = {}
FORECAST_CACHE[DEFAULT_MODEL] = fetch_forecasts(DEFAULT_MODEL)

def _date_options(forecasts, max_days=15):
    dates = forecasts[0]["dates"][:max_days] if forecasts else []
    opts = []
    for i, d in enumerate(dates):
        label = (datetime.strptime(d, "%Y-%m-%d").strftime("%-d %b") if os.name != "nt"
                 else datetime.strptime(d, "%Y-%m-%d").strftime("%#d %b"))
        opts.append({"label": label, "value": i})
    return opts, dates

_default_date_options, _default_dates = _date_options(FORECAST_CACHE[DEFAULT_MODEL])

TEMP_COLORSCALE = [
    [0.00, "#1e3a8a"], [0.15, "#3b82f6"], [0.30, "#67e8f9"],
    [0.45, "#86efac"], [0.55, "#fef08a"], [0.70, "#fb923c"],
    [0.85, "#ef4444"], [1.00, "#7c2d12"],
]


def card(title, value, sub="", color="#1a1a2e"):
    return html.Div(
        style={
            "background": "#ffffff", "border": "1px solid #e2e8f0",
            "borderRadius": "8px", "padding": "10px 14px", "minWidth": "120px",
            "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
        },
        children=[
            html.Div(title, style={"fontSize": "10px", "color": "#94a3b8",
                                   "textTransform": "uppercase", "letterSpacing": "0.06em"}),
            html.Div(value, style={"fontSize": "20px", "fontWeight": 700,
                                   "color": color, "margin": "2px 0"}),
            html.Div(sub, style={"fontSize": "11px", "color": "#94a3b8"}),
        ],
    )


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------
_TAB_STYLE = {
    "padding": "10px 20px", "fontSize": "13px", "color": "#64748b",
    "backgroundColor": "#ffffff",
}
_TAB_SELECTED_STYLE = {
    "padding": "10px 20px", "fontSize": "13px", "fontWeight": 600,
    "color": "#0f172a", "backgroundColor": "#ffffff",
    "borderTop": "2px solid #2563eb",
}

app = dash.Dash(
    __name__,
    title="US Temperature & Grid Load Forecast",
    meta_tags=[{"name": "viewport", "content": "width=device-width, initial-scale=1"}],
    suppress_callback_exceptions=True,
)

app.layout = html.Div(
    style={"fontFamily": "Inter, system-ui, sans-serif",
           "backgroundColor": "#f8fafc", "minHeight": "100vh", "color": "#1e293b"},
    children=[
        # Header
        html.Div(
            style={"padding": "14px 24px", "borderBottom": "1px solid #e2e8f0",
                   "backgroundColor": "#ffffff",
                   "display": "flex", "alignItems": "center", "justifyContent": "space-between",
                   "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"},
            children=[
                html.Div([
                    html.H1("US Temperature & Grid Load Forecast",
                            style={"margin": 0, "fontSize": "18px", "fontWeight": 600, "color": "#0f172a"}),
                    html.Span(f"Updated {datetime.now().strftime('%b %d, %Y  %I:%M %p')}",
                              style={"color": "#94a3b8", "fontSize": "12px"}),
                ]),
                html.Button("Refresh", id="refresh-btn",
                            style={"background": "#f1f5f9", "border": "1px solid #e2e8f0",
                                   "color": "#475569", "padding": "6px 14px",
                                   "borderRadius": "6px", "cursor": "pointer", "fontSize": "13px"}),
            ],
        ),

        # Tabs
        dcc.Tabs(
            id="main-tabs",
            value="temperature",
            style={"backgroundColor": "#ffffff", "borderBottom": "1px solid #e2e8f0"},
            children=[

                # ── Temperature tab ──────────────────────────────────────────
                dcc.Tab(
                    label="Temperature Map",
                    value="temperature",
                    style=_TAB_STYLE,
                    selected_style=_TAB_SELECTED_STYLE,
                    children=[
                        # Controls
                        html.Div(
                            style={"padding": "10px 24px", "borderBottom": "1px solid #e2e8f0",
                                   "backgroundColor": "#ffffff",
                                   "display": "flex", "gap": "24px", "alignItems": "center", "flexWrap": "wrap"},
                            children=[
                                html.Div([
                                    html.Label("Model:", style={"color": "#64748b", "fontSize": "12px", "marginRight": "8px"}),
                                    dcc.RadioItems(
                                        id="model-select",
                                        options=[{"label": v["label"], "value": k} for k, v in MODELS.items()],
                                        value=DEFAULT_MODEL, inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "14px", "fontSize": "13px", "color": "#334155"},
                                    ),
                                ], style={"display": "flex", "alignItems": "center"}),

                                html.Div([
                                    html.Label("Show:", style={"color": "#64748b", "fontSize": "12px", "marginRight": "8px"}),
                                    dcc.RadioItems(
                                        id="temp-type",
                                        options=[{"label": "High", "value": "hi"}, {"label": "Low", "value": "lo"}, {"label": "Avg", "value": "avg"}],
                                        value="hi", inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "14px", "fontSize": "13px", "color": "#334155"},
                                    ),
                                ], style={"display": "flex", "alignItems": "center"}),

                                html.Div([
                                    html.Label("Units:", style={"color": "#64748b", "fontSize": "12px", "marginRight": "8px"}),
                                    dcc.RadioItems(
                                        id="units",
                                        options=[{"label": "F", "value": "F"}, {"label": "C", "value": "C"}],
                                        value="F", inline=True,
                                        inputStyle={"marginRight": "4px"},
                                        labelStyle={"marginRight": "14px", "fontSize": "13px", "color": "#334155"},
                                    ),
                                ], style={"display": "flex", "alignItems": "center"}),

                                html.Div([
                                    html.Label("Forecast day:", style={"color": "#64748b", "fontSize": "12px",
                                                                        "marginRight": "8px", "whiteSpace": "nowrap"}),
                                    dcc.Slider(
                                        id="day-slider",
                                        min=0, max=len(_default_dates) - 1, step=1, value=0,
                                        marks={i: {"label": opt["label"], "style": {"color": "#64748b", "fontSize": "11px"}}
                                               for i, opt in enumerate(_default_date_options)
                                               if i % 3 == 0 or i == len(_default_dates) - 1},
                                        tooltip={"placement": "bottom", "always_visible": False},
                                        updatemode="drag",
                                    ),
                                ], style={"flex": "1", "minWidth": "300px", "display": "flex", "alignItems": "center"}),
                            ],
                        ),

                        # Map + detail
                        html.Div(
                            style={"display": "flex", "height": "calc(100vh - 180px)"},
                            children=[
                                html.Div(style={"flex": "1 1 65%", "padding": "12px 16px 12px 24px"},
                                         children=[dcc.Graph(id="us-map", style={"height": "100%"},
                                                             config={"displayModeBar": False})]),
                                html.Div(
                                    id="detail-panel",
                                    style={"flex": "0 0 340px", "borderLeft": "1px solid #e2e8f0",
                                           "backgroundColor": "#ffffff", "padding": "16px 18px",
                                           "overflowY": "auto", "display": "flex",
                                           "flexDirection": "column", "gap": "14px"},
                                    children=[
                                        html.P("Click any location on the map for a detailed forecast",
                                               style={"color": "#94a3b8", "textAlign": "center",
                                                      "marginTop": "60px", "fontSize": "13px"})
                                    ],
                                ),
                            ],
                        ),
                    ],
                ),

                # ── Grid Load tab ────────────────────────────────────────────
                dcc.Tab(
                    label="Grid Load (PJM)",
                    value="load",
                    style=_TAB_STYLE,
                    selected_style=_TAB_SELECTED_STYLE,
                    children=[
                        html.Div(
                            style={"padding": "16px 24px", "backgroundColor": "#f8fafc",
                                   "minHeight": "calc(100vh - 130px)"},
                            children=[
                                # Sub-header
                                html.Div(
                                    style={"marginBottom": "14px"},
                                    children=[
                                        html.H2("PJM Electricity Grid Load Forecast",
                                                style={"margin": "0 0 4px 0", "fontSize": "16px",
                                                       "fontWeight": 600, "color": "#0f172a"}),
                                        html.Span(
                                            "15-day load projection derived from temperature forecasts "
                                            "across 12 PJM monitoring locations. Model trained on "
                                            "2 years of historical EIA load data.",
                                            style={"color": "#64748b", "fontSize": "12px"},
                                        ),
                                    ],
                                ),

                                # Summary cards (filled by callback)
                                html.Div(id="load-cards",
                                         style={"display": "flex", "gap": "12px", "flexWrap": "wrap",
                                                "marginBottom": "16px"}),

                                # Main load chart
                                html.Div(
                                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                                           "border": "1px solid #e2e8f0", "padding": "16px",
                                           "boxShadow": "0 1px 3px rgba(0,0,0,0.06)",
                                           "marginBottom": "16px"},
                                    children=[
                                        html.Div("15-Day Load Forecast (MW)",
                                                 style={"fontSize": "12px", "color": "#64748b",
                                                        "textTransform": "uppercase",
                                                        "letterSpacing": "0.05em", "marginBottom": "8px"}),
                                        dcc.Graph(id="load-forecast-chart",
                                                  style={"height": "340px"},
                                                  config={"displayModeBar": False}),
                                    ],
                                ),

                                # Temperature driver panel
                                html.Div(
                                    style={"backgroundColor": "#ffffff", "borderRadius": "10px",
                                           "border": "1px solid #e2e8f0", "padding": "16px",
                                           "boxShadow": "0 1px 3px rgba(0,0,0,0.06)"},
                                    children=[
                                        html.Div("Temperature Driver (PJM Population-Weighted Avg)",
                                                 style={"fontSize": "12px", "color": "#64748b",
                                                        "textTransform": "uppercase",
                                                        "letterSpacing": "0.05em", "marginBottom": "8px"}),
                                        dcc.Graph(id="load-temp-chart",
                                                  style={"height": "200px"},
                                                  config={"displayModeBar": False}),
                                    ],
                                ),
                            ],
                        ),
                        # Store for load forecast data
                        dcc.Store(id="load-forecast-store"),
                    ],
                ),
            ],
        ),

        dcc.Store(id="selected-loc"),
    ],
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def to_c(f):
    return round((f - 32) * 5 / 9, 1) if f is not None else None

def fmt_temp(val, units):
    if val is None:
        return "-"
    v = val if units == "F" else to_c(val)
    return f"{v:.0f} {units}"

def temp_val(row, dtype, day_idx):
    if dtype == "avg":
        hi = row["hi"][day_idx]
        lo = row["lo"][day_idx]
        return (hi + lo) / 2 if hi is not None and lo is not None else None
    return row[dtype][day_idx]


# ---------------------------------------------------------------------------
# Temperature tab callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("day-slider", "max"),
    Output("day-slider", "marks"),
    Input("model-select", "value"),
)
def update_slider(model_key):
    forecasts = FORECAST_CACHE.get(model_key) or []
    opts, dates = _date_options(forecasts, MODELS[model_key]["days"])
    max_day = max(len(dates) - 1, 0)
    marks = {i: {"label": opt["label"], "style": {"color": "#64748b", "fontSize": "11px"}}
             for i, opt in enumerate(opts) if i % 3 == 0 or i == max_day}
    return max_day, marks


@callback(
    Output("us-map", "figure"),
    Input("day-slider", "value"),
    Input("temp-type", "value"),
    Input("units", "value"),
    Input("model-select", "value"),
)
def update_map(day_idx, temp_type, units, model_key):
    if model_key not in FORECAST_CACHE:
        FORECAST_CACHE[model_key] = fetch_forecasts(model_key)

    forecasts = FORECAST_CACHE[model_key]
    lats, lons, labels, values, hovers = [], [], [], [], []

    for row in forecasts:
        if day_idx >= len(row["dates"]):
            continue
        v = temp_val(row, temp_type, day_idx)
        if v is None:
            continue

        display_v = v if units == "F" else to_c(v)
        wmo = row["wmo"][day_idx] if day_idx < len(row["wmo"]) else 0
        wx_label, wx_icon = WMO_LABELS.get(wmo, ("", ""))
        precip = row["precip"][day_idx] if day_idx < len(row["precip"]) else 0
        precip_str = f'{precip:.2f}"' if units == "F" else f"{precip * 25.4:.1f}mm"

        lats.append(row["lat"])
        lons.append(row["lon"])
        labels.append(row["label"])
        values.append(display_v)

        hi_str = fmt_temp(row["hi"][day_idx], units) if day_idx < len(row["hi"]) else "-"
        lo_str = fmt_temp(row["lo"][day_idx], units) if day_idx < len(row["lo"]) else "-"
        hovers.append(
            f"<b>{row['label']}</b><br>"
            f"{wx_icon} {wx_label}<br>"
            f"High: {hi_str}  Low: {lo_str}<br>"
            f"Precip: {precip_str}"
        )

    all_dates = forecasts[0]["dates"] if forecasts else []
    date_str = all_dates[day_idx] if day_idx < len(all_dates) else ""
    day_label = (datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %-d") if date_str and os.name != "nt"
                 else datetime.strptime(date_str, "%Y-%m-%d").strftime("%A, %B %#d") if date_str else "")

    cmin = min(values) if values else 0
    cmax = max(values) if values else 100

    fig = go.Figure()
    fig.add_trace(go.Scattergeo(
        lat=lats, lon=lons,
        mode="markers+text",
        marker=dict(
            size=14, color=values, colorscale=TEMP_COLORSCALE, cmin=cmin, cmax=cmax,
            colorbar=dict(
                title=dict(text=f"{units}", font=dict(color="#64748b", size=11)),
                tickfont=dict(color="#64748b", size=10),
                ticksuffix=f" {units}",
                thickness=12, len=0.75, x=1.01,
            ),
            line=dict(width=1, color="#ffffff"),
        ),
        text=[f"{v:.0f}" for v in values],
        textfont=dict(size=8, color="#1e293b"),
        textposition="middle center",
        hovertemplate="%{customdata}<extra></extra>",
        customdata=hovers,
    ))

    fig.update_layout(
        title=dict(
            text=f"{['High', 'Low', 'Average'][['hi','lo','avg'].index(temp_type)]} Temperature  -  {day_label}",
            font=dict(size=13, color="#64748b"), x=0.01, y=0.98,
        ),
        geo=dict(
            scope="usa", bgcolor="#f8fafc", landcolor="#f1f5f9",
            lakecolor="#dbeafe", subunitcolor="#cbd5e1",
            coastlinecolor="#94a3b8", countrycolor="#94a3b8",
            showland=True, showlakes=True, showsubunits=True, showcoastlines=True,
            projection_type="albers usa",
        ),
        paper_bgcolor="#f8fafc",
        margin=dict(l=0, r=0, t=30, b=0),
        font=dict(color="#1e293b"),
    )
    return fig


@callback(
    Output("detail-panel", "children"),
    Input("us-map", "clickData"),
    Input("units", "value"),
    Input("model-select", "value"),
)
def update_detail(click_data, units, model_key):
    if not click_data:
        return [html.P("Click any location on the map for a detailed forecast",
                       style={"color": "#94a3b8", "textAlign": "center",
                              "marginTop": "60px", "fontSize": "13px"})]

    forecasts = FORECAST_CACHE.get(model_key, [])
    if not forecasts:
        return [html.P("Loading...", style={"color": "#94a3b8", "textAlign": "center", "marginTop": "60px"})]

    pt = click_data["points"][0]
    lat, lon = pt["lat"], pt["lon"]
    row = min(forecasts, key=lambda r: abs(r["lat"] - lat) + abs(r["lon"] - lon))

    max_days = MODELS[model_key]["days"]
    dates = row["dates"][:max_days]
    hi_vals = [v if units == "F" else to_c(v) for v in row["hi"][:max_days]]
    lo_vals = [v if units == "F" else to_c(v) for v in row["lo"][:max_days]]

    day_labels = [datetime.strptime(d, "%Y-%m-%d").strftime("%-d %b") if os.name != "nt"
                  else datetime.strptime(d, "%Y-%m-%d").strftime("%#d %b")
                  for d in dates]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=day_labels, y=lo_vals, name="Low",
        line=dict(color="#3b82f6", width=2), mode="lines+markers", marker=dict(size=5),
    ))
    fig.add_trace(go.Scatter(
        x=day_labels, y=hi_vals, name="High",
        line=dict(color="#ef4444", width=2), mode="lines+markers", marker=dict(size=5),
        fill="tonexty", fillcolor="rgba(251,146,60,0.15)",
    ))
    fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.25, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=36, r=8, t=16, b=50),
        xaxis=dict(gridcolor="#f1f5f9", tickangle=-45, tickfont=dict(size=9), linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", ticksuffix=f" {units}", linecolor="#e2e8f0"),
        height=220,
    )

    def wx_row(i):
        wmo = row["wmo"][i] if i < len(row["wmo"]) else 0
        wx_label, wx_icon = WMO_LABELS.get(wmo, ("", ""))
        precip = row["precip"][i] if i < len(row["precip"]) else 0
        precip_str = f'{precip:.2f}"' if units == "F" else f"{precip*25.4:.0f}mm"
        hi_s = fmt_temp(row["hi"][i], units) if i < len(row["hi"]) else "-"
        lo_s = fmt_temp(row["lo"][i], units) if i < len(row["lo"]) else "-"
        d = datetime.strptime(dates[i], "%Y-%m-%d")
        day_str = ("Today" if i == 0 else "Tomorrow" if i == 1
                   else (d.strftime("%-a %-d %b") if os.name != "nt" else d.strftime("%a %#d %b")))
        bg = "#f8fafc" if i % 2 == 0 else "#ffffff"
        return html.Div(
            style={"display": "grid", "gridTemplateColumns": "80px 1fr 50px 50px 44px",
                   "gap": "4px", "alignItems": "center", "padding": "6px 4px",
                   "background": bg, "borderRadius": "4px", "fontSize": "12px"},
            children=[
                html.Span(day_str, style={"color": "#475569", "fontWeight": 500}),
                html.Span(f"{wx_icon} {wx_label}", style={"color": "#64748b", "overflow": "hidden",
                                                           "textOverflow": "ellipsis", "whiteSpace": "nowrap"}),
                html.Span(hi_s, style={"color": "#dc2626", "textAlign": "right", "fontWeight": 600}),
                html.Span(lo_s, style={"color": "#2563eb", "textAlign": "right"}),
                html.Span(precip_str, style={"color": "#0ea5e9", "textAlign": "right", "fontSize": "10px"}),
            ],
        )

    header_row = html.Div(
        style={"display": "grid", "gridTemplateColumns": "80px 1fr 50px 50px 44px",
               "gap": "4px", "padding": "4px 4px 6px", "borderBottom": "1px solid #e2e8f0",
               "fontSize": "10px", "color": "#94a3b8", "textTransform": "uppercase"},
        children=[html.Span("Date"), html.Span("Conditions"),
                  html.Span("Hi", style={"textAlign": "right"}),
                  html.Span("Lo", style={"textAlign": "right"}),
                  html.Span("Precip", style={"textAlign": "right"})],
    )

    hi_today = fmt_temp(row["hi"][0], units)
    lo_today = fmt_temp(row["lo"][0], units)
    wmo0 = row["wmo"][0] if row["wmo"] else 0
    _, wx_icon0 = WMO_LABELS.get(wmo0, ("", ""))

    return [
        html.Div([
            html.H3(row["label"], style={"margin": "0 0 2px 0", "fontSize": "16px", "color": "#0f172a"}),
            html.Div(f"{row['lat']:.1f}N  {abs(row['lon']):.1f}W",
                     style={"color": "#94a3b8", "fontSize": "11px"}),
        ]),
        html.Div(style={"display": "flex", "gap": "8px"}, children=[
            card("Today High", hi_today, "", "#dc2626"),
            card("Today Low",  lo_today, "", "#2563eb"),
            card("", wx_icon0, "", "#0f172a"),
        ]),
        dcc.Graph(figure=fig, config={"displayModeBar": False}),
        header_row,
        html.Div([wx_row(i) for i in range(len(dates))]),
    ]


@callback(
    Output("us-map", "figure", allow_duplicate=True),
    Input("refresh-btn", "n_clicks"),
    State("day-slider", "value"),
    State("temp-type", "value"),
    State("units", "value"),
    State("model-select", "value"),
    prevent_initial_call=True,
)
def refresh(n, day_idx, temp_type, units, model_key):
    FORECAST_CACHE[model_key] = fetch_forecasts(model_key, force=True)
    return update_map(day_idx, temp_type, units, model_key)


# ---------------------------------------------------------------------------
# Grid Load tab callbacks
# ---------------------------------------------------------------------------

@callback(
    Output("load-forecast-store", "data"),
    Input("main-tabs", "value"),
    Input("refresh-btn", "n_clicks"),
    prevent_initial_call=False,
)
def load_forecast_data(active_tab, n_clicks):
    """Fetch load forecast when load tab is activated or refresh is clicked."""
    if active_tab != "load":
        return dash.no_update
    force = n_clicks is not None and n_clicks > 0
    data = fetch_pjm_load_forecast(force=force)
    return data


@callback(
    Output("load-cards", "children"),
    Output("load-forecast-chart", "figure"),
    Output("load-temp-chart", "figure"),
    Input("load-forecast-store", "data"),
)
def render_load_tab(data):
    empty_fig = go.Figure()
    empty_fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        xaxis=dict(visible=False), yaxis=dict(visible=False),
        margin=dict(l=0, r=0, t=0, b=0),
        annotations=[dict(text="Loading forecast data...", showarrow=False,
                          font=dict(color="#94a3b8", size=13),
                          x=0.5, y=0.5, xref="paper", yref="paper")],
    )

    if not data:
        return [], empty_fig, empty_fig

    dates     = [d["date"] for d in data]
    means     = [d["mean_load_mw"] for d in data]
    lows      = [d["low_load_mw"] for d in data]
    highs     = [d["high_load_mw"] for d in data]
    temps_f   = [d["avg_temp_f"] for d in data]
    hdds      = [d["hdd"] for d in data]
    cdds      = [d["cdd"] for d in data]

    day_labels = [
        datetime.strptime(d, "%Y-%m-%d").strftime("%#d %b") if os.name == "nt"
        else datetime.strptime(d, "%Y-%m-%d").strftime("%-d %b")
        for d in dates
    ]

    today_load  = means[0] if means else 0
    peak_load   = max(means) if means else 0
    peak_idx    = means.index(peak_load) if means else 0
    peak_date   = day_labels[peak_idx] if day_labels else "-"
    avg_load    = sum(means) / len(means) if means else 0
    total_hdd   = sum(hdds)
    total_cdd   = sum(cdds)

    # Severity color for today
    def load_color(load, avg):
        ratio = load / avg if avg else 1
        if ratio > 1.20:
            return "#ef4444"
        elif ratio > 1.08:
            return "#f97316"
        elif ratio < 0.92:
            return "#3b82f6"
        return "#22c55e"

    cards = [
        card("Today (MW)", f"{today_load:,.0f}", "GFS-based",
             load_color(today_load, avg_load)),
        card(f"Peak (MW)", f"{peak_load:,.0f}", f"on {peak_date}",
             load_color(peak_load, avg_load)),
        card("15-Day Avg (MW)", f"{avg_load:,.0f}", "baseline", "#475569"),
        card("HDD (15-day)", f"{total_hdd:.0f}", "heating demand", "#2563eb"),
        card("CDD (15-day)", f"{total_cdd:.0f}", "cooling demand", "#dc2626"),
    ]

    # Load forecast chart
    load_fig = go.Figure()

    # Uncertainty ribbon
    load_fig.add_trace(go.Scatter(
        x=day_labels + day_labels[::-1],
        y=highs + lows[::-1],
        fill="toself",
        fillcolor="rgba(251,146,60,0.15)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False,
        hoverinfo="skip",
    ))

    # Main line — color-coded by severity
    point_colors = [load_color(m, avg_load) for m in means]
    load_fig.add_trace(go.Scatter(
        x=day_labels, y=means,
        mode="lines+markers",
        name="Forecast Load",
        line=dict(color="#f97316", width=2.5),
        marker=dict(size=7, color=point_colors,
                    line=dict(color="#ffffff", width=1.5)),
        hovertemplate="<b>%{x}</b><br>Load: %{y:,.0f} MW<extra></extra>",
    ))

    # Average reference line
    load_fig.add_hline(
        y=avg_load, line_dash="dot", line_color="#94a3b8", line_width=1.5,
        annotation_text=f"15-day avg: {avg_load:,.0f} MW",
        annotation_font_size=10, annotation_font_color="#94a3b8",
        annotation_position="top left",
    )

    load_fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.25, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=60, r=20, t=10, b=60),
        xaxis=dict(gridcolor="#f8fafc", tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", tickformat=",", ticksuffix=" MW",
                   title=dict(text="Load (MW)", font=dict(size=11, color="#64748b")),
                   linecolor="#e2e8f0"),
        height=340,
    )

    # Temperature driver chart
    temp_fig = go.Figure()
    temp_fig.add_trace(go.Bar(
        x=day_labels, y=hdds,
        name="HDD",
        marker_color="#3b82f6",
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>HDD: %{y:.1f}<extra></extra>",
    ))
    temp_fig.add_trace(go.Bar(
        x=day_labels, y=cdds,
        name="CDD",
        marker_color="#ef4444",
        opacity=0.7,
        hovertemplate="<b>%{x}</b><br>CDD: %{y:.1f}<extra></extra>",
    ))
    temp_fig.add_trace(go.Scatter(
        x=day_labels, y=temps_f,
        mode="lines+markers",
        name="Avg Temp (F)",
        yaxis="y2",
        line=dict(color="#475569", width=1.5, dash="dot"),
        marker=dict(size=5),
        hovertemplate="<b>%{x}</b><br>Temp: %{y:.1f}F<extra></extra>",
    ))

    temp_fig.update_layout(
        paper_bgcolor="#ffffff", plot_bgcolor="#ffffff",
        barmode="stack",
        font=dict(color="#334155", size=11),
        legend=dict(orientation="h", y=-0.35, font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
        margin=dict(l=50, r=50, t=10, b=70),
        xaxis=dict(gridcolor="#f8fafc", tickangle=-30, tickfont=dict(size=10), linecolor="#e2e8f0"),
        yaxis=dict(gridcolor="#f1f5f9", title=dict(text="HDD / CDD", font=dict(size=11, color="#64748b")),
                   linecolor="#e2e8f0"),
        yaxis2=dict(
            title=dict(text="Avg Temp (F)", font=dict(size=11, color="#475569")),
            overlaying="y", side="right", showgrid=False,
        ),
        height=200,
    )

    return cards, load_fig, temp_fig


# ---------------------------------------------------------------------------
# Run
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    port = int(os.environ.get("PORT", 8050))
    print(f"\n  Dashboard at http://127.0.0.1:{port}\n")
    app.run(debug=False, port=port)
