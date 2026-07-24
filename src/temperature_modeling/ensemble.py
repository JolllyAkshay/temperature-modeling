"""
Multi-agent forecasting ensemble: XGBoost load model + Claude teleconnection reasoning.

The XGBoost model captures weather-to-load relationships well within its training
distribution, but it has no awareness of large-scale atmospheric patterns (MJO,
NAO, AO, PNA) that can shift temperature regimes over a 5-15 day horizon.

This module adds a Claude reasoning layer that:
  1. Fetches current NAO / AO / PNA / MJO indices from public NOAA/BOM endpoints.
  2. Passes them alongside the XGBoost forecast to Claude.
  3. Claude reasons about known teleconnection → temperature → load relationships.
  4. Returns per-day adjustment factors and a human-readable explanation.

The adjustments are applied as percentage shifts on the XGBoost mean forecast.
Low-confidence adjustments (≤ 1%) are suppressed to avoid noise.

Usage:
    from temperature_modeling.ensemble import get_ensemble_forecast
    adjusted, reasoning = get_ensemble_forecast(iso, xgb_forecasts, session)
"""

import json
import logging
from datetime import date, timedelta
from typing import Optional

import requests

from ._llm import complete, is_available
from .models import LoadForecast

log = logging.getLogger(__name__)

# Days of teleconnection history to include as context (recent trend matters)
_TELECON_LOOKBACK_DAYS = 14

_ISO_GEOGRAPHY = {
    "pjm":   "northeastern and mid-Atlantic United States (Ohio, Pennsylvania, Virginia, Illinois)",
    "caiso": "California (Pacific coast, influenced by Pacific jet stream and ENSO)",
    "ercot": "Texas (strongly influenced by Gulf moisture, subtropical ridging, and PNA pattern)",
    "miso":  "Midcontinent United States (Minnesota to Louisiana; strongly influenced by AO and PNA)",
}

_ENSEMBLE_SYSTEM = """You are a senior seasonal meteorologist and energy demand analyst.
You will receive an XGBoost-based 15-day electricity load forecast and recent atmospheric teleconnection indices
(NAO, AO, PNA, MJO). Your job is to reason about whether the XGBoost forecast is likely to be biased
based on the current large-scale atmospheric pattern, and output calibrated adjustment factors.

Key teleconnection relationships for US load:
- AO (Arctic Oscillation): strongly negative AO (< -1.5) → weakened polar vortex → cold-air outbreaks
  in mid-latitudes → heating load increase in PJM/MISO (especially days 5-12).
- NAO (North Atlantic Oscillation): negative NAO → blocking high → cold/snowy eastern US winters;
  positive NAO → mild wet pattern → reduced heating load in PJM.
- PNA (Pacific-North American pattern): positive PNA → ridge over western US / trough over eastern US
  → cold MISO/ERCOT; negative PNA → warm ridge over east.
- MJO (Madden-Julian Oscillation): active MJO phases 8/1 → enhanced cold-air intrusion into eastern US
  in winter (7-14 day lag); phases 4/5 → warm ridge over central US (warming ERCOT/MISO).
  MJO effects are strongest in boreal winter (November–March) and weakest in summer.

Output ONLY a JSON object with this exact schema (no prose, no markdown fences):
{
  "reasoning": "<2-4 sentence explanation of the dominant teleconnection signal and its expected impact>",
  "headline": "<one-sentence summary for the dashboard>",
  "confidence": "low|medium|high",
  "adjustments": [
    {"date": "YYYY-MM-DD", "adj_pct": <float, positive=increase, negative=decrease>, "note": "<brief reason>"},
    ...
  ]
}

If the current teleconnection state is neutral / mixed / in summer months with weak MJO, output
small or zero adjustments and set confidence to "low". Only output meaningful adjustments (adj_pct
outside the range -3% to +3%) when there is a clear, strong, well-established teleconnection signal.
"""


def _fetch_teleconnections(session: requests.Session) -> dict:
    """
    Fetch recent NAO, AO, PNA, and MJO indices.
    Returns dict with lists of (date, value) for each index, plus MJO amplitude/phase.
    Failures are caught and return empty lists rather than raising.
    """
    from ._teleconnections import fetch_nao_daily, fetch_ao_daily, fetch_pna_daily
    from ._mjo import fetch_mjo_daily

    cutoff = date.today() - timedelta(days=_TELECON_LOOKBACK_DAYS)
    result: dict = {"nao": [], "ao": [], "pna": [], "mjo": []}

    for name, fetcher in [
        ("nao", fetch_nao_daily),
        ("ao",  fetch_ao_daily),
        ("pna", fetch_pna_daily),
    ]:
        try:
            data = fetcher(session)
            result[name] = [
                {"date": d.isoformat(), "value": round(v, 3)}
                for d, v in sorted(data.items())
                if d >= cutoff
            ]
        except Exception:
            log.warning("Could not fetch %s index for ensemble", name.upper())

    try:
        mjo_data = fetch_mjo_daily(session)
        result["mjo"] = [
            {"date": d.isoformat(),
             "amplitude": v["mjo_amplitude"],
             "sin_phase":  v["mjo_sin_phase"],
             "cos_phase":  v["mjo_cos_phase"]}
            for d, v in sorted(mjo_data.items())
            if d >= cutoff
        ]
    except Exception:
        log.warning("Could not fetch MJO index for ensemble")

    return result


def _apply_adjustments(
    xgb_forecasts: list[LoadForecast],
    adjustments: list[dict],
) -> list[LoadForecast]:
    """Apply per-day percentage adjustments to the XGBoost LoadForecast list."""
    adj_by_date = {a["date"]: a["adj_pct"] for a in adjustments}

    adjusted = []
    for lf in xgb_forecasts:
        pct = adj_by_date.get(lf.valid_date.isoformat(), 0.0)
        if abs(pct) < 1.0:   # suppress noise below 1%
            adjusted.append(lf)
            continue
        factor = 1.0 + pct / 100.0
        adjusted.append(LoadForecast(
            valid_date=lf.valid_date,
            lead_days=lf.lead_days,
            mean_load_mw=lf.mean_load_mw * factor,
            low_load_mw=lf.low_load_mw   * factor,
            high_load_mw=lf.high_load_mw * factor,
            hdd=lf.hdd,
            cdd=lf.cdd,
            avg_temp_f=lf.avg_temp_f,
        ))
    return adjusted


def get_ensemble_forecast(
    iso: str,
    xgb_forecasts: list[LoadForecast],
    session: Optional[requests.Session] = None,
) -> tuple[list[LoadForecast], dict]:
    """
    Apply teleconnection-informed adjustments to an XGBoost load forecast.

    Parameters
    ----------
    iso:           ISO code ("pjm", "caiso", "ercot", "miso")
    xgb_forecasts: output of LoadCorrectionModel.predict_with_uncertainty()
    session:       optional requests.Session (one will be created if None)

    Returns
    -------
    (adjusted_forecasts, meta)
        adjusted_forecasts: LoadForecast list with teleconnection adjustments applied
        meta: dict with keys "reasoning", "headline", "confidence", "adjustments",
              "teleconnections" (raw index data), "ensemble_available" (bool)
    """
    _no_ensemble = {"ensemble_available": False, "reasoning": "", "headline": "",
                    "confidence": "low", "adjustments": [], "teleconnections": {}}

    if not is_available():
        return xgb_forecasts, _no_ensemble

    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "grid-dashboard-ensemble/1.0"

    # Fetch teleconnection context
    telecon = _fetch_teleconnections(session)
    if not any(telecon.values()):
        log.warning("Ensemble: no teleconnection data available — returning XGBoost-only forecast")
        return xgb_forecasts, {**_no_ensemble, "teleconnections": telecon}

    # Build the prompt
    forecast_rows = [
        {"date": lf.valid_date.isoformat(), "lead_days": lf.lead_days,
         "mean_gw": round(lf.mean_load_mw / 1000, 2),
         "low_gw":  round(lf.low_load_mw  / 1000, 2),
         "high_gw": round(lf.high_load_mw / 1000, 2)}
        for lf in xgb_forecasts
    ]

    season = _current_season()
    prompt = (
        f"ISO: {iso.upper()} — {_ISO_GEOGRAPHY.get(iso, iso.upper())}\n"
        f"Today: {date.today().isoformat()}, Season: {season}\n\n"
        f"XGBoost 15-day load forecast (GW):\n{json.dumps(forecast_rows, indent=2)}\n\n"
        f"Recent teleconnection indices (last {_TELECON_LOOKBACK_DAYS} days):\n"
        f"{json.dumps(telecon, indent=2)}\n\n"
        "Analyse the teleconnection state and output your JSON adjustment object."
    )

    try:
        raw = complete(_ENSEMBLE_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=1200)

        # Strip markdown fences if model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        parsed = json.loads(raw)
        adjustments = parsed.get("adjustments", [])

    except json.JSONDecodeError as exc:
        log.error("Ensemble Claude returned invalid JSON: %s", exc)
        return xgb_forecasts, {**_no_ensemble, "teleconnections": telecon}
    except Exception:
        log.exception("Ensemble Claude call failed")
        return xgb_forecasts, {**_no_ensemble, "teleconnections": telecon}

    adjusted = _apply_adjustments(xgb_forecasts, adjustments)
    n_adjusted = sum(1 for a in adjustments if abs(a.get("adj_pct", 0)) >= 1.0)
    log.info("Ensemble: %s — %d/%d days adjusted, confidence=%s",
             iso.upper(), n_adjusted, len(xgb_forecasts), parsed.get("confidence"))

    meta = {
        "ensemble_available": True,
        "reasoning":          parsed.get("reasoning", ""),
        "headline":           parsed.get("headline", ""),
        "confidence":         parsed.get("confidence", "low"),
        "adjustments":        adjustments,
        "teleconnections":    telecon,
    }
    return adjusted, meta


def _current_season() -> str:
    m = date.today().month
    if m in (12, 1, 2):  return "Winter"
    if m in (3, 4, 5):   return "Spring"
    if m in (6, 7, 8):   return "Summer"
    return "Autumn"
