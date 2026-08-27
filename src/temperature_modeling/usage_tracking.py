"""
Lightweight, privacy-conscious usage logging — page views, zip lookups, and
API calls, appended to api_cache/usage_log.jsonl. Seeded from the Space
repo's persisted copy on startup (best-effort, anonymous read) so history
survives a redeploy; nothing pushes back to the Hub from here — see
get_raw_log() / the internal FastAPI route this feeds for that piece.

Visitors are identified by a salted hash of IP + user-agent, never the raw
IP — enough to count unique/repeat visitors without retaining anything
directly identifying. Geography and organization (ISP/org name — often
reveals a corporate visitor, e.g. a utility's own network) come from a
best-effort, cached-per-IP lookup against ip-api.com's free tier; failures
or rate-limiting just mean that one event has no geo data, never a broken
response for the actual visitor.

Public API
----------
log_event(event_type: str, request=None, **fields) -> None   (never raises)
get_usage_summary() -> dict
get_raw_log() -> str
"""

import hashlib
import json
import logging
import os
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

import requests

log = logging.getLogger(__name__)

_CACHE_DIR = Path(__file__).parent.parent.parent / "api_cache"
_LOG_PATH = _CACHE_DIR / "usage_log.jsonl"
_SALT = os.environ.get("USAGE_SALT", "grid-dashboard-usage-v1")

_GEO_URL = "http://ip-api.com/json/{ip}"
_GEO_FIELDS = "status,country,regionName,city,isp,org,query"
_GEO_CACHE: dict = {}  # {ip: geo_dict_or_None} — process-lifetime cache
_GEO_TIMEOUT_S = 2.0

_SEEDED = False


def _seed_from_hub() -> None:
    """One-time, best-effort pull of any prior log already persisted to the
    Space repo, so a redeploy doesn't silently reset history to zero."""
    global _SEEDED
    if _SEEDED:
        return
    _SEEDED = True
    if _LOG_PATH.exists():
        return
    try:
        from huggingface_hub import hf_hub_download  # noqa: PLC0415
        _CACHE_DIR.mkdir(exist_ok=True)
        hf_hub_download(repo_id="JollyAkshay/grid-dashboard", repo_type="space",
                         filename="api_cache/usage_log.jsonl", local_dir=str(_CACHE_DIR.parent))
        log.info("usage_tracking: seeded log from Space repo")
    except Exception as exc:
        log.info("usage_tracking: no prior log to seed (%s)", exc)


def _visitor_id(ip: str, user_agent: str) -> str:
    return hashlib.sha256(f"{_SALT}:{ip}:{user_agent}".encode()).hexdigest()[:16]


def _geo_lookup(ip: str) -> Optional[dict]:
    if not ip or ip in ("127.0.0.1", "localhost", "unknown"):
        return None
    if ip in _GEO_CACHE:
        return _GEO_CACHE[ip]
    geo = None
    try:
        r = requests.get(_GEO_URL.format(ip=ip), params={"fields": _GEO_FIELDS}, timeout=_GEO_TIMEOUT_S)
        r.raise_for_status()
        data = r.json()
        if data.get("status") == "success":
            geo = {"country": data.get("country"), "region": data.get("regionName"),
                   "city": data.get("city"), "isp": data.get("isp"), "org": data.get("org")}
    except Exception:
        pass  # best-effort — a failed geo lookup should never break the actual request
    _GEO_CACHE[ip] = geo
    return geo


def _client_ip_and_ua(request) -> tuple:
    """Accepts either a Flask request (Dash callbacks) or a Starlette/FastAPI
    Request — both expose .headers, only the IP-extraction path differs."""
    if request is None:
        return "unknown", ""
    xff = request.headers.get("X-Forwarded-For", "")
    ip = xff.split(",")[0].strip() if xff else None
    if not ip:
        ip = getattr(getattr(request, "client", None), "host", None) or getattr(request, "remote_addr", None) or "unknown"
    ua = request.headers.get("User-Agent", "")
    return ip, ua


def log_event(event_type: str, request=None, **fields) -> None:
    """
    Append one usage event. Never raises — a logging failure must never
    break the actual page render or API response it's attached to.
    """
    try:
        _seed_from_hub()
        ip, ua = _client_ip_and_ua(request)
        visitor_id = _visitor_id(ip, ua)
        geo = _geo_lookup(ip)
        row = {
            "ts": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "event_type": event_type,
            "visitor_id": visitor_id,
            "user_agent": ua[:200],
            **({"geo": geo} if geo else {}),
            **fields,
        }
        _CACHE_DIR.mkdir(exist_ok=True)
        with open(_LOG_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(row) + "\n")
    except Exception:
        log.exception("usage_tracking: failed to log %s event", event_type)


def get_raw_log() -> str:
    """Raw JSONL content, for the internal sync endpoint to scrape and
    persist to the Space repo — see api.py's /v1/_internal/usage-log route."""
    _seed_from_hub()
    if not _LOG_PATH.exists():
        return ""
    return _LOG_PATH.read_text(encoding="utf-8")


def get_usage_summary() -> dict:
    """
    Aggregate stats over the full local log — total events, unique
    visitors, events by type/path, top countries/organizations, and
    activity in the last 7/30 days. Empty-but-valid shape if no log exists
    yet, never raises.
    """
    _seed_from_hub()
    empty = {
        "total_events": 0, "unique_visitors": 0, "first_event": None, "last_event": None,
        "by_event_type": {}, "by_path": {}, "by_country": {}, "by_org": {},
        "events_last_7d": 0, "events_last_30d": 0,
    }
    if not _LOG_PATH.exists():
        return empty

    rows = []
    try:
        with open(_LOG_PATH, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    rows.append(json.loads(line))
                except Exception:
                    continue
    except Exception:
        log.exception("usage_tracking: failed to read log for summary")
        return empty

    if not rows:
        return empty

    visitors = set()
    by_type: dict = {}
    by_path: dict = {}
    by_country: dict = {}
    by_org: dict = {}
    now = time.time()
    last_7d = last_30d = 0

    for row in rows:
        visitors.add(row.get("visitor_id"))
        et = row.get("event_type", "unknown")
        by_type[et] = by_type.get(et, 0) + 1
        path = row.get("path") or row.get("endpoint")
        if path:
            by_path[path] = by_path.get(path, 0) + 1
        geo = row.get("geo") or {}
        if geo.get("country"):
            by_country[geo["country"]] = by_country.get(geo["country"], 0) + 1
        if geo.get("org"):
            by_org[geo["org"]] = by_org.get(geo["org"], 0) + 1
        try:
            ts = datetime.fromisoformat(row["ts"]).timestamp()
            age_days = (now - ts) / 86400
            if age_days <= 7:
                last_7d += 1
            if age_days <= 30:
                last_30d += 1
        except Exception:
            pass

    return {
        "total_events": len(rows),
        "unique_visitors": len(visitors),
        "first_event": rows[0].get("ts"),
        "last_event": rows[-1].get("ts"),
        "by_event_type": dict(sorted(by_type.items(), key=lambda x: -x[1])),
        "by_path": dict(sorted(by_path.items(), key=lambda x: -x[1])[:20]),
        "by_country": dict(sorted(by_country.items(), key=lambda x: -x[1])[:20]),
        "by_org": dict(sorted(by_org.items(), key=lambda x: -x[1])[:20]),
        "events_last_7d": last_7d,
        "events_last_30d": last_30d,
    }
