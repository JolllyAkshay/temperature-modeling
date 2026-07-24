"""
Datacenter pipeline agent — Claude-powered, tool-calling agent that keeps
the list of announced/approved datacenter projects current.

Usage (run as a scheduled job, e.g. weekly):
    python -m temperature_modeling.datacenter_agent

The agent:
  1. Receives the existing project list as context.
  2. Calls `search_web` (Brave Search API if BRAVE_SEARCH_API_KEY is set,
     DuckDuckGo JSON API otherwise) to find recent announcements.
  3. Returns structured JSON in the same schema as _DC_PROJECTS in dashboard.py.
  4. Saves to api_cache/datacenter_pipeline.json.

Dashboard reads from that file and falls back to the hardcoded dict if it
doesn't exist or is older than PIPELINE_CACHE_TTL_DAYS days.
"""

import json
import logging
import os
import time
from datetime import date, timedelta
from pathlib import Path
from typing import Optional

import requests

from ._llm import complete, is_available

log = logging.getLogger(__name__)

PIPELINE_CACHE_TTL_DAYS: int = 7   # refresh weekly

_HERE = Path(__file__).parent.parent.parent
_PIPELINE_CACHE = _HERE / "api_cache" / "datacenter_pipeline.json"

# Search backends (priority: Brave > DuckDuckGo JSON)
_BRAVE_KEY  = os.environ.get("BRAVE_SEARCH_API_KEY", "")
_BRAVE_URL  = "https://api.search.brave.com/res/v1/web/search"
_DDG_URL    = "https://api.duckduckgo.com/"

# ---------------------------------------------------------------------------
# Hardcoded baseline — agent merges on top of this
# ---------------------------------------------------------------------------
_BASELINE: dict = {
    "pjm": [
        {"id": "stack_stafford",   "name": "STACK Stafford Technology Campus",
         "operator": "STACK Infrastructure",             "location": "Stafford County, VA",
         "mw": 1800, "status": "Approved"},
        {"id": "pax_carlisle",     "name": "Pennsylvania Digital I (PAX)",
         "operator": "PowerHouse / PA Data Center Partners", "location": "Carlisle, PA",
         "mw": 1350, "status": "Announced"},
        {"id": "edgecore_louisa",  "name": "EdgeCore Louisa County Campus",
         "operator": "EdgeCore Data Centers",             "location": "Louisa County, VA",
         "mw": 1100, "status": "Announced"},
        {"id": "cleanarc_va1",     "name": "CleanArc VA1 Campus",
         "operator": "CleanArc Data Centers",             "location": "Caroline County, VA",
         "mw": 900,  "status": "Under Construction"},
        {"id": "msft_mt_pleasant", "name": "Microsoft Mt. Pleasant",
         "operator": "Microsoft",                         "location": "Mt. Pleasant, WI",
         "mw": 1000, "status": "Under Construction"},
        {"id": "meta_dekalb",      "name": "Meta DeKalb County",
         "operator": "Meta",                              "location": "DeKalb County, GA",
         "mw": 500,  "status": "Approved"},
        {"id": "amazon_nova",      "name": "Amazon AWS Northern Virginia",
         "operator": "Amazon",                            "location": "Northern Virginia",
         "mw": 800,  "status": "Operating / Expanding"},
        {"id": "coreweave_lanc",   "name": "CoreWeave Lancaster",
         "operator": "CoreWeave",                         "location": "Lancaster, PA",
         "mw": 300,  "status": "Announced"},
    ],
    "caiso": [
        {"id": "google_sj",    "name": "Google San Jose Campus",
         "operator": "Google",               "location": "San Jose, CA",
         "mw": 400,  "status": "Announced"},
        {"id": "meta_sac",     "name": "Meta Sacramento Campus",
         "operator": "Meta",                 "location": "Sacramento, CA",
         "mw": 250,  "status": "Operating"},
        {"id": "msft_sv",      "name": "Microsoft Silicon Valley",
         "operator": "Microsoft",            "location": "San Jose, CA",
         "mw": 200,  "status": "Announced"},
        {"id": "amazon_elk",   "name": "Amazon AWS Elk Grove",
         "operator": "Amazon",               "location": "Elk Grove, CA",
         "mw": 150,  "status": "Operating"},
        {"id": "vantage_sd2",  "name": "Vantage SD2 Campus",
         "operator": "Vantage Data Centers", "location": "San Diego, CA",
         "mw": 120,  "status": "Approved"},
        {"id": "qts_richmond", "name": "QTS Richmond Campus",
         "operator": "QTS / Blackstone",     "location": "Richmond, CA",
         "mw": 180,  "status": "Under Construction"},
    ],
    "ercot": [
        {"id": "msft_abilene",   "name": "Microsoft Abilene AI Campus",
         "operator": "Microsoft",         "location": "Abilene, TX",
         "mw": 800,  "status": "Under Construction"},
        {"id": "meta_fw",        "name": "Meta Fort Worth Campus",
         "operator": "Meta",               "location": "Fort Worth, TX",
         "mw": 700,  "status": "Approved"},
        {"id": "tiktok_temple",  "name": "TikTok Temple Data Center",
         "operator": "ByteDance / TikTok", "location": "Temple, TX",
         "mw": 1200, "status": "Announced"},
        {"id": "google_rich",    "name": "Google Richardson Campus",
         "operator": "Google",             "location": "Richardson, TX",
         "mw": 300,  "status": "Announced"},
        {"id": "qts_irving",     "name": "QTS Irving Campus",
         "operator": "QTS / Blackstone",   "location": "Irving, TX",
         "mw": 500,  "status": "Operating / Expanding"},
        {"id": "amazon_sa",      "name": "Amazon AWS San Antonio",
         "operator": "Amazon",             "location": "San Antonio, TX",
         "mw": 400,  "status": "Operating / Expanding"},
    ],
    "miso": [
        {"id": "msft_racine",    "name": "Microsoft Racine County Campus",
         "operator": "Microsoft",      "location": "Racine County, WI",
         "mw": 500,  "status": "Under Construction"},
        {"id": "google_cb",      "name": "Google Council Bluffs Campus",
         "operator": "Google",          "location": "Council Bluffs, IA",
         "mw": 600,  "status": "Operating / Expanding"},
        {"id": "amazon_dmi",     "name": "Amazon AWS Des Moines",
         "operator": "Amazon",          "location": "Des Moines, IA",
         "mw": 250,  "status": "Operating"},
        {"id": "msft_chicago",   "name": "Microsoft Chicago Campus",
         "operator": "Microsoft",       "location": "Chicago, IL",
         "mw": 300,  "status": "Announced"},
        {"id": "meta_dekalb_il", "name": "Meta DeKalb Illinois Campus",
         "operator": "Meta",            "location": "DeKalb, IL",
         "mw": 800,  "status": "Approved"},
        {"id": "switch_mpls",    "name": "Switch Minneapolis Campus",
         "operator": "Switch",          "location": "Minneapolis, MN",
         "mw": 150,  "status": "Operating / Expanding"},
    ],
}


# ---------------------------------------------------------------------------
# Search tool implementations
# ---------------------------------------------------------------------------

def _search_brave(query: str, session: requests.Session, n: int = 5) -> list[dict]:
    """Call Brave Search API. Returns list of {title, url, description}."""
    try:
        r = session.get(
            _BRAVE_URL,
            headers={"Accept": "application/json", "X-Subscription-Token": _BRAVE_KEY},
            params={"q": query, "count": n, "text_decorations": False},
            timeout=15,
        )
        r.raise_for_status()
        results = r.json().get("web", {}).get("results", [])
        return [{"title": x.get("title", ""), "url": x.get("url", ""),
                 "description": x.get("description", "")} for x in results]
    except Exception as exc:
        log.warning("Brave search failed for %r: %s", query, exc)
        return []


def _search_ddg(query: str, session: requests.Session) -> list[dict]:
    """DuckDuckGo Instant Answer JSON API — no key required, limited results."""
    try:
        r = session.get(
            _DDG_URL,
            params={"q": query, "format": "json", "no_redirect": "1", "no_html": "1"},
            timeout=15,
        )
        r.raise_for_status()
        data = r.json()
        results = []
        for item in data.get("RelatedTopics", [])[:5]:
            text = item.get("Text", "")
            url  = item.get("FirstURL", "")
            if text:
                results.append({"title": text[:80], "url": url, "description": text})
        return results
    except Exception as exc:
        log.warning("DDG search failed for %r: %s", query, exc)
        return []


def _search_web(query: str, session: requests.Session) -> list[dict]:
    if _BRAVE_KEY:
        results = _search_brave(query, session)
        if results:
            return results
    return _search_ddg(query, session)


# ---------------------------------------------------------------------------
# Agent
# ---------------------------------------------------------------------------

_AGENT_SYSTEM = """You are an energy infrastructure analyst. Your job is to maintain an accurate, up-to-date list
of large datacenter projects (>= 100 MW) connected to or planned for US electricity grids.

You will be given an existing project list (baseline) and web search results. Your task is to:
1. Update statuses of existing projects where search results show changes.
2. Add NEW projects not in the baseline that are >= 100 MW and have credible sourcing.
3. Remove projects only if there is strong evidence they were cancelled.

Output ONLY a JSON object in exactly this schema (no prose, no markdown):
{
  "pjm":   [{"id": "<slug>", "name": "...", "operator": "...", "location": "...", "mw": <int>, "status": "..."}, ...],
  "caiso": [...],
  "ercot": [...],
  "miso":  [...]
}

Valid status values: "Operating", "Operating / Expanding", "Under Construction", "Approved", "Announced", "Cancelled"
The id field must be a lowercase kebab-case slug unique within the full list.
"""

_SEARCH_QUERIES = [
    "datacenter interconnection request PJM 2025 2026 MW hyperscaler",
    "new data center announced approved Texas ERCOT 2025 gigawatt",
    "California datacenter power demand CAISO interconnection queue 2025",
    "MISO Midwest datacenter announced construction 2025 2026",
    "Microsoft Meta Google Amazon datacenter US grid connection 2025 2026",
]


def run_datacenter_agent(session: Optional[requests.Session] = None) -> dict:
    """
    Run the Claude-powered agent to refresh the datacenter project list.

    Returns a dict in the _DC_PROJECTS schema. On failure returns the baseline.
    Saves result to _PIPELINE_CACHE for the dashboard to load.
    """
    if not is_available():
        log.warning("No LLM provider configured — returning baseline datacenter list")
        return _BASELINE

    if session is None:
        session = requests.Session()
        session.headers["User-Agent"] = "grid-dashboard-dc-agent/1.0"

    # Gather search results across all queries
    log.info("Datacenter agent: running %d web searches...", len(_SEARCH_QUERIES))
    search_snippets = []
    for q in _SEARCH_QUERIES:
        results = _search_web(q, session)
        for r in results:
            search_snippets.append(f"[{r['title']}] {r['description']} — {r['url']}")
        time.sleep(0.5)   # avoid rate limits

    search_block = "\n".join(search_snippets) if search_snippets else "(no search results available)"

    prompt = (
        f"Today's date: {date.today().isoformat()}\n\n"
        f"EXISTING BASELINE PROJECTS:\n{json.dumps(_BASELINE, indent=2)}\n\n"
        f"RECENT WEB SEARCH RESULTS:\n{search_block}\n\n"
        "Update the project list based on the search results. "
        "Return only the JSON object — no prose, no markdown fences."
    )

    log.info("Datacenter agent: calling LLM to merge and update project list...")
    try:
        raw = complete(_AGENT_SYSTEM, [{"role": "user", "content": prompt}], max_tokens=4096)

        # Strip markdown fences if model adds them despite instructions
        if raw.startswith("```"):
            raw = raw.split("```")[1]
            if raw.startswith("json"):
                raw = raw[4:]
        raw = raw.strip()

        updated = json.loads(raw)
        # Validate schema
        for iso in ("pjm", "caiso", "ercot", "miso"):
            if iso not in updated:
                raise ValueError(f"Missing ISO key: {iso}")
            for p in updated[iso]:
                assert "id" in p and "name" in p and "mw" in p and "status" in p

    except (json.JSONDecodeError, ValueError, AssertionError) as exc:
        log.error("Datacenter agent returned invalid JSON — using baseline: %s", exc)
        updated = _BASELINE
    except Exception:
        log.exception("Datacenter agent Claude call failed — using baseline")
        updated = _BASELINE

    # Persist to cache
    try:
        _PIPELINE_CACHE.parent.mkdir(parents=True, exist_ok=True)
        _PIPELINE_CACHE.write_text(json.dumps({
            "updated": date.today().isoformat(),
            "projects": updated,
        }))
        total = sum(len(v) for v in updated.values())
        log.info("Datacenter pipeline saved: %d projects across 4 ISOs", total)
    except OSError:
        log.warning("Could not write datacenter pipeline cache")

    return updated


# ---------------------------------------------------------------------------
# Dashboard loader (called at dashboard startup)
# ---------------------------------------------------------------------------

def load_datacenter_projects() -> dict:
    """
    Load datacenter projects from agent-maintained cache, or return baseline.
    Auto-triggers the agent if the cache is older than PIPELINE_CACHE_TTL_DAYS.
    """
    if _PIPELINE_CACHE.exists():
        try:
            age_days = (time.time() - _PIPELINE_CACHE.stat().st_mtime) / 86400
            data = json.loads(_PIPELINE_CACHE.read_text())
            projects = data.get("projects", {})
            if projects and age_days < PIPELINE_CACHE_TTL_DAYS:
                log.info("Loaded datacenter pipeline from cache (%s, %.1f days old)",
                         data.get("updated"), age_days)
                return projects
        except (json.JSONDecodeError, OSError):
            pass

    # Cache stale or missing — trigger refresh in background
    import threading
    log.info("Datacenter pipeline cache stale — refreshing in background")
    threading.Thread(target=run_datacenter_agent, daemon=True).start()

    # Return baseline while agent runs
    return _BASELINE


# ---------------------------------------------------------------------------
# CLI entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO,
                        format="%(asctime)s [%(levelname)s] %(message)s")
    result = run_datacenter_agent()
    print(json.dumps(result, indent=2))
