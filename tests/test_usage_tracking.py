"""
Unit tests for usage_tracking.py.

Each test isolates the module's file-level state (_CACHE_DIR/_LOG_PATH) to
a pytest tmp_path so tests never touch the real usage_log.jsonl. Geo
lookups are exercised against real IPs (Google's public DNS 8.8.8.8, a
stable, always-resolvable address) rather than mocked, matching this
project's convention of testing against real data over mocks — a failed
geo lookup degrading gracefully (no "geo" key) is itself covered by the
malformed/private-IP cases below, which skip the network call entirely.

Run with:  pytest tests/test_usage_tracking.py -v
"""

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from temperature_modeling import usage_tracking as ut


class _FakeRequest:
    def __init__(self, ip="8.8.8.8", ua="pytest-agent/1.0"):
        self.headers = {"User-Agent": ua, "X-Forwarded-For": ip}


@pytest.fixture(autouse=True)
def isolated_log(tmp_path, monkeypatch):
    """Every test gets its own empty log file/dir, and skips the hub-seed pull."""
    monkeypatch.setattr(ut, "_CACHE_DIR", tmp_path)
    monkeypatch.setattr(ut, "_LOG_PATH", tmp_path / "usage_log.jsonl")
    monkeypatch.setattr(ut, "_SEEDED", True)
    yield


def test_visitor_id_deterministic_for_same_ip_and_ua():
    id1 = ut._visitor_id("1.2.3.4", "some-browser/1.0")
    id2 = ut._visitor_id("1.2.3.4", "some-browser/1.0")
    assert id1 == id2


def test_visitor_id_differs_for_different_ip():
    id1 = ut._visitor_id("1.2.3.4", "some-browser/1.0")
    id2 = ut._visitor_id("5.6.7.8", "some-browser/1.0")
    assert id1 != id2


def test_visitor_id_never_contains_raw_ip():
    """The whole point of hashing — the id must not leak the raw IP back out."""
    vid = ut._visitor_id("203.0.113.42", "some-browser/1.0")
    assert "203.0.113.42" not in vid


def test_empty_summary_shape_when_no_log_exists():
    summary = ut.get_usage_summary()
    assert summary["total_events"] == 0
    assert summary["unique_visitors"] == 0
    assert summary["by_event_type"] == {}


def test_get_raw_log_empty_string_when_no_log_exists():
    assert ut.get_raw_log() == ""


def test_log_event_never_raises_on_bad_request_object():
    """A malformed/None request object must never break the caller's actual response."""
    ut.log_event("page_view", request=None, path="/")
    ut.log_event("page_view", request=object(), path="/")  # no .headers at all


def test_log_event_and_summary_roundtrip():
    req = _FakeRequest(ip="8.8.8.8")
    ut.log_event("page_view", request=req, path="/my-electricity")
    ut.log_event("zip_lookup", request=req, zip="07029", iso="pjm", found=True)
    ut.log_event("api_call", request=req, path="/v1/forecast/pjm", status_code=200)

    summary = ut.get_usage_summary()
    assert summary["total_events"] == 3
    assert summary["unique_visitors"] == 1  # same fake IP+UA across all three events
    assert summary["by_event_type"] == {"page_view": 1, "zip_lookup": 1, "api_call": 1}
    assert summary["by_path"].get("/my-electricity") == 1
    assert summary["by_path"].get("/v1/forecast/pjm") == 1

    raw = ut.get_raw_log()
    assert raw.count("\n") == 3  # one JSON line per event
    assert "07029" in raw


def test_different_visitors_counted_separately():
    ut.log_event("page_view", request=_FakeRequest(ip="8.8.8.8"), path="/")
    ut.log_event("page_view", request=_FakeRequest(ip="1.1.1.1"), path="/")
    summary = ut.get_usage_summary()
    assert summary["unique_visitors"] == 2


def test_geo_lookup_private_ip_returns_none_no_network_call():
    assert ut._geo_lookup("127.0.0.1") is None
    assert ut._geo_lookup("") is None
    assert ut._geo_lookup(None) is None
