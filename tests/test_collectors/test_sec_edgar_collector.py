"""Tests du `SECEdgarCollector` — mocks HTTP + filtrage par forme/fenêtre."""
from __future__ import annotations

from datetime import datetime, timedelta
from unittest.mock import MagicMock, patch

import pytest

from collectors.sec_edgar_collector import SECEdgarCollector


def _fake_response(status: int = 200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    return r


def _submission_payload(
    accessions: list[str], forms: list[str], dates: list[str]
) -> dict:
    return {
        "name": "Fake Corp",
        "filings": {
            "recent": {
                "accessionNumber": accessions,
                "form": forms,
                "filingDate": dates,
                "primaryDocument": [f"{a}.htm" for a in accessions],
            }
        },
    }


def test_collect_filters_by_form_and_window():
    c = SECEdgarCollector(ticker_to_cik={"TEST": "12345"})
    c.request_delay = 0
    today = datetime(2026, 4, 15)
    data = _submission_payload(
        accessions=["a1", "a2", "a3", "a4"],
        forms=["10-K", "CORRESP", "8-K", "10-Q"],   # CORRESP ignoré
        dates=[
            (today - timedelta(days=5)).strftime("%Y-%m-%d"),
            (today - timedelta(days=6)).strftime("%Y-%m-%d"),
            (today - timedelta(days=200)).strftime("%Y-%m-%d"),  # hors fenêtre
            (today - timedelta(days=10)).strftime("%Y-%m-%d"),
        ],
    )
    with patch("collectors.sec_edgar_collector.requests.get",
               return_value=_fake_response(json_data=data)):
        out = c.collect(since=today - timedelta(days=30), until=today)

    accs = {i["accession"] for i in out}
    assert accs == {"a1", "a4"}
    for item in out:
        assert item["ticker"] == "TEST"


def test_collect_skips_on_http_error():
    c = SECEdgarCollector(ticker_to_cik={"X": "1"})
    c.request_delay = 0
    with patch("collectors.sec_edgar_collector.requests.get",
               return_value=_fake_response(status=500)):
        out = c.collect(since=datetime(2026, 1, 1), until=datetime(2026, 4, 15))
    assert out == []


def test_normalize_maps_fields():
    c = SECEdgarCollector(ticker_to_cik={"X": "1"})
    raw = {
        "ticker": "NVDA", "cik": "1045810",
        "accession": "0001045810-26-000001",
        "form": "10-Q", "filing_date": "2026-03-01",
        "primary_doc": "doc.htm", "company_name": "NVIDIA",
    }
    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "sec_filing"
    assert item["entity_id"] == "0001045810-26-000001"
    assert item["content_at"] == datetime(2026, 3, 1)
    assert item["payload"]["form"] == "10-Q"


def test_normalize_invalid_date_returns_none():
    c = SECEdgarCollector(ticker_to_cik={"X": "1"})
    raw = {"accession": "a", "filing_date": "not-a-date"}
    assert c.normalize(raw) is None


def test_normalize_missing_fields_returns_none():
    c = SECEdgarCollector(ticker_to_cik={"X": "1"})
    assert c.normalize({}) is None
    assert c.normalize({"accession": "a"}) is None
