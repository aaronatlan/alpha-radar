"""Tests du `USASpendingCollector` — mocks API spending_by_award."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from collectors.usaspending_collector import (
    USASpendingCollector,
    _defense_sponsors,
    _parse_date,
    _to_float,
)


def _fake_response(status: int = 200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    return r


def _award(
    *,
    award_id: str = "FA8625-21-C-0001",
    recipient: str = "Lockheed Martin Corp",
    amount: float = 250_000_000.0,
    description: str = "F-35 lot procurement",
    action_date: str = "2024-09-15",
    agency: str = "Department of Defense",
) -> dict:
    return {
        "Award ID": award_id,
        "Recipient Name": recipient,
        "Award Amount": amount,
        "Description": description,
        "Action Date": action_date,
        "Awarding Agency": agency,
        "Awarding Sub Agency": "U.S. Air Force",
        "Award Type": "Definitive Contract",
        "Period of Performance Start Date": "2024-10-01",
        "Period of Performance Current End Date": "2026-09-30",
    }


# ---------------------------------------------------------------- helpers


def test_parse_date_iso():
    assert _parse_date("2024-09-15") == datetime(2024, 9, 15)


def test_parse_date_invalid():
    assert _parse_date("9/15/2024") is None
    assert _parse_date(None) is None
    assert _parse_date("") is None


def test_to_float_handles_str_and_number():
    assert _to_float("123.45") == 123.45
    assert _to_float(100) == 100.0
    assert _to_float(None) is None
    assert _to_float("") is None
    assert _to_float("not-a-number") is None


def test_defense_sponsors_includes_space_tickers():
    s = _defense_sponsors()
    # LMT et RKLB sont sector=space dans la watchlist.
    assert "LMT" in s
    assert "RKLB" in s
    # MRNA n'a pas le sector space.
    assert "MRNA" not in s
    # Override de nom appliqué.
    assert s["LMT"] == "Lockheed Martin"
    assert s["RKLB"] == "Rocket Lab"


# ------------------------------------------------------------ collect


def test_collect_calls_api_per_sponsor_and_tags_ticker():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    c.request_delay = 0
    awards = [_award()]
    with patch("collectors.usaspending_collector.requests.post",
               return_value=_fake_response(json_data={"results": awards})) as m:
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert m.call_count == 1
    assert len(out) == 1
    assert out[0]["_ticker"] == "LMT"
    assert out[0]["_sponsor_query"] == "Lockheed Martin"


def test_collect_uses_post_with_json_body():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    c.request_delay = 0
    with patch("collectors.usaspending_collector.requests.post",
               return_value=_fake_response(json_data={"results": []})) as m:
        c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    body = m.call_args.kwargs["json"]
    assert body["filters"]["recipient_search_text"] == ["Lockheed Martin"]
    assert body["filters"]["time_period"][0]["start_date"] == "2024-01-01"
    assert body["filters"]["time_period"][0]["end_date"] == "2024-12-31"
    # Filtre award type = contrats stricts.
    assert set(body["filters"]["award_type_codes"]) >= {"A", "B", "C", "D"}


def test_collect_handles_http_error():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    c.request_delay = 0
    with patch("collectors.usaspending_collector.requests.post",
               return_value=_fake_response(status=500)):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert out == []


def test_collect_handles_invalid_json():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    c.request_delay = 0
    bad = MagicMock()
    bad.status_code = 200
    bad.json.side_effect = ValueError("bad")
    with patch("collectors.usaspending_collector.requests.post",
               return_value=bad):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert out == []


def test_collect_continues_on_request_exception():
    import requests as requests_mod
    c = USASpendingCollector(
        sponsors={"LMT": "Lockheed Martin", "RKLB": "Rocket Lab"},
    )
    c.request_delay = 0
    side_effects = [
        requests_mod.ConnectTimeout("nope"),
        _fake_response(json_data={"results": [_award()]}),
    ]
    with patch("collectors.usaspending_collector.requests.post",
               side_effect=side_effects):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert len(out) == 1


# ------------------------------------------------------------ normalize


def test_normalize_extracts_core_fields():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    raw = _award()
    raw["_ticker"] = "LMT"
    raw["_sponsor_query"] = "Lockheed Martin"

    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "gov_contract"
    assert item["entity_id"] == "FA8625-21-C-0001"
    assert item["content_at"] == datetime(2024, 9, 15)
    p = item["payload"]
    assert p["ticker"] == "LMT"
    assert p["award_id"] == "FA8625-21-C-0001"
    assert p["award_amount"] == 250_000_000.0
    assert p["awarding_agency"] == "Department of Defense"
    assert "F-35" in (p["description"] or "")


def test_normalize_returns_none_without_award_id():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    raw = {"Action Date": "2024-09-15"}
    assert c.normalize(raw) is None


def test_normalize_returns_none_with_invalid_date():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    raw = {"Award ID": "x", "Action Date": "garbage"}
    assert c.normalize(raw) is None


def test_normalize_handles_missing_amount():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    raw = _award(amount=None)
    raw["_ticker"] = "LMT"
    item = c.normalize(raw)
    assert item is not None
    assert item["payload"]["award_amount"] is None


def test_normalize_handles_string_amount():
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    raw = _award()
    raw["Award Amount"] = "250000000"   # string from JSON
    raw["_ticker"] = "LMT"
    item = c.normalize(raw)
    assert item["payload"]["award_amount"] == 250_000_000.0


# --------------------------------------------------------------- run (e2e)


def test_run_inserts_contracts(tmp_db):
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    c.request_delay = 0
    awards = [_award(award_id="A1"), _award(award_id="A2")]
    with patch("collectors.usaspending_collector.requests.post",
               return_value=_fake_response(json_data={"results": awards})):
        n = c.run(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert n == 2


def test_run_idempotent(tmp_db):
    c = USASpendingCollector(sponsors={"LMT": "Lockheed Martin"})
    c.request_delay = 0
    awards = [_award(award_id="A1")]
    with patch("collectors.usaspending_collector.requests.post",
               return_value=_fake_response(json_data={"results": awards})):
        n1 = c.run(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
        n2 = c.run(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert n1 == 1
    assert n2 == 0
