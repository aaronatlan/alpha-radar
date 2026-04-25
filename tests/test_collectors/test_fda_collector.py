"""Tests du `FDACollector` — mocks openFDA drug/drugsfda."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from collectors.fda_collector import (
    FDACollector,
    _biotech_sponsors,
    _parse_yyyymmdd,
)


def _fake_response(status: int = 200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    return r


def _application(
    *,
    sponsor: str = "MODERNATX",
    application_number: str = "BLA125752",
    submissions: list[dict] | None = None,
    products: list[dict] | None = None,
) -> dict:
    if submissions is None:
        submissions = [{
            "submission_status": "AP",
            "submission_status_date": "20240131",
            "submission_number": "1",
            "submission_type": "ORIG",
            "submission_class_code": "BLA",
        }]
    if products is None:
        products = [{
            "brand_name": "SPIKEVAX",
            "active_ingredients": [
                {"name": "ELASOMERAN", "strength": "100MG/0.5ML"},
            ],
            "dosage_form": "INJECTION",
        }]
    return {
        "sponsor_name": sponsor,
        "application_number": application_number,
        "submissions": submissions,
        "products": products,
    }


# ---------------------------------------------------------------- helpers


def test_parse_yyyymmdd_full():
    assert _parse_yyyymmdd("20240131") == datetime(2024, 1, 31)


def test_parse_yyyymmdd_invalid():
    assert _parse_yyyymmdd("2024-01-31") is None
    assert _parse_yyyymmdd("") is None
    assert _parse_yyyymmdd(None) is None
    assert _parse_yyyymmdd("99999999") is None


def test_biotech_sponsors_uses_overrides():
    """MRNA est référencé sous MODERNATX dans openFDA — override appliqué."""
    s = _biotech_sponsors()
    assert s.get("MRNA") == "MODERNATX"


def test_biotech_sponsors_includes_watchlist_only():
    s = _biotech_sponsors()
    assert "MRNA" in s
    assert "CRSP" in s
    assert "NVDA" not in s


# ------------------------------------------------------------ collect


def test_collect_explodes_submissions_per_application():
    """Une application avec 2 submissions AP → 2 items distincts."""
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    app = _application(submissions=[
        {"submission_status": "AP", "submission_status_date": "20240131",
         "submission_number": "1", "submission_type": "ORIG"},
        {"submission_status": "AP", "submission_status_date": "20240601",
         "submission_number": "2", "submission_type": "SUPPL"},
    ])
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(json_data={"results": [app]})):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert len(out) == 2
    submission_numbers = {item["submission"]["submission_number"] for item in out}
    assert submission_numbers == {"1", "2"}


def test_collect_skips_non_approved_submissions():
    """Les submissions sans status=AP sont filtrées."""
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    app = _application(submissions=[
        {"submission_status": "AP", "submission_status_date": "20240131",
         "submission_number": "1"},
        {"submission_status": "TA", "submission_status_date": "20240301",
         "submission_number": "2"},   # tentative approval — pas AP
    ])
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(json_data={"results": [app]})):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert len(out) == 1


def test_collect_handles_404_as_empty():
    """openFDA renvoie 404 quand 0 résultat — comportement attendu."""
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(status=404)):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert out == []


def test_collect_handles_http_500():
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(status=500)):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert out == []


def test_collect_passes_yyyymmdd_dates_in_search():
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(json_data={"results": []})) as m:
        c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    params = m.call_args.kwargs["params"]
    assert "20240101" in params["search"]
    assert "20241231" in params["search"]
    assert 'sponsor_name:"MODERNATX"' in params["search"]


def test_collect_continues_on_request_exception():
    import requests as requests_mod
    c = FDACollector(
        sponsors={"MRNA": "MODERNATX", "CRSP": "CRISPR THERAPEUTICS"},
    )
    c.request_delay = 0
    side_effects = [
        requests_mod.ConnectTimeout("nope"),
        _fake_response(json_data={"results": [_application()]}),
    ]
    with patch("collectors.fda_collector.requests.get",
               side_effect=side_effects):
        out = c.collect(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert len(out) == 1


# ------------------------------------------------------------ normalize


def test_normalize_extracts_core_fields():
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    raw = {
        "ticker": "MRNA",
        "sponsor_query": "MODERNATX",
        "application": _application(),
        "submission": {
            "submission_status": "AP",
            "submission_status_date": "20240131",
            "submission_number": "1",
            "submission_type": "ORIG",
            "submission_class_code": "BLA",
        },
    }
    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "fda_approval"
    assert item["entity_id"] == "BLA125752-1"
    assert item["content_at"] == datetime(2024, 1, 31)
    p = item["payload"]
    assert p["ticker"] == "MRNA"
    assert p["application_number"] == "BLA125752"
    assert p["submission_number"] == "1"
    assert p["submission_status"] == "AP"
    assert p["brand_name"] == "SPIKEVAX"
    assert p["active_ingredients"] == ["ELASOMERAN"]


def test_normalize_returns_none_without_app_or_sub_number():
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    raw = {"application": {"sponsor_name": "X"}, "submission": {}}
    assert c.normalize(raw) is None


def test_normalize_returns_none_with_invalid_date():
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    raw = {
        "application": _application(),
        "submission": {
            "submission_status": "AP",
            "submission_status_date": "bad-date",
            "submission_number": "1",
        },
    }
    assert c.normalize(raw) is None


def test_normalize_handles_missing_products():
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    app = _application(products=[])
    raw = {
        "ticker": "MRNA",
        "application": app,
        "submission": app["submissions"][0],
    }
    item = c.normalize(raw)
    assert item is not None
    p = item["payload"]
    assert p["brand_name"] is None
    assert p["active_ingredients"] == []
    assert p["n_products"] == 0


# --------------------------------------------------------------- run (e2e)


def test_run_inserts_approvals(tmp_db):
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(json_data={"results": [_application()]})):
        n = c.run(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert n == 1


def test_run_idempotent(tmp_db):
    c = FDACollector(sponsors={"MRNA": "MODERNATX"})
    c.request_delay = 0
    with patch("collectors.fda_collector.requests.get",
               return_value=_fake_response(json_data={"results": [_application()]})):
        n1 = c.run(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
        n2 = c.run(
            since=datetime(2024, 1, 1), until=datetime(2024, 12, 31),
        )
    assert n1 == 1
    assert n2 == 0
