"""Tests du `ClinicalTrialsCollector` — mocks HTTP API v2."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from collectors.clinicaltrials_collector import (
    ClinicalTrialsCollector,
    _biotech_sponsors,
    _most_advanced_phase,
    _parse_date,
    _struct_date,
)


def _fake_response(status: int = 200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    return r


def _study(
    *,
    nct_id: str,
    sponsor: str = "Moderna",
    title: str = "A Test Trial",
    phases: list[str] | None = None,
    status: str = "RECRUITING",
    last_update: str | None = "2026-04-15",
    conditions: list[str] | None = None,
    interventions: list[str] | None = None,
    start_date: str | None = "2026-01-01",
    completion_date: str | None = "2027-12-31",
) -> dict:
    return {
        "protocolSection": {
            "identificationModule": {
                "nctId": nct_id,
                "briefTitle": title,
            },
            "statusModule": {
                "overallStatus": status,
                "lastUpdatePostDateStruct": (
                    {"date": last_update} if last_update else {}
                ),
                "startDateStruct": (
                    {"date": start_date} if start_date else {}
                ),
                "completionDateStruct": (
                    {"date": completion_date} if completion_date else {}
                ),
                "primaryCompletionDateStruct": (
                    {"date": completion_date} if completion_date else {}
                ),
            },
            "designModule": {"phases": phases or ["PHASE2"]},
            "sponsorCollaboratorsModule": {
                "leadSponsor": {"name": sponsor},
            },
            "conditionsModule": {
                "conditions": conditions or ["Cancer"],
            },
            "armsInterventionsModule": {
                "interventions": [
                    {"name": iv} for iv in (interventions or ["mRNA-1234"])
                ],
            },
        }
    }


# ----------------------------------------------------------- helpers


def test_most_advanced_phase_picks_latest():
    assert _most_advanced_phase(["PHASE1", "PHASE3"]) == "PHASE3"
    assert _most_advanced_phase(["PHASE2", "PHASE2/PHASE3"]) == "PHASE2/PHASE3"


def test_most_advanced_phase_handles_empty():
    assert _most_advanced_phase([]) is None


def test_most_advanced_phase_handles_unknown():
    """Une phase inconnue ne doit pas crasher mais ne doit pas gagner."""
    assert _most_advanced_phase(["PHASE1", "MYSTERY"]) == "PHASE1"


def test_parse_date_full():
    assert _parse_date("2026-04-15") == datetime(2026, 4, 15)


def test_parse_date_yyyy_mm():
    assert _parse_date("2026-04") == datetime(2026, 4, 1)


def test_parse_date_invalid():
    assert _parse_date("not-a-date") is None
    assert _parse_date(None) is None
    assert _parse_date("") is None


def test_struct_date_extracts():
    assert _struct_date({"date": "2026-04-15", "type": "ACTUAL"}) == "2026-04-15"


def test_struct_date_handles_none():
    assert _struct_date(None) is None
    assert _struct_date({}) is None


def test_biotech_sponsors_includes_watchlist_biotech():
    sponsors = _biotech_sponsors()
    # MRNA, CRSP, NTLA sont biotech dans la watchlist.
    assert "MRNA" in sponsors
    assert "CRSP" in sponsors
    assert "NTLA" in sponsors
    # NVDA n'est pas biotech.
    assert "NVDA" not in sponsors


# ------------------------------------------------------------ collect


def test_collect_calls_api_per_sponsor_and_tags_ticker():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    c.request_delay = 0
    studies = [_study(nct_id="NCT01", sponsor="Moderna")]
    payload = {"studies": studies}
    with patch("collectors.clinicaltrials_collector.requests.get",
               return_value=_fake_response(json_data=payload)) as m:
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert m.call_count == 1
    assert len(out) == 1
    assert out[0]["_ticker"] == "MRNA"
    assert out[0]["_sponsor_query"] == "Moderna"


def test_collect_passes_date_range_in_filter():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    c.request_delay = 0
    with patch("collectors.clinicaltrials_collector.requests.get",
               return_value=_fake_response(json_data={"studies": []})) as m:
        c.collect(since=datetime(2026, 1, 1), until=datetime(2026, 4, 30))
    params = m.call_args.kwargs["params"]
    assert "filter.advanced" in params
    assert "2026-01-01" in params["filter.advanced"]
    assert "2026-04-30" in params["filter.advanced"]


def test_collect_handles_http_error():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    c.request_delay = 0
    with patch("collectors.clinicaltrials_collector.requests.get",
               return_value=_fake_response(status=500)):
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert out == []


def test_collect_handles_invalid_json():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    c.request_delay = 0
    bad = MagicMock()
    bad.status_code = 200
    bad.json.side_effect = ValueError("bad json")
    with patch("collectors.clinicaltrials_collector.requests.get",
               return_value=bad):
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert out == []


def test_collect_continues_on_request_exception():
    """Une RequestException sur un sponsor n'empêche pas les autres."""
    import requests as requests_mod
    c = ClinicalTrialsCollector(
        sponsors={"MRNA": "Moderna", "CRSP": "CRISPR Therapeutics"},
    )
    c.request_delay = 0
    side_effects = [
        requests_mod.ConnectTimeout("nope"),
        _fake_response(json_data={"studies": [_study(nct_id="NCT02")]}),
    ]
    with patch("collectors.clinicaltrials_collector.requests.get",
               side_effect=side_effects):
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert len(out) == 1
    assert out[0]["protocolSection"]["identificationModule"]["nctId"] == "NCT02"


# ---------------------------------------------------------- normalize


def test_normalize_extracts_core_fields():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    raw = _study(
        nct_id="NCT04567890",
        sponsor="Moderna",
        title="Phase 3 mRNA cancer trial",
        phases=["PHASE3"],
        status="ACTIVE_NOT_RECRUITING",
        last_update="2026-04-15",
        conditions=["Lung Cancer", "Solid Tumors"],
        interventions=["mRNA-4157"],
        start_date="2025-09-01",
        completion_date="2028-06-01",
    )
    raw["_ticker"] = "MRNA"
    raw["_sponsor_query"] = "Moderna"

    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "clinical_trial"
    assert item["entity_id"] == "NCT04567890"
    assert item["content_at"] == datetime(2026, 4, 15)
    p = item["payload"]
    assert p["nct_id"] == "NCT04567890"
    assert p["ticker"] == "MRNA"
    assert p["lead_sponsor"] == "Moderna"
    assert p["phase"] == "PHASE3"
    assert p["overall_status"] == "ACTIVE_NOT_RECRUITING"
    assert p["conditions"] == ["Lung Cancer", "Solid Tumors"]
    assert p["interventions"] == ["mRNA-4157"]
    assert p["start_date"] == "2025-09-01"
    assert p["completion_date"] == "2028-06-01"
    assert p["last_update_post_date"] == "2026-04-15"


def test_normalize_returns_none_without_nct_id():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    raw = _study(nct_id="X")
    raw["protocolSection"]["identificationModule"]["nctId"] = None
    assert c.normalize(raw) is None


def test_normalize_returns_none_without_last_update():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    raw = _study(nct_id="NCT01", last_update=None)
    assert c.normalize(raw) is None


def test_normalize_handles_missing_optional_modules():
    """Un payload minimal (juste nctId + lastUpdate) ne doit pas crasher."""
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    raw = {
        "protocolSection": {
            "identificationModule": {"nctId": "NCT99"},
            "statusModule": {
                "lastUpdatePostDateStruct": {"date": "2026-04-01"},
            },
        }
    }
    item = c.normalize(raw)
    assert item is not None
    p = item["payload"]
    assert p["nct_id"] == "NCT99"
    assert p["phase"] is None
    assert p["lead_sponsor"] is None
    assert p["conditions"] == []
    assert p["interventions"] == []


def test_normalize_picks_most_advanced_phase():
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    raw = _study(nct_id="NCT01", phases=["PHASE1", "PHASE2", "PHASE3"])
    item = c.normalize(raw)
    assert item["payload"]["phase"] == "PHASE3"


# ------------------------------------------------------------- run (e2e)


def test_run_inserts_studies(tmp_db):
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    c.request_delay = 0
    studies = [
        _study(nct_id="NCT01", sponsor="Moderna"),
        _study(nct_id="NCT02", sponsor="Moderna"),
    ]
    with patch("collectors.clinicaltrials_collector.requests.get",
               return_value=_fake_response(json_data={"studies": studies})):
        n = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert n == 2


def test_run_idempotent(tmp_db):
    c = ClinicalTrialsCollector(sponsors={"MRNA": "Moderna"})
    c.request_delay = 0
    studies = [_study(nct_id="NCT01", sponsor="Moderna")]
    with patch("collectors.clinicaltrials_collector.requests.get",
               return_value=_fake_response(json_data={"studies": studies})):
        n1 = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
        n2 = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert n1 == 1
    assert n2 == 0  # même hash → INSERT OR IGNORE
