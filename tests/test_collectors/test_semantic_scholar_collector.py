"""Tests du `SemanticScholarCollector` — mocks API graph/v1/paper/search."""
from __future__ import annotations

from datetime import datetime
from unittest.mock import MagicMock, patch

from collectors.semantic_scholar_collector import (
    SemanticScholarCollector,
    _parse_date,
)


def _fake_response(status: int = 200, json_data=None):
    r = MagicMock()
    r.status_code = status
    r.json.return_value = json_data or {}
    return r


def _paper(
    *,
    paper_id: str = "p1",
    title: str = "Attention is all you need 2",
    citation_count: int = 250,
    influential: int = 30,
    pub_date: str | None = "2026-02-01",
    year: int | None = 2026,
    authors: list[str] | None = None,
    venue: str = "NeurIPS",
    doi: str | None = "10.1/x",
    arxiv: str | None = "2602.0001",
) -> dict:
    return {
        "paperId": paper_id,
        "title": title,
        "abstract": "Long abstract " * 100,
        "year": year,
        "publicationDate": pub_date,
        "citationCount": citation_count,
        "influentialCitationCount": influential,
        "referenceCount": 42,
        "venue": venue,
        "authors": [{"name": n} for n in (authors or ["Alice", "Bob"])],
        "externalIds": {"DOI": doi, "ArXiv": arxiv},
    }


# ---------------------------------------------------------------- helpers


def test_parse_date_iso():
    assert _parse_date("2026-02-01") == datetime(2026, 2, 1)


def test_parse_date_invalid():
    assert _parse_date("invalid") is None
    assert _parse_date(None) is None
    assert _parse_date("") is None


# ------------------------------------------------------------ collect


def test_collect_iterates_sectors_and_tags_meta():
    c = SemanticScholarCollector(sector_ids=["ai_ml", "biotech"])
    c.request_delay = 0
    payload = {"data": [_paper(paper_id="p_one")]}
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data=payload)) as m:
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    # Une requête par secteur.
    assert m.call_count == 2
    # Chaque item est tagué avec son sector_id.
    tagged = {item["_sector_id"] for item in out}
    assert tagged == {"ai_ml", "biotech"}


def test_collect_uses_keyword_query_and_recent_filter():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    c.request_delay = 0
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data={"data": []})) as m:
        c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    params = m.call_args.kwargs["params"]
    assert "publicationDateOrYear" in params
    assert params["publicationDateOrYear"].endswith(":")
    # Query = 2 premiers keywords joints
    assert "language model" in params["query"] or "transformer" in params["query"]


def test_collect_handles_429_rate_limit():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    c.request_delay = 0
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(status=429)):
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert out == []


def test_collect_handles_http_error():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    c.request_delay = 0
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(status=500)):
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert out == []


def test_collect_continues_on_request_exception():
    import requests as requests_mod
    c = SemanticScholarCollector(sector_ids=["ai_ml", "biotech"])
    c.request_delay = 0
    side_effects = [
        requests_mod.ConnectTimeout("nope"),
        _fake_response(json_data={"data": [_paper(paper_id="p_b")]}),
    ]
    with patch("collectors.semantic_scholar_collector.requests.get",
               side_effect=side_effects):
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert len(out) == 1


def test_collect_skips_sectors_without_keywords():
    """Robustesse : un secteur sans keywords ne crashe pas."""
    c = SemanticScholarCollector()
    c.request_delay = 0
    # On force un sector sans keywords.
    c._sectors = [
        {"id": "fake", "name": "Fake", "category": "x",
         "keywords": [], "arxiv_categories": []},
    ]
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data={"data": []})) as m:
        out = c.collect(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert out == []
    assert m.call_count == 0


# ------------------------------------------------------------ normalize


def test_normalize_extracts_core_fields():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    raw = _paper(paper_id="abc123", title="LLM Scaling", citation_count=500)
    raw["_sector_id"] = "ai_ml"
    raw["_query"] = "language model"

    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "paper"
    assert item["entity_id"] == "abc123"
    assert item["content_at"] == datetime(2026, 2, 1)
    p = item["payload"]
    assert p["paper_id"] == "abc123"
    assert p["sector_id"] == "ai_ml"
    assert p["citation_count"] == 500
    assert p["title"] == "LLM Scaling"
    assert p["doi"] == "10.1/x"
    assert p["arxiv_id"] == "2602.0001"


def test_normalize_returns_none_without_paper_id():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    raw = {"title": "X", "publicationDate": "2026-02-01"}
    assert c.normalize(raw) is None


def test_normalize_falls_back_to_year_when_pub_date_missing():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    raw = _paper(paper_id="x", pub_date=None, year=2025)
    raw["_sector_id"] = "ai_ml"
    item = c.normalize(raw)
    assert item is not None
    assert item["content_at"] == datetime(2025, 1, 1)


def test_normalize_handles_missing_authors_externalids():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    raw = {
        "paperId": "x",
        "publicationDate": "2026-01-01",
    }
    item = c.normalize(raw)
    assert item is not None
    assert item["payload"]["authors"] == []
    assert item["payload"]["doi"] is None
    assert item["payload"]["arxiv_id"] is None


def test_normalize_truncates_long_abstract():
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    raw = _paper(paper_id="x")
    item = c.normalize(raw)
    assert len(item["payload"]["abstract"]) <= 1000


# --------------------------------------------------------------- run (e2e)


def test_run_inserts_papers(tmp_db):
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    c.request_delay = 0
    papers = [_paper(paper_id="p1"), _paper(paper_id="p2")]
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data={"data": papers})):
        n = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert n == 2


def test_run_creates_new_row_on_citation_count_change(tmp_db):
    """Snapshot : un changement de citation_count = nouvelle ligne raw_data."""
    c = SemanticScholarCollector(sector_ids=["ai_ml"])
    c.request_delay = 0

    # 1er run : citationCount=100
    p_low = _paper(paper_id="p_evolve", citation_count=100)
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data={"data": [p_low]})):
        n1 = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    # 2e run même payload → INSERT OR IGNORE (no-op).
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data={"data": [p_low]})):
        n_dup = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    # 3e run : citationCount=300 → hash différent → nouvelle ligne.
    p_high = _paper(paper_id="p_evolve", citation_count=300)
    with patch("collectors.semantic_scholar_collector.requests.get",
               return_value=_fake_response(json_data={"data": [p_high]})):
        n2 = c.run(
            since=datetime(2026, 1, 1), until=datetime(2026, 4, 30),
        )
    assert n1 == 1
    assert n_dup == 0
    assert n2 == 1   # nouvelle ligne snapshot
