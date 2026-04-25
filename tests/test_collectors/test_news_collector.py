"""Tests du `NewsAPICollector` — comportement normal + mode dégradé
quand la clé est absente."""
from __future__ import annotations

from datetime import datetime, timezone
from unittest.mock import MagicMock, patch

import pytest

from collectors.news_collector import NewsAPICollector, _build_query


SECTOR = {
    "id": "ai_ml", "name": "IA", "category": "software",
    "keywords": ["large language model", "transformer"],
    "arxiv_categories": ["cs.LG"],
}


def _fake_ok(articles):
    r = MagicMock()
    r.status_code = 200
    r.json.return_value = {"status": "ok", "articles": articles}
    return r


# ---- query builder


def test_build_query_quotes_multiword_terms():
    q = _build_query(["large language model", "transformer", "", "  "])
    assert q == '"large language model" OR transformer'


# ---- collect (no key)


def test_collect_is_noop_without_key(monkeypatch):
    from collectors import news_collector as mod
    monkeypatch.setattr(mod.settings, "newsapi_key", None, raising=False)
    c = NewsAPICollector(sectors=[SECTOR])
    c.request_delay = 0
    assert c.collect(datetime(2026, 4, 1), datetime(2026, 4, 15)) == []


# ---- collect (with key)


def test_collect_parses_articles(monkeypatch):
    from collectors import news_collector as mod
    monkeypatch.setattr(mod.settings, "newsapi_key", "SECRET", raising=False)
    c = NewsAPICollector(sectors=[SECTOR])
    c.request_delay = 0

    articles = [
        {
            "title": "Breaking AI news",
            "description": "...",
            "content": "...",
            "url": "https://x.com/a1",
            "source": {"name": "TechCrunch"},
            "publishedAt": "2026-04-10T12:00:00Z",
            "author": "Alice",
        },
    ]
    with patch("collectors.news_collector.requests.get",
               return_value=_fake_ok(articles)):
        out = c.collect(datetime(2026, 4, 1), datetime(2026, 4, 15))
    assert len(out) == 1
    assert out[0]["sector_id"] == "ai_ml"
    assert out[0]["article"]["url"] == "https://x.com/a1"


def test_collect_breaks_on_rate_limit(monkeypatch):
    from collectors import news_collector as mod
    monkeypatch.setattr(mod.settings, "newsapi_key", "SECRET", raising=False)
    sectors = [SECTOR, {**SECTOR, "id": "biotech", "keywords": ["crispr"]}]
    c = NewsAPICollector(sectors=sectors)
    c.request_delay = 0
    resp = MagicMock(status_code=429)
    with patch("collectors.news_collector.requests.get", return_value=resp) as m:
        out = c.collect(datetime(2026, 4, 1), datetime(2026, 4, 15))
    assert out == []
    assert m.call_count == 1


# ---- normalize


def test_normalize_parses_iso_timestamp():
    c = NewsAPICollector(sectors=[SECTOR])
    raw = {
        "sector_id": "ai_ml",
        "article": {
            "title": "t", "description": "d", "content": "c",
            "url": "https://x.com/a", "source": {"name": "X"},
            "publishedAt": "2026-04-10T12:00:00Z",
            "author": "A",
        },
    }
    item = c.normalize(raw)
    assert item is not None
    assert item["entity_type"] == "news_article"
    assert item["entity_id"] == "https://x.com/a"
    assert item["content_at"] == datetime(2026, 4, 10, 12, 0)


def test_normalize_missing_url_returns_none():
    c = NewsAPICollector(sectors=[SECTOR])
    assert c.normalize({"article": {}}) is None


def test_normalize_bad_timestamp_keeps_content_at_none():
    c = NewsAPICollector(sectors=[SECTOR])
    raw = {"article": {"url": "https://x/a", "publishedAt": "garbage"}}
    item = c.normalize(raw)
    assert item is not None
    assert item["content_at"] is None
