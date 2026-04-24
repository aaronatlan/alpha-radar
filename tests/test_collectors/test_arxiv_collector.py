"""Tests unitaires du collecteur arXiv — sans I/O réseau.

Les appels à `arxiv.Client.results()` sont mockés. Les tests vérifient :
  - la normalisation (structure canonique de sortie),
  - le filtrage temporel avec sortie anticipée,
  - l'idempotence via la déduplication en base.
"""
from __future__ import annotations

from datetime import datetime, timedelta, timezone
from types import SimpleNamespace
from unittest.mock import patch

from collectors.arxiv_collector import ArxivCollector


def _fake_result(entry_id: str, title: str, published: datetime):
    """Fabrique un faux `arxiv.Result` avec les attributs minimaux utilisés."""
    return SimpleNamespace(
        entry_id=entry_id,
        title=title,
        summary="abstract text",
        authors=[SimpleNamespace(name="Alice"), SimpleNamespace(name="Bob")],
        categories=["cs.LG"],
        primary_category="cs.LG",
        published=published,
        updated=published,
        pdf_url=f"{entry_id}.pdf",
        doi=None,
    )


# --------------------------------------------------------------- normalize


def test_normalize_produces_canonical_item():
    c = ArxivCollector(categories=["cs.LG"])
    pub = datetime(2026, 1, 15, 12, 0, tzinfo=timezone.utc)
    r = _fake_result("http://arxiv.org/abs/2601.00001v1", "Test paper", pub)

    item = c.normalize(r)

    assert item is not None
    assert item["entity_type"] == "paper"
    assert item["entity_id"] == "http://arxiv.org/abs/2601.00001v1"
    # content_at doit être naïf (UTC), convention du projet.
    assert item["content_at"] == pub.replace(tzinfo=None)
    assert item["content_at"].tzinfo is None
    assert item["payload"]["title"] == "Test paper"
    assert item["payload"]["authors"] == ["Alice", "Bob"]
    assert item["payload"]["primary_category"] == "cs.LG"


def test_normalize_returns_none_for_none_input():
    c = ArxivCollector(categories=["cs.LG"])
    assert c.normalize(None) is None


# ------------------------------------------------------------------ collect


def test_collect_filters_window_and_stops_on_older():
    """Dès qu'un résultat plus ancien que `since` arrive, on sort tôt.

    Vérifie aussi qu'aucune information postérieure à `until` ne passe —
    garantie point-in-time de la fenêtre.
    """
    c = ArxivCollector(categories=["cs.LG"])
    c.request_delay = 0
    now = datetime.now(timezone.utc)
    in_window = _fake_result("id1", "t1", now - timedelta(hours=5))
    too_old = _fake_result("id2", "t2", now - timedelta(days=30))
    never_reached = _fake_result("id3", "t3", now - timedelta(days=60))

    def fake_results(search):
        yield in_window
        yield too_old          # déclenche le break
        yield never_reached    # ne doit pas être consommé

    with patch.object(c._client, "results", side_effect=fake_results):
        out = c.collect(
            since=now - timedelta(days=1),
            until=now + timedelta(minutes=1),
        )

    assert [r.entry_id for r in out] == ["id1"]


def test_collect_skips_category_on_exception():
    """Mode dégradé : une catégorie qui plante ne doit pas faire sauter la collecte."""
    c = ArxivCollector(categories=["cs.LG", "cs.CR"])
    c.request_delay = 0
    now = datetime.now(timezone.utc)

    call_count = {"n": 0}

    def fake_results(search):
        call_count["n"] += 1
        if call_count["n"] == 1:
            raise RuntimeError("API down")
        yield _fake_result("id_good", "ok", now - timedelta(hours=1))

    with patch.object(c._client, "results", side_effect=fake_results):
        out = c.collect(
            since=now - timedelta(days=1),
            until=now + timedelta(minutes=1),
        )

    assert [r.entry_id for r in out] == ["id_good"]


# ----------------------------------------------------------------- run / store


def test_run_stores_then_deduplicates(tmp_db):
    """Deuxième run sur la même donnée : aucun nouveau insert (UNIQUE)."""
    c = ArxivCollector(categories=["cs.LG"])
    c.request_delay = 0
    now = datetime.now(timezone.utc)
    r = _fake_result(
        "http://arxiv.org/abs/xxx", "paper",
        now - timedelta(hours=1),
    )

    def fake_results(search):
        yield r

    with patch.object(c._client, "results", side_effect=fake_results):
        inserted_1 = c.run(now - timedelta(days=1), now + timedelta(minutes=1))
        inserted_2 = c.run(now - timedelta(days=1), now + timedelta(minutes=1))

    assert inserted_1 == 1
    assert inserted_2 == 0  # doublon filtré par la contrainte UNIQUE


def test_run_returns_zero_when_collect_raises(tmp_db):
    """Exception en collect() : mode dégradé → 0 inséré, pas d'exception remontée."""
    c = ArxivCollector(categories=["cs.LG"])
    c.request_delay = 0
    with patch.object(c, "collect", side_effect=RuntimeError("boom")):
        now = datetime.now(timezone.utc).replace(tzinfo=None)
        assert c.run(now, now) == 0
