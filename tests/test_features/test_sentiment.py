"""Tests de `features.sentiment` avec un classifieur stubbé (pas de FinBERT)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from features.sentiment import FinBERTSentimentAnalyzer, NewsSentimentSectorFeature
from memory.database import RawData, session_scope


def _stub_classifier(mapping: dict[str, tuple[float, float]]):
    """Retourne un callable qui simule le pipeline HF.

    `mapping` : texte → (p_pos, p_neg). Les textes inconnus → neutre.
    """
    def _call(texts: list[str]):
        out = []
        for t in texts:
            p_pos, p_neg = mapping.get(t, (0.33, 0.33))
            p_neu = max(0.0, 1.0 - p_pos - p_neg)
            out.append([
                {"label": "positive", "score": p_pos},
                {"label": "negative", "score": p_neg},
                {"label": "neutral",  "score": p_neu},
            ])
        return out
    return _call


def _seed_article(sector_id: str, title: str, content_at: datetime,
                  fetched_at: datetime | None = None, description: str = "") -> None:
    fetched_at = fetched_at or content_at
    payload = {
        "sector_id": sector_id,
        "title": title,
        "description": description,
        "url": f"https://example.test/{title.replace(' ', '-')}",
    }
    with session_scope() as s:
        s.add(RawData(
            source="newsapi",
            entity_type="news_article",
            entity_id=payload["url"],
            fetched_at=fetched_at,
            content_at=content_at,
            payload_json=json.dumps(payload),
            hash=f"h-{title}",
        ))


# -------------------------------------------------------------- analyzer


def test_analyzer_reduces_scores_to_pos_minus_neg():
    stub = _stub_classifier({
        "great beat": (0.9, 0.05),
        "disaster":   (0.05, 0.9),
        "meh":        (0.2, 0.2),
    })
    a = FinBERTSentimentAnalyzer(classifier=stub)
    out = a.score_texts(["great beat", "disaster", "meh"])
    assert out[0] == pytest.approx(0.85)
    assert out[1] == pytest.approx(-0.85)
    assert out[2] == pytest.approx(0.0)


def test_analyzer_handles_empty_and_none():
    a = FinBERTSentimentAnalyzer(classifier=_stub_classifier({}))
    out = a.score_texts(["", None, "real"])
    assert out == [0.0, 0.0, 0.0]  # "real" inconnu → neutre 0.33/0.33


def test_analyzer_truncates_long_text():
    captured: list[str] = []
    def stub(texts):
        captured.extend(texts)
        return [[{"label": "neutral", "score": 1.0}] for _ in texts]
    a = FinBERTSentimentAnalyzer(classifier=stub)
    long_text = "x" * 2000
    a.score_texts([long_text])
    assert len(captured[0]) == FinBERTSentimentAnalyzer.max_chars_per_text


# -------------------------------------------------------------- feature


def test_compute_returns_none_below_threshold(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_article("ai_ml", "positive beat", as_of - timedelta(days=1))
    _seed_article("ai_ml", "another",        as_of - timedelta(days=2))

    feat = NewsSentimentSectorFeature(
        sector_ids=["ai_ml"],
        analyzer=FinBERTSentimentAnalyzer(classifier=_stub_classifier({})),
    )
    # min_articles = 3, on n'en a que 2 → None.
    assert feat.compute("ai_ml", as_of) is None


def test_compute_weights_recent_articles_more(tmp_db):
    as_of = datetime(2026, 4, 20)
    # 3 articles, les plus récents positifs, le plus ancien négatif.
    # Moyenne NON pondérée = (1 + 1 − 1) / 3 ≈ 0.33.
    # Avec half-life 3j et Δ=1,2,6j : w ≈ 0.79, 0.63, 0.25 → pondérée ≈
    # (0.79 + 0.63 − 0.25) / (0.79+0.63+0.25) ≈ 0.70.
    _seed_article("ai_ml", "pos today",   as_of - timedelta(days=1),
                  description="")
    _seed_article("ai_ml", "pos 2d ago",  as_of - timedelta(days=2))
    _seed_article("ai_ml", "neg 6d ago",  as_of - timedelta(days=6))

    stub = _stub_classifier({
        "pos today":   (0.95, 0.0),
        "pos 2d ago":  (0.95, 0.0),
        "neg 6d ago":  (0.0, 0.95),
    })
    feat = NewsSentimentSectorFeature(
        sector_ids=["ai_ml"],
        analyzer=FinBERTSentimentAnalyzer(classifier=stub),
    )
    value, meta = feat.compute("ai_ml", as_of)
    # La pondération doit donner un score > la moyenne non pondérée.
    unweighted = (0.95 + 0.95 - 0.95) / 3
    assert value > unweighted
    assert meta["n_articles"] == 3


def test_compute_respects_pit_on_fetched_at(tmp_db):
    as_of = datetime(2026, 4, 20)
    # 3 articles, dont 1 fetched APRES as_of → ignoré.
    _seed_article("ai_ml", "a", as_of - timedelta(days=1))
    _seed_article("ai_ml", "b", as_of - timedelta(days=2))
    _seed_article("ai_ml", "c", content_at=as_of - timedelta(days=3),
                  fetched_at=as_of + timedelta(days=1))  # hors PIT

    stub = _stub_classifier({
        "a": (0.8, 0.0), "b": (0.8, 0.0), "c": (0.0, 0.9),
    })
    feat = NewsSentimentSectorFeature(
        sector_ids=["ai_ml"],
        analyzer=FinBERTSentimentAnalyzer(classifier=stub),
    )
    # 2 articles réellement disponibles → sous min_articles → None.
    assert feat.compute("ai_ml", as_of) is None


def test_compute_filters_by_sector(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_article("ai_ml", "ai1", as_of - timedelta(days=1))
    _seed_article("ai_ml", "ai2", as_of - timedelta(days=2))
    _seed_article("ai_ml", "ai3", as_of - timedelta(days=3))
    _seed_article("biotech", "bio1", as_of - timedelta(days=1))
    _seed_article("biotech", "bio2", as_of - timedelta(days=2))

    stub = _stub_classifier({
        "ai1": (0.9, 0.0), "ai2": (0.9, 0.0), "ai3": (0.9, 0.0),
        "bio1": (0.0, 0.9), "bio2": (0.0, 0.9),
    })
    feat = NewsSentimentSectorFeature(
        sector_ids=["ai_ml", "biotech"],
        analyzer=FinBERTSentimentAnalyzer(classifier=stub),
    )
    ai_v, _ = feat.compute("ai_ml", as_of)
    assert ai_v > 0.5  # très positif
    # biotech n'a que 2 articles → sous min_articles.
    assert feat.compute("biotech", as_of) is None


def test_run_stores_only_available_sectors(tmp_db):
    as_of = datetime(2026, 4, 20)
    for t in ("a1", "a2", "a3"):
        _seed_article("ai_ml", t, as_of - timedelta(days=1))

    stub = _stub_classifier({
        "a1": (0.9, 0.0), "a2": (0.9, 0.0), "a3": (0.9, 0.0),
    })
    feat = NewsSentimentSectorFeature(
        sector_ids=["ai_ml", "biotech"],
        analyzer=FinBERTSentimentAnalyzer(classifier=stub),
    )
    n = feat.run(as_of=as_of)
    assert n == 1  # ai_ml seul, biotech n'a pas d'articles


def test_unknown_sector_raises(tmp_db):
    with pytest.raises(ValueError):
        NewsSentimentSectorFeature(sector_ids=["made_up"])
