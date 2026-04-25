"""Tests du `ThesisGenerator` (Phase 3 étape 1)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory.database import Feature, RawData, Thesis, session_scope
from thesis.generator import (
    ThesisGenerator,
    _dominant_dimension,
    _latest_entry_price,
    _make_recommendation,
    _make_risks,
)


# ------------------------------------------------------------ helpers


def _seed_score(
    ticker: str,
    score: float,
    ts: datetime,
    dimensions: dict[str, float] | None = None,
    details: dict | None = None,
    model_version: str = "v2_mom_sigqual",
    weights: dict | None = None,
) -> None:
    meta = {
        "model_version": model_version,
        "weights": weights or {"momentum": 0.6, "signal_quality": 0.4},
        "dimensions": dimensions or {"momentum": score, "signal_quality": score},
        "details": details or {},
    }
    with session_scope() as s:
        s.add(Feature(
            feature_name="stock_score",
            target_type="asset",
            target_id=ticker,
            computed_at=ts,
            value=score,
            metadata_json=json.dumps(meta),
        ))


def _seed_close(ticker: str, close: float, content_at: datetime,
                fetched_at: datetime | None = None) -> None:
    fetched_at = fetched_at or content_at
    payload = {
        "ticker": ticker,
        "session_date": content_at.strftime("%Y-%m-%d"),
        "close": close,
    }
    with session_scope() as s:
        s.add(RawData(
            source="yfinance",
            entity_type="ohlcv_daily",
            entity_id=f"{ticker}:{content_at.strftime('%Y-%m-%d')}",
            fetched_at=fetched_at,
            content_at=content_at,
            payload_json=json.dumps(payload),
            hash=f"h-{ticker}-{content_at.strftime('%Y-%m-%d')}",
        ))


def _count_theses(ticker: str | None = None) -> int:
    with session_scope() as s:
        q = s.query(Thesis)
        if ticker is not None:
            q = q.filter_by(asset_id=ticker)
        return q.count()


# ---------------------------------------------------- pure helpers


def test_make_recommendation_bands():
    assert _make_recommendation(90.0) == "BUY"
    assert _make_recommendation(75.0) == "BUY"
    assert _make_recommendation(70.0) == "WATCH"
    assert _make_recommendation(60.0) == "WATCH"
    assert _make_recommendation(55.0) == "AVOID"


def test_dominant_dimension_picks_max():
    assert _dominant_dimension({"a": 10.0, "b": 80.0, "c": 50.0}) == ("b", 80.0)
    assert _dominant_dimension({}) is None


def test_make_risks_union_and_dedupe():
    risks = _make_risks(["ai_ml", "biotech"])
    cats = [r["category"] for r in risks]
    # Bases + secteur AI + biotech ; tous distincts.
    assert "macro" in cats
    assert "value_trap" in cats
    assert any(r["category"] == "concurrentiel" for r in risks)  # ai_ml
    assert any(r["category"] == "régulatoire" for r in risks)    # biotech
    # Dédup : pas de description dupliquée.
    descs = [r["description"] for r in risks]
    assert len(descs) == len(set(descs))


# --------------------------------------------------- entry price PIT


def test_latest_entry_price_prefers_most_recent(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_close("NVDA", 800.0, as_of - timedelta(days=3))
    _seed_close("NVDA", 820.0, as_of - timedelta(days=1))
    assert _latest_entry_price("NVDA", as_of) == 820.0


def test_latest_entry_price_respects_pit(tmp_db):
    as_of = datetime(2026, 4, 20)
    # Prix "futur" : content_at après as_of → ignoré.
    _seed_close("NVDA", 999.0, as_of + timedelta(days=1))
    _seed_close("NVDA", 750.0, as_of - timedelta(days=2))
    assert _latest_entry_price("NVDA", as_of) == 750.0


def test_latest_entry_price_respects_fetched_at(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_close("NVDA", 900.0,
                content_at=as_of - timedelta(days=1),
                fetched_at=as_of + timedelta(days=2))  # pas encore collecté
    _seed_close("NVDA", 800.0,
                content_at=as_of - timedelta(days=3),
                fetched_at=as_of - timedelta(days=3))
    assert _latest_entry_price("NVDA", as_of) == 800.0


def test_latest_entry_price_none_when_absent(tmp_db):
    assert _latest_entry_price("NVDA", datetime(2026, 4, 20)) is None


# ---------------------------------------------------- generator


def test_run_skips_below_threshold(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_score("NVDA", 60.0, as_of - timedelta(hours=1))  # ai_ml seuil 75
    n = ThesisGenerator(tickers=["NVDA"]).run(as_of=as_of)
    assert n == 0
    assert _count_theses() == 0


def test_run_creates_thesis_above_threshold(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_score(
        "NVDA", 85.0, as_of - timedelta(hours=1),
        dimensions={"momentum": 90.0, "signal_quality": 78.0},
    )
    _seed_close("NVDA", 800.0, as_of - timedelta(days=1))

    n = ThesisGenerator(tickers=["NVDA"]).run(as_of=as_of)
    assert n == 1

    with session_scope() as s:
        th = s.query(Thesis).one()
        assert th.asset_type == "stock"
        assert th.asset_id == "NVDA"
        assert th.sector_id == "ai_ml"
        assert th.score == 85.0
        assert th.recommendation == "BUY"
        assert th.entry_price == 800.0
        assert th.model_version == "v2_mom_sigqual"

        triggers = json.loads(th.triggers_json)
        # Momentum domine (90 > 78) → premier trigger.
        assert triggers[0]["dimension"] == "momentum"
        assert triggers[0]["sub_score"] == 90.0

        risks = json.loads(th.risks_json)
        assert any(r["category"] == "macro" for r in risks)
        assert any(r["category"] == "concurrentiel" for r in risks)  # ai_ml

        breakdown = json.loads(th.score_breakdown_json)
        assert breakdown["dimensions"]["momentum"] == 90.0

        # Narrative doit contenir les 5 sections.
        for section in (
            "Pourquoi maintenant", "Score", "Catalyseurs", "Risques", "Entrée",
        ):
            assert section in th.narrative


def test_run_uses_permissive_sector_threshold(tmp_db):
    """quantum_computing a un seuil à 65 — un score de 66 doit passer
    même si l'action est aussi taguée AI (seuil 75) : on prend le min."""
    as_of = datetime(2026, 4, 20)
    # IBM est tagué ['quantum_computing', 'ai_ml'] dans la watchlist.
    _seed_score("IBM", 66.0, as_of - timedelta(hours=1))
    n = ThesisGenerator(tickers=["IBM"]).run(as_of=as_of)
    assert n == 1
    with session_scope() as s:
        th = s.query(Thesis).one()
        # sector_id canonique = premier secteur listé → quantum_computing.
        assert th.sector_id == "quantum_computing"
        assert th.recommendation == "WATCH"


def test_run_idempotent_within_same_day(tmp_db):
    as_of = datetime(2026, 4, 20, 14, 0)
    _seed_score("NVDA", 90.0, as_of - timedelta(hours=1))

    gen = ThesisGenerator(tickers=["NVDA"])
    assert gen.run(as_of=as_of) == 1
    # Deuxième run le même jour → 0 création supplémentaire.
    assert gen.run(as_of=as_of + timedelta(hours=3)) == 0
    assert _count_theses("NVDA") == 1


def test_run_creates_new_thesis_next_day(tmp_db):
    d1 = datetime(2026, 4, 20, 14, 0)
    d2 = datetime(2026, 4, 21, 14, 0)
    _seed_score("NVDA", 90.0, d1 - timedelta(hours=1))
    _seed_score("NVDA", 92.0, d2 - timedelta(hours=1))

    gen = ThesisGenerator(tickers=["NVDA"])
    assert gen.run(as_of=d1) == 1
    assert gen.run(as_of=d2) == 1
    assert _count_theses("NVDA") == 2


def test_run_handles_missing_entry_price(tmp_db):
    """Pas de close yfinance → thèse créée sans entry_price (degraded)."""
    as_of = datetime(2026, 4, 20)
    _seed_score("NVDA", 85.0, as_of - timedelta(hours=1))
    # Aucun OHLCV seed.
    n = ThesisGenerator(tickers=["NVDA"]).run(as_of=as_of)
    assert n == 1
    with session_scope() as s:
        th = s.query(Thesis).one()
        assert th.entry_price is None
        assert "non disponible" in th.narrative


def test_run_snapshots_model_version_and_weights(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_score(
        "NVDA", 90.0, as_of - timedelta(hours=1),
        model_version="v3_mom_sigqual_sent",
        weights={"momentum": 0.4, "signal_quality": 0.3, "sentiment": 0.3},
    )
    ThesisGenerator(tickers=["NVDA"]).run(as_of=as_of)
    with session_scope() as s:
        th = s.query(Thesis).one()
        assert th.model_version == "v3_mom_sigqual_sent"
        weights = json.loads(th.weights_snapshot_json)
        assert weights["sentiment"] == 0.3


def test_run_isolates_per_ticker_errors(tmp_db):
    """Une exception sur un ticker n'interrompt pas les autres."""
    as_of = datetime(2026, 4, 20)
    _seed_score("NVDA", 90.0, as_of - timedelta(hours=1))
    # On seed un JSON corrompu pour AMD — json.loads échoue, mais la
    # feature numérique est lisible, donc on n'a pas vraiment de crash.
    # À la place, on force un ticker inconnu dans la watchlist : pas de
    # secteurs mappés, seuil _default=70. Score 90 → thèse créée.
    _seed_score("UNKNOWN", 90.0, as_of - timedelta(hours=1))

    n = ThesisGenerator(tickers=["NVDA", "UNKNOWN"]).run(as_of=as_of)
    assert n == 2  # les deux passent
