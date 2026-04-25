"""Tests du `StockScorer` (phase 2 v1 : dimension momentum seule)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory.database import Feature, session_scope
from scoring.stock_scorer import (
    StockScorer,
    _momentum_to_score,
    _rsi_to_score,
    _volume_to_score,
)


def _seed_feature(
    name: str,
    target_id: str,
    value: float,
    ts: datetime,
    target_type: str = "asset",
) -> None:
    with session_scope() as session:
        session.add(
            Feature(
                feature_name=name,
                target_type=target_type,
                target_id=target_id,
                computed_at=ts,
                value=value,
                metadata_json=None,
            )
        )


def _get_stock_score(ticker: str) -> Feature | None:
    with session_scope() as session:
        return (
            session.query(Feature)
            .filter_by(feature_name="stock_score", target_id=ticker)
            .order_by(Feature.computed_at.desc())
            .first()
        )


# ---------------------------------------------------------- sub-score mappings


@pytest.mark.parametrize(
    "rsi,expected",
    [(0.0, 0.0), (60.0, 100.0), (30.0, 50.0), (100.0, 0.0), (80.0, 50.0)],
)
def test_rsi_mapping(rsi, expected):
    assert _rsi_to_score(rsi) == pytest.approx(expected, rel=1e-6)


@pytest.mark.parametrize(
    "m,expected",
    [(-0.20, 0.0), (0.0, 50.0), (0.20, 100.0), (0.10, 75.0), (-0.50, 0.0)],
)
def test_momentum_mapping(m, expected):
    assert _momentum_to_score(m) == pytest.approx(expected)


@pytest.mark.parametrize(
    "vr,expected",
    [(0.0, 0.0), (1.0, 50.0), (2.0, 100.0), (5.0, 100.0), (-1.0, 0.0)],
)
def test_volume_mapping(vr, expected):
    assert _volume_to_score(vr) == pytest.approx(expected)


# -------------------------------------------------------------- compute


def test_compute_returns_none_without_any_input(tmp_db):
    scorer = StockScorer(tickers=["NVDA"], model_version="v1_momentum_only")
    assert scorer.compute("NVDA", datetime(2026, 2, 15)) is None


def test_compute_with_all_three_inputs(tmp_db):
    as_of = datetime(2026, 2, 15)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "NVDA", 60.0, ts)           # sub = 100
    _seed_feature("momentum_30d", "NVDA", 0.10, ts)     # sub = 75
    _seed_feature("volume_ratio_7_30", "NVDA", 1.5, ts) # sub = 75

    scorer = StockScorer(tickers=["NVDA"], model_version="v1_momentum_only")
    value, meta = scorer.compute("NVDA", as_of)

    # intra-dimension : 0.35*100 + 0.45*75 + 0.20*75 = 35 + 33.75 + 15 = 83.75
    # dimension unique (momentum, poids 1.0) → score composite = 83.75
    assert value == pytest.approx(83.75, rel=1e-6)
    assert meta["dimensions"]["momentum"] == pytest.approx(83.75, rel=1e-6)
    assert meta["model_version"] == "v1_momentum_only"


def test_compute_renormalizes_on_partial_inputs(tmp_db):
    """Si seul `rsi_14` est disponible, le sous-score momentum utilise
    les 0.35 comme dénominateur → équivaut à renvoyer le sub-score RSI brut."""
    as_of = datetime(2026, 2, 15)
    _seed_feature("rsi_14", "NVDA", 60.0, as_of - timedelta(hours=1))

    scorer = StockScorer(tickers=["NVDA"], model_version="v1_momentum_only")
    value, _ = scorer.compute("NVDA", as_of)
    assert value == pytest.approx(100.0, rel=1e-6)  # sub-score RSI à 60 = 100


def test_compute_respects_pit(tmp_db):
    """Features calculées après as_of → ignorées."""
    as_of = datetime(2026, 2, 15)
    _seed_feature("rsi_14", "NVDA", 60.0, as_of + timedelta(days=1))
    scorer = StockScorer(tickers=["NVDA"], model_version="v1_momentum_only")
    assert scorer.compute("NVDA", as_of) is None


# ---------------------------------------------------------------- run


def test_run_stores_for_multiple_tickers(tmp_db):
    as_of = datetime(2026, 2, 15)
    ts = as_of - timedelta(hours=1)
    for ticker in ("NVDA", "AMD"):
        _seed_feature("rsi_14", ticker, 60.0, ts)
        _seed_feature("momentum_30d", ticker, 0.10, ts)

    scorer = StockScorer(tickers=["NVDA", "AMD"], model_version="v1_momentum_only")
    n = scorer.run(as_of=as_of)
    assert n == 2

    nvda = _get_stock_score("NVDA")
    assert nvda is not None
    meta = json.loads(nvda.metadata_json)
    assert meta["model_version"] == "v1_momentum_only"
    assert "momentum" in meta["dimensions"]


def test_run_skips_tickers_without_inputs(tmp_db):
    as_of = datetime(2026, 2, 15)
    _seed_feature("rsi_14", "NVDA", 60.0, as_of - timedelta(hours=1))
    scorer = StockScorer(tickers=["NVDA", "AMD"], model_version="v1_momentum_only")
    n = scorer.run(as_of=as_of)
    assert n == 1
    assert _get_stock_score("NVDA") is not None
    assert _get_stock_score("AMD") is None


# --------------------------------------------- v2 (signal_quality dimension)


def test_v2_uses_sector_heat_when_available(tmp_db):
    """signal_quality intègre le Heat Score moyen des secteurs de l'action."""
    as_of = datetime(2026, 2, 15)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "NVDA", 60.0, ts)                # momentum=100
    _seed_feature("momentum_30d", "NVDA", 0.20, ts)          # =100
    _seed_feature("volume_ratio_7_30", "NVDA", 2.0, ts)      # =100
    # NVDA → sectors=['ai_ml']. Heat à 80. Pas de sec_data → sec_score
    # est absent → signal_quality = heat moyen = 80
    _seed_feature("sector_heat_score", "ai_ml", 80.0, ts, target_type="sector")

    scorer = StockScorer(tickers=["NVDA"], model_version="v2_mom_sigqual")
    value, meta = scorer.compute("NVDA", as_of)
    # momentum=100, signal_quality=80, weights 0.6/0.4 → 92
    assert value == pytest.approx(92.0, rel=1e-6)
    assert "signal_quality" in meta["dimensions"]


def test_v2_skips_signal_quality_when_absent(tmp_db):
    """Ni heat_score ni SEC data → signal_quality None → dimension sautée
    → score = momentum seul."""
    as_of = datetime(2026, 2, 15)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "NVDA", 60.0, ts)

    scorer = StockScorer(tickers=["NVDA"], model_version="v2_mom_sigqual")
    value, meta = scorer.compute("NVDA", as_of)
    assert "signal_quality" not in meta["dimensions"]
    assert value == pytest.approx(100.0, rel=1e-6)


def test_v2_signal_quality_counts_sec_filings(tmp_db):
    """Présence de filings SEC → sec_filings alimenté ; 2/3 formes
    "signal" → sec_score ≈ 66.67."""
    as_of = datetime(2026, 2, 15)

    from memory.database import RawData, session_scope as scope
    import json as _json

    def _add_filing(form, acc):
        with scope() as s:
            s.add(RawData(
                source="sec_edgar", entity_type="sec_filing",
                entity_id=acc,
                fetched_at=as_of - timedelta(days=1),
                content_at=as_of - timedelta(days=3),
                payload_json=_json.dumps({"ticker": "NVDA", "form": form}),
                hash=acc,
            ))
    _add_filing("SC 13D", "a1")
    _add_filing("SC 13D", "a2")
    _add_filing("10-Q", "a3")  # non-signal → n'incrémente pas

    _seed_feature("rsi_14", "NVDA", 60.0, as_of - timedelta(hours=1))

    scorer = StockScorer(tickers=["NVDA"], model_version="v2_mom_sigqual")
    _, meta = scorer.compute("NVDA", as_of)
    sq_detail = meta["details"]["signal_quality"]
    assert sq_detail["inputs"]["n_signal_filings_30d"] == 2
    assert sq_detail["inputs"]["has_sec_data"] is True


# --------------------------------------------- v3 (sentiment dimension)


def test_v3_sentiment_uses_sector_news_sentiment(tmp_db):
    """La dimension sentiment moyenne le `news_sentiment_sector` des secteurs
    de l'action et mappe [−1, +1] → [0, 100] (0 → 50)."""
    as_of = datetime(2026, 2, 15)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "NVDA", 60.0, ts)                      # momentum=100
    _seed_feature("momentum_30d", "NVDA", 0.20, ts)                # =100
    _seed_feature("volume_ratio_7_30", "NVDA", 2.0, ts)            # =100
    _seed_feature("sector_heat_score", "ai_ml", 80.0, ts, target_type="sector")
    # sentiment sectoriel à +0.5 → sub-score = 75
    _seed_feature("news_sentiment_sector", "ai_ml", 0.5, ts, target_type="sector")

    scorer = StockScorer(tickers=["NVDA"], model_version="v3_mom_sigqual_sent")
    value, meta = scorer.compute("NVDA", as_of)
    # momentum=100 (0.4), signal_quality=80 (0.3), sentiment=75 (0.3)
    # → 0.4*100 + 0.3*80 + 0.3*75 = 40 + 24 + 22.5 = 86.5
    assert value == pytest.approx(86.5, rel=1e-6)
    assert "sentiment" in meta["dimensions"]
    assert meta["dimensions"]["sentiment"] == pytest.approx(75.0, rel=1e-6)


def test_v3_skips_sentiment_when_absent(tmp_db):
    """Pas de sentiment publié → dimension sautée, renormalisation sur
    les dimensions restantes (momentum + signal_quality)."""
    as_of = datetime(2026, 2, 15)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "NVDA", 60.0, ts)
    _seed_feature("sector_heat_score", "ai_ml", 80.0, ts, target_type="sector")

    scorer = StockScorer(tickers=["NVDA"], model_version="v3_mom_sigqual_sent")
    _, meta = scorer.compute("NVDA", as_of)
    assert "sentiment" not in meta["dimensions"]
