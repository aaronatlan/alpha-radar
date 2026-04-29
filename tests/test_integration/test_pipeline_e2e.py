"""Tests d'intégration bout-en-bout du pipeline Alpha Radar.

Ces tests exercent plusieurs modules de bout en bout, sur des données
seedées en base, sans toucher au réseau ni aux APIs externes :

- `test_score_to_thesis_pipeline` : OHLCV → features techniques →
  scoring composite → génération de thèse. Vérifie qu'un mouvement
  haussier produit bien une thèse BUY avec narrative + entry_price.

- `test_thesis_to_backtest_pipeline` : thèse en base → évaluateur
  (jalons 30/90 avec peer benchmark) → simulateur portefeuille. Vérifie
  que les évaluations retombent sur les bons returns et que le simulateur
  ouvre la position correspondante.

Ces deux tests n'essaient pas d'être exhaustifs — chaque module a ses
propres tests unitaires. Ils valident que les contrats inter-modules
(noms de features, formats de metadata, jalons) sont alignés.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from backtesting.portfolio import PortfolioSimulator
from features.technical import Momentum30DFeature
from memory.database import (
    Evaluation,
    Feature,
    RawData,
    Thesis,
    session_scope,
)
from scoring.stock_scorer import StockScorer
from thesis.evaluator import ThesisEvaluator
from thesis.generator import ThesisGenerator


# ---------------------------------------------------------------- helpers


def _seed_ohlcv_series(
    ticker: str,
    start: datetime,
    closes: list[float],
    *,
    volume: int = 10_000_000,
) -> None:
    """Insère une série quotidienne yfinance ohlcv_daily en base.

    `start` est inclusif. `len(closes)` séances générées (jours civils,
    pas trading days — c'est suffisant pour les tests, les features
    lisent au format calendrier).
    """
    with session_scope() as s:
        for i, close in enumerate(closes):
            d = start + timedelta(days=i)
            payload = {
                "ticker": ticker,
                "session_date": d.strftime("%Y-%m-%d"),
                "open": close * 0.99,
                "high": close * 1.01,
                "low": close * 0.98,
                "close": close,
                "adj_close": close,
                "volume": volume,
            }
            s.add(RawData(
                source="yfinance",
                entity_type="ohlcv_daily",
                entity_id=f"{ticker}:{d.strftime('%Y-%m-%d')}",
                fetched_at=d,
                content_at=d,
                payload_json=json.dumps(payload),
                hash=f"h-{ticker}-{d.strftime('%Y-%m-%d-%H-%M-%S')}",
            ))


# ---------------------------------------------------------------- pipeline 1


def test_score_to_thesis_pipeline(tmp_db):
    """Mouvement haussier sur NVDA → thèse BUY générée."""
    # 60 jours de prix montant linéairement de 100 à 160 (+60%) — momentum
    # 30j fortement positif, RSI saturé en zone d'achat.
    start = datetime(2026, 1, 1)
    closes = [100.0 + i * 1.0 for i in range(60)]
    _seed_ohlcv_series("NVDA", start, closes)

    # `as_of` après la dernière séance — toutes les closes seedées sont PIT.
    as_of = start + timedelta(days=len(closes) + 1)

    # 1) Features techniques. On ne lance que momentum_30d pour ce test :
    #    une rampe monotone donne un RSI saturé qui pénaliserait le score
    #    via le mapping triangulaire (peak à 60). On exerce ainsi aussi
    #    la renormalisation du scorer sur les dimensions disponibles.
    n_mom = Momentum30DFeature(tickers=["NVDA"]).run(as_of=as_of)
    assert n_mom == 1

    # 2) Scoring composite. v1 = momentum-only — pas de dépendance à
    #    GitHub stars / SEC qui ne sont pas seedés ici.
    scorer = StockScorer(tickers=["NVDA"], model_version="v1_momentum_only")
    n_scores = scorer.run(as_of=as_of)
    assert n_scores == 1

    # Vérifie le score persisté + ses dimensions / model_version.
    with session_scope() as s:
        row = s.execute(
            Feature.__table__.select()
            .where(Feature.feature_name == "stock_score")
            .where(Feature.target_id == "NVDA")
        ).fetchone()
        assert row is not None
        score = float(row.value)
        meta = json.loads(row.metadata_json)
    # +60% sur 30j → momentum_30d clippé à +20% → sous-score 100. Avec
    # RSI/volume manquants, renormalisation → momentum dimension = 100.
    assert score == pytest.approx(100.0, abs=0.5)
    assert meta["model_version"] == "v1_momentum_only"
    assert "momentum" in meta["dimensions"]

    # 3) Génération de thèse. Seuil bas pour garantir l'éligibilité.
    gen = ThesisGenerator(
        tickers=["NVDA"],
        thresholds={"_default": 50.0},
    )
    n_theses = gen.run(as_of=as_of)
    assert n_theses == 1

    # 4) La thèse doit être complète et reproductible.
    with session_scope() as s:
        th = s.query(Thesis).one()
        assert th.asset_id == "NVDA"
        assert th.recommendation in ("BUY", "WATCH")
        assert th.entry_price == pytest.approx(closes[-1], rel=1e-9)
        assert th.score == pytest.approx(score, rel=1e-9)
        assert th.model_version == "v1_momentum_only"
        # Snapshot des poids → reproductibilité du score.
        assert json.loads(th.weights_snapshot_json) == {"momentum": 1.0}
        # Narrative non vide, mentionne le ticker.
        assert "NVDA" in th.narrative
        # Score breakdown contient les dimensions du moment.
        breakdown = json.loads(th.score_breakdown_json)
        assert "momentum" in breakdown["dimensions"]


# ---------------------------------------------------------------- pipeline 2


def test_thesis_to_backtest_pipeline(tmp_db):
    """Thèse en base → évaluateur jalon 30 → simulateur portefeuille."""
    open_date = datetime(2026, 1, 1)

    # Prix : NVDA +20% à J+30, AMD (peer ai_ml) +5% à J+30 → alpha = +15%.
    _seed_ohlcv_series("NVDA", open_date,
                       [100.0 + 20.0 * i / 30 for i in range(31)])
    _seed_ohlcv_series("AMD", open_date,
                       [200.0 + 10.0 * i / 30 for i in range(31)])

    # 1) Insère une thèse BUY directement (on isole les modules aval).
    with session_scope() as s:
        th = Thesis(
            created_at=open_date,
            asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
            score=82.0,
            score_breakdown_json=json.dumps({"dimensions": {"momentum": 82.0}}),
            recommendation="BUY",
            horizon_days=180,
            entry_price=100.0,
            entry_conditions_json=json.dumps({"band_pct": 0.02}),
            triggers_json="[]", risks_json="[]", catalysts_json="[]",
            narrative="…",
            model_version="v1_momentum_only",
            weights_snapshot_json=json.dumps({"momentum": 1.0}),
        )
        s.add(th)
        s.flush()
        thesis_id = th.id

    # 2) Évaluateur sur le jalon 30. NVDA +20% vs AMD +5% → alpha ≈ +15%,
    #    statut 'active' car days < 180 (classification minimum).
    eval_at = open_date + timedelta(days=30)
    n_evals = ThesisEvaluator(milestones=[30, 90]).run(as_of=eval_at)
    assert n_evals == 1

    with session_scope() as s:
        ev = s.query(Evaluation).filter_by(thesis_id=thesis_id).one()
        assert ev.days_since_thesis == 30
        assert ev.return_pct == pytest.approx(0.20, abs=0.005)
        assert ev.benchmark_return_pct == pytest.approx(0.05, abs=0.01)
        assert ev.alpha_pct == pytest.approx(0.15, abs=0.01)
        assert ev.status == "active"  # < 180j

    # 3) Simulateur portefeuille sur la fenêtre. Frais=0 pour valider le
    #    P&L brut du mark-to-market — l'effet des frais a ses propres tests.
    sim = PortfolioSimulator(fee_bps=0, slippage_bps=0)
    res = sim.run(start=open_date, end=open_date + timedelta(days=31))

    assert res.positions_taken == 1
    # NVDA +20% → equity finale autour de 1.20.
    assert res.equity_curve[-1] == pytest.approx(1.20, abs=0.02)
    assert res.metrics["total_return"] == pytest.approx(0.20, abs=0.02)
    assert res.metrics["sharpe"] is not None
