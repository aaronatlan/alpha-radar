"""Tests du `PortfolioSimulator` (Phase 5 étape 1)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from backtesting.portfolio import PortfolioSimulator, _daily_grid
from memory.database import RawData, Thesis, session_scope


# ------------------------------------------------------------------ helpers


def _seed_thesis(
    *,
    asset_id: str = "NVDA",
    created_at: datetime,
    entry_price: float | None = 100.0,
    horizon_days: int = 30,
    recommendation: str = "BUY",
    score: float = 80.0,
) -> int:
    with session_scope() as s:
        th = Thesis(
            created_at=created_at,
            asset_type="stock",
            asset_id=asset_id,
            sector_id="ai_ml",
            score=score,
            score_breakdown_json=json.dumps({"dimensions": {"momentum": score}}),
            recommendation=recommendation,
            horizon_days=horizon_days,
            entry_price=entry_price,
            entry_conditions_json=None,
            triggers_json="[]",
            risks_json="[]",
            catalysts_json="[]",
            narrative="…",
            model_version="v1_test",
            weights_snapshot_json="{}",
        )
        s.add(th)
        s.flush()
        thesis_id = th.id
    return thesis_id


def _seed_close(
    ticker: str, close: float, content_at: datetime,
    fetched_at: datetime | None = None,
) -> None:
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
            hash=f"h-{ticker}-{content_at.strftime('%Y-%m-%d-%H-%M')}",
        ))


# ----------------------------------------------------------- _daily_grid


def test_daily_grid_inclusive_start_exclusive_end():
    out = _daily_grid(datetime(2026, 4, 1), datetime(2026, 4, 5))
    assert out == [
        datetime(2026, 4, 1),
        datetime(2026, 4, 2),
        datetime(2026, 4, 3),
        datetime(2026, 4, 4),
    ]


def test_daily_grid_empty_when_end_before_start():
    out = _daily_grid(datetime(2026, 4, 5), datetime(2026, 4, 1))
    assert out == []


# ----------------------------------------------------------- constructor


def test_simulator_validates_capital():
    with pytest.raises(ValueError):
        PortfolioSimulator(initial_capital=0)
    with pytest.raises(ValueError):
        PortfolioSimulator(initial_capital=-100)


def test_simulator_validates_recommendations():
    with pytest.raises(ValueError):
        PortfolioSimulator(accepted_recommendations=())


def test_run_validates_window():
    sim = PortfolioSimulator()
    with pytest.raises(ValueError):
        sim.run(start=datetime(2026, 4, 5), end=datetime(2026, 4, 1))


# ----------------------------------------------------------- simulation


def test_run_no_theses_returns_flat_curve(tmp_db):
    sim = PortfolioSimulator(initial_capital=1.0)
    res = sim.run(start=datetime(2026, 4, 1), end=datetime(2026, 4, 11))
    assert res.positions_taken == 0
    assert res.equity_curve == [1.0] * 10
    assert res.metrics["total_return"] == 0.0


def test_run_opens_position_on_buy_thesis(tmp_db):
    """Une thèse BUY avec entry valide ouvre une position et le P&L
    est calculé via les closes successifs."""
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, horizon_days=5)
    # Close à 100 le jour d'ouverture, +10% au jour suivant, plat ensuite.
    _seed_close("NVDA", 100.0, open_date)
    _seed_close("NVDA", 110.0, open_date + timedelta(days=1))
    _seed_close("NVDA", 110.0, open_date + timedelta(days=2))

    # Frais à zéro ici pour vérifier le mark-to-market pur. Un test dédié
    # plus bas vérifie l'effet des frais.
    sim = PortfolioSimulator(initial_capital=1.0, fee_bps=0, slippage_bps=0)
    res = sim.run(start=open_date, end=open_date + timedelta(days=3))
    assert res.positions_taken == 1
    # Equity avant prise de position = 1.0, après +10% le 2e jour = 1.10.
    assert res.equity_curve[-1] == pytest.approx(1.10, rel=1e-9)


def test_run_skips_avoid_recommendation(tmp_db):
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, recommendation="AVOID")
    sim = PortfolioSimulator()
    res = sim.run(start=open_date, end=open_date + timedelta(days=5))
    assert res.positions_taken == 0


def test_run_skips_thesis_without_entry_price(tmp_db):
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date, entry_price=None)
    sim = PortfolioSimulator()
    res = sim.run(start=open_date, end=open_date + timedelta(days=5))
    assert res.positions_taken == 0


def test_run_closes_position_after_horizon(tmp_db):
    """Après horizon_days, la position cesse d'impacter l'equity."""
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, horizon_days=2)
    _seed_close("NVDA", 100.0, open_date)
    _seed_close("NVDA", 110.0, open_date + timedelta(days=1))   # +10%
    _seed_close("NVDA", 200.0, open_date + timedelta(days=3))   # post-close

    sim = PortfolioSimulator(fee_bps=0, slippage_bps=0)
    res = sim.run(start=open_date, end=open_date + timedelta(days=5))
    # Position fermée le 3e jour → equity finale ≈ 1.10 (pas 2.0).
    assert res.equity_curve[-1] == pytest.approx(1.10, rel=1e-2)


def test_run_equipondere_multiple_positions(tmp_db):
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="A", created_at=open_date,
                 entry_price=100.0, horizon_days=3)
    _seed_thesis(asset_id="B", created_at=open_date,
                 entry_price=100.0, horizon_days=3)
    _seed_close("A", 100.0, open_date)
    _seed_close("B", 100.0, open_date)
    # A monte +20%, B descend -10% → moyenne +5%.
    _seed_close("A", 120.0, open_date + timedelta(days=1))
    _seed_close("B", 90.0, open_date + timedelta(days=1))

    sim = PortfolioSimulator(fee_bps=0, slippage_bps=0)
    res = sim.run(start=open_date, end=open_date + timedelta(days=2))
    # Equity = 1.0 * 1.05.
    assert res.equity_curve[-1] == pytest.approx(1.05, rel=1e-2)


def test_run_uses_explicit_theses_param(tmp_db):
    """Si on passe `theses=` explicitement, on ne lit pas la base."""
    sim = PortfolioSimulator()
    # Aucun seed en base, mais on fournit la thèse en argument.
    th = Thesis(
        created_at=datetime(2026, 4, 1),
        asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
        score=80.0,
        score_breakdown_json="{}",
        recommendation="BUY", horizon_days=2, entry_price=100.0,
        triggers_json="[]", risks_json="[]", catalysts_json="[]",
        narrative="…", model_version="v1", weights_snapshot_json="{}",
    )
    res = sim.run(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 5),
        theses=[th],
    )
    assert res.positions_taken == 1


def test_run_metrics_populated(tmp_db):
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, horizon_days=10)
    _seed_close("NVDA", 100.0, open_date)
    _seed_close("NVDA", 105.0, open_date + timedelta(days=1))
    _seed_close("NVDA", 108.0, open_date + timedelta(days=2))
    _seed_close("NVDA", 110.0, open_date + timedelta(days=3))

    sim = PortfolioSimulator()
    res = sim.run(start=open_date, end=open_date + timedelta(days=4))
    m = res.metrics
    assert "total_return" in m
    assert "sharpe" in m
    assert "max_drawdown" in m
    assert m["total_return"] > 0


def test_run_with_benchmark_computes_alpha(tmp_db):
    open_date = datetime(2026, 4, 1)
    # Portefeuille : NVDA +10%
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, horizon_days=10)
    _seed_close("NVDA", 100.0, open_date)
    _seed_close("NVDA", 110.0, open_date + timedelta(days=1))
    # Benchmark : SPY +2%
    _seed_close("SPY", 400.0, open_date)
    _seed_close("SPY", 408.0, open_date + timedelta(days=1))

    sim = PortfolioSimulator(benchmark_tickers=["SPY"])
    res = sim.run(start=open_date, end=open_date + timedelta(days=2))
    assert res.benchmark_curve
    assert res.metrics["alpha_vs_benchmark"] is not None
    assert res.metrics["alpha_vs_benchmark"] > 0


def test_run_loads_theses_from_db_when_unspecified(tmp_db):
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date, entry_price=100.0)
    _seed_thesis(asset_id="AMD", created_at=open_date, entry_price=200.0)

    sim = PortfolioSimulator()
    res = sim.run(start=open_date, end=open_date + timedelta(days=3))
    # Les 2 thèses doivent être chargées et donner 2 positions.
    assert res.positions_taken == 2


def test_run_filters_theses_outside_window(tmp_db):
    """Une thèse hors fenêtre ne doit pas ouvrir de position."""
    _seed_thesis(asset_id="NVDA", created_at=datetime(2026, 1, 1),
                 entry_price=100.0)
    sim = PortfolioSimulator()
    res = sim.run(start=datetime(2026, 4, 1), end=datetime(2026, 4, 10))
    assert res.positions_taken == 0


# ----------------------------------------------------------- frais / slippage


def test_constructor_rejects_negative_costs():
    with pytest.raises(ValueError):
        PortfolioSimulator(fee_bps=-1)
    with pytest.raises(ValueError):
        PortfolioSimulator(slippage_bps=-0.01)


def test_zero_costs_match_pre_fees_baseline(tmp_db):
    """fee_bps=0 + slippage_bps=0 : equity exactement le P&L sous-jacent."""
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, horizon_days=5)
    _seed_close("NVDA", 100.0, open_date)
    _seed_close("NVDA", 110.0, open_date + timedelta(days=1))
    sim = PortfolioSimulator(fee_bps=0, slippage_bps=0)
    res = sim.run(start=open_date, end=open_date + timedelta(days=2))
    assert res.equity_curve[-1] == pytest.approx(1.10, rel=1e-9)


def test_costs_reduce_equity_for_single_position(tmp_db):
    """Sur 1 position seule, drag = 2 × cost_per_side. Avec
    slippage_bps=10 et fee_bps=5 → 2 × 15 bps = 30 bps = 0.003."""
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="NVDA", created_at=open_date,
                 entry_price=100.0, horizon_days=5)
    _seed_close("NVDA", 100.0, open_date)
    _seed_close("NVDA", 110.0, open_date + timedelta(days=1))

    sim_no = PortfolioSimulator(fee_bps=0, slippage_bps=0)
    sim_yes = PortfolioSimulator(fee_bps=5, slippage_bps=10)

    eq_no = sim_no.run(
        start=open_date, end=open_date + timedelta(days=2)
    ).equity_curve[-1]
    eq_yes = sim_yes.run(
        start=open_date, end=open_date + timedelta(days=2)
    ).equity_curve[-1]

    expected_drag = 2 * (5 + 10) / 10_000.0  # 30 bps round-trip
    assert eq_yes == pytest.approx(eq_no * (1 - expected_drag), rel=1e-9)


def test_costs_dilute_with_more_positions(tmp_db):
    """Le drag par position est pondéré par 1/N_active : 2 positions
    simultanées → drag/position diminue de moitié, mais on en paye 2."""
    open_date = datetime(2026, 4, 1)
    _seed_thesis(asset_id="A", created_at=open_date,
                 entry_price=100.0, horizon_days=5)
    _seed_thesis(asset_id="B", created_at=open_date,
                 entry_price=100.0, horizon_days=5)
    _seed_close("A", 100.0, open_date)
    _seed_close("B", 100.0, open_date)
    # Pas de mouvement de prix : on isole l'effet des frais.
    sim = PortfolioSimulator(fee_bps=0, slippage_bps=10)
    res = sim.run(start=open_date, end=open_date + timedelta(days=1))
    # 2 positions, weight = 0.5 chacune, drag = 2 × 0.5 × 2 × 10 bps
    # = 2 × 0.001 = 0.002 (somme = même drag que 1 position seule à 10 bps×2).
    assert res.equity_curve[0] == pytest.approx(1 - 0.002, rel=1e-9)


def test_default_costs_are_realistic(tmp_db):
    """Sanity : par défaut, on a un slippage non nul. Un Sharpe pur
    sans frais aurait été optimiste — c'est précisément ce que ces
    defaults corrigent."""
    sim = PortfolioSimulator()
    assert sim._cost_per_side > 0
