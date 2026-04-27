"""Tests du walk-forward (Phase 5 étape 3)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Iterable
from unittest.mock import MagicMock

import pytest

from backtesting.walk_forward import (
    SELECTION_METRICS,
    FoldResult,
    WalkForwardResult,
    _select_best_weight,
    run_walk_forward,
    split_folds,
)
from memory.database import Thesis


# ---------------------------------------------------------------- helpers


def _make_thesis(
    *,
    asset_id: str = "NVDA",
    created_at: datetime,
    entry_price: float = 100.0,
    horizon_days: int = 10,
    recommendation: str = "BUY",
) -> Thesis:
    return Thesis(
        created_at=created_at,
        asset_type="stock", asset_id=asset_id, sector_id="ai_ml",
        score=80.0,
        score_breakdown_json="{}",
        recommendation=recommendation,
        horizon_days=horizon_days, entry_price=entry_price,
        triggers_json="[]", risks_json="[]", catalysts_json="[]",
        narrative="…", model_version="v_test", weights_snapshot_json="{}",
    )


# --------------------------------------------------------- split_folds


def test_split_folds_creates_n_folds():
    splits = split_folds(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 12, 31),
        n_folds=3,
    )
    assert len(splits) == 3


def test_split_folds_train_precedes_test():
    splits = split_folds(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 12, 31),
        n_folds=3,
    )
    for tr_s, tr_e, te_s, te_e in splits:
        assert tr_s < tr_e <= te_s < te_e


def test_split_folds_test_ratio_respected():
    splits = split_folds(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 1, 11),   # 10 jours
        n_folds=1,
        test_ratio=0.2,
    )
    tr_s, tr_e, te_s, te_e = splits[0]
    train_dur = (tr_e - tr_s).total_seconds()
    test_dur = (te_e - te_s).total_seconds()
    total = train_dur + test_dur
    assert test_dur / total == pytest.approx(0.2, rel=1e-9)


def test_split_folds_validates_window():
    with pytest.raises(ValueError):
        split_folds(start=datetime(2026, 4, 5), end=datetime(2026, 4, 1),
                    n_folds=2)


def test_split_folds_validates_n_folds():
    with pytest.raises(ValueError):
        split_folds(start=datetime(2026, 1, 1), end=datetime(2026, 12, 31),
                    n_folds=0)


def test_split_folds_validates_test_ratio():
    with pytest.raises(ValueError):
        split_folds(start=datetime(2026, 1, 1), end=datetime(2026, 12, 31),
                    n_folds=3, test_ratio=0.0)
    with pytest.raises(ValueError):
        split_folds(start=datetime(2026, 1, 1), end=datetime(2026, 12, 31),
                    n_folds=3, test_ratio=1.0)


def test_split_folds_progress_in_time():
    """Les folds successifs avancent dans le temps (walk-forward)."""
    splits = split_folds(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
        n_folds=3,
    )
    for prev, curr in zip(splits, splits[1:]):
        assert prev[0] < curr[0]   # train_start avance
        assert prev[3] <= curr[3]  # test_end avance


# ------------------------------------------------- _select_best_weight


def test_select_best_weight_picks_max():
    metrics = {
        "v1": {"sharpe": 1.0},
        "v2": {"sharpe": 2.5},
        "v3": {"sharpe": 0.5},
    }
    assert _select_best_weight(["v1", "v2", "v3"], metrics, "sharpe") == "v2"


def test_select_best_weight_treats_none_as_minus_inf():
    metrics = {
        "v1": {"sharpe": None},
        "v2": {"sharpe": 0.1},
    }
    assert _select_best_weight(["v1", "v2"], metrics, "sharpe") == "v2"


def test_select_best_weight_all_none_returns_first():
    metrics = {
        "v1": {"sharpe": None},
        "v2": {"sharpe": None},
    }
    # Tous égaux à -inf, le premier dans `candidates` gagne.
    assert _select_best_weight(["v1", "v2"], metrics, "sharpe") == "v1"


def test_select_best_weight_empty_candidates_returns_none():
    assert _select_best_weight([], {}, "sharpe") is None


# ------------------------------------------------------------- run_walk_forward


def test_run_walk_forward_calls_build_theses_per_fold_per_weight():
    calls: list[tuple[str, datetime, datetime]] = []

    def build_theses(weight, start, end):
        calls.append((weight, start, end))
        return []

    run_walk_forward(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
        n_folds=3,
        weight_candidates=["v1", "v2"],
        build_theses=build_theses,
    )
    # 3 folds × 2 candidats sur train = 6 + N test calls (1 par fold gagnant).
    # Avec aucune thèse (returns 0), pas de Sharpe → all None → premier candidat gagne.
    # Donc 3 calls test pour "v1" + 6 calls train.
    assert len(calls) == 9
    # Au moins toutes les combinaisons train sont présentes.
    weights_called = {(w, s) for w, s, _ in calls}
    assert any(w == "v1" for w, _ in weights_called)
    assert any(w == "v2" for w, _ in weights_called)


def test_run_walk_forward_picks_winner_with_best_metric():
    """Construire des thèses telles que v2 surperforme v1 sur le train."""
    open_dates = {
        "v1": datetime(2026, 1, 5),
        "v2": datetime(2026, 1, 5),
    }

    def build_theses(weight, start, end):
        # v2 retourne 1 thèse fictive ; v1 n'en retourne pas.
        if weight == "v2" and open_dates[weight] >= start \
                and open_dates[weight] < end:
            return [_make_thesis(created_at=open_dates[weight])]
        return []

    fake_sim = MagicMock()
    # Mocker les métriques : v1 → Sharpe None, v2 → Sharpe 1.5
    def fake_run(*, start, end, theses):
        from backtesting.portfolio import PortfolioResult
        if theses:
            return PortfolioResult(
                dates=[start], equity_curve=[1.05], metrics={"sharpe": 1.5},
            )
        return PortfolioResult(
            dates=[start], equity_curve=[1.0], metrics={"sharpe": None},
        )
    fake_sim.run.side_effect = fake_run

    res = run_walk_forward(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
        n_folds=2,
        weight_candidates=["v1", "v2"],
        build_theses=build_theses,
        portfolio_simulator=fake_sim,
    )
    # Le 1er fold (jan-fev) doit retenir v2 (la thèse v2 tombe dans le train).
    fold0 = res.folds[0]
    assert fold0.best_weights == "v2"


def test_run_walk_forward_isolates_failing_train():
    """Une exception sur un candidat n'empêche pas l'évaluation des autres."""
    def build_theses(weight, start, end):
        if weight == "boom":
            raise RuntimeError("boom")
        return []

    fake_sim = MagicMock()
    from backtesting.portfolio import PortfolioResult
    fake_sim.run.return_value = PortfolioResult(
        dates=[datetime(2026, 1, 1)], equity_curve=[1.0],
        metrics={"sharpe": 0.1},
    )

    res = run_walk_forward(
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
        n_folds=2,
        weight_candidates=["boom", "good"],
        build_theses=build_theses,
        portfolio_simulator=fake_sim,
    )
    # `good` doit gagner partout (boom est invalide).
    assert all(f.best_weights == "good" for f in res.folds)


def test_run_walk_forward_validates_args():
    def noop(w, s, e): return []
    with pytest.raises(ValueError):
        run_walk_forward(
            start=datetime(2026, 1, 1), end=datetime(2026, 4, 1),
            n_folds=2, weight_candidates=[], build_theses=noop,
        )
    with pytest.raises(ValueError):
        run_walk_forward(
            start=datetime(2026, 1, 1), end=datetime(2026, 4, 1),
            n_folds=2, weight_candidates=["v1"],
            build_theses=noop, selection_metric="bogus",
        )


def test_run_walk_forward_records_train_metrics_per_weight():
    def build_theses(w, s, e):
        return []

    fake_sim = MagicMock()
    from backtesting.portfolio import PortfolioResult
    fake_sim.run.return_value = PortfolioResult(
        dates=[datetime(2026, 1, 1)], equity_curve=[1.0],
        metrics={"sharpe": 0.5, "total_return": 0.1},
    )

    res = run_walk_forward(
        start=datetime(2026, 1, 1), end=datetime(2026, 7, 1),
        n_folds=2, weight_candidates=["v1", "v2"],
        build_theses=build_theses,
        portfolio_simulator=fake_sim,
    )
    for fold in res.folds:
        assert "v1" in fold.train_metrics
        assert "v2" in fold.train_metrics
        assert fold.train_metrics["v1"]["sharpe"] == 0.5


def test_walk_forward_result_average_test_metric():
    res = WalkForwardResult(selection_metric="sharpe")
    res.folds = [
        FoldResult(
            fold_index=0,
            train_start=datetime(2026, 1, 1), train_end=datetime(2026, 2, 1),
            test_start=datetime(2026, 2, 1), test_end=datetime(2026, 3, 1),
            test_metrics={"sharpe": 1.0},
        ),
        FoldResult(
            fold_index=1,
            train_start=datetime(2026, 3, 1), train_end=datetime(2026, 4, 1),
            test_start=datetime(2026, 4, 1), test_end=datetime(2026, 5, 1),
            test_metrics={"sharpe": 2.0},
        ),
        FoldResult(
            fold_index=2,
            train_start=datetime(2026, 5, 1), train_end=datetime(2026, 6, 1),
            test_start=datetime(2026, 6, 1), test_end=datetime(2026, 7, 1),
            test_metrics={"sharpe": None},   # ignoré
        ),
    ]
    assert res.average_test_metric("sharpe") == pytest.approx(1.5)
    assert res.average_test_metric("inexistant") is None


def test_walk_forward_result_best_per_fold():
    res = WalkForwardResult()
    res.folds = [
        FoldResult(fold_index=0,
                   train_start=datetime(2026, 1, 1), train_end=datetime(2026, 2, 1),
                   test_start=datetime(2026, 2, 1), test_end=datetime(2026, 3, 1),
                   best_weights="v1"),
        FoldResult(fold_index=1,
                   train_start=datetime(2026, 3, 1), train_end=datetime(2026, 4, 1),
                   test_start=datetime(2026, 4, 1), test_end=datetime(2026, 5, 1),
                   best_weights="v2"),
    ]
    assert res.best_per_fold() == [(0, "v1"), (1, "v2")]


def test_selection_metrics_constant_includes_core():
    assert "sharpe" in SELECTION_METRICS
    assert "total_return" in SELECTION_METRICS
    assert "alpha_vs_benchmark" in SELECTION_METRICS
