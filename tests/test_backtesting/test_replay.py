"""Tests du `HistoricalReplay` (Phase 5 étape 2)."""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any
from unittest.mock import MagicMock

import pytest

from backtesting.replay import HistoricalReplay, _daily_grid


# ---------------------------------------------------------------- helpers


class _FakeScorer:
    """Faux BaseFeature : enregistre les appels, retourne un compteur fixe."""

    def __init__(self, name: str = "fake", n_per_call: int = 1) -> None:
        self.feature_name = name
        self.target_type = "asset"
        self.calls: list[datetime] = []
        self._n = n_per_call

    def run(self, as_of: datetime | None = None) -> int:
        self.calls.append(as_of)
        return self._n


class _RaisingScorer:
    """Faux scorer qui lève à chaque appel."""

    def __init__(self) -> None:
        self.feature_name = "boom"

    def run(self, as_of: datetime | None = None) -> int:
        raise RuntimeError("boom")


class _FakeGenerator:
    """Faux thesis_generator : enregistre les appels."""

    def __init__(self, n_per_call: int = 1) -> None:
        self.calls: list[datetime] = []
        self._n = n_per_call

    def run(self, as_of: datetime | None = None) -> int:
        self.calls.append(as_of)
        return self._n


# ----------------------------------------------------------- _daily_grid


def test_daily_grid_step_1():
    out = _daily_grid(datetime(2026, 4, 1), datetime(2026, 4, 5))
    assert len(out) == 4
    assert out[0] == datetime(2026, 4, 1)
    assert out[-1] == datetime(2026, 4, 4)


def test_daily_grid_step_2():
    out = _daily_grid(
        datetime(2026, 4, 1), datetime(2026, 4, 11), step_days=2,
    )
    assert out == [
        datetime(2026, 4, 1),
        datetime(2026, 4, 3),
        datetime(2026, 4, 5),
        datetime(2026, 4, 7),
        datetime(2026, 4, 9),
    ]


def test_daily_grid_empty_when_end_before_start():
    assert _daily_grid(datetime(2026, 4, 5), datetime(2026, 4, 1)) == []


# -------------------------------------------------- HistoricalReplay


def test_run_validates_window():
    r = HistoricalReplay()
    with pytest.raises(ValueError):
        r.run(start=datetime(2026, 4, 5), end=datetime(2026, 4, 1))


def test_run_validates_step_days():
    r = HistoricalReplay()
    with pytest.raises(ValueError):
        r.run(
            start=datetime(2026, 4, 1),
            end=datetime(2026, 4, 5),
            step_days=0,
        )


def test_run_calls_scorers_for_each_date():
    s1 = _FakeScorer("s1", n_per_call=2)
    s2 = _FakeScorer("s2", n_per_call=3)
    r = HistoricalReplay(scorers=[s1, s2])
    res = r.run(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 4),
    )
    # 3 dates × 2 scorers chacun.
    assert res.n_dates == 3
    assert len(s1.calls) == 3
    assert len(s2.calls) == 3
    assert res.n_features_inserted == 3 * (2 + 3)


def test_run_preserves_scorer_order():
    """Les scorers sont appelés dans l'ordre passé (séquentiel par date)."""
    log: list[str] = []

    class _Tracking:
        feature_name = "x"

        def __init__(self, label: str) -> None:
            self.label = label

        def run(self, as_of):
            log.append(self.label)
            return 0

    a = _Tracking("A")
    b = _Tracking("B")
    HistoricalReplay(scorers=[a, b]).run(
        start=datetime(2026, 4, 1), end=datetime(2026, 4, 3),
    )
    assert log == ["A", "B", "A", "B"]


def test_run_calls_thesis_generator_for_each_date():
    gen = _FakeGenerator(n_per_call=4)
    r = HistoricalReplay(thesis_generator=gen)
    res = r.run(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 4),
    )
    assert len(gen.calls) == 3
    assert res.n_theses_created == 12


def test_run_skips_thesis_generator_when_none():
    s = _FakeScorer()
    r = HistoricalReplay(scorers=[s], thesis_generator=None)
    res = r.run(
        start=datetime(2026, 4, 1), end=datetime(2026, 4, 3),
    )
    # Pas d'erreur — thesis_generator absent simplement skippé.
    assert res.n_theses_created == 0


def test_run_isolates_scorer_exceptions():
    """Une règle qui crashe ne tue pas le replay."""
    bad = _RaisingScorer()
    good = _FakeScorer()
    r = HistoricalReplay(scorers=[bad, good])
    res = r.run(
        start=datetime(2026, 4, 1), end=datetime(2026, 4, 4),
    )
    # 3 dates × 1 erreur sur bad + 3 calls sur good.
    assert res.n_errors == 3
    assert len(good.calls) == 3
    assert res.n_features_inserted == 3


def test_run_isolates_thesis_generator_exceptions():
    gen = MagicMock()
    gen.run.side_effect = RuntimeError("boom")
    r = HistoricalReplay(thesis_generator=gen)
    res = r.run(
        start=datetime(2026, 4, 1), end=datetime(2026, 4, 3),
    )
    assert res.n_errors == 2   # 2 dates → 2 erreurs


def test_run_step_days_reduces_dates():
    s = _FakeScorer()
    r = HistoricalReplay(scorers=[s])
    res = r.run(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 11),
        step_days=2,
    )
    # 5 dates: 1, 3, 5, 7, 9.
    assert res.n_dates == 5
    assert len(s.calls) == 5


def test_run_returns_zero_when_window_is_empty_grid():
    """Une fenêtre [d, d+step) donne 1 date ; (d, d) est rejeté en haut."""
    s = _FakeScorer()
    r = HistoricalReplay(scorers=[s])
    res = r.run(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 1) + timedelta(days=1),
    )
    assert res.n_dates == 1


def test_replay_result_summary_fields():
    s = _FakeScorer(n_per_call=2)
    gen = _FakeGenerator(n_per_call=1)
    r = HistoricalReplay(scorers=[s], thesis_generator=gen)
    res = r.run(
        start=datetime(2026, 4, 1),
        end=datetime(2026, 4, 4),
    )
    assert res.start == datetime(2026, 4, 1)
    assert res.end == datetime(2026, 4, 4)
    assert res.step_days == 1
    assert res.n_dates == 3
    assert res.n_features_inserted == 6
    assert res.n_theses_created == 3


# ----------------------------------------------------- intégration légère


def test_replay_with_real_thesis_generator(tmp_db):
    """Replay end-to-end : un faux scorer alimente une feature stock_score
    → le ThesisGenerator réel doit générer une thèse à la date passée."""
    import json

    from memory.database import Feature, Thesis, session_scope
    from thesis.generator import ThesisGenerator

    # Seed d'un score de 90/100 pour NVDA à plusieurs dates.
    for day in range(1, 4):
        with session_scope() as s:
            s.add(Feature(
                feature_name="stock_score",
                target_type="asset", target_id="NVDA",
                computed_at=datetime(2026, 4, day, 1, 0),
                value=90.0,
                metadata_json=json.dumps({
                    "model_version": "bt_test",
                    "weights": {"momentum": 1.0},
                    "dimensions": {"momentum": 90.0},
                    "details": {},
                }),
            ))

    gen = ThesisGenerator(tickers=["NVDA"])
    r = HistoricalReplay(thesis_generator=gen)
    res = r.run(
        start=datetime(2026, 4, 1, 12, 0),
        end=datetime(2026, 4, 4, 12, 0),
    )
    # Une thèse par jour (3 jours, idempotence par jour UTC).
    assert res.n_theses_created == 3
    with session_scope() as s:
        n = s.query(Thesis).count()
    assert n == 3
