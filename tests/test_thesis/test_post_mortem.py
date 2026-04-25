"""Tests du `PostMortemAnalyzer` (Phase 3 étape 3)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory.database import (
    Evaluation,
    SignalPerformance,
    Thesis,
    session_scope,
)
from thesis.post_mortem import (
    PostMortemAnalyzer,
    SUCCESS_STATUS,
    TERMINAL_STATUSES,
    _extract_signals,
)


# ------------------------------------------------------------------ helpers


def _seed_thesis(
    *,
    asset_id: str = "NVDA",
    sector_id: str = "ai_ml",
    dimensions: dict[str, float] | None = None,
    score: float = 80.0,
    created_at: datetime | None = None,
    breakdown_override: str | None = None,
) -> int:
    """Insère une thèse minimaliste avec breakdown configurable. Retourne id."""
    dims = dimensions if dimensions is not None else {"momentum": 80.0}
    breakdown = breakdown_override or json.dumps({"dimensions": dims})
    with session_scope() as s:
        th = Thesis(
            created_at=created_at or datetime(2026, 1, 1),
            asset_type="stock",
            asset_id=asset_id,
            sector_id=sector_id,
            score=score,
            score_breakdown_json=breakdown,
            recommendation="BUY",
            horizon_days=180,
            entry_price=100.0,
            entry_conditions_json=None,
            triggers_json=json.dumps([]),
            risks_json=json.dumps([]),
            catalysts_json=json.dumps([]),
            narrative="…",
            model_version="v1_test",
            weights_snapshot_json=json.dumps({}),
        )
        s.add(th)
        s.flush()
        thesis_id = th.id
    return thesis_id


def _seed_eval(
    thesis_id: int,
    *,
    days: int = 365,
    status: str = "success",
    alpha_pct: float | None = 0.10,
) -> None:
    with session_scope() as s:
        s.add(Evaluation(
            thesis_id=thesis_id,
            evaluated_at=datetime(2027, 1, 1),
            days_since_thesis=days,
            current_price=110.0,
            return_pct=0.10,
            benchmark_return_pct=0.0,
            alpha_pct=alpha_pct,
            status=status,
            notes=None,
        ))


def _signal_rows() -> list[SignalPerformance]:
    with session_scope() as s:
        rows = s.query(SignalPerformance).order_by(
            SignalPerformance.signal_name,
            SignalPerformance.sector_id,
            SignalPerformance.horizon_days,
        ).all()
        for r in rows:
            s.expunge(r)
    return rows


def _row_for(
    signal: str, sector: str | None, horizon: int
) -> SignalPerformance | None:
    for r in _signal_rows():
        if r.signal_name == signal and r.sector_id == sector and r.horizon_days == horizon:
            return r
    return None


# ----------------------------------------------------------- pure helpers


def test_extract_signals_returns_dimension_names():
    th = Thesis(
        created_at=datetime(2026, 1, 1),
        asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
        score=80.0,
        score_breakdown_json=json.dumps({
            "dimensions": {"momentum": 80.0, "sentiment": 60.0},
        }),
        recommendation="BUY", horizon_days=180, entry_price=100.0,
        triggers_json="[]", risks_json="[]", catalysts_json="[]",
        narrative="…", model_version="v1", weights_snapshot_json="{}",
    )
    assert sorted(_extract_signals(th)) == ["momentum", "sentiment"]


def test_extract_signals_empty_when_breakdown_malformed():
    th = Thesis(
        created_at=datetime(2026, 1, 1),
        asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
        score=80.0, score_breakdown_json="not a json",
        recommendation="BUY", horizon_days=180, entry_price=100.0,
        triggers_json="[]", risks_json="[]", catalysts_json="[]",
        narrative="…", model_version="v1", weights_snapshot_json="{}",
    )
    assert _extract_signals(th) == []


def test_extract_signals_empty_when_no_dimensions_key():
    th = Thesis(
        created_at=datetime(2026, 1, 1),
        asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
        score=80.0, score_breakdown_json=json.dumps({"other": 1}),
        recommendation="BUY", horizon_days=180, entry_price=100.0,
        triggers_json="[]", risks_json="[]", catalysts_json="[]",
        narrative="…", model_version="v1", weights_snapshot_json="{}",
    )
    assert _extract_signals(th) == []


def test_terminal_statuses_constant():
    assert SUCCESS_STATUS == "success"
    assert set(TERMINAL_STATUSES) == {"success", "failure", "partial"}


# --------------------------------------------------------------- analyzer


def test_run_no_evaluations_returns_zero(tmp_db):
    assert PostMortemAnalyzer().run() == 0
    assert _signal_rows() == []


def test_run_skips_active_evaluations(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0})
    _seed_eval(th, status="active", alpha_pct=None)

    n = PostMortemAnalyzer().run()
    assert n == 0
    assert _signal_rows() == []


def test_run_writes_one_signal_per_dimension(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0, "sentiment": 60.0})
    _seed_eval(th, status="success", alpha_pct=0.20)

    PostMortemAnalyzer().run()
    rows = _signal_rows()
    # 2 signaux × (sector + tous secteurs) = 4 lignes.
    assert len(rows) == 4
    signals = {r.signal_name for r in rows}
    assert signals == {"momentum", "sentiment"}


def test_run_creates_sector_and_all_sectors_rows(tmp_db):
    th = _seed_thesis(
        sector_id="ai_ml", dimensions={"momentum": 80.0}
    )
    _seed_eval(th, status="success", alpha_pct=0.20)

    PostMortemAnalyzer().run()
    sec_row = _row_for("momentum", "ai_ml", 365)
    all_row = _row_for("momentum", None, 365)
    assert sec_row is not None
    assert all_row is not None
    # Avec une seule thèse, sec et all ont les mêmes stats.
    assert sec_row.n_predictions == all_row.n_predictions == 1
    assert sec_row.n_successes == all_row.n_successes == 1
    assert sec_row.accuracy == all_row.accuracy == pytest.approx(1.0)


def test_run_aggregates_multiple_theses_same_sector(tmp_db):
    t1 = _seed_thesis(asset_id="NVDA", dimensions={"momentum": 80.0})
    t2 = _seed_thesis(asset_id="AMD", dimensions={"momentum": 70.0})
    t3 = _seed_thesis(asset_id="GOOGL", dimensions={"momentum": 75.0})
    _seed_eval(t1, status="success", alpha_pct=0.20)
    _seed_eval(t2, status="failure", alpha_pct=-0.15)
    _seed_eval(t3, status="partial", alpha_pct=0.02)

    PostMortemAnalyzer().run()
    row = _row_for("momentum", "ai_ml", 365)
    assert row.n_predictions == 3
    assert row.n_successes == 1
    assert row.accuracy == pytest.approx(1 / 3)
    assert row.avg_alpha == pytest.approx((0.20 - 0.15 + 0.02) / 3)


def test_run_separates_horizons(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0})
    _seed_eval(th, days=180, status="success", alpha_pct=0.10)
    _seed_eval(th, days=365, status="failure", alpha_pct=-0.15)

    PostMortemAnalyzer().run()
    row_180 = _row_for("momentum", "ai_ml", 180)
    row_365 = _row_for("momentum", "ai_ml", 365)
    assert row_180.n_successes == 1
    assert row_365.n_successes == 0


def test_run_separates_sectors(tmp_db):
    t_ai = _seed_thesis(
        asset_id="NVDA", sector_id="ai_ml", dimensions={"momentum": 80.0}
    )
    t_bio = _seed_thesis(
        asset_id="MRNA", sector_id="biotech", dimensions={"momentum": 80.0}
    )
    _seed_eval(t_ai, status="success", alpha_pct=0.20)
    _seed_eval(t_bio, status="failure", alpha_pct=-0.10)

    PostMortemAnalyzer().run()
    ai = _row_for("momentum", "ai_ml", 365)
    bio = _row_for("momentum", "biotech", 365)
    all_ = _row_for("momentum", None, 365)
    assert ai.n_successes == 1 and ai.accuracy == pytest.approx(1.0)
    assert bio.n_successes == 0 and bio.accuracy == pytest.approx(0.0)
    # « Tous secteurs » agrège : 2 prédictions, 1 succès.
    assert all_.n_predictions == 2
    assert all_.n_successes == 1
    assert all_.accuracy == pytest.approx(0.5)


def test_run_avg_alpha_skips_none(tmp_db):
    t1 = _seed_thesis(asset_id="NVDA", dimensions={"momentum": 80.0})
    t2 = _seed_thesis(asset_id="AMD", dimensions={"momentum": 70.0})
    _seed_eval(t1, status="success", alpha_pct=0.20)
    _seed_eval(t2, status="failure", alpha_pct=None)  # alpha indisponible

    PostMortemAnalyzer().run()
    row = _row_for("momentum", "ai_ml", 365)
    assert row.n_predictions == 2
    # avg_alpha calculé sur la seule thèse avec alpha disponible.
    assert row.avg_alpha == pytest.approx(0.20)


def test_run_avg_alpha_none_when_all_alphas_missing(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0})
    _seed_eval(th, status="failure", alpha_pct=None)

    PostMortemAnalyzer().run()
    row = _row_for("momentum", "ai_ml", 365)
    assert row.n_predictions == 1
    assert row.avg_alpha is None


def test_run_idempotent(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0})
    _seed_eval(th, status="success", alpha_pct=0.10)

    pm = PostMortemAnalyzer()
    n1 = pm.run()
    n2 = pm.run()
    assert n1 == n2
    rows = _signal_rows()
    # 1 signal × 2 (secteur + all) = 2 lignes, pas de doublon malgré 2 runs.
    assert len(rows) == 2


def test_run_recomputes_on_new_evaluation(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0})
    _seed_eval(th, status="success", alpha_pct=0.10)
    PostMortemAnalyzer().run()
    assert _row_for("momentum", "ai_ml", 365).n_predictions == 1

    # Nouvelle thèse + éval → re-run doit refléter le nouvel agrégat.
    t2 = _seed_thesis(asset_id="AMD", dimensions={"momentum": 70.0})
    _seed_eval(t2, status="failure", alpha_pct=-0.05)
    PostMortemAnalyzer().run()
    row = _row_for("momentum", "ai_ml", 365)
    assert row.n_predictions == 2
    assert row.n_successes == 1


def test_run_skips_thesis_with_malformed_breakdown(tmp_db):
    """Une thèse cassée ne fait pas tomber le batch ; elle ne contribue pas."""
    bad = _seed_thesis(
        asset_id="BAD",
        dimensions=None,
        breakdown_override="not a json",
    )
    good = _seed_thesis(asset_id="NVDA", dimensions={"momentum": 80.0})
    _seed_eval(bad, status="success", alpha_pct=0.10)
    _seed_eval(good, status="success", alpha_pct=0.20)

    PostMortemAnalyzer().run()
    row = _row_for("momentum", "ai_ml", 365)
    assert row.n_predictions == 1  # bad thesis exclue
    assert row.avg_alpha == pytest.approx(0.20)


def test_run_handles_empty_dimensions(tmp_db):
    """Une thèse avec dimensions={} ne contribue à aucun signal."""
    th = _seed_thesis(
        breakdown_override=json.dumps({"dimensions": {}}),
    )
    _seed_eval(th, status="success", alpha_pct=0.10)

    n = PostMortemAnalyzer().run()
    assert n == 0
    assert _signal_rows() == []


def test_invalid_terminal_statuses_raise():
    with pytest.raises(ValueError):
        PostMortemAnalyzer(terminal_statuses=())


def test_run_uses_custom_terminal_statuses(tmp_db):
    th = _seed_thesis(dimensions={"momentum": 80.0})
    _seed_eval(th, status="success", alpha_pct=0.10)
    _seed_eval(th, days=180, status="partial", alpha_pct=0.02)

    # On ne compte que les success — les partial sont ignorés.
    PostMortemAnalyzer(terminal_statuses=("success",)).run()
    row_365 = _row_for("momentum", "ai_ml", 365)
    row_180 = _row_for("momentum", "ai_ml", 180)
    assert row_365 is not None and row_365.n_predictions == 1
    assert row_180 is None
