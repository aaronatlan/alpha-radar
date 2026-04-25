"""Tests du `ThesisEvaluator` (Phase 3 étape 2)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory.database import Evaluation, RawData, Thesis, session_scope
from thesis.evaluator import (
    MILESTONES_DAYS,
    ThesisEvaluator,
    _benchmark_return,
    _classify_status,
    _peers_for,
)


# ------------------------------------------------------------ helpers


def _seed_thesis(
    asset_id: str,
    *,
    created_at: datetime,
    sector_id: str = "ai_ml",
    entry_price: float | None = 100.0,
    score: float = 80.0,
    recommendation: str = "BUY",
) -> int:
    """Insère une thèse minimaliste pour le test. Retourne son id."""
    with session_scope() as s:
        th = Thesis(
            created_at=created_at,
            asset_type="stock",
            asset_id=asset_id,
            sector_id=sector_id,
            score=score,
            score_breakdown_json=json.dumps({"dimensions": {"momentum": score}}),
            recommendation=recommendation,
            horizon_days=180,
            entry_price=entry_price,
            entry_conditions_json=None,
            triggers_json=json.dumps([]),
            risks_json=json.dumps([]),
            catalysts_json=json.dumps([]),
            narrative="…",
            model_version="v1_test",
            weights_snapshot_json=json.dumps({"momentum": 1.0}),
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
            hash=f"h-{ticker}-{content_at.strftime('%Y-%m-%d-%H')}",
        ))


def _evaluations_for(thesis_id: int) -> list[Evaluation]:
    with session_scope() as s:
        rows = (
            s.query(Evaluation)
            .filter_by(thesis_id=thesis_id)
            .order_by(Evaluation.days_since_thesis.asc())
            .all()
        )
        for r in rows:
            s.expunge(r)
    return rows


# ---------------------------------------------------- pure helpers


def test_classify_status_active_before_180():
    assert _classify_status(30, alpha_pct=0.20) == "active"
    assert _classify_status(90, alpha_pct=-0.30) == "active"


def test_classify_status_thresholds_at_180():
    assert _classify_status(180, alpha_pct=0.10) == "success"
    assert _classify_status(180, alpha_pct=-0.10) == "failure"
    assert _classify_status(180, alpha_pct=0.02) == "partial"
    assert _classify_status(180, alpha_pct=-0.02) == "partial"
    # Bornes strictes : exactement +5% n'est pas success.
    assert _classify_status(180, alpha_pct=0.05) == "partial"


def test_classify_status_active_when_alpha_unknown():
    assert _classify_status(365, alpha_pct=None) == "active"


def test_peers_excludes_self():
    peers = _peers_for("NVDA", "ai_ml")
    assert "NVDA" not in peers
    assert "AMD" in peers  # même secteur dans la watchlist


def test_peers_returns_empty_for_unknown_sector():
    assert _peers_for("NVDA", "made_up_sector") == []


def test_benchmark_return_averages_peers(tmp_db):
    t0 = datetime(2026, 1, 1)
    t1 = t0 + timedelta(days=30)
    _seed_close("AMD", 100.0, t0)
    _seed_close("AMD", 110.0, t1)            # +10%
    _seed_close("GOOGL", 200.0, t0)
    _seed_close("GOOGL", 180.0, t1)          # -10%
    avg = _benchmark_return(["AMD", "GOOGL"], t0, t1)
    assert avg == pytest.approx(0.0, abs=1e-9)


def test_benchmark_return_skips_peers_with_missing_prices(tmp_db):
    t0 = datetime(2026, 1, 1)
    t1 = t0 + timedelta(days=30)
    _seed_close("AMD", 100.0, t0)
    _seed_close("AMD", 120.0, t1)            # +20% — seul peer valide
    # GOOGL : pas de prix → skippé.
    avg = _benchmark_return(["AMD", "GOOGL"], t0, t1)
    assert avg == pytest.approx(0.20, rel=1e-6)


def test_benchmark_return_none_when_no_valid_peers(tmp_db):
    t0 = datetime(2026, 1, 1)
    t1 = t0 + timedelta(days=30)
    assert _benchmark_return(["AMD", "GOOGL"], t0, t1) is None


# ---------------------------------------------------- evaluator


def test_skips_thesis_too_recent(tmp_db):
    created = datetime(2026, 4, 1)
    thid = _seed_thesis("NVDA", created_at=created)
    n = ThesisEvaluator().run(as_of=created + timedelta(days=15))
    assert n == 0
    assert _evaluations_for(thid) == []


def test_creates_milestone_30(tmp_db):
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    _seed_close("NVDA", 110.0, created + timedelta(days=30))

    n = ThesisEvaluator().run(as_of=created + timedelta(days=30))
    assert n == 1
    evs = _evaluations_for(thid)
    assert len(evs) == 1
    ev = evs[0]
    assert ev.days_since_thesis == 30
    assert ev.return_pct == pytest.approx(0.10, rel=1e-6)
    assert ev.status == "active"  # < 180j


def test_creates_multiple_milestones_at_once(tmp_db):
    """Si on évalue une thèse créée il y a 100j, jalons 30 et 90 doivent
    être tous deux créés en un seul run."""
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    _seed_close("NVDA", 105.0, created + timedelta(days=30))
    _seed_close("NVDA", 115.0, created + timedelta(days=90))

    n = ThesisEvaluator().run(as_of=created + timedelta(days=100))
    assert n == 2
    evs = _evaluations_for(thid)
    assert [e.days_since_thesis for e in evs] == [30, 90]
    assert evs[0].return_pct == pytest.approx(0.05)
    assert evs[1].return_pct == pytest.approx(0.15)


def test_idempotent_per_milestone(tmp_db):
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    _seed_close("NVDA", 110.0, created + timedelta(days=30))

    ev = ThesisEvaluator()
    assert ev.run(as_of=created + timedelta(days=30)) == 1
    # Re-run le même jour → pas de doublon.
    assert ev.run(as_of=created + timedelta(days=31)) == 0
    assert len(_evaluations_for(thid)) == 1


def test_status_success_at_180(tmp_db):
    """Return +20% / benchmark +5% → alpha +15% → success."""
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    t180 = created + timedelta(days=180)
    _seed_close("NVDA", 120.0, t180)         # +20%
    # Peers AI : AMD, GOOGL, MSFT, META, IBM, PLTR, TSLA, AAPL.
    # On simule un benchmark global +5% en seedant un seul peer suffisant.
    _seed_close("AMD", 100.0, created)
    _seed_close("AMD", 105.0, t180)          # +5%

    n = ThesisEvaluator(milestones=[180]).run(as_of=t180 + timedelta(days=1))
    assert n == 1
    ev = _evaluations_for(thid)[0]
    assert ev.status == "success"
    assert ev.alpha_pct == pytest.approx(0.15, rel=1e-6)


def test_status_failure_at_180(tmp_db):
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    t180 = created + timedelta(days=180)
    _seed_close("NVDA", 90.0, t180)          # -10%
    _seed_close("AMD", 100.0, created)
    _seed_close("AMD", 105.0, t180)          # benchmark +5% → alpha -15%

    ThesisEvaluator(milestones=[180]).run(as_of=t180 + timedelta(days=1))
    ev = _evaluations_for(thid)[0]
    assert ev.status == "failure"
    assert ev.alpha_pct == pytest.approx(-0.15, rel=1e-6)


def test_status_partial_when_alpha_in_band(tmp_db):
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    t180 = created + timedelta(days=180)
    _seed_close("NVDA", 102.0, t180)         # +2%
    _seed_close("AMD", 100.0, created)
    _seed_close("AMD", 100.0, t180)          # benchmark 0% → alpha +2%

    ThesisEvaluator(milestones=[180]).run(as_of=t180 + timedelta(days=1))
    ev = _evaluations_for(thid)[0]
    assert ev.status == "partial"


def test_status_active_at_180_when_no_benchmark(tmp_db):
    """Sans peer valide, alpha=None → status=active même à 180j."""
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    t180 = created + timedelta(days=180)
    _seed_close("NVDA", 130.0, t180)         # return calculé OK
    # Aucun peer seed → benchmark None → alpha None.

    ThesisEvaluator(milestones=[180]).run(as_of=t180 + timedelta(days=1))
    ev = _evaluations_for(thid)[0]
    assert ev.status == "active"
    assert ev.return_pct == pytest.approx(0.30)
    assert ev.benchmark_return_pct is None
    assert ev.alpha_pct is None


def test_handles_missing_current_price(tmp_db):
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=100.0)
    # Pas de close pour NVDA.
    n = ThesisEvaluator().run(as_of=created + timedelta(days=30))
    assert n == 1
    ev = _evaluations_for(thid)[0]
    assert ev.current_price is None
    assert ev.return_pct is None
    assert ev.alpha_pct is None
    assert "non disponible" in (ev.notes or "")


def test_handles_missing_entry_price(tmp_db):
    created = datetime(2026, 1, 1)
    thid = _seed_thesis("NVDA", created_at=created, entry_price=None)
    _seed_close("NVDA", 110.0, created + timedelta(days=30))

    n = ThesisEvaluator().run(as_of=created + timedelta(days=30))
    assert n == 1
    ev = _evaluations_for(thid)[0]
    assert ev.return_pct is None
    assert "entry_price absent" in (ev.notes or "")


def test_default_milestones_constant_matches_spec():
    assert MILESTONES_DAYS == [30, 90, 180, 365, 540]


def test_invalid_milestones_raise():
    with pytest.raises(ValueError):
        ThesisEvaluator(milestones=[])
    with pytest.raises(ValueError):
        ThesisEvaluator(milestones=[0, 30])
    with pytest.raises(ValueError):
        ThesisEvaluator(milestones=[-30])
