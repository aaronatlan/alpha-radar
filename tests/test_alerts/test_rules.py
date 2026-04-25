"""Tests des règles d'alerte (Phase 3 étape 4)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from alerts.rules import (
    AlertCandidate,
    EvaluationVerdictRule,
    NewThesisRule,
    SectorHeatSurgeRule,
)
from memory.database import (
    Evaluation,
    Feature,
    Thesis,
    session_scope,
)


# ------------------------------------------------------------------ helpers


def _seed_thesis(
    *,
    asset_id: str = "NVDA",
    sector_id: str = "ai_ml",
    score: float = 80.0,
    created_at: datetime | None = None,
    recommendation: str = "BUY",
) -> int:
    with session_scope() as s:
        th = Thesis(
            created_at=created_at or datetime(2026, 4, 1),
            asset_type="stock",
            asset_id=asset_id,
            sector_id=sector_id,
            score=score,
            score_breakdown_json=json.dumps({"dimensions": {"momentum": score}}),
            recommendation=recommendation,
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
    days: int = 180,
    status: str = "success",
    alpha_pct: float | None = 0.10,
    evaluated_at: datetime | None = None,
) -> int:
    with session_scope() as s:
        ev = Evaluation(
            thesis_id=thesis_id,
            evaluated_at=evaluated_at or datetime(2026, 10, 1),
            days_since_thesis=days,
            current_price=110.0,
            return_pct=0.10,
            benchmark_return_pct=0.0,
            alpha_pct=alpha_pct,
            status=status,
            notes=None,
        )
        s.add(ev)
        s.flush()
        eval_id = ev.id
    return eval_id


def _seed_heat(
    sector_id: str, value: float, computed_at: datetime,
) -> None:
    with session_scope() as s:
        s.add(Feature(
            feature_name="sector_heat_score",
            target_type="sector",
            target_id=sector_id,
            computed_at=computed_at,
            value=value,
            metadata_json=None,
        ))


# ----------------------------------------------------------- AlertCandidate


def test_alert_candidate_serialises_dedupe_key():
    c = AlertCandidate(
        rule_name="r", severity="info",
        message="m", dedupe_key="k:1",
        data={"x": 42},
    )
    parsed = json.loads(c.data_json())
    assert parsed["dedupe_key"] == "k:1"
    assert parsed["x"] == 42


# ----------------------------------------------------------- NewThesisRule


def test_new_thesis_rule_emits_above_threshold(tmp_db):
    _seed_thesis(asset_id="NVDA", score=80.0,
                 created_at=datetime(2026, 4, 20))
    cands = NewThesisRule(score_threshold=70.0).evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    c = cands[0]
    assert c.rule_name == "new_thesis"
    assert c.severity == "info"
    assert c.asset_id == "NVDA"
    assert c.dedupe_key.startswith("thesis:")


def test_new_thesis_rule_skips_below_threshold(tmp_db):
    _seed_thesis(asset_id="NVDA", score=65.0,
                 created_at=datetime(2026, 4, 20))
    cands = NewThesisRule(score_threshold=70.0).evaluate(datetime(2026, 4, 25))
    assert cands == []


def test_new_thesis_rule_respects_lookback(tmp_db):
    # Thèse vieille de 30j, lookback 7j → ignorée.
    _seed_thesis(asset_id="NVDA", score=80.0,
                 created_at=datetime(2026, 3, 25))
    cands = NewThesisRule(score_threshold=70.0, lookback_days=7).evaluate(
        datetime(2026, 4, 25)
    )
    assert cands == []


def test_new_thesis_rule_dedupe_key_is_thesis_id(tmp_db):
    th = _seed_thesis(asset_id="NVDA", score=80.0,
                      created_at=datetime(2026, 4, 20))
    cand = NewThesisRule().evaluate(datetime(2026, 4, 25))[0]
    assert cand.dedupe_key == f"thesis:{th}"
    assert cand.thesis_id == th


# ------------------------------------------------------ EvaluationVerdictRule


def test_eval_verdict_emits_for_success(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    _seed_eval(th, status="success", alpha_pct=0.15,
               evaluated_at=datetime(2026, 10, 1))
    cands = EvaluationVerdictRule().evaluate(datetime(2026, 10, 5))
    assert len(cands) == 1
    c = cands[0]
    assert c.rule_name == "eval_verdict"
    assert "success" in c.message.lower()
    assert c.thesis_id == th


def test_eval_verdict_emits_for_failure(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    _seed_eval(th, status="failure", alpha_pct=-0.20,
               evaluated_at=datetime(2026, 10, 1))
    cands = EvaluationVerdictRule().evaluate(datetime(2026, 10, 5))
    assert len(cands) == 1
    assert cands[0].severity == "info"
    assert "failure" in cands[0].message.lower()


def test_eval_verdict_skips_partial_by_default(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    _seed_eval(th, status="partial", alpha_pct=0.02,
               evaluated_at=datetime(2026, 10, 1))
    assert EvaluationVerdictRule().evaluate(datetime(2026, 10, 5)) == []


def test_eval_verdict_skips_active(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    _seed_eval(th, status="active", alpha_pct=None, days=30,
               evaluated_at=datetime(2026, 5, 1))
    assert EvaluationVerdictRule().evaluate(datetime(2026, 5, 5)) == []


def test_eval_verdict_dedupe_key_is_eval_id(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    eid = _seed_eval(th, status="success", evaluated_at=datetime(2026, 10, 1))
    c = EvaluationVerdictRule().evaluate(datetime(2026, 10, 5))[0]
    assert c.dedupe_key == f"eval:{eid}"


def test_eval_verdict_respects_lookback(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    _seed_eval(th, status="success", evaluated_at=datetime(2026, 1, 1))
    # Évaluation vieille de >7j → ignorée.
    assert EvaluationVerdictRule(lookback_days=7).evaluate(
        datetime(2026, 10, 5)
    ) == []


def test_eval_verdict_custom_statuses(tmp_db):
    th = _seed_thesis(asset_id="NVDA")
    _seed_eval(th, status="partial", alpha_pct=0.02,
               evaluated_at=datetime(2026, 10, 1))
    rule = EvaluationVerdictRule(statuses=("partial",))
    cands = rule.evaluate(datetime(2026, 10, 5))
    assert len(cands) == 1


def test_eval_verdict_empty_statuses_raise():
    with pytest.raises(ValueError):
        EvaluationVerdictRule(statuses=())


# ------------------------------------------------------ SectorHeatSurgeRule


def test_heat_surge_emits_when_delta_exceeds_threshold(tmp_db):
    t1 = datetime(2026, 4, 23, 12, 0)   # -48h
    t2 = datetime(2026, 4, 25, 12, 0)
    _seed_heat("ai_ml", 50.0, t1)
    _seed_heat("ai_ml", 75.0, t2)       # +25 pts

    cands = SectorHeatSurgeRule(delta_threshold=20.0,
                                 window_hours=48).evaluate(t2)
    assert len(cands) == 1
    c = cands[0]
    assert c.rule_name == "sector_heat_surge"
    assert c.severity == "critical"
    assert c.sector_id == "ai_ml"
    assert c.data["delta"] == pytest.approx(25.0)


def test_heat_surge_skips_when_delta_below(tmp_db):
    t1 = datetime(2026, 4, 23, 12, 0)
    t2 = datetime(2026, 4, 25, 12, 0)
    _seed_heat("ai_ml", 50.0, t1)
    _seed_heat("ai_ml", 60.0, t2)       # +10 pts < 20

    assert SectorHeatSurgeRule(delta_threshold=20.0).evaluate(t2) == []


def test_heat_surge_skips_when_no_baseline(tmp_db):
    """Aucune mesure ≤ as_of - window → skip (pas d'imputation à 0)."""
    t2 = datetime(2026, 4, 25, 12, 0)
    _seed_heat("ai_ml", 75.0, t2)
    assert SectorHeatSurgeRule().evaluate(t2) == []


def test_heat_surge_dedupe_key_uses_observation_date(tmp_db):
    t1 = datetime(2026, 4, 23, 6, 0)    # bien avant as_of - 48h
    t2 = datetime(2026, 4, 25, 9, 0)
    _seed_heat("ai_ml", 50.0, t1)
    _seed_heat("ai_ml", 80.0, t2)
    c = SectorHeatSurgeRule().evaluate(t2)[0]
    # Date du jour de l'observation courante (9h le 25) → 2026-04-25.
    assert c.dedupe_key == "heat_surge:ai_ml:2026-04-25"


def test_heat_surge_handles_multiple_sectors(tmp_db):
    t1 = datetime(2026, 4, 23, 12, 0)
    t2 = datetime(2026, 4, 25, 12, 0)
    _seed_heat("ai_ml", 50.0, t1)
    _seed_heat("ai_ml", 80.0, t2)       # +30 pts → alerte
    _seed_heat("biotech", 40.0, t1)
    _seed_heat("biotech", 45.0, t2)     # +5 pts → pas d'alerte

    cands = SectorHeatSurgeRule(delta_threshold=20.0).evaluate(t2)
    sectors = {c.sector_id for c in cands}
    assert sectors == {"ai_ml"}


def test_heat_surge_invalid_args_raise():
    with pytest.raises(ValueError):
        SectorHeatSurgeRule(delta_threshold=0)
    with pytest.raises(ValueError):
        SectorHeatSurgeRule(delta_threshold=-5)
    with pytest.raises(ValueError):
        SectorHeatSurgeRule(window_hours=0)
