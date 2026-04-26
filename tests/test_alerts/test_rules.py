"""Tests des règles d'alerte (Phase 3 étape 4)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from alerts.rules import (
    AlertCandidate,
    BuybackMentionRule,
    CitationVelocityRule,
    EvaluationVerdictRule,
    Form13DRule,
    LargeGovContractRule,
    NewThesisRule,
    PDUFANearRule,
    SectorHeatSurgeRule,
)
from memory.database import (
    Evaluation,
    Feature,
    RawData,
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


# ----------------------------------------------------------- PDUFANearRule


def _seed_score(ticker: str, value: float, ts: datetime) -> None:
    """Seed une feature stock_score pour `ticker`."""
    with session_scope() as s:
        s.add(Feature(
            feature_name="stock_score",
            target_type="asset",
            target_id=ticker,
            computed_at=ts,
            value=value,
            metadata_json=None,
        ))


def test_pdufa_near_rule_emits_when_within_window_and_score_high(
    tmp_db, monkeypatch,
):
    from alerts import rules as rules_mod
    pdufa = [{
        "ticker": "MRNA",
        "target_action_date": "2026-05-10",
        "drug": "mRNA-1234",
        "indication": "Cancer du poumon",
    }]
    monkeypatch.setattr(rules_mod, "upcoming_pdufas", lambda d: pdufa)

    _seed_score("MRNA", 80.0, datetime(2026, 4, 25))
    cands = PDUFANearRule().evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    c = cands[0]
    assert c.severity == "critical"
    assert c.asset_id == "MRNA"
    assert "MRNA" in c.message
    assert c.data["days_left"] == 15
    assert c.data["score"] == 80.0


def test_pdufa_near_rule_skips_when_score_below_threshold(
    tmp_db, monkeypatch,
):
    from alerts import rules as rules_mod
    monkeypatch.setattr(rules_mod, "upcoming_pdufas", lambda d: [{
        "ticker": "MRNA",
        "target_action_date": "2026-05-10",
        "drug": "X",
    }])
    _seed_score("MRNA", 65.0, datetime(2026, 4, 25))   # < 70
    cands = PDUFANearRule().evaluate(datetime(2026, 4, 25))
    assert cands == []


def test_pdufa_near_rule_skips_when_pdufa_too_far(
    tmp_db, monkeypatch,
):
    from alerts import rules as rules_mod
    monkeypatch.setattr(rules_mod, "upcoming_pdufas", lambda d: [{
        "ticker": "MRNA",
        "target_action_date": "2026-08-01",   # >30 j
        "drug": "X",
    }])
    _seed_score("MRNA", 85.0, datetime(2026, 4, 25))
    cands = PDUFANearRule().evaluate(datetime(2026, 4, 25))
    assert cands == []


def test_pdufa_near_rule_skips_when_no_score_yet(
    tmp_db, monkeypatch,
):
    from alerts import rules as rules_mod
    monkeypatch.setattr(rules_mod, "upcoming_pdufas", lambda d: [{
        "ticker": "MRNA",
        "target_action_date": "2026-05-10",
        "drug": "X",
    }])
    # Pas de seed → score introuvable.
    cands = PDUFANearRule().evaluate(datetime(2026, 4, 25))
    assert cands == []


def test_pdufa_near_rule_dedupe_key_per_pdufa_date(
    tmp_db, monkeypatch,
):
    from alerts import rules as rules_mod
    monkeypatch.setattr(rules_mod, "upcoming_pdufas", lambda d: [{
        "ticker": "MRNA",
        "target_action_date": "2026-05-10",
        "drug": "X",
    }])
    _seed_score("MRNA", 80.0, datetime(2026, 4, 25))
    c = PDUFANearRule().evaluate(datetime(2026, 4, 25))[0]
    assert c.dedupe_key == "pdufa:MRNA:2026-05-10"


def test_pdufa_near_rule_skips_invalid_date(
    tmp_db, monkeypatch,
):
    from alerts import rules as rules_mod
    monkeypatch.setattr(rules_mod, "upcoming_pdufas", lambda d: [{
        "ticker": "MRNA",
        "target_action_date": "not-a-date",
        "drug": "X",
    }])
    _seed_score("MRNA", 80.0, datetime(2026, 4, 25))
    assert PDUFANearRule().evaluate(datetime(2026, 4, 25)) == []


def test_pdufa_near_rule_invalid_args_raise():
    with pytest.raises(ValueError):
        PDUFANearRule(days_ahead=0)
    with pytest.raises(ValueError):
        PDUFANearRule(days_ahead=-5)


def test_pdufa_calendar_helper_filters_past():
    from datetime import date

    from config.pdufa_calendar import upcoming_pdufas

    # Avec un calendrier vide (état initial Phase 4), helper renvoie [].
    assert upcoming_pdufas(date(2026, 4, 25)) == []


# -------------------------------------------------------- LargeGovContractRule


def _seed_contract(
    *,
    award_id: str = "A1",
    amount: float = 250_000_000.0,
    ticker: str = "LMT",
    recipient: str = "Lockheed Martin",
    agency: str = "DoD",
    description: str | None = "F-35 procurement",
    content_at: datetime | None = None,
) -> None:
    """Insère une ligne raw_data simulant un contrat USASpending."""
    payload = {
        "ticker": ticker,
        "award_id": award_id,
        "award_amount": amount,
        "recipient_name": recipient,
        "awarding_agency": agency,
        "description": description,
        "action_date": (content_at or datetime(2026, 4, 20)).date().isoformat(),
    }
    with session_scope() as s:
        s.add(RawData(
            source="usaspending",
            entity_type="gov_contract",
            entity_id=award_id,
            fetched_at=content_at or datetime(2026, 4, 20),
            content_at=content_at or datetime(2026, 4, 20),
            payload_json=json.dumps(payload),
            hash=f"h-{award_id}",
        ))


def test_large_contract_emits_above_threshold(tmp_db):
    _seed_contract(amount=250_000_000.0)
    cands = LargeGovContractRule().evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    c = cands[0]
    assert c.severity == "warning"
    assert c.asset_id == "LMT"
    assert "250" in c.message  # amount /1e6 = 250.0
    assert c.dedupe_key.startswith("contract:")


def test_large_contract_skips_below_threshold(tmp_db):
    _seed_contract(amount=50_000_000.0)
    assert LargeGovContractRule().evaluate(datetime(2026, 4, 25)) == []


def test_large_contract_skips_when_amount_none(tmp_db):
    _seed_contract(amount=None)  # type: ignore[arg-type]
    assert LargeGovContractRule().evaluate(datetime(2026, 4, 25)) == []


def test_large_contract_respects_lookback(tmp_db):
    """Contrat hors fenêtre lookback → pas d'alerte."""
    _seed_contract(amount=200_000_000.0,
                   content_at=datetime(2026, 1, 1))   # >7j avant as_of
    rule = LargeGovContractRule(lookback_days=7)
    assert rule.evaluate(datetime(2026, 4, 25)) == []


def test_large_contract_dedupe_key_uses_award_id(tmp_db):
    _seed_contract(award_id="FA8625-21-C-0001", amount=200_000_000.0)
    c = LargeGovContractRule().evaluate(datetime(2026, 4, 25))[0]
    assert c.dedupe_key == "contract:FA8625-21-C-0001"


def test_large_contract_skips_malformed_payload(tmp_db):
    """Payload JSON cassé → pas de crash, pas d'alerte."""
    with session_scope() as s:
        s.add(RawData(
            source="usaspending", entity_type="gov_contract",
            entity_id="bad", fetched_at=datetime(2026, 4, 20),
            content_at=datetime(2026, 4, 20),
            payload_json="not a json",
            hash="h-bad",
        ))
    assert LargeGovContractRule().evaluate(datetime(2026, 4, 25)) == []


def test_large_contract_custom_threshold(tmp_db):
    _seed_contract(amount=10_000_000.0)
    rule = LargeGovContractRule(threshold_usd=5_000_000.0)
    assert len(rule.evaluate(datetime(2026, 4, 25))) == 1


def test_large_contract_invalid_args_raise():
    with pytest.raises(ValueError):
        LargeGovContractRule(threshold_usd=0)
    with pytest.raises(ValueError):
        LargeGovContractRule(threshold_usd=-1)
    with pytest.raises(ValueError):
        LargeGovContractRule(lookback_days=0)


def test_large_contract_message_includes_amount_in_millions(tmp_db):
    _seed_contract(amount=125_500_000.0)
    c = LargeGovContractRule().evaluate(datetime(2026, 4, 25))[0]
    assert "125.5M" in c.message


# ----------------------------------------------------- CitationVelocityRule


def _seed_paper_snapshot(
    *,
    paper_id: str,
    citation_count: int,
    fetched_at: datetime,
    title: str = "A great paper",
    sector_id: str = "ai_ml",
) -> None:
    """Seed une ligne raw_data simulant un snapshot semantic_scholar."""
    payload = {
        "paper_id": paper_id,
        "title": title,
        "citation_count": citation_count,
        "sector_id": sector_id,
    }
    with session_scope() as s:
        s.add(RawData(
            source="semantic_scholar",
            entity_type="paper",
            entity_id=paper_id,
            fetched_at=fetched_at,
            content_at=fetched_at,
            payload_json=json.dumps(payload),
            # Hash unique par snapshot (citation_count + fetched_at).
            hash=f"h-{paper_id}-{citation_count}-{fetched_at.isoformat()}",
        ))


def test_citation_velocity_emits_when_delta_above_threshold(tmp_db):
    paper = "p1"
    _seed_paper_snapshot(paper_id=paper, citation_count=50,
                         fetched_at=datetime(2026, 4, 20))
    _seed_paper_snapshot(paper_id=paper, citation_count=200,
                         fetched_at=datetime(2026, 4, 25))
    cands = CitationVelocityRule().evaluate(datetime(2026, 4, 26))
    assert len(cands) == 1
    c = cands[0]
    assert c.severity == "warning"
    assert c.data["delta"] == 150
    assert c.data["from"] == 50
    assert c.data["to"] == 200
    assert c.sector_id == "ai_ml"


def test_citation_velocity_skips_when_delta_below(tmp_db):
    _seed_paper_snapshot(paper_id="p1", citation_count=50,
                         fetched_at=datetime(2026, 4, 20))
    _seed_paper_snapshot(paper_id="p1", citation_count=80,
                         fetched_at=datetime(2026, 4, 25))
    assert CitationVelocityRule().evaluate(datetime(2026, 4, 26)) == []


def test_citation_velocity_skips_single_snapshot(tmp_db):
    """Sans baseline (1 seul snapshot) → delta = 0 → pas d'alerte."""
    _seed_paper_snapshot(paper_id="p1", citation_count=500,
                         fetched_at=datetime(2026, 4, 25))
    assert CitationVelocityRule().evaluate(datetime(2026, 4, 26)) == []


def test_citation_velocity_respects_window(tmp_db):
    """Snapshot hors window n'est pas considéré comme baseline."""
    paper = "p1"
    _seed_paper_snapshot(paper_id=paper, citation_count=10,
                         fetched_at=datetime(2026, 1, 1))   # >7j
    _seed_paper_snapshot(paper_id=paper, citation_count=200,
                         fetched_at=datetime(2026, 4, 25))
    # Avec window=7, l'oldest dans la fenêtre = 200 → delta=0 → pas d'alerte.
    rule = CitationVelocityRule(window_days=7)
    assert rule.evaluate(datetime(2026, 4, 26)) == []


def test_citation_velocity_dedupe_key_per_paper_per_day(tmp_db):
    paper = "abc"
    _seed_paper_snapshot(paper_id=paper, citation_count=10,
                         fetched_at=datetime(2026, 4, 20))
    _seed_paper_snapshot(paper_id=paper, citation_count=200,
                         fetched_at=datetime(2026, 4, 25, 9, 0))
    c = CitationVelocityRule().evaluate(datetime(2026, 4, 26))[0]
    assert c.dedupe_key == "citation:abc:2026-04-25"


def test_citation_velocity_handles_multiple_papers(tmp_db):
    _seed_paper_snapshot(paper_id="A", citation_count=10,
                         fetched_at=datetime(2026, 4, 20))
    _seed_paper_snapshot(paper_id="A", citation_count=300,
                         fetched_at=datetime(2026, 4, 25))   # +290
    _seed_paper_snapshot(paper_id="B", citation_count=10,
                         fetched_at=datetime(2026, 4, 20))
    _seed_paper_snapshot(paper_id="B", citation_count=50,
                         fetched_at=datetime(2026, 4, 25))   # +40 → skip
    cands = CitationVelocityRule().evaluate(datetime(2026, 4, 26))
    paper_ids = {c.data["paper_id"] for c in cands}
    assert paper_ids == {"A"}


def test_citation_velocity_handles_malformed_payload(tmp_db):
    with session_scope() as s:
        s.add(RawData(
            source="semantic_scholar", entity_type="paper",
            entity_id="bad", fetched_at=datetime(2026, 4, 25),
            content_at=datetime(2026, 4, 25),
            payload_json="not a json",
            hash="h-bad",
        ))
    # Pas de crash, pas d'alerte.
    assert CitationVelocityRule().evaluate(datetime(2026, 4, 26)) == []


def test_citation_velocity_invalid_args_raise():
    with pytest.raises(ValueError):
        CitationVelocityRule(delta_threshold=0)
    with pytest.raises(ValueError):
        CitationVelocityRule(delta_threshold=-5)
    with pytest.raises(ValueError):
        CitationVelocityRule(window_days=0)


# ----------------------------------------------------------- Form13DRule


def _seed_sec_filing(
    *,
    accession: str = "0001000000-26-000001",
    form: str = "SC 13D",
    ticker: str = "NVDA",
    company: str = "NVIDIA",
    content_at: datetime | None = None,
) -> None:
    """Insère une ligne raw_data simulant un filing SEC EDGAR."""
    payload = {
        "accession": accession,
        "form": form,
        "ticker": ticker,
        "company_name": company,
        "filing_date": (content_at or datetime(2026, 4, 20)).date().isoformat(),
    }
    with session_scope() as s:
        s.add(RawData(
            source="sec_edgar",
            entity_type="sec_filing",
            entity_id=accession,
            fetched_at=content_at or datetime(2026, 4, 20),
            content_at=content_at or datetime(2026, 4, 20),
            payload_json=json.dumps(payload),
            hash=f"h-{accession}",
        ))


def test_form_13d_emits_for_new_filing(tmp_db):
    _seed_sec_filing(form="SC 13D", ticker="NVDA")
    cands = Form13DRule().evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    c = cands[0]
    assert c.severity == "critical"
    assert c.asset_id == "NVDA"
    assert "SC 13D" in c.message
    assert c.dedupe_key.startswith("13d:")


def test_form_13d_emits_for_amended(tmp_db):
    _seed_sec_filing(form="SC 13D/A", accession="0001-26-000099")
    cands = Form13DRule().evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    assert "SC 13D/A" in cands[0].message


def test_form_13d_skips_other_forms(tmp_db):
    _seed_sec_filing(form="10-K")
    _seed_sec_filing(form="SC 13G", accession="diff")
    assert Form13DRule().evaluate(datetime(2026, 4, 25)) == []


def test_form_13d_respects_lookback(tmp_db):
    _seed_sec_filing(form="SC 13D", content_at=datetime(2026, 1, 1))
    rule = Form13DRule(lookback_days=7)
    assert rule.evaluate(datetime(2026, 4, 25)) == []


def test_form_13d_dedupe_key_uses_accession(tmp_db):
    _seed_sec_filing(form="SC 13D", accession="myaccession")
    c = Form13DRule().evaluate(datetime(2026, 4, 25))[0]
    assert c.dedupe_key == "13d:myaccession"


def test_form_13d_invalid_args_raise():
    with pytest.raises(ValueError):
        Form13DRule(lookback_days=0)
    with pytest.raises(ValueError):
        Form13DRule(lookback_days=-1)


def test_form_13d_skips_malformed_payload(tmp_db):
    with session_scope() as s:
        s.add(RawData(
            source="sec_edgar", entity_type="sec_filing",
            entity_id="bad", fetched_at=datetime(2026, 4, 20),
            content_at=datetime(2026, 4, 20),
            payload_json="not a json",
            hash="h-bad",
        ))
    assert Form13DRule().evaluate(datetime(2026, 4, 25)) == []


# --------------------------------------------------- BuybackMentionRule


def _seed_news(
    *,
    url: str = "https://x.com/article-1",
    title: str = "",
    description: str = "",
    content_at: datetime | None = None,
) -> None:
    payload = {
        "url": url,
        "title": title,
        "description": description,
    }
    with session_scope() as s:
        s.add(RawData(
            source="newsapi",
            entity_type="article",
            entity_id=url,
            fetched_at=content_at or datetime(2026, 4, 20),
            content_at=content_at or datetime(2026, 4, 20),
            payload_json=json.dumps(payload),
            hash=f"h-{url}",
        ))


def test_buyback_emits_when_keyword_and_company_match(tmp_db):
    _seed_news(
        url="https://example.com/a1",
        title="NVIDIA announces $50B share repurchase program",
        description="The chipmaker boosts its buyback authorization.",
    )
    cands = BuybackMentionRule().evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    c = cands[0]
    assert c.severity == "warning"
    assert c.asset_id == "NVDA"
    assert "buyback" in c.message.lower()


def test_buyback_skips_news_without_keyword(tmp_db):
    _seed_news(
        title="NVIDIA reports strong earnings",
        description="Revenue beat expectations.",
    )
    assert BuybackMentionRule().evaluate(datetime(2026, 4, 25)) == []


def test_buyback_skips_news_without_watchlist_match(tmp_db):
    """Buyback mentionné mais pas de société watchlist → skip."""
    _seed_news(
        title="ExxonMobil announces share repurchase",
        description="Oil major returns capital.",
    )
    assert BuybackMentionRule().evaluate(datetime(2026, 4, 25)) == []


def test_buyback_dedupe_key_uses_url(tmp_db):
    _seed_news(
        url="https://x.com/unique",
        title="Apple share buyback expanded",
    )
    c = BuybackMentionRule().evaluate(datetime(2026, 4, 25))[0]
    assert c.dedupe_key == "buyback:AAPL:https://x.com/unique"


def test_buyback_respects_lookback(tmp_db):
    _seed_news(
        title="NVIDIA buyback program",
        content_at=datetime(2026, 1, 1),
    )
    rule = BuybackMentionRule(lookback_days=7)
    assert rule.evaluate(datetime(2026, 4, 25)) == []


def test_buyback_handles_multiple_keywords(tmp_db):
    """rachat d'actions (FR) doit aussi matcher."""
    _seed_news(
        title="NVIDIA annonce un rachat d'actions",
        description="Programme de retour aux actionnaires.",
    )
    cands = BuybackMentionRule().evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1


def test_buyback_invalid_args_raise():
    with pytest.raises(ValueError):
        BuybackMentionRule(lookback_days=0)
    with pytest.raises(ValueError):
        BuybackMentionRule(keywords=())


def test_buyback_skips_malformed_payload(tmp_db):
    with session_scope() as s:
        s.add(RawData(
            source="newsapi", entity_type="article",
            entity_id="bad", fetched_at=datetime(2026, 4, 20),
            content_at=datetime(2026, 4, 20),
            payload_json="not a json",
            hash="h-bad",
        ))
    assert BuybackMentionRule().evaluate(datetime(2026, 4, 25)) == []


def test_buyback_uses_custom_ticker_map(tmp_db):
    _seed_news(
        title="ACME Corp announces share buyback",
        description="ACME plans $1B repurchase.",
    )
    rule = BuybackMentionRule(ticker_to_company={"ACME": "ACME Corp"})
    cands = rule.evaluate(datetime(2026, 4, 25))
    assert len(cands) == 1
    assert cands[0].asset_id == "ACME"
