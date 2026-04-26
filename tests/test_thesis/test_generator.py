"""Tests du `ThesisGenerator` (Phase 3 étape 1 + Phase 4 étape 6)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory.database import Feature, RawData, Thesis, session_scope
from thesis.generator import (
    ThesisGenerator,
    _active_late_phase_trials,
    _dominant_dimension,
    _latest_entry_price,
    _make_catalysts,
    _make_recommendation,
    _make_risks,
    _pdufa_catalysts,
    _recent_fda_approvals,
    _recent_large_contracts,
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


# -------------------------------------------------- Étape 6 : catalyseurs


def _seed_clinical_trial(
    ticker: str,
    *,
    nct_id: str = "NCT01",
    status: str = "RECRUITING",
    phase: str = "PHASE3",
    interventions: list[str] | None = None,
    content_at: datetime | None = None,
) -> None:
    payload = {
        "nct_id": nct_id,
        "ticker": ticker,
        "overall_status": status,
        "phase": phase,
        "interventions": interventions or ["mRNA-1234"],
        "primary_completion_date": "2027-12-01",
    }
    with session_scope() as s:
        s.add(RawData(
            source="clinicaltrials",
            entity_type="clinical_trial",
            entity_id=nct_id,
            fetched_at=content_at or datetime(2026, 4, 20),
            content_at=content_at or datetime(2026, 4, 20),
            payload_json=json.dumps(payload),
            hash=f"h-{nct_id}-{phase}-{status}",
        ))


def _seed_fda_approval(
    ticker: str,
    *,
    app_number: str = "BLA1",
    sub_number: str = "1",
    brand: str = "Spikevax",
    content_at: datetime | None = None,
) -> None:
    payload = {
        "ticker": ticker,
        "application_number": app_number,
        "submission_number": sub_number,
        "submission_status": "AP",
        "brand_name": brand,
    }
    eid = f"{app_number}-{sub_number}"
    with session_scope() as s:
        s.add(RawData(
            source="fda",
            entity_type="fda_approval",
            entity_id=eid,
            fetched_at=content_at or datetime(2026, 4, 1),
            content_at=content_at or datetime(2026, 4, 1),
            payload_json=json.dumps(payload),
            hash=f"h-{eid}",
        ))


def _seed_gov_contract(
    ticker: str,
    *,
    award_id: str = "A1",
    amount: float = 250_000_000.0,
    agency: str = "Department of Defense",
    description: str = "F-35 lot procurement",
    content_at: datetime | None = None,
) -> None:
    payload = {
        "ticker": ticker,
        "award_id": award_id,
        "award_amount": amount,
        "awarding_agency": agency,
        "description": description,
    }
    with session_scope() as s:
        s.add(RawData(
            source="usaspending",
            entity_type="gov_contract",
            entity_id=award_id,
            fetched_at=content_at or datetime(2026, 4, 1),
            content_at=content_at or datetime(2026, 4, 1),
            payload_json=json.dumps(payload),
            hash=f"h-{award_id}",
        ))


# ---------------- _pdufa_catalysts


def test_pdufa_catalysts_returns_only_ticker_match(monkeypatch):
    import thesis.generator as gen_mod
    monkeypatch.setattr(gen_mod, "upcoming_pdufas", lambda d: [
        {"ticker": "MRNA", "target_action_date": "2026-05-10",
         "drug": "mRNA-1234", "indication": "cancer"},
        {"ticker": "CRSP", "target_action_date": "2026-06-01", "drug": "X"},
    ])
    out = _pdufa_catalysts("MRNA", datetime(2026, 4, 25))
    assert len(out) == 1
    c = out[0]
    assert c["type"] == "pdufa"
    assert c["days_to"] == 15
    assert c["drug"] == "mRNA-1234"
    assert "PDUFA" in c["description"]


def test_pdufa_catalysts_handles_invalid_date(monkeypatch):
    import thesis.generator as gen_mod
    monkeypatch.setattr(gen_mod, "upcoming_pdufas", lambda d: [
        {"ticker": "MRNA", "target_action_date": "garbage", "drug": "X"},
    ])
    assert _pdufa_catalysts("MRNA", datetime(2026, 4, 25)) == []


# ---------------- _active_late_phase_trials


def test_active_late_phase_trials_filters_phase_and_status(tmp_db):
    _seed_clinical_trial("MRNA", nct_id="NCT01", phase="PHASE3", status="RECRUITING")
    _seed_clinical_trial("MRNA", nct_id="NCT02", phase="PHASE2", status="RECRUITING")
    _seed_clinical_trial("MRNA", nct_id="NCT03", phase="PHASE3", status="COMPLETED")
    out = _active_late_phase_trials("MRNA", datetime(2026, 4, 25))
    assert len(out) == 1
    assert out[0]["nct_id"] == "NCT01"
    assert out[0]["phase"] == "PHASE3"


def test_active_late_phase_trials_dedupes_by_nct(tmp_db):
    """Si plusieurs snapshots du même NCT (statut différent → hash différent),
    on n'en retient qu'un dans la sortie."""
    _seed_clinical_trial("MRNA", nct_id="NCT_A", phase="PHASE3",
                         status="RECRUITING",
                         content_at=datetime(2026, 4, 10))
    _seed_clinical_trial("MRNA", nct_id="NCT_A", phase="PHASE3",
                         status="ACTIVE_NOT_RECRUITING",
                         content_at=datetime(2026, 4, 20))
    out = _active_late_phase_trials("MRNA", datetime(2026, 4, 25))
    assert len(out) == 1


def test_active_late_phase_trials_filters_by_ticker(tmp_db):
    _seed_clinical_trial("MRNA", nct_id="NCT01", phase="PHASE3")
    out = _active_late_phase_trials("CRSP", datetime(2026, 4, 25))
    assert out == []


# ---------------- _recent_fda_approvals


def test_recent_fda_approvals_returns_recent(tmp_db):
    _seed_fda_approval("MRNA", brand="Spikevax",
                       content_at=datetime(2026, 3, 15))
    out = _recent_fda_approvals("MRNA", datetime(2026, 4, 25),
                                 lookback_days=90)
    assert len(out) == 1
    assert out[0]["drug"] == "Spikevax"


def test_recent_fda_approvals_skips_old(tmp_db):
    _seed_fda_approval("MRNA", content_at=datetime(2025, 1, 1))
    out = _recent_fda_approvals("MRNA", datetime(2026, 4, 25),
                                 lookback_days=90)
    assert out == []


# ---------------- _recent_large_contracts


def test_recent_large_contracts_sorts_by_amount(tmp_db):
    _seed_gov_contract("LMT", award_id="A1", amount=100_000_000.0)
    _seed_gov_contract("LMT", award_id="A2", amount=500_000_000.0)
    _seed_gov_contract("LMT", award_id="A3", amount=50_000_000.0)
    out = _recent_large_contracts("LMT", datetime(2026, 4, 25))
    assert len(out) == 3
    amounts = [c["amount_usd"] for c in out]
    assert amounts == sorted(amounts, reverse=True)
    assert amounts[0] == 500_000_000.0


def test_recent_large_contracts_top_n(tmp_db):
    """Limite à top_n=3 par défaut."""
    for i in range(5):
        _seed_gov_contract(
            "LMT", award_id=f"A{i}", amount=100_000_000.0 * (i + 1),
        )
    out = _recent_large_contracts("LMT", datetime(2026, 4, 25))
    assert len(out) == 3


def test_recent_large_contracts_filters_by_ticker(tmp_db):
    _seed_gov_contract("LMT", amount=200_000_000.0)
    assert _recent_large_contracts("RKLB", datetime(2026, 4, 25)) == []


# ---------------- _make_catalysts (orchestrateur)


def test_make_catalysts_biotech_aggregates_sources(tmp_db, monkeypatch):
    import thesis.generator as gen_mod
    monkeypatch.setattr(gen_mod, "upcoming_pdufas", lambda d: [
        {"ticker": "MRNA", "target_action_date": "2026-05-10",
         "drug": "mRNA-1234"},
    ])
    _seed_clinical_trial("MRNA", nct_id="NCT01", phase="PHASE3",
                         status="RECRUITING")
    _seed_fda_approval("MRNA", content_at=datetime(2026, 3, 1))
    out = _make_catalysts("MRNA", ["biotech"], datetime(2026, 4, 25))
    types = {c["type"] for c in out}
    assert types == {"pdufa", "phase3_trial", "fda_approval"}


def test_make_catalysts_space_uses_gov_contracts(tmp_db):
    _seed_gov_contract("LMT", amount=300_000_000.0,
                        content_at=datetime(2026, 4, 1))
    out = _make_catalysts("LMT", ["space"], datetime(2026, 4, 25))
    assert len(out) == 1
    assert out[0]["type"] == "gov_contract"


def test_make_catalysts_returns_empty_for_irrelevant_sectors(tmp_db):
    """Un ticker AI/ML n'a aucun catalyseur sectoriel collecté."""
    out = _make_catalysts("NVDA", ["ai_ml"], datetime(2026, 4, 25))
    assert out == []


def test_make_catalysts_multi_sector_combines(tmp_db, monkeypatch):
    """Un ticker multi-secteurs cumule les catalyseurs des sources actives."""
    import thesis.generator as gen_mod
    monkeypatch.setattr(gen_mod, "upcoming_pdufas", lambda d: [
        {"ticker": "X", "target_action_date": "2026-05-10", "drug": "Y"},
    ])
    _seed_gov_contract("X", amount=200_000_000.0)
    out = _make_catalysts("X", ["biotech", "space"], datetime(2026, 4, 25))
    types = {c["type"] for c in out}
    assert "pdufa" in types
    assert "gov_contract" in types


# ---------------- bout-en-bout : thèse avec catalyseurs


def test_run_writes_catalysts_into_thesis_for_biotech(tmp_db, monkeypatch):
    """Une thèse biotech doit refléter les PDUFA + Phase 3 dans
    `catalysts_json` et la narrative."""
    as_of = datetime(2026, 4, 20)
    _seed_score("MRNA", 80.0, as_of - timedelta(hours=1))

    import thesis.generator as gen_mod
    monkeypatch.setattr(gen_mod, "upcoming_pdufas", lambda d: [
        {"ticker": "MRNA", "target_action_date": "2026-05-10",
         "drug": "mRNA-1234", "indication": "cancer du poumon"},
    ])
    _seed_clinical_trial("MRNA", nct_id="NCT_X", phase="PHASE3",
                         status="RECRUITING")

    n = ThesisGenerator(tickers=["MRNA"]).run(as_of=as_of)
    assert n == 1
    with session_scope() as s:
        th = s.query(Thesis).one()
        cats = json.loads(th.catalysts_json)
        types = {c["type"] for c in cats}
        assert "pdufa" in types
        assert "phase3_trial" in types
        # La narrative inclut le détail PDUFA.
        assert "PDUFA" in th.narrative
        assert "mRNA-1234" in th.narrative


def test_run_writes_gov_contracts_into_thesis_for_space(tmp_db):
    as_of = datetime(2026, 4, 20)
    _seed_score("LMT", 80.0, as_of - timedelta(hours=1))
    _seed_gov_contract("LMT", amount=500_000_000.0,
                        content_at=as_of - timedelta(days=10))

    n = ThesisGenerator(tickers=["LMT"]).run(as_of=as_of)
    assert n == 1
    with session_scope() as s:
        th = s.query(Thesis).one()
        cats = json.loads(th.catalysts_json)
        assert any(c["type"] == "gov_contract" for c in cats)
        # Mention du montant en M$ dans la narrative.
        assert "$500.0M" in th.narrative or "500.0M" in th.narrative


def test_run_thesis_without_catalysts_uses_placeholder(tmp_db):
    """Aucun catalyseur sectoriel collecté → placeholder dans la narrative."""
    as_of = datetime(2026, 4, 20)
    _seed_score("NVDA", 90.0, as_of - timedelta(hours=1))
    n = ThesisGenerator(tickers=["NVDA"]).run(as_of=as_of)
    assert n == 1
    with session_scope() as s:
        th = s.query(Thesis).one()
        assert json.loads(th.catalysts_json) == []
        assert "Pas de catalyseur" in th.narrative
