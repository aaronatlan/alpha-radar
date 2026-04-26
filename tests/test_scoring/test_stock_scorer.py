"""Tests du `StockScorer` (phase 2 v1 : dimension momentum seule)."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from memory.database import Feature, RawData, session_scope
from scoring.stock_scorer import (
    StockScorer,
    _amount_to_score,
    _momentum_to_score,
    _rsi_to_score,
    _trial_count_to_score,
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


# ---------------------------------------------- pharma_pipeline mappings


@pytest.mark.parametrize(
    "n,expected",
    [(0, 0.0), (1, 20.0), (5, 50.0), (20, 80.0), (50, 100.0), (100, 100.0)],
)
def test_trial_count_mapping(n, expected):
    assert _trial_count_to_score(n) == pytest.approx(expected)


# ---------------------------------------------- amount_to_score (gov)


@pytest.mark.parametrize(
    "amount,expected",
    [
        (0.0, 0.0),
        (10_000_000.0, 20.0),
        (100_000_000.0, 50.0),
        (1_000_000_000.0, 80.0),
        (10_000_000_000.0, 100.0),
        (1e12, 100.0),
    ],
)
def test_amount_to_score_paliers(amount, expected):
    assert _amount_to_score(amount) == pytest.approx(expected, rel=1e-6)


def test_amount_to_score_below_first_palier_is_linear():
    # $1M = 1/10 du premier palier ($10M) → 1/10 * 20 = 2.0
    assert _amount_to_score(1_000_000.0) == pytest.approx(2.0, rel=1e-6)


# ----------------------------- helpers : seed clinical_trial / fda / contract


def _seed_clinical_trial(
    ticker: str,
    *,
    nct_id: str = "NCT01",
    status: str = "RECRUITING",
    phase: str = "PHASE2",
    content_at: datetime | None = None,
) -> None:
    payload = {
        "nct_id": nct_id,
        "ticker": ticker,
        "overall_status": status,
        "phase": phase,
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
    content_at: datetime | None = None,
) -> None:
    payload = {
        "ticker": ticker,
        "application_number": app_number,
        "submission_number": sub_number,
        "submission_status": "AP",
    }
    eid = f"{app_number}-{sub_number}"
    with session_scope() as s:
        s.add(RawData(
            source="fda",
            entity_type="fda_approval",
            entity_id=eid,
            fetched_at=content_at or datetime(2026, 4, 20),
            content_at=content_at or datetime(2026, 4, 20),
            payload_json=json.dumps(payload),
            hash=f"h-{eid}",
        ))


def _seed_gov_contract(
    ticker: str,
    *,
    award_id: str = "A1",
    amount: float = 100_000_000.0,
    content_at: datetime | None = None,
) -> None:
    payload = {
        "ticker": ticker,
        "award_id": award_id,
        "award_amount": amount,
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


# ----------------------------------------- pharma_pipeline dimension


def test_pharma_pipeline_returns_none_without_data(tmp_db):
    """Sans trial ni approval → dimension absente."""
    scorer = StockScorer(tickers=["MRNA"], model_version="v4_sectoral")
    _seed_feature("rsi_14", "MRNA", 60.0, datetime(2026, 4, 20))
    _, meta = scorer.compute("MRNA", datetime(2026, 4, 25))
    assert "pharma_pipeline" not in meta["dimensions"]


def test_pharma_pipeline_counts_active_trials_and_phase(tmp_db):
    _seed_clinical_trial("MRNA", nct_id="NCT01", status="RECRUITING", phase="PHASE2")
    _seed_clinical_trial("MRNA", nct_id="NCT02", status="ACTIVE_NOT_RECRUITING",
                         phase="PHASE3")
    _seed_clinical_trial("MRNA", nct_id="NCT03", status="COMPLETED", phase="PHASE3")

    scorer = StockScorer(tickers=["MRNA"], model_version="v4_sectoral")
    _, meta = scorer.compute("MRNA", datetime(2026, 4, 25))
    assert "pharma_pipeline" in meta["dimensions"]
    detail = meta["details"]["pharma_pipeline"]
    # 2 actifs (RECRUITING + ACTIVE) — COMPLETED ne compte pas.
    assert detail["inputs"]["n_active_trials_365d"] == 2
    # Phase la plus avancée = PHASE3 = 75.
    assert detail["inputs"]["phase_score"] == 75.0


def test_pharma_pipeline_fda_bonus_applied(tmp_db):
    _seed_clinical_trial("MRNA", phase="PHASE3", status="RECRUITING")
    _seed_fda_approval("MRNA")
    scorer = StockScorer(tickers=["MRNA"], model_version="v4_sectoral")
    score, meta = scorer.compute("MRNA", datetime(2026, 4, 25))
    detail = meta["details"]["pharma_pipeline"]
    # Score = (n=20 + phase=75) / 2 + 20 (bonus) = 47.5 + 20 = 67.5
    assert detail["inputs"]["fda_approvals_365d"] == 1
    assert meta["dimensions"]["pharma_pipeline"] == pytest.approx(67.5, rel=1e-6)


def test_pharma_pipeline_score_capped_at_100(tmp_db):
    """50 trials actifs (n_score=100) + phase 4 (=85) + bonus 20 → cap à 100."""
    for i in range(50):
        _seed_clinical_trial(
            "MRNA", nct_id=f"NCT{i:03d}", status="RECRUITING", phase="PHASE4",
        )
    _seed_fda_approval("MRNA")
    scorer = StockScorer(tickers=["MRNA"], model_version="v4_sectoral")
    _, meta = scorer.compute("MRNA", datetime(2026, 4, 25))
    assert meta["dimensions"]["pharma_pipeline"] == 100.0


def test_pharma_pipeline_filters_by_ticker(tmp_db):
    """Trial sur MRNA ne contribue pas au score CRSP."""
    _seed_clinical_trial("MRNA", phase="PHASE3", status="RECRUITING")
    _seed_feature("rsi_14", "CRSP", 60.0, datetime(2026, 4, 20))   # pour avoir un score
    scorer = StockScorer(tickers=["CRSP"], model_version="v4_sectoral")
    _, meta = scorer.compute("CRSP", datetime(2026, 4, 25))
    assert "pharma_pipeline" not in meta["dimensions"]


def test_pharma_pipeline_pit_filters_future_trial(tmp_db):
    """Un trial avec content_at > as_of est ignoré."""
    _seed_clinical_trial(
        "MRNA", phase="PHASE3", status="RECRUITING",
        content_at=datetime(2026, 6, 1),
    )
    scorer = StockScorer(tickers=["MRNA"], model_version="v4_sectoral")
    _, meta = scorer.compute("MRNA", datetime(2026, 4, 25)) or (0.0, {"dimensions": {}})
    assert "pharma_pipeline" not in (meta.get("dimensions") or {})


# ----------------------------------------- gov_contracts dimension


def test_gov_contracts_returns_none_without_data(tmp_db):
    _seed_feature("rsi_14", "LMT", 60.0, datetime(2026, 4, 20))
    scorer = StockScorer(tickers=["LMT"], model_version="v4_sectoral")
    _, meta = scorer.compute("LMT", datetime(2026, 4, 25))
    assert "gov_contracts" not in meta["dimensions"]


def test_gov_contracts_aggregates_amounts(tmp_db):
    _seed_gov_contract("LMT", award_id="A1", amount=50_000_000.0)
    _seed_gov_contract("LMT", award_id="A2", amount=50_000_000.0)
    scorer = StockScorer(tickers=["LMT"], model_version="v4_sectoral")
    _, meta = scorer.compute("LMT", datetime(2026, 4, 25))
    detail = meta["details"]["gov_contracts"]
    assert detail["inputs"]["n_contracts_365d"] == 2
    assert detail["inputs"]["total_amount_usd"] == pytest.approx(100_000_000.0)
    # $100M = palier 50.
    assert meta["dimensions"]["gov_contracts"] == pytest.approx(50.0, rel=1e-6)


def test_gov_contracts_filters_by_ticker(tmp_db):
    _seed_gov_contract("LMT", amount=500_000_000.0)
    _seed_feature("rsi_14", "RKLB", 60.0, datetime(2026, 4, 20))
    scorer = StockScorer(tickers=["RKLB"], model_version="v4_sectoral")
    _, meta = scorer.compute("RKLB", datetime(2026, 4, 25))
    assert "gov_contracts" not in meta["dimensions"]


def test_gov_contracts_skips_amount_none(tmp_db):
    """Un contrat sans award_amount n'est pas comptabilisé."""
    payload = {
        "ticker": "LMT", "award_id": "A1", "award_amount": None,
    }
    with session_scope() as s:
        s.add(RawData(
            source="usaspending", entity_type="gov_contract",
            entity_id="A1",
            fetched_at=datetime(2026, 4, 20),
            content_at=datetime(2026, 4, 20),
            payload_json=json.dumps(payload),
            hash="h-A1-none",
        ))
    # Seed RSI pour que le scorer ait au moins une dimension à retourner.
    _seed_feature("rsi_14", "LMT", 60.0, datetime(2026, 4, 20))
    scorer = StockScorer(tickers=["LMT"], model_version="v4_sectoral")
    _, meta = scorer.compute("LMT", datetime(2026, 4, 25))
    assert "gov_contracts" not in meta["dimensions"]


# ----------------------------------------- v4_sectoral end-to-end


def test_v4_combines_all_available_dimensions(tmp_db):
    """v4_sectoral combine les dimensions disponibles avec renormalisation."""
    as_of = datetime(2026, 4, 25)
    ts = as_of - timedelta(hours=1)
    # Momentum (RSI 60 → 100), signal_quality (heat 80), pas de sentiment,
    # pas de pharma, pas de gov.
    _seed_feature("rsi_14", "NVDA", 60.0, ts)
    _seed_feature("sector_heat_score", "ai_ml", 80.0, ts, target_type="sector")

    scorer = StockScorer(tickers=["NVDA"], model_version="v4_sectoral")
    score, meta = scorer.compute("NVDA", as_of)
    assert set(meta["dimensions"].keys()) == {"momentum", "signal_quality"}
    # Renormalisé sur 2 dimensions (momentum 0.30 + signal_quality 0.20 = 0.50).
    # → score = (0.30*100 + 0.20*80) / 0.50 = (30 + 16) / 0.5 = 92.0
    assert score == pytest.approx(92.0, rel=1e-6)


def test_v4_includes_pharma_for_biotech(tmp_db):
    """Sur un ticker biotech, pharma_pipeline contribue au score."""
    as_of = datetime(2026, 4, 25)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "MRNA", 60.0, ts)
    _seed_clinical_trial("MRNA", phase="PHASE3", status="RECRUITING")

    scorer = StockScorer(tickers=["MRNA"], model_version="v4_sectoral")
    _, meta = scorer.compute("MRNA", as_of)
    assert "momentum" in meta["dimensions"]
    assert "pharma_pipeline" in meta["dimensions"]


def test_v4_includes_gov_for_defense(tmp_db):
    as_of = datetime(2026, 4, 25)
    ts = as_of - timedelta(hours=1)
    _seed_feature("rsi_14", "LMT", 60.0, ts)
    _seed_gov_contract("LMT", amount=500_000_000.0)

    scorer = StockScorer(tickers=["LMT"], model_version="v4_sectoral")
    _, meta = scorer.compute("LMT", as_of)
    assert "gov_contracts" in meta["dimensions"]
