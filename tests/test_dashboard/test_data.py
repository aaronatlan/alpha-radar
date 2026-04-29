"""Tests des accesseurs lecture seule du dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from dashboard._data import (
    acknowledge_alert,
    get_alerts,
    get_alpha_by_horizon,
    get_collector_health,
    get_feature_freshness,
    get_performance_summary,
    get_price_history,
    get_sector_heat_scores,
    get_signal_performance,
    get_stock_scores,
    get_theses_history,
    get_thesis_detail,
    list_thesis_ids,
)
from memory.database import (
    Alert,
    Evaluation,
    Feature,
    RawData,
    SignalPerformance,
    Thesis,
    session_scope,
)


def _seed_feature(
    name: str, target_id: str, value: float, ts: datetime,
    target_type: str, metadata: dict | None = None,
) -> None:
    with session_scope() as s:
        s.add(Feature(
            feature_name=name,
            target_type=target_type,
            target_id=target_id,
            computed_at=ts,
            value=value,
            metadata_json=json.dumps(metadata) if metadata else None,
        ))


def _seed_raw(source: str, entity_id: str, content_at: datetime,
              fetched_at: datetime) -> None:
    with session_scope() as s:
        s.add(RawData(
            source=source, entity_type="x", entity_id=entity_id,
            fetched_at=fetched_at, content_at=content_at,
            payload_json="{}", hash=f"h-{source}-{entity_id}",
        ))


# ------------------------------------------------------------- heat scores


def test_get_sector_heat_scores_returns_all_sectors_even_without_score(tmp_db):
    df = get_sector_heat_scores()
    # Tous les secteurs listés même sans score calculé.
    assert not df.empty
    assert df["heat_score"].isna().all()
    assert {"sector_id", "sector_name", "category"}.issubset(df.columns)


def test_get_sector_heat_scores_uses_latest_per_sector(tmp_db):
    now = datetime(2026, 4, 1, 12)
    _seed_feature("sector_heat_score", "ai_ml", 40.0, now - timedelta(days=2),
                  target_type="sector")
    _seed_feature("sector_heat_score", "ai_ml", 75.0, now - timedelta(hours=1),
                  target_type="sector")
    df = get_sector_heat_scores(as_of=now)
    ai = df[df["sector_id"] == "ai_ml"].iloc[0]
    assert ai["heat_score"] == 75.0


def test_get_sector_heat_scores_pit(tmp_db):
    now = datetime(2026, 4, 1, 12)
    _seed_feature("sector_heat_score", "ai_ml", 40.0, now - timedelta(days=2),
                  target_type="sector")
    _seed_feature("sector_heat_score", "ai_ml", 99.0, now + timedelta(hours=1),
                  target_type="sector")  # futur — doit être ignoré
    df = get_sector_heat_scores(as_of=now)
    ai = df[df["sector_id"] == "ai_ml"].iloc[0]
    assert ai["heat_score"] == 40.0


# --------------------------------------------------------------- stocks


def test_get_stock_scores_empty(tmp_db):
    df = get_stock_scores()
    assert df.empty or df["stock_score"].isna().all()


def test_get_stock_scores_extracts_dimensions(tmp_db):
    now = datetime(2026, 4, 1, 12)
    _seed_feature(
        "stock_score", "NVDA", 83.75, now,
        target_type="asset",
        metadata={
            "model_version": "v1_momentum_only",
            "dimensions": {"momentum": 83.75},
        },
    )
    df = get_stock_scores(as_of=now)
    row = df[df["ticker"] == "NVDA"].iloc[0]
    assert row["stock_score"] == 83.75
    assert row["momentum"] == 83.75
    # signal_quality absent → None (pas 0, pour ne pas induire en erreur)
    assert row["signal_quality"] is None
    # Le name provient de la watchlist
    assert row["name"] == "NVIDIA"


# -------------------------------------------------------- collector health


def test_get_collector_health_aggregates_per_source(tmp_db):
    t = datetime(2026, 4, 1)
    _seed_raw("arxiv", "p1", t, t)
    _seed_raw("arxiv", "p2", t + timedelta(hours=1), t + timedelta(hours=1))
    _seed_raw("newsapi", "n1", t, t)
    df = get_collector_health()
    assert set(df["source"]) == {"arxiv", "newsapi"}
    arx = df[df["source"] == "arxiv"].iloc[0]
    assert arx["n_rows"] == 2
    assert arx["last_content_at"] == t + timedelta(hours=1)


def test_get_feature_freshness_groups_by_name_and_type(tmp_db):
    t = datetime(2026, 4, 1)
    _seed_feature("rsi_14", "NVDA", 60.0, t, target_type="asset")
    _seed_feature("rsi_14", "AMD", 55.0, t, target_type="asset")
    _seed_feature("sector_heat_score", "ai_ml", 70.0, t, target_type="sector")
    df = get_feature_freshness()
    rsi_row = df[df["feature_name"] == "rsi_14"].iloc[0]
    assert rsi_row["n_targets"] == 2
    assert rsi_row["target_type"] == "asset"
    hs_row = df[df["feature_name"] == "sector_heat_score"].iloc[0]
    assert hs_row["target_type"] == "sector"


# ------------------------------------------------------- helpers supplémentaires


def _seed_thesis(
    asset_id: str = "NVDA",
    sector_id: str = "ai_ml",
    score: float = 80.0,
    created_at: datetime | None = None,
) -> int:
    with session_scope() as s:
        th = Thesis(
            created_at=created_at or datetime(2026, 1, 1),
            asset_type="stock", asset_id=asset_id, sector_id=sector_id,
            score=score,
            score_breakdown_json=json.dumps({"dimensions": {"momentum": score}}),
            recommendation="BUY", horizon_days=180, entry_price=100.0,
            triggers_json="[]", risks_json="[]", catalysts_json="[]",
            narrative="…", model_version="v1", weights_snapshot_json="{}",
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
            evaluated_at=evaluated_at or datetime(2026, 7, 1),
            days_since_thesis=days,
            current_price=110.0,
            return_pct=0.10,
            benchmark_return_pct=0.0,
            alpha_pct=alpha_pct,
            status=status,
        )
        s.add(ev)
        s.flush()
        eid = ev.id
    return eid


def _seed_alert(
    rule_name: str = "new_thesis",
    severity: str = "info",
    sector_id: str | None = "ai_ml",
    acknowledged: bool = False,
    created_at: datetime | None = None,
) -> int:
    with session_scope() as s:
        a = Alert(
            created_at=created_at or datetime(2026, 4, 25),
            rule_name=rule_name,
            severity=severity,
            sector_id=sector_id,
            message="test alert",
            data_json=json.dumps({"dedupe_key": f"k:{rule_name}"}),
            acknowledged=acknowledged,
        )
        s.add(a)
        s.flush()
        aid = a.id
    return aid


def _seed_signal_perf(
    signal: str, sector_id: str | None, horizon: int,
    n_pred: int = 10, n_succ: int = 7, avg_alpha: float | None = 0.05,
) -> None:
    with session_scope() as s:
        s.add(SignalPerformance(
            signal_name=signal,
            sector_id=sector_id,
            horizon_days=horizon,
            n_predictions=n_pred,
            n_successes=n_succ,
            accuracy=n_succ / n_pred,
            avg_alpha=avg_alpha,
            last_updated=datetime(2026, 4, 25),
        ))


# --------------------------------------------------------- performance_summary


def test_performance_summary_empty(tmp_db):
    s = get_performance_summary()
    assert s["n_theses"] == 0
    assert s["n_evaluated"] == 0
    assert s["success_rate"] is None
    assert s["avg_alpha"] is None


def test_performance_summary_counts_correctly(tmp_db):
    t1 = _seed_thesis("NVDA")
    t2 = _seed_thesis("AMD")
    t3 = _seed_thesis("GOOGL")
    _seed_eval(t1, days=180, status="success", alpha_pct=0.20)
    _seed_eval(t2, days=180, status="failure", alpha_pct=-0.10)
    _seed_eval(t3, days=30, status="active", alpha_pct=None)  # non terminal

    s = get_performance_summary()
    assert s["n_theses"] == 3
    assert s["n_evaluated"] == 2   # 2 thèses avec éval terminale
    assert s["n_success"] == 1
    assert s["n_failure"] == 1
    assert s["n_partial"] == 0
    assert s["success_rate"] == pytest.approx(0.5)
    assert s["avg_alpha"] == pytest.approx(0.05)  # (0.20 - 0.10) / 2


# --------------------------------------------------------- signal_performance


def test_signal_performance_empty(tmp_db):
    df = get_signal_performance()
    assert df.empty


def test_signal_performance_returns_all_sectors_rows(tmp_db):
    _seed_signal_perf("momentum", None, 365, n_pred=5, n_succ=3)
    _seed_signal_perf("momentum", "ai_ml", 365, n_pred=2, n_succ=2)
    df_all = get_signal_performance(sector_id=None)
    assert len(df_all) == 1
    assert df_all.iloc[0]["signal_name"] == "momentum"
    assert df_all.iloc[0]["n_predictions"] == 5


def test_signal_performance_accuracy_computed(tmp_db):
    _seed_signal_perf("sentiment", None, 180, n_pred=8, n_succ=6)
    df = get_signal_performance(sector_id=None)
    row = df[df["signal_name"] == "sentiment"].iloc[0]
    assert row["accuracy"] == pytest.approx(0.75)


# --------------------------------------------------------- get_alpha_by_horizon


def test_alpha_by_horizon_empty(tmp_db):
    df = get_alpha_by_horizon()
    assert df.empty


def test_alpha_by_horizon_aggregates_by_milestone(tmp_db):
    t1 = _seed_thesis("NVDA")
    t2 = _seed_thesis("AMD")
    _seed_eval(t1, days=180, status="success", alpha_pct=0.20)
    _seed_eval(t2, days=180, status="failure", alpha_pct=-0.10)
    _seed_eval(t1, days=365, status="success", alpha_pct=0.30)

    df = get_alpha_by_horizon()
    r180 = df[df["horizon_days"] == 180].iloc[0]
    r365 = df[df["horizon_days"] == 365].iloc[0]

    assert r180["n_predictions"] == 2
    assert r180["mean_alpha"] == pytest.approx(0.05)  # (0.20 - 0.10) / 2
    assert r180["success_rate"] == pytest.approx(0.5)

    assert r365["n_predictions"] == 1
    assert r365["mean_alpha"] == pytest.approx(0.30)
    assert r365["success_rate"] == pytest.approx(1.0)


def test_alpha_by_horizon_excludes_active(tmp_db):
    t = _seed_thesis()
    _seed_eval(t, days=30, status="active", alpha_pct=None)
    assert get_alpha_by_horizon().empty


# ---------------------------------------------------------------- get_alerts


def test_get_alerts_empty(tmp_db):
    df = get_alerts()
    assert df.empty


def test_get_alerts_returns_all(tmp_db):
    _seed_alert("new_thesis", "info")
    _seed_alert("sector_heat_surge", "critical")
    df = get_alerts()
    assert len(df) == 2


def test_get_alerts_filter_severity(tmp_db):
    _seed_alert("r1", "info")
    _seed_alert("r2", "critical")
    df = get_alerts(severity="critical")
    assert len(df) == 1
    assert df.iloc[0]["severity"] == "critical"


def test_get_alerts_filter_acknowledged(tmp_db):
    _seed_alert("r1", acknowledged=False)
    _seed_alert("r2", acknowledged=True)
    df_unack = get_alerts(acknowledged=False)
    df_ack = get_alerts(acknowledged=True)
    assert len(df_unack) == 1
    assert len(df_ack) == 1


# ------------------------------------------------------- acknowledge_alert


def test_acknowledge_alert_marks_as_treated(tmp_db):
    aid = _seed_alert("r1", acknowledged=False)
    result = acknowledge_alert(aid)
    assert result is True
    df = get_alerts(acknowledged=True)
    assert any(df["id"] == aid)


def test_acknowledge_alert_returns_false_for_unknown_id(tmp_db):
    assert acknowledge_alert(999) is False


# ----------------------------------------------------- get_theses_history


def test_get_theses_history_empty(tmp_db):
    df = get_theses_history()
    assert df.empty


def test_get_theses_history_returns_all_theses(tmp_db):
    _seed_thesis("NVDA")
    _seed_thesis("AMD")
    df = get_theses_history()
    assert len(df) == 2


def test_get_theses_history_latest_status(tmp_db):
    th = _seed_thesis("NVDA")
    _seed_eval(th, days=30, status="active", alpha_pct=None)
    _seed_eval(th, days=180, status="success", alpha_pct=0.15)

    df = get_theses_history()
    row = df[df["thesis_id"] == th].iloc[0]
    # Le jalon le plus récent (180) doit être retourné.
    assert row["latest_status"] == "success"
    assert row["latest_days"] == 180


def test_get_theses_history_no_eval_shows_none(tmp_db):
    _seed_thesis("NVDA")
    df = get_theses_history()
    assert df.iloc[0]["latest_status"] is None


def test_get_theses_history_filter_by_status(tmp_db):
    t1 = _seed_thesis("NVDA")
    t2 = _seed_thesis("AMD")
    _seed_eval(t1, status="success", alpha_pct=0.10)
    _seed_eval(t2, status="failure", alpha_pct=-0.10)

    df = get_theses_history(status_filter=["success"])
    assert len(df) == 1
    assert df.iloc[0]["asset_id"] == "NVDA"


def test_get_theses_history_filter_by_date(tmp_db):
    _seed_thesis("NVDA", created_at=datetime(2026, 1, 1))
    _seed_thesis("AMD", created_at=datetime(2026, 6, 1))

    df = get_theses_history(date_to=datetime(2026, 3, 1))
    assert len(df) == 1
    assert df.iloc[0]["asset_id"] == "NVDA"


# ----------------------------------------------------- get_thesis_detail


def _seed_close(
    ticker: str, close: float, content_at: datetime,
    fetched_at: datetime | None = None,
) -> None:
    fetched_at = fetched_at or content_at
    with session_scope() as s:
        s.add(RawData(
            source="yfinance",
            entity_type="ohlcv_daily",
            entity_id=f"{ticker}:{content_at.strftime('%Y-%m-%d')}",
            fetched_at=fetched_at,
            content_at=content_at,
            payload_json=json.dumps({"ticker": ticker, "close": close}),
            hash=f"h-{ticker}-{content_at.strftime('%Y-%m-%d-%H-%M')}",
        ))


def test_list_thesis_ids_orders_by_recent_first(tmp_db):
    t1 = _seed_thesis("NVDA", created_at=datetime(2026, 1, 1))
    t2 = _seed_thesis("AMD", created_at=datetime(2026, 4, 1))
    out = list_thesis_ids()
    assert [tid for tid, _, _ in out] == [t2, t1]
    assert out[0][1] == "AMD"


def test_get_thesis_detail_unknown_returns_none(tmp_db):
    assert get_thesis_detail(99999) is None


def test_get_thesis_detail_returns_full_payload(tmp_db):
    # Thèse riche : breakdown, triggers, risks, catalysts, entry_conditions.
    with session_scope() as s:
        th = Thesis(
            created_at=datetime(2026, 1, 1),
            asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
            score=82.5,
            score_breakdown_json=json.dumps({
                "dimensions": {"momentum": 90.0, "sentiment": 70.0},
                "details": {"momentum": {"inputs": {"rsi_14": 65.0}}},
            }),
            recommendation="BUY", horizon_days=180, entry_price=120.0,
            entry_conditions_json=json.dumps({"band_pct": 0.03}),
            triggers_json=json.dumps([{"dimension": "momentum", "sub_score": 90.0}]),
            risks_json=json.dumps([
                {"category": "macro", "description": "Taux"},
            ]),
            catalysts_json=json.dumps([
                {"type": "fda_approval", "date": "2026-03-01",
                 "description": "Approval", "source": "fda"},
            ]),
            narrative="Markdown narrative.",
            model_version="v1_test",
            weights_snapshot_json="{}",
        )
        s.add(th)
        s.flush()
        thesis_id = th.id

    _seed_eval(thesis_id, days=30, status="active", alpha_pct=None)
    _seed_eval(thesis_id, days=180, status="success", alpha_pct=0.12)

    detail = get_thesis_detail(thesis_id)
    assert detail is not None
    assert detail["thesis"]["asset_id"] == "NVDA"
    assert detail["thesis"]["score"] == pytest.approx(82.5)
    assert detail["dimensions"] == {"momentum": 90.0, "sentiment": 70.0}
    assert detail["entry_conditions"]["band_pct"] == pytest.approx(0.03)
    assert len(detail["triggers"]) == 1
    assert len(detail["risks"]) == 1
    assert detail["catalysts"][0]["type"] == "fda_approval"
    assert len(detail["evaluations"]) == 2
    # Le dernier jalon (180) doit être le "latest".
    assert detail["latest_status"] == "success"
    assert detail["latest_alpha"] == pytest.approx(0.12)


def test_get_thesis_detail_handles_malformed_json(tmp_db):
    """Un JSON invalide ne doit pas faire planter la page détail."""
    with session_scope() as s:
        th = Thesis(
            created_at=datetime(2026, 1, 1),
            asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
            score=70.0,
            score_breakdown_json="not json",
            recommendation="WATCH", horizon_days=180, entry_price=100.0,
            triggers_json="not json",
            risks_json="not json",
            catalysts_json="not json",
            narrative="…",
            model_version="v1",
            weights_snapshot_json="{}",
        )
        s.add(th)
        s.flush()
        tid = th.id

    detail = get_thesis_detail(tid)
    assert detail is not None
    assert detail["dimensions"] == {}
    assert detail["triggers"] == []
    assert detail["risks"] == []
    assert detail["catalysts"] == []


# ----------------------------------------------------- get_price_history


def test_get_price_history_empty(tmp_db):
    df = get_price_history(
        "NVDA",
        start=datetime(2026, 1, 1),
        end=datetime(2026, 4, 1),
    )
    assert df.empty
    assert list(df.columns) == ["date", "close"]


def test_get_price_history_filters_window_and_ticker(tmp_db):
    _seed_close("NVDA", 100.0, datetime(2026, 3, 1))
    _seed_close("NVDA", 110.0, datetime(2026, 3, 5))
    _seed_close("AMD", 50.0, datetime(2026, 3, 1))    # autre ticker
    _seed_close("NVDA", 200.0, datetime(2025, 1, 1))  # hors fenêtre

    df = get_price_history(
        "NVDA",
        start=datetime(2026, 2, 1),
        end=datetime(2026, 4, 1),
    )
    assert list(df["close"]) == [100.0, 110.0]


def test_get_price_history_respects_fetched_before(tmp_db):
    """Un close fetché APRÈS `fetched_before` n'est pas visible (PIT)."""
    # Close datant du 5/3 mais ingéré le 10/3.
    _seed_close(
        "NVDA", 110.0,
        content_at=datetime(2026, 3, 5),
        fetched_at=datetime(2026, 3, 10),
    )
    df = get_price_history(
        "NVDA",
        start=datetime(2026, 2, 1),
        end=datetime(2026, 4, 1),
        fetched_before=datetime(2026, 3, 7),  # avant la fetch → exclu
    )
    assert df.empty
