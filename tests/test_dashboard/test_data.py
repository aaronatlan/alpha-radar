"""Tests des accesseurs lecture seule du dashboard."""
from __future__ import annotations

import json
from datetime import datetime, timedelta

from dashboard._data import (
    get_collector_health,
    get_feature_freshness,
    get_sector_heat_scores,
    get_stock_scores,
)
from memory.database import Feature, RawData, session_scope


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
