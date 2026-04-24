"""Tests du schéma SQLite et des contraintes d'intégrité."""
from __future__ import annotations

import pytest
from sqlalchemy import inspect, select
from sqlalchemy.exc import IntegrityError

from memory.database import (
    RawData,
    Thesis,
    get_engine,
    session_scope,
    utc_now,
)


def test_init_db_creates_all_tables(tmp_db):
    """Les 7 tables du schéma (section 5) doivent être créées."""
    insp = inspect(get_engine())
    expected = {
        "raw_data", "features", "sectors", "theses",
        "evaluations", "signal_performance", "alerts",
    }
    assert expected.issubset(set(insp.get_table_names()))


def test_raw_data_insert_and_query(tmp_db):
    """Insertion simple puis lecture."""
    now = utc_now()
    with session_scope() as s:
        s.add(
            RawData(
                source="arxiv", entity_type="paper", entity_id="abc",
                fetched_at=now, content_at=now,
                payload_json='{"k":1}', hash="h1",
            )
        )
    with session_scope() as s:
        rows = s.execute(select(RawData)).scalars().all()
        assert len(rows) == 1
        assert rows[0].source == "arxiv"
        assert rows[0].entity_id == "abc"


def test_raw_data_unique_constraint_rejects_duplicate(tmp_db):
    """La contrainte UNIQUE(source, entity_id, hash) doit bloquer les doublons."""
    now = utc_now()
    with session_scope() as s:
        s.add(
            RawData(
                source="arxiv", entity_type="paper", entity_id="abc",
                fetched_at=now, payload_json="{}", hash="h1",
            )
        )
    with pytest.raises(IntegrityError):
        with session_scope() as s:
            s.add(
                RawData(
                    source="arxiv", entity_type="paper", entity_id="abc",
                    fetched_at=now, payload_json="{}", hash="h1",
                )
            )


def test_raw_data_allows_same_entity_different_hash(tmp_db):
    """Deux versions d'une même entité (hash différent) sont autorisées.

    C'est le mécanisme prévu par la spec pour le versioning implicite :
    une correction d'une donnée arrive comme nouvelle ligne sans écraser
    l'ancienne (section 5.2).
    """
    now = utc_now()
    with session_scope() as s:
        s.add(RawData(source="arxiv", entity_type="paper", entity_id="abc",
                      fetched_at=now, payload_json="{}", hash="h1"))
        s.add(RawData(source="arxiv", entity_type="paper", entity_id="abc",
                      fetched_at=now, payload_json="{}", hash="h2"))
    with session_scope() as s:
        rows = s.execute(select(RawData)).scalars().all()
        assert len(rows) == 2


def test_theses_basic_insert(tmp_db):
    """Insertion d'une thèse avec tous les champs obligatoires."""
    with session_scope() as s:
        s.add(
            Thesis(
                asset_type="stock", asset_id="NVDA", sector_id="ai_ml",
                score=78.0, score_breakdown_json="{}", recommendation="BUY",
                horizon_days=180, triggers_json="[]", risks_json="[]",
                narrative="test", model_version="rules-v0",
                weights_snapshot_json="{}",
            )
        )
    with session_scope() as s:
        rows = s.execute(select(Thesis)).scalars().all()
        assert len(rows) == 1
        assert rows[0].recommendation == "BUY"
        assert rows[0].score == 78.0
