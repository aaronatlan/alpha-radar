"""Tests du scoreur `SectorHeatScorer`.

On seede directement `features` avec des valeurs d'`arxiv_velocity` et
vérifie la conversion en Heat Score 0-100, la discipline PIT et les
cas limites.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from features.velocity import ArxivVelocityFeature
from memory.database import Feature, session_scope
from scoring.sector_heat import SectorHeatScorer, _velocity_to_heat_score


def _seed_velocity(sector_id: str, value: float, computed_at: datetime) -> None:
    with session_scope() as session:
        session.add(
            Feature(
                feature_name="arxiv_velocity",
                target_type="sector",
                target_id=sector_id,
                computed_at=computed_at,
                value=value,
                metadata_json=None,
            )
        )


def _get_heat(sector_id: str) -> Feature | None:
    with session_scope() as session:
        return (
            session.query(Feature)
            .filter_by(feature_name="sector_heat_score", target_id=sector_id)
            .order_by(Feature.computed_at.desc())
            .first()
        )


# --- mapping velocity → score ----------------------------------------------


@pytest.mark.parametrize(
    "velocity,expected",
    [
        (0.0, 0.0),
        (0.5, 25.0),
        (1.0, 50.0),
        (2.0, 50.0 + 50.0 * (1.0 / (ArxivVelocityFeature.max_ratio - 1.0))),
        (ArxivVelocityFeature.max_ratio, 100.0),
        (50.0, 100.0),   # clip au-delà du max
        (-5.0, 0.0),     # clip en dessous de 0
    ],
)
def test_velocity_mapping(velocity, expected):
    got = _velocity_to_heat_score(velocity, max_ratio=ArxivVelocityFeature.max_ratio)
    assert got == pytest.approx(expected, rel=1e-9)


# --- compute ---------------------------------------------------------------


def test_compute_reads_velocity_and_produces_score(tmp_db):
    as_of = datetime(2026, 2, 1)
    _seed_velocity("ai_ml", value=2.0, computed_at=as_of - timedelta(hours=1))

    scorer = SectorHeatScorer(sector_ids=["ai_ml"])
    result = scorer.compute("ai_ml", as_of)

    assert result is not None
    score, meta = result
    expected = 50.0 + 50.0 * (2.0 - 1.0) / (ArxivVelocityFeature.max_ratio - 1.0)
    assert score == pytest.approx(expected, rel=1e-9)
    assert meta["model_version"] == "v1_velocity_only"
    assert meta["inputs"]["arxiv_velocity"] == 2.0


def test_compute_returns_none_when_no_velocity(tmp_db):
    """Pas de feature arxiv_velocity disponible → on ne fabrique pas de score."""
    scorer = SectorHeatScorer(sector_ids=["ai_ml"])
    assert scorer.compute("ai_ml", datetime(2026, 2, 1)) is None


def test_compute_ignores_velocity_computed_after_as_of(tmp_db):
    """PIT : une valeur calculée APRÈS as_of ne doit pas être lue."""
    as_of = datetime(2026, 2, 1)
    _seed_velocity("ai_ml", value=5.0, computed_at=as_of + timedelta(days=1))

    scorer = SectorHeatScorer(sector_ids=["ai_ml"])
    assert scorer.compute("ai_ml", as_of) is None


def test_compute_uses_latest_velocity_before_as_of(tmp_db):
    """Plusieurs valeurs avant as_of → on prend la plus récente."""
    as_of = datetime(2026, 2, 1)
    _seed_velocity("ai_ml", value=1.0, computed_at=as_of - timedelta(days=5))
    _seed_velocity("ai_ml", value=3.0, computed_at=as_of - timedelta(days=1))

    scorer = SectorHeatScorer(sector_ids=["ai_ml"])
    _, meta = scorer.compute("ai_ml", as_of)
    assert meta["inputs"]["arxiv_velocity"] == 3.0


# --- run -------------------------------------------------------------------


def test_run_stores_and_deduplicates(tmp_db):
    as_of = datetime(2026, 2, 1)
    _seed_velocity("ai_ml", value=1.0, computed_at=as_of - timedelta(hours=1))

    scorer = SectorHeatScorer(sector_ids=["ai_ml"])
    n1 = scorer.run(as_of=as_of)
    n2 = scorer.run(as_of=as_of)

    assert n1 == 1
    assert n2 == 0

    row = _get_heat("ai_ml")
    assert row is not None
    assert row.value == pytest.approx(50.0)
    meta = json.loads(row.metadata_json)
    assert meta["model_version"] == "v1_velocity_only"


def test_run_skips_sectors_without_input(tmp_db):
    """Un secteur sans vélocité stockée est simplement ignoré."""
    as_of = datetime(2026, 2, 1)
    _seed_velocity("ai_ml", value=2.0, computed_at=as_of - timedelta(hours=1))

    scorer = SectorHeatScorer(sector_ids=["ai_ml", "biotech"])
    n = scorer.run(as_of=as_of)
    assert n == 1
    assert _get_heat("ai_ml") is not None
    assert _get_heat("biotech") is None


def test_unknown_sector_id_is_rejected():
    with pytest.raises(ValueError):
        SectorHeatScorer(sector_ids=["not_a_sector"])
