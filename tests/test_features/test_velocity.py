"""Tests de la feature `ArxivVelocityFeature`.

On seede directement `raw_data` avec des papiers fabriqués de toutes
pièces — pas d'I/O réseau. Les tests couvrent :
  - le calcul du ratio dans le cas nominal,
  - les cas limites (référence nulle, fenêtres vides),
  - le double filtre PIT (content_at + fetched_at),
  - le filtre par catégorie,
  - le mode dégradé (target invalide n'interrompt pas `run`).
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Iterable

import pytest

from features.velocity import ArxivVelocityFeature
from memory.database import Feature, RawData, session_scope


def _seed_paper(
    entity_id: str,
    primary_category: str,
    content_at: datetime,
    fetched_at: datetime,
) -> None:
    """Insère un papier arXiv synthétique dans raw_data."""
    payload = {
        "primary_category": primary_category,
        "title": f"Paper {entity_id}",
    }
    with session_scope() as session:
        session.add(
            RawData(
                source="arxiv",
                entity_type="paper",
                entity_id=entity_id,
                fetched_at=fetched_at,
                content_at=content_at,
                payload_json=json.dumps(payload),
                hash=entity_id,  # unique → pas de collision dédupe
            )
        )


def _seed_many(specs: Iterable[tuple[str, str, datetime, datetime]]) -> None:
    for spec in specs:
        _seed_paper(*spec)


def _get_feature(name: str, target_id: str) -> Feature | None:
    with session_scope() as session:
        return (
            session.query(Feature)
            .filter_by(feature_name=name, target_id=target_id)
            .order_by(Feature.computed_at.desc())
            .first()
        )


# ----------------------------------------------------------------- compute

def test_compute_nominal_ratio(tmp_db):
    """recent_rate / ref_rate avec deux fenêtres non vides."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=1)

    # Fenêtre récente [as_of-7j, as_of) : 7 papiers (→ 1/jour)
    specs = [
        (f"r{i}", "cs.LG", as_of - timedelta(days=1, hours=i), fetched)
        for i in range(7)
    ]
    # Fenêtre référence [as_of-37j, as_of-7j) : 15 papiers (→ 0.5/jour)
    specs += [
        (f"b{i}", "cs.LG", as_of - timedelta(days=20, hours=i), fetched)
        for i in range(15)
    ]
    _seed_many(specs)

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    result = feat.compute("ai_ml", as_of)

    assert result is not None
    value, meta = result
    # recent=7/7=1.0 ; ref=15/30=0.5 ; ratio=2.0
    assert value == pytest.approx(2.0, rel=1e-6)
    assert meta["n_recent"] == 7
    assert meta["n_reference"] == 15


def test_compute_zero_activity_returns_zero(tmp_db):
    """Aucun papier dans aucune des deux fenêtres → 0.0, pas None."""
    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    result = feat.compute("ai_ml", datetime(2026, 2, 1))
    assert result is not None
    value, meta = result
    assert value == 0.0
    assert meta["n_recent"] == 0 and meta["n_reference"] == 0


def test_compute_empty_reference_saturates(tmp_db):
    """Référence vide, récent non vide → vélocité plafonnée à max_ratio."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=1)
    _seed_many([
        (f"r{i}", "cs.LG", as_of - timedelta(hours=i + 1), fetched)
        for i in range(3)
    ])

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    value, _ = feat.compute("ai_ml", as_of)
    assert value == feat.max_ratio


def test_compute_ratio_is_capped(tmp_db):
    """Pic d'activité > max_ratio → saturation."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=1)
    # 100 papiers récents, 1 seul en référence → ratio théorique ~429 >> 10
    specs = [
        (f"r{i}", "cs.LG", as_of - timedelta(hours=i + 1), fetched)
        for i in range(100)
    ]
    specs += [("b0", "cs.LG", as_of - timedelta(days=20), fetched)]
    _seed_many(specs)

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    value, _ = feat.compute("ai_ml", as_of)
    assert value == feat.max_ratio


# ------------------------------------------------------------ PIT filtering


def test_compute_ignores_content_after_as_of(tmp_db):
    """Un papier publié après `as_of` ne doit jamais compter."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=10)
    _seed_many([
        ("future", "cs.LG", as_of + timedelta(days=1), fetched),
        ("past_ref", "cs.LG", as_of - timedelta(days=20), fetched),
    ])

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    _, meta = feat.compute("ai_ml", as_of)
    assert meta["n_recent"] == 0  # rien dans [as_of-7, as_of)
    assert meta["n_reference"] == 1  # le "past_ref"


def test_compute_ignores_rows_fetched_after_as_of(tmp_db):
    """Un papier avec content_at valide mais fetched_at > as_of est ignoré —
    on ne pouvait pas "connaître" cette donnée à as_of."""
    as_of = datetime(2026, 2, 1)
    _seed_paper(
        entity_id="backfilled",
        primary_category="cs.LG",
        content_at=as_of - timedelta(days=3),   # dans la fenêtre récente
        fetched_at=as_of + timedelta(days=5),   # mais collecté APRÈS as_of
    )

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    _, meta = feat.compute("ai_ml", as_of)
    assert meta["n_recent"] == 0


def test_compute_filters_by_primary_category(tmp_db):
    """Un papier d'une autre catégorie ne doit pas être compté."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=1)
    _seed_many([
        ("ai", "cs.LG", as_of - timedelta(days=1), fetched),      # ai_ml
        ("bio", "q-bio", as_of - timedelta(days=1), fetched),      # biotech
        ("vision", "cs.CV", as_of - timedelta(days=1), fetched),   # computer_vision
    ])

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    _, meta = feat.compute("ai_ml", as_of)
    assert meta["n_recent"] == 1  # seul "ai" compte pour ai_ml


# ----------------------------------------------------------------- run


def test_run_stores_and_deduplicates(tmp_db):
    """Deux runs au même `as_of` : le second est un no-op (UNIQUE)."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=1)
    _seed_paper("r0", "cs.LG", as_of - timedelta(days=1), fetched)

    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])
    n1 = feat.run(as_of=as_of)
    n2 = feat.run(as_of=as_of)

    assert n1 == 1
    assert n2 == 0

    row = _get_feature("arxiv_velocity", "ai_ml")
    assert row is not None
    assert row.target_type == "sector"
    assert row.computed_at == as_of
    meta = json.loads(row.metadata_json)
    assert meta["n_recent"] == 1


def test_run_skips_none_results_silently(tmp_db):
    """compute() → None ne génère aucune ligne (pas d'erreur, pas d'écriture)."""
    as_of = datetime(2026, 2, 1)
    feat = ArxivVelocityFeature(sector_ids=["ai_ml"])

    # Si recent et ref sont tous deux vides, on renvoie 0.0 (informatif)
    # donc pour tester le skip None il faut forcer compute → None.
    feat.compute = lambda target_id, ts: None  # type: ignore[method-assign]
    assert feat.run(as_of=as_of) == 0


def test_run_continues_after_target_exception(tmp_db):
    """Mode dégradé : une exception sur un target n'interrompt pas les autres."""
    as_of = datetime(2026, 2, 1)
    fetched = as_of - timedelta(days=1)
    _seed_paper("r0", "cs.CR", as_of - timedelta(days=1), fetched)

    feat = ArxivVelocityFeature(sector_ids=["ai_ml", "cybersecurity"])

    original_compute = feat.compute

    def flaky(target_id, ts):
        if target_id == "ai_ml":
            raise RuntimeError("boom")
        return original_compute(target_id, ts)

    feat.compute = flaky  # type: ignore[method-assign]
    n = feat.run(as_of=as_of)
    assert n == 1  # seul cybersecurity a été écrit

    assert _get_feature("arxiv_velocity", "ai_ml") is None
    assert _get_feature("arxiv_velocity", "cybersecurity") is not None


# ----------------------------------------------------------------- validation


def test_unknown_sector_id_is_rejected():
    with pytest.raises(ValueError):
        ArxivVelocityFeature(sector_ids=["not_a_sector"])
