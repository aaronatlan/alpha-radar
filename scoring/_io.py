"""Helpers d'accès à la table `features` pour le module scoring.

Isolé ici pour être réutilisé par tous les scoreurs sans dupliquer la
logique de lookup point-in-time.
"""
from __future__ import annotations

from datetime import datetime

from memory.database import Feature, session_scope


def latest_feature_value(
    feature_name: str,
    target_type: str,
    target_id: str,
    as_of: datetime,
) -> float | None:
    """Dernière valeur d'une feature calculée **à ou avant** `as_of`.

    Discipline PIT : on refuse toute ligne `computed_at > as_of`. Si
    aucune ligne ne qualifie, on retourne `None` (absence de donnée,
    pas substitution par 0).
    """
    with session_scope() as session:
        row = (
            session.query(Feature)
            .filter(
                Feature.feature_name == feature_name,
                Feature.target_type == target_type,
                Feature.target_id == target_id,
                Feature.computed_at <= as_of,
            )
            .order_by(Feature.computed_at.desc())
            .first()
        )
        return float(row.value) if row is not None else None
