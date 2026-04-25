"""Classe abstraite commune à toutes les features.

Une feature est une fonction **pure et point-in-time** : pour un
`target_id` et un instant `as_of` donnés, elle produit une valeur
numérique calculée exclusivement à partir de données disponibles à
`as_of` ou avant. Deux calculs à la même date produisent la même
valeur — d'où la contrainte UNIQUE de `features`.

Chaque feature concrète :
  1. Définit `feature_name` (str) et `target_type` ('sector' | 'asset').
  2. Implémente `targets()` — liste des `target_id` à évaluer.
  3. Implémente `compute(target_id, as_of)` — calcul pur PIT.

La méthode `run(as_of)` itère sur `targets()`, appelle `compute()` et
stocke les résultats dans `features` via `INSERT OR IGNORE`.

Mode dégradé
------------
Comme pour les collecteurs, une exception sur un target isolé est
loggée mais n'interrompt pas le run complet. Un calcul qui renvoie
`None` signifie "valeur non définie à cette date" (ex. fenêtre de
données vide) — rien n'est écrit en base.
"""
from __future__ import annotations

import json
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any, Literal

from loguru import logger
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from memory.database import Feature, session_scope, utc_now


TargetType = Literal["sector", "asset"]


class BaseFeature(ABC):
    """Base abstraite pour toute feature calculée."""

    #: Nom stable de la feature (colonne `features.feature_name`).
    feature_name: str = ""

    #: Type de cible : 'sector' ou 'asset'.
    target_type: TargetType = "asset"

    def __init__(self) -> None:
        if not self.feature_name:
            raise ValueError(
                f"{type(self).__name__} doit définir `feature_name`."
            )
        if self.target_type not in ("sector", "asset"):
            raise ValueError(
                f"{type(self).__name__}.target_type doit valoir "
                "'sector' ou 'asset'."
            )

    # --- API à implémenter ----------------------------------------------

    @abstractmethod
    def targets(self) -> list[str]:
        """Liste des `target_id` pour lesquels calculer la feature."""

    @abstractmethod
    def compute(
        self, target_id: str, as_of: datetime
    ) -> float | tuple[float, dict[str, Any]] | None:
        """Calcule la valeur PIT pour `target_id` à `as_of`.

        Retourne :
          - un `float` (valeur simple), ou
          - un tuple `(value, metadata)` si des méta-infos sont à tracer
            (ex. nombre de points utilisés, fenêtre retenue), ou
          - `None` si la valeur est indéfinie — rien n'est écrit.
        """

    # --- Orchestration --------------------------------------------------

    def run(self, as_of: datetime | None = None) -> int:
        """Calcule la feature pour tous les targets et stocke les résultats.

        Retourne le nombre de lignes effectivement insérées (hors
        doublons via UNIQUE et `None` skippés).
        """
        ts = as_of or utc_now()
        targets = self.targets()
        logger.info(
            "[feature:{}] Calcul pour {} target(s) à {}",
            self.feature_name, len(targets), ts,
        )

        rows: list[dict[str, Any]] = []
        for target_id in targets:
            try:
                result = self.compute(target_id, ts)
            except Exception as exc:
                logger.warning(
                    "[feature:{}] compute({}) a échoué : {}",
                    self.feature_name, target_id, exc,
                )
                continue
            if result is None:
                continue

            if isinstance(result, tuple):
                value, metadata = result
            else:
                value, metadata = result, None

            rows.append(
                {
                    "feature_name": self.feature_name,
                    "target_type": self.target_type,
                    "target_id": target_id,
                    "computed_at": ts,
                    "value": float(value),
                    "metadata_json": (
                        json.dumps(metadata, default=str, ensure_ascii=False)
                        if metadata else None
                    ),
                }
            )

        if not rows:
            return 0

        inserted = 0
        with session_scope() as session:
            for row in rows:
                stmt = (
                    sqlite_insert(Feature).values(**row).prefix_with("OR IGNORE")
                )
                inserted += session.execute(stmt).rowcount or 0

        logger.info(
            "[feature:{}] Terminé : {} calculé(s), {} inséré(s)",
            self.feature_name, len(rows), inserted,
        )
        return inserted
