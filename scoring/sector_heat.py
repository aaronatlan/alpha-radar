"""Heat Score sectoriel — agrégation des signaux sectoriels en 0-100.

Le Heat Score mesure à quel point un secteur "chauffe" à un instant
donné : c'est la porte d'entrée du système (niveau 1 de l'architecture
à trois niveaux — SPEC §3.1). Il est lu par les scoreurs d'actifs pour
moduler leur recommandation.

Convention de mapping velocity → score (v1) :

    v = 0            →   0  (secteur inactif)
    v = 1            →  50  (régime stationnaire, rythme historique)
    v = max_ratio    → 100  (saturation, pic clair d'activité)

Interpolation linéaire par morceaux : pénalité symétrique pour les
ralentissements (v < 1) et bonus pour les accélérations (v > 1). Les
seuils sont choisis pour qu'un score ≥ 70 corresponde à un régime
2× au-dessus de la normale — c'est le seuil au-delà duquel le SPEC
§7.7 déclenche des alertes "sector heat".

Phase 2 v1 : n'utilise que `arxiv_velocity`. Les signaux GitHub /
SEC / news seront pondérés dans une v2 (étape 4+) sans toucher au
mapping final.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any

from config.sectors import SECTORS, SECTORS_BY_ID
from features.base import BaseFeature
from features.velocity import ArxivVelocityFeature, GitHubStarsVelocityFeature
from scoring._io import latest_feature_value
from scoring.weights import SECTOR_HEAT_WEIGHTS


# Chaque input de Heat Score porte un `max_ratio` propre (cf. feature
# d'origine). Extensible : ajouter ici quand un nouvel input est intégré.
_INPUT_MAX_RATIO: dict[str, float] = {
    "arxiv_velocity": ArxivVelocityFeature.max_ratio,
    "github_stars_velocity": GitHubStarsVelocityFeature.max_ratio,
}


class SectorHeatScorer(BaseFeature):
    """Agrège les signaux sectoriels en un score 0-100 par secteur.

    Hérite de `BaseFeature` pour réutiliser l'orchestration
    `run()` / `INSERT OR IGNORE`. Le résultat est persisté dans
    `features` avec `feature_name="sector_heat_score"`.
    """

    feature_name = "sector_heat_score"
    target_type = "sector"
    model_version = "v1_velocity_only"

    def __init__(self, sector_ids: list[str] | None = None) -> None:
        super().__init__()
        if sector_ids is not None:
            unknown = set(sector_ids) - set(SECTORS_BY_ID)
            if unknown:
                raise ValueError(f"Secteurs inconnus : {sorted(unknown)}")
            self._sector_ids = list(sector_ids)
        else:
            self._sector_ids = [s["id"] for s in SECTORS]
        self._weights = SECTOR_HEAT_WEIGHTS[self.model_version]

    def targets(self) -> list[str]:
        return list(self._sector_ids)

    def compute(
        self, target_id: str, as_of: datetime
    ) -> tuple[float, dict[str, Any]] | None:
        # Chaque input est lu PIT et mappé individuellement sur [0, 100]
        # via `_velocity_to_heat_score` (tous nos inputs actuels sont des
        # vélocités bornées). Les inputs manquants sont simplement sautés
        # et on renormalise les poids sur ceux qui restent.
        inputs: dict[str, float] = {}
        subscores: dict[str, float] = {}
        for input_name, weight in self._weights.items():
            raw = latest_feature_value(input_name, "sector", target_id, as_of)
            if raw is None:
                continue
            max_r = _INPUT_MAX_RATIO.get(input_name)
            if max_r is None:
                # Input non mappé — on s'abstient plutôt que d'appliquer
                # une transformation par défaut qui serait silencieuse.
                continue
            inputs[input_name] = raw
            subscores[input_name] = _velocity_to_heat_score(raw, max_ratio=max_r)

        if not subscores:
            return None

        total_w = sum(self._weights[k] for k in subscores)
        score = sum(self._weights[k] * subscores[k] for k in subscores) / total_w

        metadata = {
            "model_version": self.model_version,
            "weights": dict(self._weights),
            "inputs": inputs,
            "subscores": subscores,
        }
        return score, metadata


def _velocity_to_heat_score(v: float, max_ratio: float) -> float:
    """Mappe une vélocité ∈ [0, max_ratio] sur [0, 100].

    Linéaire par morceaux avec v=1 → 50 (régime stationnaire).
    Valeurs hors bornes clippées aux extrêmes.
    """
    v = max(0.0, min(v, max_ratio))
    if v <= 1.0:
        return 50.0 * v
    return 50.0 + 50.0 * (v - 1.0) / (max_ratio - 1.0)
