"""Features de vélocité — dérivée temporelle d'un compteur d'événements.

La vélocité est le **ratio des taux** entre une fenêtre récente et une
fenêtre de référence plus longue qui la précède. Un ratio supérieur à
1 indique une accélération du flux d'événements, inférieur à 1 un
ralentissement.

Motivation : capter l'inflexion plus tôt que le volume brut. Un secteur
qui génère 50 papiers/semaine depuis 6 mois et passe soudain à 100 est
un signal fort ; 50 en absolu serait trivialement noyé dans le bruit.

Définition retenue
------------------
Pour une fenêtre récente R (par défaut 7 j) et une fenêtre de référence B
(par défaut 30 j, **non chevauchante**, immédiatement avant R) :

    recent_rate    = N_recent / |R|        (événements/jour)
    reference_rate = N_reference / |B|
    velocity       = recent_rate / reference_rate

Cas limites :
  - reference_rate == 0 et recent_rate == 0 : 0.0 (secteur inactif)
  - reference_rate == 0 et recent_rate  > 0 : capped à `max_ratio`
    (saturation — un "ex nihilo" a bien une signification mais on
    évite des valeurs infinies dans les agrégations)
  - sinon : ratio plafonné à `max_ratio`

Point-in-time
-------------
Filtrage double sur `raw_data` :
  - `content_at` dans la fenêtre temporelle considérée
  - `fetched_at <= as_of` — on ne peut pas prétendre connaître à
    `as_of` une donnée qu'on n'avait pas matériellement collectée.
Cette discipline garantit qu'un backfill a posteriori ne fausse pas
un calcul historique.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from config.github_repos import GITHUB_REPOS, repo_full_name
from config.sectors import SECTORS, SECTORS_BY_ID
from features.base import BaseFeature
from memory.database import RawData, session_scope


class ArxivVelocityFeature(BaseFeature):
    """Vélocité du flux arXiv par secteur.

    Un secteur regroupe plusieurs catégories arXiv (cf. `config.sectors`).
    Les papiers comptés sont ceux dont `primary_category` est listée
    parmi celles du secteur.
    """

    feature_name = "arxiv_velocity"
    target_type = "sector"

    #: Fenêtre récente en jours.
    recent_window_days = 7
    #: Fenêtre de référence en jours, immédiatement avant la fenêtre récente.
    reference_window_days = 30
    #: Plafond appliqué au ratio pour borner les saturations.
    max_ratio = 10.0

    def __init__(self, sector_ids: list[str] | None = None) -> None:
        super().__init__()
        if sector_ids is not None:
            unknown = set(sector_ids) - set(SECTORS_BY_ID)
            if unknown:
                raise ValueError(f"Secteurs inconnus : {sorted(unknown)}")
            self._sector_ids = list(sector_ids)
        else:
            self._sector_ids = [s["id"] for s in SECTORS]

    def targets(self) -> list[str]:
        return list(self._sector_ids)

    def compute(
        self, target_id: str, as_of: datetime
    ) -> tuple[float, dict[str, Any]] | None:
        sector = SECTORS_BY_ID[target_id]
        categories = set(sector["arxiv_categories"])
        if not categories:
            return None

        recent_end = as_of
        recent_start = as_of - timedelta(days=self.recent_window_days)
        ref_end = recent_start
        ref_start = recent_start - timedelta(days=self.reference_window_days)

        n_recent = self._count_papers(categories, recent_start, recent_end, as_of)
        n_ref = self._count_papers(categories, ref_start, ref_end, as_of)

        recent_rate = n_recent / self.recent_window_days
        ref_rate = n_ref / self.reference_window_days

        if ref_rate == 0.0:
            value = 0.0 if recent_rate == 0.0 else self.max_ratio
        else:
            value = min(recent_rate / ref_rate, self.max_ratio)

        metadata = {
            "n_recent": n_recent,
            "n_reference": n_ref,
            "recent_window_days": self.recent_window_days,
            "reference_window_days": self.reference_window_days,
            "categories": sorted(categories),
        }
        return value, metadata

    # --- helpers --------------------------------------------------------

    @staticmethod
    def _count_papers(
        categories: set[str],
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> int:
        """Compte les papiers arXiv dans [start, end) dont `primary_category`
        appartient à `categories`, en respectant le double filtre PIT.

        Le champ `primary_category` vit dans `payload_json` ; on filtre
        d'abord par source + fenêtre temporelle en SQL, puis en Python
        sur la catégorie. Le volume reste raisonnable aux échelles
        visées (< 10k papiers/mois toutes catégories confondues).
        """
        stmt = (
            select(RawData.payload_json)
            .where(RawData.source == "arxiv")
            .where(RawData.entity_type == "paper")
            .where(RawData.content_at >= start)
            .where(RawData.content_at < end)
            .where(RawData.fetched_at <= as_of)
        )
        count = 0
        with session_scope() as session:
            for (payload_json,) in session.execute(stmt):
                try:
                    payload = json.loads(payload_json)
                except (TypeError, ValueError):
                    continue
                if payload.get("primary_category") in categories:
                    count += 1
        return count


class GitHubStarsVelocityFeature(BaseFeature):
    """Vélocité des stars GitHub par secteur.

    Pour chaque secteur, on agrège les dépôts listés dans
    `config.github_repos` qui déclarent ce secteur, et on mesure la
    **croissance absolue** des stars sur une fenêtre récente de 7 j vs
    une fenêtre de référence de 30 j (non chevauchantes, référence
    précédant la récente) — même convention que la vélocité arXiv.

    Les snapshots GitHub (`entity_type='github_repo_snapshot'`) étant
    quotidiens, on lit le premier et le dernier snapshot de chaque
    fenêtre pour calculer `delta_stars`. Si un dépôt manque un
    snapshot dans une fenêtre, il est ignoré **pour cette fenêtre-là**
    (on ne fabrique pas de delta imputé).

    Cas limites identiques à arXiv : les deux rates à 0 donnent 0.0 ;
    référence nulle + récent > 0 sature à `max_ratio`.
    """

    feature_name = "github_stars_velocity"
    target_type = "sector"
    recent_window_days = 7
    reference_window_days = 30
    max_ratio = 10.0

    def __init__(self, sector_ids: list[str] | None = None) -> None:
        super().__init__()
        if sector_ids is not None:
            unknown = set(sector_ids) - set(SECTORS_BY_ID)
            if unknown:
                raise ValueError(f"Secteurs inconnus : {sorted(unknown)}")
            self._sector_ids = list(sector_ids)
        else:
            self._sector_ids = [s["id"] for s in SECTORS]

    def targets(self) -> list[str]:
        return list(self._sector_ids)

    def compute(self, target_id: str, as_of: datetime):
        # Liste des dépôts rattachés à ce secteur.
        repo_names = [
            repo_full_name(r) for r in GITHUB_REPOS
            if target_id in r["sectors"]
        ]
        if not repo_names:
            return None

        recent_end = as_of
        recent_start = as_of - timedelta(days=self.recent_window_days)
        ref_end = recent_start
        ref_start = recent_start - timedelta(days=self.reference_window_days)

        d_recent = self._delta_stars(repo_names, recent_start, recent_end, as_of)
        d_ref = self._delta_stars(repo_names, ref_start, ref_end, as_of)

        recent_rate = d_recent / self.recent_window_days
        ref_rate = d_ref / self.reference_window_days

        if ref_rate <= 0.0:
            value = 0.0 if recent_rate <= 0.0 else self.max_ratio
        else:
            value = max(0.0, min(recent_rate / ref_rate, self.max_ratio))

        metadata = {
            "delta_stars_recent": d_recent,
            "delta_stars_reference": d_ref,
            "repos": repo_names,
        }
        return value, metadata

    @staticmethod
    def _delta_stars(
        repo_names: list[str],
        start: datetime,
        end: datetime,
        as_of: datetime,
    ) -> float:
        """Somme sur les dépôts de (dernier - premier) snapshot stars
        dans [start, end), sous contrainte PIT `fetched_at <= as_of`.

        Un dépôt avec un seul snapshot dans la fenêtre renvoie 0
        (pas de delta calculable). Les deltas négatifs (rare, mais
        possibles en cas de purge de bots) sont conservés tels quels —
        la normalisation en ratio s'en occupe.
        """
        total = 0.0
        with session_scope() as session:
            for full_name in repo_names:
                stmt = (
                    select(RawData.content_at, RawData.payload_json)
                    .where(RawData.source == "github")
                    .where(RawData.entity_type == "github_repo_snapshot")
                    .where(RawData.content_at >= start)
                    .where(RawData.content_at < end)
                    .where(RawData.fetched_at <= as_of)
                    .order_by(RawData.content_at.asc())
                )
                snapshots: list[tuple[datetime, int]] = []
                for content_at, payload_json in session.execute(stmt):
                    try:
                        payload = json.loads(payload_json)
                    except (TypeError, ValueError):
                        continue
                    if payload.get("full_name") != full_name:
                        continue
                    stars = payload.get("stars")
                    if stars is None:
                        continue
                    snapshots.append((content_at, int(stars)))
                if len(snapshots) >= 2:
                    total += float(snapshots[-1][1] - snapshots[0][1])
        return total
