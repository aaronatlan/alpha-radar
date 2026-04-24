"""Collecteur arXiv.

Utilise le client Python officiel `arxiv`. Aucune clé API n'est requise.
Couvre les catégories retournées par `config.sectors.all_arxiv_categories()`
(dédupliquées entre secteurs).

Point-in-time
-------------
Chaque papier est stocké avec `content_at = paper.published` (date de
première soumission sur arXiv). C'est cette date qui fait foi pour
toute feature calculée en aval. `fetched_at` (instant de l'appel) est
également conservé dans `raw_data`. En backtesting, une feature à
l'instant T devra filtrer :

    WHERE content_at  <= T_prédiction
      AND fetched_at  <= T_prédiction

Cette double condition empêche toute fuite d'information du futur même
si on a backfillé des données a posteriori : on ne peut pas prétendre
"connaître" à T une donnée qu'on n'avait matériellement pas collectée
à ce moment-là.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import arxiv
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.sectors import all_arxiv_categories


class ArxivCollector(BaseCollector):
    """Collecte les preprints arXiv pour les catégories configurées."""

    source_name = "arxiv"
    request_delay = 3.0              # politesse entre pages de résultats
    page_size = 100
    max_results_per_category = 500   # borne haute par run/catégorie

    def __init__(self, categories: list[str] | None = None) -> None:
        super().__init__()
        self.categories: list[str] = categories or all_arxiv_categories()
        self._client = arxiv.Client(
            page_size=self.page_size,
            delay_seconds=self.request_delay,
            num_retries=3,
        )

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[Any]:
        """Récupère les papiers publiés dans [since, until], par catégorie.

        arXiv renvoie les résultats triés par date de soumission
        décroissante. On itère et on sort dès qu'on dépasse `since` vers
        le passé — pas besoin de paginer l'intégralité de l'historique.
        """
        since_utc = _to_utc(since)
        until_utc = _to_utc(until)

        collected: list[Any] = []
        for category in self.categories:
            search = arxiv.Search(
                query=f"cat:{category}",
                max_results=self.max_results_per_category,
                sort_by=arxiv.SortCriterion.SubmittedDate,
                sort_order=arxiv.SortOrder.Descending,
            )
            n_kept = 0
            try:
                for result in self._client.results(search):
                    published = result.published
                    if published.tzinfo is None:
                        published = published.replace(tzinfo=timezone.utc)

                    if published < since_utc:
                        # Résultats triés décroissants : inutile de continuer.
                        break
                    if published > until_utc:
                        continue

                    collected.append(result)
                    n_kept += 1
            except Exception as exc:
                # Mode dégradé : une catégorie qui plante ne doit pas
                # empêcher les autres de s'exécuter.
                logger.warning(
                    "[arxiv] échec sur catégorie {} : {}", category, exc
                )
                continue

            logger.debug("[arxiv] {} : {} papiers retenus", category, n_kept)

        return collected

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: Any) -> NormalizedItem | None:
        """Convertit un `arxiv.Result` en item canonique.

        Le `content_at` retenu est `raw.published` — date de première
        soumission du preprint, en UTC, stocké naïf (convention du projet
        pour tous les timestamps en base).
        """
        if raw is None:
            return None

        entity_id = raw.entry_id  # URL canonique incluant la version

        published = raw.published
        if published is not None and published.tzinfo is not None:
            published = published.astimezone(timezone.utc).replace(tzinfo=None)

        payload = {
            "entry_id": raw.entry_id,
            "title": (raw.title or "").strip(),
            "summary": (raw.summary or "").strip(),
            "authors": [a.name for a in (raw.authors or [])],
            "categories": list(raw.categories or []),
            "primary_category": raw.primary_category,
            "published": raw.published.isoformat() if raw.published else None,
            "updated": raw.updated.isoformat() if raw.updated else None,
            "pdf_url": raw.pdf_url,
            "doi": raw.doi,
        }

        return NormalizedItem(
            entity_type="paper",
            entity_id=entity_id,
            content_at=published,
            payload=payload,
        )


def _to_utc(dt: datetime) -> datetime:
    """Retourne `dt` aware en UTC. Un `dt` naïf est supposé UTC."""
    if dt.tzinfo is None:
        return dt.replace(tzinfo=timezone.utc)
    return dt.astimezone(timezone.utc)
