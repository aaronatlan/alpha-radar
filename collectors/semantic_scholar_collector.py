"""Collecteur Semantic Scholar — papers à fort impact par secteur suivi.

API publique : `GET https://api.semanticscholar.org/graph/v1/paper/search`.
Pas de clé requise pour ~1 req/s (limite par IP). On reste poli avec
`request_delay=1.5`.

Stratégie de collecte
---------------------
Pour chaque secteur de `SECTORS`, on requête le premier mot-clé
représentatif puis on stocke les top N papers récents (dernières 90 j)
triés par citation count. Le `paperId` est stable côté Semantic Scholar
— c'est notre `entity_id`.

Snapshots multiples
-------------------
La `citationCount` évolue dans le temps. Comme `payload_json` inclut
ce champ, le `hash` SHA-256 change quand le compteur évolue → une
**nouvelle ligne raw_data** est créée à chaque mise à jour. La règle
`CitationVelocityRule` (Phase 4 v1) consomme ces snapshots pour
détecter une accélération brutale (>100 citations en 7 jours).

Paper récent vs ancien
----------------------
On filtre sur `publicationDateOrYear=YYYY-MM-DD:` (depuis la date)
pour ne garder que les papers récemment publiés — c'est là que
l'effet d'amplification est le plus pertinent. Les classics
(Transformer, ResNet…) ont déjà 100k+ citations, leur Δ est faible
en proportion.
"""
from __future__ import annotations

from datetime import datetime, timedelta
from typing import Any

import requests
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.sectors import SECTORS
from memory.database import utc_now


class SemanticScholarCollector(BaseCollector):
    """Snapshot des papers à fort impact par secteur."""

    source_name = "semantic_scholar"
    request_delay = 1.5  # API gratuite : ~1 req/s, on garde de la marge
    timeout = 30.0
    base_url = "https://api.semanticscholar.org/graph/v1/paper/search"

    #: Nb de papers récupérés par secteur. L'API limite à 100 par requête.
    page_size = 50

    #: Fenêtre de publication considérée comme "récente" (jours).
    recent_window_days = 90

    #: Champs minimaux à demander à l'API.
    fields: list[str] = [
        "paperId",
        "title",
        "abstract",
        "year",
        "publicationDate",
        "citationCount",
        "influentialCitationCount",
        "referenceCount",
        "venue",
        "authors",
        "externalIds",
    ]

    def __init__(self, sector_ids: list[str] | None = None) -> None:
        super().__init__()
        if sector_ids is not None:
            self._sectors = [
                s for s in SECTORS if s["id"] in set(sector_ids)
            ]
        else:
            self._sectors = list(SECTORS)

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        # since/until non utilisés (on filtre côté API par publicationDate
        # sur une fenêtre fixe `recent_window_days`). Ils sont conservés
        # dans la signature pour conformité au scheduler.
        recent_since = (
            utc_now() - timedelta(days=self.recent_window_days)
        ).date().isoformat()

        items: list[dict[str, Any]] = []
        for sector in self._sectors:
            keywords = sector.get("keywords") or []
            if not keywords:
                continue
            # On combine les 2 premiers mots-clés pour focaliser le sujet.
            query = " ".join(keywords[:2])
            papers = self._fetch_query(query, recent_since)
            for paper in papers:
                # Copie superficielle pour éviter de muter l'objet API
                # partagé (ou un mock qui renverrait la même référence).
                item = dict(paper)
                item["_sector_id"] = sector["id"]
                item["_query"] = query
                items.append(item)
            self._throttle()
        return items

    def _fetch_query(self, query: str, since_iso: str) -> list[dict[str, Any]]:
        params = {
            "query": query,
            "limit": self.page_size,
            "fields": ",".join(self.fields),
            "publicationDateOrYear": f"{since_iso}:",
        }
        try:
            r = requests.get(self.base_url, params=params, timeout=self.timeout)
        except requests.RequestException as exc:
            logger.warning("[semantic_scholar] GET '{}' : {}", query, exc)
            return []

        if r.status_code == 429:
            # Rate-limited : on log, on saute. Le prochain run rattrapera.
            logger.warning("[semantic_scholar] rate limit (429) sur '{}'", query)
            return []
        if r.status_code != 200:
            logger.warning(
                "[semantic_scholar] '{}' : HTTP {}", query, r.status_code,
            )
            return []
        try:
            data = r.json()
        except ValueError:
            logger.warning("[semantic_scholar] '{}' : JSON invalide", query)
            return []
        return list(data.get("data") or [])

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        paper_id = raw.get("paperId")
        if not paper_id:
            return None

        pub_date_str = raw.get("publicationDate")
        content_at = _parse_date(pub_date_str)
        if content_at is None:
            year = raw.get("year")
            if isinstance(year, int):
                # Fallback : 1er janvier de l'année si publicationDate absent.
                content_at = datetime(year, 1, 1)
            else:
                # Sans date du tout, on stamp avec l'instant courant —
                # le PIT est dégradé mais l'item reste exploitable
                # (la rule citation se base sur fetched_at).
                content_at = datetime.utcnow()

        authors = [
            a.get("name") for a in (raw.get("authors") or [])
            if a.get("name")
        ]

        payload = {
            "paper_id": paper_id,
            "sector_id": raw.get("_sector_id"),
            "query": raw.get("_query"),
            "title": raw.get("title"),
            "abstract": (raw.get("abstract") or "")[:1000],  # tronqué
            "year": raw.get("year"),
            "publication_date": pub_date_str,
            "citation_count": raw.get("citationCount"),
            "influential_citation_count": raw.get("influentialCitationCount"),
            "reference_count": raw.get("referenceCount"),
            "venue": raw.get("venue"),
            "authors": authors,
            "doi": (raw.get("externalIds") or {}).get("DOI"),
            "arxiv_id": (raw.get("externalIds") or {}).get("ArXiv"),
        }
        return NormalizedItem(
            entity_type="paper",
            entity_id=str(paper_id),
            content_at=content_at,
            payload=payload,
        )


# ------------------------------------------------------------------ helpers


def _parse_date(value: str | None) -> datetime | None:
    """Parse une date ISO YYYY-MM-DD, None si invalide."""
    if not value:
        return None
    try:
        return datetime.strptime(value, "%Y-%m-%d")
    except ValueError:
        return None
