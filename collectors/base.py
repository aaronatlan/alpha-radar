"""Classe abstraite commune à tous les collecteurs.

Chaque collecteur concret :
  1. Définit l'attribut de classe `source_name` (str).
  2. Implémente `collect(since, until)` — récupération brute.
  3. Implémente `normalize(raw)` — conversion en item canonique.

La méthode `run()` orchestre collect → normalize → store et garantit
le **mode dégradé** : toute exception est loggée mais n'est pas
remontée, pour qu'un collecteur fautif ne tue pas le scheduler.

Le stockage utilise `INSERT OR IGNORE` contre la contrainte UNIQUE
`(source, entity_id, hash)` de `raw_data` : la collecte est donc
idempotente — re-collecter la même donnée est un no-op silencieux.
"""
from __future__ import annotations

import hashlib
import json
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import Any

from loguru import logger
from sqlalchemy.dialects.sqlite import insert as sqlite_insert

from memory.database import RawData, session_scope, utc_now


class NormalizedItem(dict):
    """Forme canonique attendue en sortie de `normalize()`.

    Clés obligatoires :
      - `entity_type` (str) : type logique de l'entité ('paper', 'ohlcv_daily'…).
      - `entity_id` (str) : identifiant unique côté source.
      - `content_at` (datetime | None) : timestamp intrinsèque de la donnée.
      - `payload` (dict) : contenu sérialisable JSON.
    """


class BaseCollector(ABC):
    """Classe de base pour tous les collecteurs."""

    #: Nom court et stable de la source (colonne `raw_data.source`).
    source_name: str = ""

    #: Délai courtois entre deux requêtes (secondes). Surchargeable.
    request_delay: float = 1.0

    def __init__(self) -> None:
        if not self.source_name:
            raise ValueError(
                f"{type(self).__name__} doit définir l'attribut de classe "
                "`source_name`."
            )

    # --- API à implémenter par les sous-classes --------------------------

    @abstractmethod
    def collect(self, since: datetime, until: datetime) -> list[Any]:
        """Récupère les données brutes sur la période [since, until].

        Les items retournés peuvent être dans leur format natif (objets
        `arxiv.Result`, dicts JSON, lignes de DataFrame, etc.) — ils
        seront ensuite passés un par un à `normalize()`.
        """

    @abstractmethod
    def normalize(self, raw: Any) -> NormalizedItem | None:
        """Normalise un item brut. Retourne `None` pour skipper un item invalide."""

    # --- Helpers ---------------------------------------------------------

    @staticmethod
    def _hash_payload(payload: dict) -> str:
        """Hash déterministe (SHA-256) d'un payload JSON pour déduplication.

        Sérialisation triée par clé : deux payloads équivalents produisent
        le même hash même si l'ordre des clés diffère.
        """
        serialized = json.dumps(
            payload, sort_keys=True, default=str, ensure_ascii=False
        )
        return hashlib.sha256(serialized.encode("utf-8")).hexdigest()

    def _throttle(self) -> None:
        """Pause courte pour respecter les rate limits."""
        if self.request_delay > 0:
            time.sleep(self.request_delay)

    # --- Stockage --------------------------------------------------------

    def store(self, items: list[NormalizedItem]) -> int:
        """Insère les items dans `raw_data` et retourne le nombre inséré.

        Utilise `INSERT OR IGNORE` : les doublons (même UNIQUE key) sont
        silencieusement ignorés. Le `rowcount` retourné par SQLAlchemy
        reflète le nombre réel de lignes insérées.
        """
        if not items:
            return 0

        now = utc_now()
        rows: list[dict[str, Any]] = []
        for item in items:
            payload = item["payload"]
            rows.append(
                {
                    "source": self.source_name,
                    "entity_type": item["entity_type"],
                    "entity_id": str(item["entity_id"]),
                    "fetched_at": now,
                    "content_at": item.get("content_at"),
                    "payload_json": json.dumps(
                        payload, default=str, ensure_ascii=False
                    ),
                    "hash": self._hash_payload(payload),
                }
            )

        inserted = 0
        with session_scope() as session:
            for row in rows:
                stmt = sqlite_insert(RawData).values(**row).prefix_with("OR IGNORE")
                result = session.execute(stmt)
                inserted += result.rowcount or 0
        return inserted

    # --- Orchestration ---------------------------------------------------

    def run(self, since: datetime, until: datetime) -> int:
        """Exécute un cycle complet collect → normalize → store.

        Mode dégradé : toute exception est loggée mais pas remontée.
        Retourne le nombre d'items effectivement insérés.
        """
        logger.info(
            "[{source}] Début collecte {since} → {until}",
            source=self.source_name, since=since, until=until,
        )
        try:
            raw_items = self.collect(since, until)
        except Exception as exc:
            logger.exception(
                "[{}] Échec de collect() : {}", self.source_name, exc
            )
            return 0

        normalized: list[NormalizedItem] = []
        for raw in raw_items:
            try:
                item = self.normalize(raw)
            except Exception as exc:
                logger.warning(
                    "[{}] normalize() a échoué sur un item : {}",
                    self.source_name, exc,
                )
                continue
            if item is not None:
                normalized.append(item)

        try:
            inserted = self.store(normalized)
        except Exception as exc:
            logger.exception(
                "[{}] Échec de store() : {}", self.source_name, exc
            )
            return 0

        logger.info(
            "[{source}] Terminé : {n_raw} récupérés, {n_norm} normalisés, "
            "{n_ins} insérés (hors doublons)",
            source=self.source_name, n_raw=len(raw_items),
            n_norm=len(normalized), n_ins=inserted,
        )
        return inserted
