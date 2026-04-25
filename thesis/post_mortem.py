"""Post-mortem & track record — Phase 3 étape 3.

Agrège les évaluations terminales (`status ∈ {success, failure, partial}`)
en lignes `signal_performance` indexées par `(signal_name, sector_id,
horizon_days)`. Chaque thèse contribue à autant de signaux que de
dimensions présentes dans son `score_breakdown_json` :

- une ligne par dimension × secteur de la thèse × jalon évalué,
- plus une ligne « tous secteurs » (sector_id = NULL) pour la même
  paire (dimension, jalon).

Définition d'un succès (SPEC §7.5)
----------------------------------
- `status == 'success'` → succès.
- `status == 'failure'` → échec (compte comme prédiction).
- `status == 'partial'` → ni succès ni échec (compte comme prédiction).
- `status == 'active'` → exclus (verdict non rendu).

Recalcul complet
----------------
Chaque appel à `run()` repart de zéro : on relit toutes les évaluations
pertinentes, on vide `signal_performance` et on réécrit toutes les
lignes dans la même transaction. Pourquoi pas un UPSERT ? La contrainte
UNIQUE de SQLite ne traite pas `NULL = NULL` (cf. standard SQL), donc
les lignes « tous secteurs » (sector_id = NULL) doubleraient à chaque
run avec ON CONFLICT. Le delete-then-insert est sûr et reste pas cher
(volumes très bas en Phase 3).

Recommandations
---------------
On ne traite ici que les thèses dont la `recommendation` n'est pas
`AVOID` — l'AVOID inverse le sens du succès et n'est de toute façon pas
généré par le pipeline en Phase 3. À revisiter quand le shorting/hedging
entrera dans le SPEC.
"""
from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime

from loguru import logger
from sqlalchemy import select

from memory.database import (
    Evaluation,
    SignalPerformance,
    Thesis,
    session_scope,
    utc_now,
)

#: Statuts qui comptent comme une prédiction « rendue ».
TERMINAL_STATUSES: tuple[str, ...] = ("success", "failure", "partial")

#: Statut considéré comme un succès dans `n_successes`.
SUCCESS_STATUS: str = "success"


# ----------------------------------------------------------------- helpers


def _extract_signals(thesis: Thesis) -> list[str]:
    """Liste des dimensions présentes dans le breakdown de la thèse.

    Un breakdown malformé (JSON cassé, dimensions absentes) renvoie une
    liste vide — la thèse ne contribue alors à aucun signal mais ne fait
    pas échouer le batch.
    """
    try:
        breakdown = json.loads(thesis.score_breakdown_json or "{}")
    except (TypeError, ValueError):
        return []
    dims = breakdown.get("dimensions")
    if not isinstance(dims, dict):
        return []
    return list(dims.keys())


@dataclass
class _Bucket:
    """Compteurs intermédiaires avant calcul de l'accuracy."""

    n_predictions: int = 0
    n_successes: int = 0
    alphas: list[float] = field(default_factory=list)

    def absorb(self, status: str, alpha_pct: float | None) -> None:
        self.n_predictions += 1
        if status == SUCCESS_STATUS:
            self.n_successes += 1
        if alpha_pct is not None:
            self.alphas.append(alpha_pct)

    @property
    def accuracy(self) -> float:
        return self.n_successes / self.n_predictions if self.n_predictions else 0.0

    @property
    def avg_alpha(self) -> float | None:
        return sum(self.alphas) / len(self.alphas) if self.alphas else None


# --------------------------------------------------------------- aggregator


class PostMortemAnalyzer:
    """Recalcule `signal_performance` depuis les évaluations terminales.

    Idempotent : un appel suivi d'un autre sur le même état de base
    produit les mêmes lignes. Mode dégradé : si une thèse a un breakdown
    invalide, elle est sautée mais le batch continue.
    """

    def __init__(
        self,
        *,
        terminal_statuses: tuple[str, ...] = TERMINAL_STATUSES,
    ) -> None:
        if not terminal_statuses:
            raise ValueError("terminal_statuses ne peut pas être vide")
        self._terminal = tuple(terminal_statuses)

    def run(self, as_of: datetime | None = None) -> int:
        """Reconstruit `signal_performance` depuis les évaluations.

        Retourne le nombre de lignes upsertées.
        """
        ts = as_of or utc_now()
        pairs = self._load_terminal_pairs()
        logger.info(
            "[post_mortem] {} évaluation(s) terminale(s) à analyser", len(pairs),
        )
        buckets = self._aggregate(pairs)
        n = self._upsert(buckets, last_updated=ts)
        logger.info("[post_mortem] {} ligne(s) signal_performance upsertée(s)", n)
        return n

    # --- pipeline ------------------------------------------------------

    def _load_terminal_pairs(self) -> list[tuple[Thesis, Evaluation]]:
        """Charge (thèse, évaluation) pour chaque évaluation terminale."""
        stmt = (
            select(Thesis, Evaluation)
            .join(Evaluation, Evaluation.thesis_id == Thesis.id)
            .where(Evaluation.status.in_(self._terminal))
        )
        with session_scope() as session:
            rows = session.execute(stmt).all()
            pairs = [(th, ev) for th, ev in rows]
            # Une même thèse peut apparaître plusieurs fois (un eval par jalon)
            # → expunge unique par identité pour éviter "not present in session".
            seen: set[int] = set()
            for th, ev in pairs:
                if id(th) not in seen:
                    session.expunge(th)
                    seen.add(id(th))
                session.expunge(ev)
        return pairs

    def _aggregate(
        self, pairs: list[tuple[Thesis, Evaluation]]
    ) -> dict[tuple[str, str | None, int], _Bucket]:
        """Regroupe en buckets `(signal, sector_id|None, horizon_days)`."""
        buckets: dict[tuple[str, str | None, int], _Bucket] = defaultdict(_Bucket)
        for thesis, ev in pairs:
            signals = _extract_signals(thesis)
            if not signals:
                continue
            horizon = int(ev.days_since_thesis)
            sector = thesis.sector_id
            for signal in signals:
                # Ligne secteur-spécifique.
                buckets[(signal, sector, horizon)].absorb(ev.status, ev.alpha_pct)
                # Ligne « tous secteurs » (sector_id = NULL).
                buckets[(signal, None, horizon)].absorb(ev.status, ev.alpha_pct)
        return buckets

    def _upsert(
        self,
        buckets: dict[tuple[str, str | None, int], _Bucket],
        *,
        last_updated: datetime,
    ) -> int:
        """Vide `signal_performance` puis ré-insère tous les buckets.

        Tout dans la même transaction — un consommateur qui lit pendant
        le run verra l'état pré-batch ou post-batch, jamais d'état hybride.
        """
        with session_scope() as session:
            session.query(SignalPerformance).delete()
            for (signal, sector, horizon), b in buckets.items():
                session.add(SignalPerformance(
                    signal_name=signal,
                    sector_id=sector,
                    horizon_days=horizon,
                    n_predictions=b.n_predictions,
                    n_successes=b.n_successes,
                    accuracy=b.accuracy,
                    avg_alpha=b.avg_alpha,
                    last_updated=last_updated,
                ))
        return len(buckets)
