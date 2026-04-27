"""Replay PIT historique — Phase 5 étape 2.

Itère sur une fenêtre historique jour-par-jour, ré-exécute scorers et
générateur de thèses pour chaque date. Les composants existants
(`SectorHeatScorer`, `StockScorer`, `ThesisGenerator`) acceptent déjà un
paramètre `as_of` — on les pilote depuis ici sans les modifier.

Prérequis
---------
Les **données brutes** (`raw_data`) et les **features amont**
(`features` : RSI, momentum, velocity, sentiment, …) doivent déjà couvrir
la fenêtre. Le replay ne re-collecte pas (les APIs n'exposent pas leur
historique facilement) et ne recalcule pas les features amont qui sont
elles-mêmes PIT — il rejoue uniquement la **couche scoring + thèses**,
là où les poids se calibrent.

Idempotence
-----------
Tous les inserts sont protégés par `UNIQUE` (features) et par le check
`_has_thesis_today` (thèses), donc re-rejouer la même fenêtre avec la
même config est un no-op silencieux.

Versionnement
-------------
Pour comparer plusieurs jeux de poids sur la même période, instancier
`StockScorer(model_version="v2_mom_sigqual")` et `StockScorer(model_version="v3_...")`
dans deux replays distincts. Les `features.metadata_json` stockent le
`model_version` → les thèses générées le copient → traçabilité totale.
"""
from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Iterable

from loguru import logger

from features.base import BaseFeature
from thesis.generator import ThesisGenerator


@dataclass
class ReplayResult:
    """Résumé d'un replay."""

    start: datetime
    end: datetime
    step_days: int
    n_dates: int = 0
    n_features_inserted: int = 0
    n_theses_created: int = 0
    n_errors: int = 0


class HistoricalReplay:
    """Re-joue scorers + générateur de thèses sur une fenêtre historique.

    Paramètres
    ----------
    scorers
        Liste de `BaseFeature` (typiquement `SectorHeatScorer` puis
        `StockScorer`). Ils sont exécutés **dans l'ordre** chaque jour
        — important quand un scorer aval lit la sortie d'un scorer amont
        (ex. stock_score lit sector_heat_score via `signal_quality`).
    thesis_generator
        Instance de `ThesisGenerator`. Si `None`, les thèses ne sont pas
        générées (utile pour le mode features-only).
    """

    def __init__(
        self,
        *,
        scorers: Iterable[BaseFeature] | None = None,
        thesis_generator: ThesisGenerator | None = None,
    ) -> None:
        self._scorers: list[BaseFeature] = list(scorers) if scorers else []
        self._thesis_generator = thesis_generator

    # --- API publique ----------------------------------------------------

    def run(
        self,
        *,
        start: datetime,
        end: datetime,
        step_days: int = 1,
    ) -> ReplayResult:
        """Exécute le replay entre `start` et `end` par pas de `step_days`.

        Une exception sur un jour donné est loggée mais n'interrompt pas
        le replay — la fenêtre suivante peut quand même être validée.
        """
        if end <= start:
            raise ValueError("end doit être > start")
        if step_days <= 0:
            raise ValueError("step_days doit être strictement positif")

        result = ReplayResult(start=start, end=end, step_days=step_days)
        dates = _daily_grid(start, end, step_days)
        result.n_dates = len(dates)
        logger.info(
            "[replay] Démarrage : {} jour(s) entre {} et {}",
            len(dates), start, end,
        )

        for current in dates:
            for scorer in self._scorers:
                try:
                    result.n_features_inserted += scorer.run(as_of=current)
                except Exception as exc:
                    result.n_errors += 1
                    logger.warning(
                        "[replay] {} @ {} a échoué : {}",
                        scorer.feature_name, current, exc,
                    )

            if self._thesis_generator is not None:
                try:
                    result.n_theses_created += self._thesis_generator.run(
                        as_of=current
                    )
                except Exception as exc:
                    result.n_errors += 1
                    logger.warning(
                        "[replay] thesis_generator @ {} a échoué : {}",
                        current, exc,
                    )

        logger.info(
            "[replay] Terminé : {} features, {} thèses, {} erreurs",
            result.n_features_inserted,
            result.n_theses_created,
            result.n_errors,
        )
        return result


# ------------------------------------------------------------------ helpers


def _daily_grid(
    start: datetime, end: datetime, step_days: int = 1,
) -> list[datetime]:
    """Liste des dates start..<end par pas de `step_days` jours."""
    out: list[datetime] = []
    cursor = start
    delta = timedelta(days=step_days)
    while cursor < end:
        out.append(cursor)
        cursor += delta
    return out
