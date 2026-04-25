"""Règles d'alerte — Phase 3 étape 4.

Chaque règle :
- déclare un `name` (devient `rule_name` dans la table `alerts`),
- déclare un `severity` (`info` | `warning` | `critical`),
- expose `evaluate(as_of) -> list[AlertCandidate]`.

Les règles sont stateless : la déduplication est gérée par le moteur via
un `dedupe_key` stable porté par chaque `AlertCandidate`. Une règle
peut donc être appelée plusieurs fois sans craindre les doublons —
c'est le moteur qui vérifie l'existant avant insertion.

Périmètre Phase 3 v1
--------------------
Seules les règles dont les données sont déjà collectées en Phase 3 sont
implémentées :

- `NewThesisRule` (info) : nouvelle thèse avec score ≥ seuil.
- `EvaluationVerdictRule` (info) : évaluation passe en success / failure.
- `SectorHeatSurgeRule` (critical) : Heat Score +Δ points en N heures.

Les règles SPEC restantes (Form 13D, PDUFA, citations >100, contrats
gouvernementaux, buybacks) attendent les collecteurs spécialisés de la
Phase 4 — placeholders prévus, pas implémentés.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Any, Iterable

from sqlalchemy import select

from config.pdufa_calendar import upcoming_pdufas
from memory.database import (
    Evaluation,
    Feature,
    Thesis,
    session_scope,
)


#: Seuil par défaut au-delà duquel une thèse génère une alerte info.
DEFAULT_NEW_THESIS_SCORE = 70.0

#: Statuts d'évaluation qui déclenchent une alerte (le `partial` est
#: volontairement exclu — trop bruyant pour cette v1).
DEFAULT_VERDICT_STATUSES: tuple[str, ...] = ("success", "failure")

#: Hausse minimale du Heat Score sectoriel (en points 0-100) sur la
#: fenêtre `surge_window_hours` qui déclenche une alerte critique.
DEFAULT_HEAT_SURGE_DELTA = 20.0
DEFAULT_HEAT_SURGE_WINDOW_HOURS = 48

#: Seuil par défaut SPEC §7.7 : PDUFA <30 j avec score >70 = critique.
DEFAULT_PDUFA_DAYS_AHEAD = 30
DEFAULT_PDUFA_SCORE_THRESHOLD = 70.0


# ----------------------------------------------------------------- types


@dataclass
class AlertCandidate:
    """Événement candidat à devenir une ligne `alerts`.

    Le moteur consulte `dedupe_key` avant insertion. Une clé déjà vue
    pour ce `rule_name` est skippée.
    """

    rule_name: str
    severity: str
    message: str
    dedupe_key: str
    asset_type: str | None = None
    asset_id: str | None = None
    sector_id: str | None = None
    thesis_id: int | None = None
    data: dict[str, Any] = field(default_factory=dict)

    def data_json(self) -> str:
        """Serialise `data` + `dedupe_key` pour la colonne data_json."""
        payload = {**self.data, "dedupe_key": self.dedupe_key}
        return json.dumps(payload, default=str, sort_keys=True)


class Rule:
    """Interface minimale d'une règle d'alerte."""

    name: str = ""
    severity: str = "info"

    def evaluate(self, as_of: datetime) -> Iterable[AlertCandidate]:
        raise NotImplementedError


# ----------------------------------------------------------------- rules


class NewThesisRule(Rule):
    """Émet une alerte info pour chaque thèse récente au-dessus du seuil."""

    name = "new_thesis"
    severity = "info"

    def __init__(
        self,
        *,
        score_threshold: float = DEFAULT_NEW_THESIS_SCORE,
        lookback_days: int = 7,
    ) -> None:
        self._threshold = float(score_threshold)
        self._lookback_days = int(lookback_days)

    def evaluate(self, as_of: datetime) -> list[AlertCandidate]:
        since = as_of - timedelta(days=self._lookback_days)
        stmt = (
            select(Thesis)
            .where(Thesis.created_at >= since)
            .where(Thesis.created_at <= as_of)
            .where(Thesis.score >= self._threshold)
            .order_by(Thesis.created_at.asc())
        )
        with session_scope() as session:
            theses = list(session.execute(stmt).scalars().all())
            for th in theses:
                session.expunge(th)

        out: list[AlertCandidate] = []
        for th in theses:
            msg = (
                f"Nouvelle thèse {th.recommendation} sur {th.asset_id} "
                f"({th.sector_id}) — score {th.score:.1f}/100, "
                f"horizon {th.horizon_days}j."
            )
            out.append(AlertCandidate(
                rule_name=self.name,
                severity=self.severity,
                message=msg,
                dedupe_key=f"thesis:{th.id}",
                asset_type=th.asset_type,
                asset_id=th.asset_id,
                sector_id=th.sector_id,
                thesis_id=th.id,
                data={"score": float(th.score),
                      "recommendation": th.recommendation},
            ))
        return out


class EvaluationVerdictRule(Rule):
    """Émet une alerte info quand une évaluation passe success / failure."""

    name = "eval_verdict"
    severity = "info"

    def __init__(
        self,
        *,
        statuses: tuple[str, ...] = DEFAULT_VERDICT_STATUSES,
        lookback_days: int = 7,
    ) -> None:
        if not statuses:
            raise ValueError("statuses ne peut pas être vide")
        self._statuses = tuple(statuses)
        self._lookback_days = int(lookback_days)

    def evaluate(self, as_of: datetime) -> list[AlertCandidate]:
        since = as_of - timedelta(days=self._lookback_days)
        stmt = (
            select(Evaluation, Thesis)
            .join(Thesis, Thesis.id == Evaluation.thesis_id)
            .where(Evaluation.evaluated_at >= since)
            .where(Evaluation.evaluated_at <= as_of)
            .where(Evaluation.status.in_(self._statuses))
            .order_by(Evaluation.evaluated_at.asc())
        )
        with session_scope() as session:
            rows = session.execute(stmt).all()
            pairs = [(ev, th) for ev, th in rows]
            for ev, th in pairs:
                session.expunge(ev)
                session.expunge(th)

        out: list[AlertCandidate] = []
        for ev, th in pairs:
            alpha_str = (
                f"alpha {ev.alpha_pct * 100:+.1f}%"
                if ev.alpha_pct is not None else "alpha n/a"
            )
            msg = (
                f"Thèse #{th.id} ({th.asset_id}) : verdict "
                f"**{ev.status}** à J+{ev.days_since_thesis}j ({alpha_str})."
            )
            out.append(AlertCandidate(
                rule_name=self.name,
                severity=self.severity,
                message=msg,
                dedupe_key=f"eval:{ev.id}",
                asset_type=th.asset_type,
                asset_id=th.asset_id,
                sector_id=th.sector_id,
                thesis_id=th.id,
                data={
                    "status": ev.status,
                    "days_since_thesis": int(ev.days_since_thesis),
                    "alpha_pct": (
                        float(ev.alpha_pct) if ev.alpha_pct is not None else None
                    ),
                },
            ))
        return out


class SectorHeatSurgeRule(Rule):
    """Émet une alerte critique quand un Heat Score grimpe brutalement."""

    name = "sector_heat_surge"
    severity = "critical"
    feature_name = "sector_heat_score"

    def __init__(
        self,
        *,
        delta_threshold: float = DEFAULT_HEAT_SURGE_DELTA,
        window_hours: int = DEFAULT_HEAT_SURGE_WINDOW_HOURS,
    ) -> None:
        if delta_threshold <= 0:
            raise ValueError("delta_threshold doit être strictement positif")
        if window_hours <= 0:
            raise ValueError("window_hours doit être strictement positif")
        self._delta = float(delta_threshold)
        self._window = timedelta(hours=int(window_hours))

    def evaluate(self, as_of: datetime) -> list[AlertCandidate]:
        # Pour chaque secteur observé, on récupère le dernier heat score à
        # `as_of` et le dernier observé `as_of - window`. La différence est
        # comparée à `delta`.
        sectors = self._observed_sectors(as_of)
        out: list[AlertCandidate] = []
        for sector_id in sectors:
            current = self._latest_value(sector_id, as_of)
            previous = self._latest_value(sector_id, as_of - self._window)
            if current is None or previous is None:
                continue
            current_value, current_at = current
            delta = current_value - previous[0]
            if delta < self._delta:
                continue
            msg = (
                f"Heat Score {sector_id} : +{delta:.1f} pts en "
                f"{int(self._window.total_seconds() // 3600)}h "
                f"(de {previous[0]:.1f} à {current_value:.1f})."
            )
            out.append(AlertCandidate(
                rule_name=self.name,
                severity=self.severity,
                message=msg,
                # Une seule alerte par sector × jour de l'observation
                # courante : si le surge persiste plusieurs jours, on
                # alerte une fois par jour, pas à chaque tick.
                dedupe_key=(
                    f"heat_surge:{sector_id}:{current_at.date().isoformat()}"
                ),
                sector_id=sector_id,
                data={
                    "current": float(current_value),
                    "previous": float(previous[0]),
                    "delta": float(delta),
                    "window_hours": int(self._window.total_seconds() // 3600),
                },
            ))
        return out

    # --- helpers ------------------------------------------------------

    def _observed_sectors(self, as_of: datetime) -> list[str]:
        """Liste des secteurs ayant au moins une feature heat_score ≤ as_of."""
        stmt = (
            select(Feature.target_id)
            .where(Feature.feature_name == self.feature_name)
            .where(Feature.target_type == "sector")
            .where(Feature.computed_at <= as_of)
            .distinct()
        )
        with session_scope() as session:
            return [row[0] for row in session.execute(stmt).all()]

    def _latest_value(
        self, sector_id: str, at: datetime
    ) -> tuple[float, datetime] | None:
        """Dernière valeur PIT du heat score pour `sector_id` ≤ at."""
        stmt = (
            select(Feature.value, Feature.computed_at)
            .where(Feature.feature_name == self.feature_name)
            .where(Feature.target_type == "sector")
            .where(Feature.target_id == sector_id)
            .where(Feature.computed_at <= at)
            .order_by(Feature.computed_at.desc())
            .limit(1)
        )
        with session_scope() as session:
            row = session.execute(stmt).first()
            if row is None:
                return None
            return float(row[0]), row[1]


class PDUFANearRule(Rule):
    """Émet une alerte critique pour chaque PDUFA proche d'un actif scoré.

    Source du calendrier : `config/pdufa_calendar.py` (manuel — voir le
    docstring du module pour la justification). Pour chaque PDUFA dont
    la date est dans `[as_of, as_of + days_ahead]` jours :

    1. on récupère le dernier `stock_score` du ticker (PIT à `as_of`),
    2. si `score >= score_threshold`, on émet l'alerte,
    3. dedupe_key = `pdufa:{ticker}:{date}` (une alerte par PDUFA, jamais
       répétée même si la PDUFA approche jour après jour).
    """

    name = "pdufa_near"
    severity = "critical"
    feature_name = "stock_score"

    def __init__(
        self,
        *,
        days_ahead: int = DEFAULT_PDUFA_DAYS_AHEAD,
        score_threshold: float = DEFAULT_PDUFA_SCORE_THRESHOLD,
    ) -> None:
        if days_ahead <= 0:
            raise ValueError("days_ahead doit être strictement positif")
        self._days_ahead = int(days_ahead)
        self._threshold = float(score_threshold)

    def evaluate(self, as_of: datetime) -> list[AlertCandidate]:
        as_of_date = as_of.date()
        horizon_date = as_of_date + timedelta(days=self._days_ahead)
        out: list[AlertCandidate] = []
        for entry in upcoming_pdufas(as_of_date):
            try:
                pdufa_date = datetime.strptime(
                    entry["target_action_date"], "%Y-%m-%d"
                ).date()
            except (KeyError, ValueError):
                continue
            if pdufa_date > horizon_date:
                continue   # trop loin pour alerter
            score = self._latest_score(entry["ticker"], as_of)
            if score is None or score < self._threshold:
                continue
            days_left = (pdufa_date - as_of_date).days
            drug = entry.get("drug", "n/a")
            msg = (
                f"PDUFA dans {days_left} j sur {entry['ticker']} "
                f"({drug}) — score {score:.1f}/100. "
                f"Date cible : {entry['target_action_date']}."
            )
            out.append(AlertCandidate(
                rule_name=self.name,
                severity=self.severity,
                message=msg,
                dedupe_key=(
                    f"pdufa:{entry['ticker']}:{entry['target_action_date']}"
                ),
                asset_type="stock",
                asset_id=entry["ticker"],
                data={
                    "pdufa_date": entry["target_action_date"],
                    "days_left": days_left,
                    "score": float(score),
                    "drug": drug,
                    "indication": entry.get("indication"),
                },
            ))
        return out

    def _latest_score(
        self, ticker: str, as_of: datetime,
    ) -> float | None:
        """Dernière feature `stock_score` PIT pour `ticker`."""
        stmt = (
            select(Feature.value)
            .where(Feature.feature_name == self.feature_name)
            .where(Feature.target_type == "asset")
            .where(Feature.target_id == ticker)
            .where(Feature.computed_at <= as_of)
            .order_by(Feature.computed_at.desc())
            .limit(1)
        )
        with session_scope() as session:
            row = session.execute(stmt).first()
            return float(row[0]) if row is not None else None


#: Règles activées par défaut dans `AlertsEngine`. Ajouter ici les futures
#: règles Phase 4+ (Form 13D, citations papers, contrats gouv…).
DEFAULT_RULES: list[Rule] = [
    NewThesisRule(),
    EvaluationVerdictRule(),
    SectorHeatSurgeRule(),
    PDUFANearRule(),
]


__all__ = [
    "AlertCandidate",
    "Rule",
    "NewThesisRule",
    "EvaluationVerdictRule",
    "SectorHeatSurgeRule",
    "PDUFANearRule",
    "DEFAULT_RULES",
    "DEFAULT_NEW_THESIS_SCORE",
    "DEFAULT_VERDICT_STATUSES",
    "DEFAULT_HEAT_SURGE_DELTA",
    "DEFAULT_HEAT_SURGE_WINDOW_HOURS",
    "DEFAULT_PDUFA_DAYS_AHEAD",
    "DEFAULT_PDUFA_SCORE_THRESHOLD",
]
