"""Évaluateur de thèses — Phase 3 étape 2.

Pour chaque thèse stockée, calcule les évaluations dues aux jalons
30 / 90 / 180 / 365 / 540 jours après `created_at`. Une ligne par
jalon dans `evaluations`, append-only — l'idempotence est garantie
par un check `(thesis_id, days_since_thesis)` avant insertion.

Calcul du return / alpha
------------------------
- `entry_price` : copié depuis la thèse (snapshot figé à la création).
- `current_price` : dernier close PIT à `created_at + N jours`.
- `benchmark_return_pct` : moyenne des returns des autres tickers de la
  watchlist sur le **même secteur** que la thèse, sur la même période.
  Si aucune paire valide n'est trouvable (secteur sans peers, prix
  manquants), le benchmark vaut `None` et l'alpha aussi — on ne
  fabrique pas de zéro.

Classification de statut (SPEC §7.5)
------------------------------------
- jalons 30 / 90 : `status = active` (la thèse mûrit encore).
- jalons 180+ : on tranche
    - `success` si `alpha > +5%`
    - `failure` si `alpha < -5%`
    - `partial` sinon (ou `active` si l'alpha est indéterminé).

Les recommandations 'AVOID' (non générées en Phase 3 v1) inversent
l'attendu — le code actuel les traite comme des thèses normales ; à
revisiter quand le shorting / hedging entrera dans le SPEC.
"""
from __future__ import annotations

from datetime import datetime, timedelta

from loguru import logger
from sqlalchemy import select

from config.watchlists import STOCK_WATCHLIST
from memory.database import Evaluation, Thesis, session_scope, utc_now
from thesis._io import latest_close_at


#: Jalons standards SPEC §7.5. Modifiable via paramètre du constructeur.
MILESTONES_DAYS: list[int] = [30, 90, 180, 365, 540]

#: À partir du jalon 180j, alpha > +5% → success ; alpha < -5% → failure.
SUCCESS_ALPHA_THRESHOLD = 0.05
FAILURE_ALPHA_THRESHOLD = -0.05

#: Premier jalon où la classification non-`active` peut s'appliquer.
CLASSIFICATION_MIN_DAYS = 180


# ----------------------------------------------------------------- helpers


def _classify_status(days: int, alpha_pct: float | None) -> str:
    """Statut SPEC §7.5 selon le jalon et l'alpha calculé."""
    if days < CLASSIFICATION_MIN_DAYS or alpha_pct is None:
        return "active"
    if alpha_pct > SUCCESS_ALPHA_THRESHOLD:
        return "success"
    if alpha_pct < FAILURE_ALPHA_THRESHOLD:
        return "failure"
    return "partial"


def _peers_for(ticker: str, sector_id: str) -> list[str]:
    """Tickers de la watchlist sur le même secteur, hors `ticker`."""
    return [
        w["ticker"] for w in STOCK_WATCHLIST
        if sector_id in w["sectors"] and w["ticker"] != ticker
    ]


def _benchmark_return(
    peers: list[str], t_start: datetime, t_end: datetime
) -> float | None:
    """Moyenne arithmétique des returns des `peers` entre `t_start` et `t_end`.

    Un peer dont l'un des deux prix est manquant ou nul est exclu —
    pas d'imputation silencieuse à 0.
    """
    returns: list[float] = []
    for peer in peers:
        p_start = latest_close_at(peer, t_start)
        p_end = latest_close_at(peer, t_end)
        if p_start is None or p_end is None or p_start <= 0:
            continue
        returns.append((p_end - p_start) / p_start)
    if not returns:
        return None
    return sum(returns) / len(returns)


def _existing_milestones(thesis_id: int) -> set[int]:
    """Jalons déjà évalués pour `thesis_id`."""
    stmt = (
        select(Evaluation.days_since_thesis)
        .where(Evaluation.thesis_id == thesis_id)
    )
    with session_scope() as session:
        return {row[0] for row in session.execute(stmt).all()}


# --------------------------------------------------------------- evaluator


class ThesisEvaluator:
    """Calcule les évaluations dues à `as_of` pour toutes les thèses.

    Idempotent : un jalon déjà présent en base est sauté. Une exception
    sur une thèse n'interrompt pas le run global (mode dégradé).
    """

    def __init__(self, milestones: list[int] | None = None) -> None:
        ms = list(milestones) if milestones is not None else list(MILESTONES_DAYS)
        if not ms or any(m <= 0 for m in ms):
            raise ValueError("milestones doivent être strictement positifs")
        self._milestones = sorted(ms)

    def run(self, as_of: datetime | None = None) -> int:
        """Itère sur toutes les thèses et écrit les évaluations dues.

        Retourne le nombre de lignes `evaluations` créées.
        """
        ts = as_of or utc_now()
        with session_scope() as session:
            theses = session.query(Thesis).all()
            for th in theses:
                session.expunge(th)

        logger.info(
            "[evaluator] {} thèse(s) à examiner à {}", len(theses), ts,
        )
        created = 0
        for th in theses:
            try:
                created += self._evaluate_one(th, ts)
            except Exception as exc:
                logger.warning(
                    "[evaluator] thèse #{} a échoué : {}", th.id, exc,
                )
        logger.info("[evaluator] Terminé : {} évaluation(s) créée(s)", created)
        return created

    # --- unité ---------------------------------------------------------

    def _evaluate_one(self, thesis: Thesis, as_of: datetime) -> int:
        days_since = (as_of - thesis.created_at).days
        if days_since < self._milestones[0]:
            return 0  # avant le 1er jalon : rien à faire

        existing = _existing_milestones(thesis.id)
        peers = _peers_for(thesis.asset_id, thesis.sector_id)
        n_created = 0

        for m in self._milestones:
            if m > days_since:
                break
            if m in existing:
                continue

            t_eval = thesis.created_at + timedelta(days=m)
            current_price = latest_close_at(thesis.asset_id, t_eval)

            return_pct: float | None = None
            if (
                current_price is not None
                and thesis.entry_price is not None
                and thesis.entry_price > 0
            ):
                return_pct = (current_price - thesis.entry_price) / thesis.entry_price

            benchmark = _benchmark_return(peers, thesis.created_at, t_eval)

            alpha_pct: float | None = None
            if return_pct is not None and benchmark is not None:
                alpha_pct = return_pct - benchmark

            status = _classify_status(m, alpha_pct)

            notes = None
            if current_price is None:
                notes = "Prix non disponible à la date d'évaluation."
            elif thesis.entry_price is None:
                notes = "entry_price absent dans la thèse — return non calculé."

            with session_scope() as session:
                session.add(Evaluation(
                    thesis_id=thesis.id,
                    evaluated_at=as_of,
                    days_since_thesis=m,
                    current_price=current_price,
                    return_pct=return_pct,
                    benchmark_return_pct=benchmark,
                    alpha_pct=alpha_pct,
                    status=status,
                    notes=notes,
                ))
            n_created += 1
            logger.info(
                "[evaluator] +1 thèse=#{} J+{}j → status={} alpha={}",
                thesis.id, m, status,
                f"{alpha_pct * 100:+.2f}%" if alpha_pct is not None else "n/a",
            )
        return n_created
