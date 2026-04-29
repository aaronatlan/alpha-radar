"""Accès base pour le dashboard — **lecture seule**.

Helpers purs qui renvoient des `pandas.DataFrame` pour Streamlit /
Plotly. Aucun couplage à Streamlit ici : testable en isolation.

Les fonctions respectent la discipline point-in-time (on lit
systématiquement la dernière valeur valide à `as_of`, par défaut
`utc_now()`). Aucun calcul n'est fait — on lit simplement les valeurs
persistées par les features / scorers.
"""
from __future__ import annotations

import json
from datetime import datetime
from typing import Any, Optional

import pandas as pd
from sqlalchemy import func, select, update

from config.sectors import SECTORS_BY_ID
from config.watchlists import STOCK_WATCHLIST
from memory.database import (
    Alert,
    Evaluation,
    Feature,
    RawData,
    SignalPerformance,
    Thesis,
    session_scope,
    utc_now,
)


# --------------------------------------------------------------- helpers


def _latest_per_target(
    feature_name: str, target_type: str, as_of: datetime
) -> pd.DataFrame:
    """Retourne un DataFrame (target_id, value, computed_at, metadata_json)
    avec la dernière valeur PIT pour chaque target."""
    with session_scope() as session:
        sub = (
            select(
                Feature.target_id,
                func.max(Feature.computed_at).label("max_ts"),
            )
            .where(Feature.feature_name == feature_name)
            .where(Feature.target_type == target_type)
            .where(Feature.computed_at <= as_of)
            .group_by(Feature.target_id)
            .subquery()
        )
        stmt = (
            select(
                Feature.target_id,
                Feature.value,
                Feature.computed_at,
                Feature.metadata_json,
            )
            .join(
                sub,
                (Feature.target_id == sub.c.target_id)
                & (Feature.computed_at == sub.c.max_ts),
            )
            .where(Feature.feature_name == feature_name)
            .where(Feature.target_type == target_type)
        )
        rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows, columns=["target_id", "value", "computed_at", "metadata_json"]
    )


# -------------------------------------------------------------- sector heat


def get_sector_heat_scores(as_of: Optional[datetime] = None) -> pd.DataFrame:
    """DataFrame enrichi des Heat Scores sectoriels.

    Colonnes : sector_id, sector_name, category, heat_score, computed_at,
    metadata_json. Les secteurs sans score sont inclus avec heat_score=NaN
    (utile pour colorer une treemap complète).
    """
    as_of = as_of or utc_now()
    df = _latest_per_target("sector_heat_score", "sector", as_of)
    if not df.empty:
        df = df.rename(columns={"target_id": "sector_id", "value": "heat_score"})
    else:
        df = pd.DataFrame(
            columns=["sector_id", "heat_score", "computed_at", "metadata_json"]
        )

    meta = [
        {
            "sector_id": s["id"],
            "sector_name": s["name"],
            "category": s["category"],
        }
        for s in SECTORS_BY_ID.values()
    ]
    meta_df = pd.DataFrame(meta)
    return meta_df.merge(df, on="sector_id", how="left")


# --------------------------------------------------------------- stocks


def get_stock_scores(as_of: Optional[datetime] = None) -> pd.DataFrame:
    """Classement d'actions : ticker, score, dimensions, secteurs."""
    as_of = as_of or utc_now()
    df = _latest_per_target("stock_score", "asset", as_of)
    if df.empty:
        df = pd.DataFrame(
            columns=["target_id", "value", "computed_at", "metadata_json"]
        )
    df = df.rename(columns={"target_id": "ticker", "value": "stock_score"})

    name_by = {w["ticker"]: w["name"] for w in STOCK_WATCHLIST}
    sectors_by = {w["ticker"]: w["sectors"] for w in STOCK_WATCHLIST}

    def _dims(meta_json: Any) -> dict[str, float]:
        if not meta_json:
            return {}
        try:
            import json
            return dict((json.loads(meta_json) or {}).get("dimensions", {}))
        except (TypeError, ValueError):
            return {}

    df["name"] = df["ticker"].map(name_by)
    df["sectors"] = df["ticker"].map(lambda t: ", ".join(sectors_by.get(t, [])))
    dims_series = df["metadata_json"].map(_dims)
    df["momentum"] = dims_series.map(lambda d: d.get("momentum"))
    df["signal_quality"] = dims_series.map(lambda d: d.get("signal_quality"))
    df["sentiment"] = dims_series.map(lambda d: d.get("sentiment"))

    cols = [
        "ticker", "name", "sectors", "stock_score",
        "momentum", "signal_quality", "sentiment", "computed_at",
    ]
    return df[cols].sort_values("stock_score", ascending=False, na_position="last")


# -------------------------------------------------------------- data health


def get_collector_health() -> pd.DataFrame:
    """Résumé par source : dernière fetch, dernier content_at, nb lignes."""
    with session_scope() as session:
        stmt = (
            select(
                RawData.source,
                func.count(RawData.id).label("n_rows"),
                func.max(RawData.fetched_at).label("last_fetched_at"),
                func.max(RawData.content_at).label("last_content_at"),
            )
            .group_by(RawData.source)
        )
        rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows, columns=["source", "n_rows", "last_fetched_at", "last_content_at"]
    ).sort_values("source")


def get_feature_freshness() -> pd.DataFrame:
    """Fraîcheur par feature : nb targets, dernier computed_at."""
    with session_scope() as session:
        stmt = (
            select(
                Feature.feature_name,
                Feature.target_type,
                func.count(func.distinct(Feature.target_id)).label("n_targets"),
                func.max(Feature.computed_at).label("last_computed_at"),
            )
            .group_by(Feature.feature_name, Feature.target_type)
        )
        rows = session.execute(stmt).all()
    return pd.DataFrame(
        rows,
        columns=[
            "feature_name", "target_type", "n_targets", "last_computed_at",
        ],
    ).sort_values(["target_type", "feature_name"])


# -------------------------------------------------------------- performance


def get_performance_summary() -> dict[str, Any]:
    """KPIs globaux du track record.

    Retourne un dictionnaire avec :
    - n_theses : total de thèses générées
    - n_evaluated : thèses ayant au moins une évaluation non-active
    - n_success / n_failure / n_partial : par statut terminal (horizon 180+)
    - success_rate : n_success / (n_success + n_failure + n_partial)
    - avg_alpha : alpha moyen sur les évaluations 180+ avec alpha connu
    """
    terminal = ("success", "failure", "partial")
    with session_scope() as session:
        n_theses = session.query(func.count(Thesis.id)).scalar() or 0

        # Thèses avec au moins une éval terminale.
        n_evaluated = session.execute(
            select(func.count(func.distinct(Evaluation.thesis_id)))
            .where(Evaluation.status.in_(terminal))
        ).scalar() or 0

        # Comptes par statut sur jalons ≥ 180j.
        for status in terminal:
            count = session.execute(
                select(func.count(Evaluation.id))
                .where(Evaluation.status == status)
                .where(Evaluation.days_since_thesis >= 180)
            ).scalar() or 0
            if status == "success":
                n_success = count
            elif status == "failure":
                n_failure = count
            else:
                n_partial = count

        # Alpha moyen sur jalons ≥ 180j.
        avg_alpha_raw = session.execute(
            select(func.avg(Evaluation.alpha_pct))
            .where(Evaluation.status.in_(terminal))
            .where(Evaluation.days_since_thesis >= 180)
            .where(Evaluation.alpha_pct.is_not(None))
        ).scalar()

    n_terminal = n_success + n_failure + n_partial
    success_rate = n_success / n_terminal if n_terminal > 0 else None
    return {
        "n_theses": int(n_theses),
        "n_evaluated": int(n_evaluated),
        "n_success": int(n_success),
        "n_failure": int(n_failure),
        "n_partial": int(n_partial),
        "success_rate": float(success_rate) if success_rate is not None else None,
        "avg_alpha": float(avg_alpha_raw) if avg_alpha_raw is not None else None,
    }


def get_signal_performance(
    sector_id: str | None = "_all",
) -> pd.DataFrame:
    """Tableau signal_performance filtré par secteur.

    Passe `sector_id=None` pour le vrai NULL SQL (tous secteurs agrégés).
    Passe `sector_id="_all"` (défaut) pour idem via la sentinelle.
    Retourne colonnes : signal_name, horizon_days, n_predictions,
    n_successes, accuracy, avg_alpha, last_updated.
    """
    with session_scope() as session:
        stmt = select(SignalPerformance)
        if sector_id is None or sector_id == "_all":
            stmt = stmt.where(SignalPerformance.sector_id.is_(None))
        else:
            stmt = stmt.where(SignalPerformance.sector_id == sector_id)
        rows = session.execute(stmt).scalars().all()
        data = [
            {
                "signal_name": r.signal_name,
                "sector_id": r.sector_id,
                "horizon_days": r.horizon_days,
                "n_predictions": r.n_predictions,
                "n_successes": r.n_successes,
                "accuracy": r.accuracy,
                "avg_alpha": r.avg_alpha,
                "last_updated": r.last_updated,
            }
            for r in rows
        ]
    if not data:
        return pd.DataFrame(columns=[
            "signal_name", "sector_id", "horizon_days", "n_predictions",
            "n_successes", "accuracy", "avg_alpha", "last_updated",
        ])
    return pd.DataFrame(data).sort_values(
        ["horizon_days", "accuracy"], ascending=[True, False]
    )


def get_alpha_by_horizon() -> pd.DataFrame:
    """Alpha moyen et taux de succès par jalon, tous secteurs confondus.

    Colonnes : horizon_days, mean_alpha, success_rate, n_predictions.
    Seules les évaluations terminales avec alpha connu sont incluses.
    """
    terminal = ("success", "failure", "partial")
    with session_scope() as session:
        stmt = (
            select(
                Evaluation.days_since_thesis,
                func.avg(Evaluation.alpha_pct).label("mean_alpha"),
                func.count(Evaluation.id).label("n_predictions"),
                func.sum(
                    (Evaluation.status == "success").cast(type_=None)
                    if False else func.cast(
                        Evaluation.status == "success", type_=None
                    )
                ).label("n_success_raw"),
            )
            .where(Evaluation.status.in_(terminal))
            .group_by(Evaluation.days_since_thesis)
            .order_by(Evaluation.days_since_thesis)
        )
        # Simpler: fetch raw and compute in pandas.
        raw_stmt = (
            select(
                Evaluation.days_since_thesis,
                Evaluation.status,
                Evaluation.alpha_pct,
            )
            .where(Evaluation.status.in_(terminal))
        )
        rows = session.execute(raw_stmt).all()

    if not rows:
        return pd.DataFrame(columns=[
            "horizon_days", "mean_alpha", "success_rate", "n_predictions"
        ])

    df = pd.DataFrame(rows, columns=["horizon_days", "status", "alpha_pct"])
    agg = (
        df.groupby("horizon_days")
        .agg(
            mean_alpha=("alpha_pct", "mean"),
            n_predictions=("alpha_pct", "count"),
            n_success=("status", lambda s: (s == "success").sum()),
        )
        .reset_index()
    )
    agg["success_rate"] = agg["n_success"] / agg["n_predictions"]
    return agg[["horizon_days", "mean_alpha", "success_rate", "n_predictions"]]


# --------------------------------------------------------------- alerts


def get_alerts(
    *,
    severity: str | None = None,
    sector_id: str | None = None,
    acknowledged: bool | None = None,
    limit: int = 200,
) -> pd.DataFrame:
    """Alertes filtrables.

    Colonnes : id, created_at, rule_name, severity, asset_id, sector_id,
    message, thesis_id, acknowledged.
    """
    with session_scope() as session:
        stmt = select(Alert).order_by(Alert.created_at.desc()).limit(limit)
        if severity is not None:
            stmt = stmt.where(Alert.severity == severity)
        if sector_id is not None:
            stmt = stmt.where(Alert.sector_id == sector_id)
        if acknowledged is not None:
            stmt = stmt.where(Alert.acknowledged == acknowledged)
        rows = session.execute(stmt).scalars().all()
        data = [
            {
                "id": r.id,
                "created_at": r.created_at,
                "rule_name": r.rule_name,
                "severity": r.severity,
                "asset_id": r.asset_id,
                "sector_id": r.sector_id,
                "message": r.message,
                "thesis_id": r.thesis_id,
                "acknowledged": r.acknowledged,
            }
            for r in rows
        ]
    if not data:
        return pd.DataFrame(columns=[
            "id", "created_at", "rule_name", "severity",
            "asset_id", "sector_id", "message", "thesis_id", "acknowledged",
        ])
    return pd.DataFrame(data)


def acknowledge_alert(alert_id: int) -> bool:
    """Marque une alerte comme traitée. Retourne True si trouvée."""
    with session_scope() as session:
        stmt = (
            update(Alert)
            .where(Alert.id == alert_id)
            .values(acknowledged=True)
        )
        result = session.execute(stmt)
    return (result.rowcount or 0) > 0


# --------------------------------------------------------------- theses history


def list_thesis_ids() -> list[tuple[int, str, datetime]]:
    """IDs disponibles pour le sélecteur de la page détail.

    Liste triée du plus récent au plus ancien. Tuple (id, ticker, created_at)
    pour pouvoir construire un libellé lisible côté UI sans 2e requête.
    """
    with session_scope() as session:
        stmt = (
            select(Thesis.id, Thesis.asset_id, Thesis.created_at)
            .order_by(Thesis.created_at.desc())
        )
        return [(row[0], row[1], row[2]) for row in session.execute(stmt)]


def get_thesis_detail(thesis_id: int) -> dict[str, Any] | None:
    """Détail complet d'une thèse pour la page §7.9 Page 3.

    Retourne `None` si la thèse n'existe pas. Sinon un dict avec :
      - thesis : champs scalaires de la thèse (id, ticker, score, …)
      - dimensions : sous-scores (peut être vide)
      - triggers, risks, catalysts : listes JSON désérialisées
      - evaluations : liste de jalons (dicts)
      - latest_status / latest_alpha : raccourci sur le dernier jalon
    Les JSON malformés se dégradent en `[]` / `{}` plutôt que crasher.
    """
    with session_scope() as session:
        th = session.get(Thesis, thesis_id)
        if th is None:
            return None
        evals_rows = session.execute(
            select(Evaluation)
            .where(Evaluation.thesis_id == thesis_id)
            .order_by(Evaluation.days_since_thesis.asc())
        ).scalars().all()

        thesis_dict = {
            "id": th.id,
            "created_at": th.created_at,
            "asset_type": th.asset_type,
            "asset_id": th.asset_id,
            "sector_id": th.sector_id,
            "score": th.score,
            "recommendation": th.recommendation,
            "horizon_days": th.horizon_days,
            "entry_price": th.entry_price,
            "narrative": th.narrative,
            "model_version": th.model_version,
        }
        breakdown = _safe_json(th.score_breakdown_json) or {}
        triggers = _safe_json(th.triggers_json) or []
        risks = _safe_json(th.risks_json) or []
        catalysts = _safe_json(th.catalysts_json) or []
        entry_conditions = _safe_json(th.entry_conditions_json) or {}

        evaluations = [
            {
                "days_since_thesis": e.days_since_thesis,
                "evaluated_at": e.evaluated_at,
                "current_price": e.current_price,
                "return_pct": e.return_pct,
                "benchmark_return_pct": e.benchmark_return_pct,
                "alpha_pct": e.alpha_pct,
                "status": e.status,
                "notes": e.notes,
            }
            for e in evals_rows
        ]

    latest = evaluations[-1] if evaluations else None
    return {
        "thesis": thesis_dict,
        "dimensions": (breakdown.get("dimensions") if isinstance(breakdown, dict) else {}) or {},
        "details": (breakdown.get("details") if isinstance(breakdown, dict) else {}) or {},
        "triggers": triggers if isinstance(triggers, list) else [],
        "risks": risks if isinstance(risks, list) else [],
        "catalysts": catalysts if isinstance(catalysts, list) else [],
        "entry_conditions": entry_conditions if isinstance(entry_conditions, dict) else {},
        "evaluations": evaluations,
        "latest_status": latest["status"] if latest else None,
        "latest_alpha": latest["alpha_pct"] if latest else None,
    }


def get_price_history(
    ticker: str,
    *,
    start: datetime,
    end: datetime,
    fetched_before: datetime | None = None,
) -> pd.DataFrame:
    """Closes journaliers PIT pour le graphique de la page détail.

    Filtre `content_at` dans `[start, end]` et `fetched_at <= fetched_before`
    (par défaut `end`) — on ne montre que des prix qui auraient été
    observables à `fetched_before`. Colonnes : `date`, `close`.
    """
    cutoff = fetched_before or end
    with session_scope() as session:
        stmt = (
            select(RawData.content_at, RawData.payload_json)
            .where(RawData.source == "yfinance")
            .where(RawData.entity_type == "ohlcv_daily")
            .where(RawData.content_at >= start)
            .where(RawData.content_at <= end)
            .where(RawData.fetched_at <= cutoff)
            .order_by(RawData.content_at.asc())
        )
        rows = session.execute(stmt).all()

    out: list[dict[str, Any]] = []
    for content_at, payload_json in rows:
        payload = _safe_json(payload_json) or {}
        if not isinstance(payload, dict) or payload.get("ticker") != ticker:
            continue
        close = payload.get("close")
        if close is None:
            close = payload.get("adj_close")
        if close is None:
            continue
        try:
            out.append({"date": content_at, "close": float(close)})
        except (TypeError, ValueError):
            continue
    return pd.DataFrame(out, columns=["date", "close"])


def _safe_json(blob: Any) -> Any:
    """JSON parse silencieux (renvoie `None` si vide ou invalide)."""
    if not blob:
        return None
    try:
        return json.loads(blob)
    except (TypeError, ValueError):
        return None


def get_theses_history(
    *,
    status_filter: list[str] | None = None,
    date_from: datetime | None = None,
    date_to: datetime | None = None,
) -> pd.DataFrame:
    """Historique complet des thèses avec leur dernier statut d'évaluation.

    Colonnes : thesis_id, asset_id, sector_id, recommendation, score,
    created_at, entry_price, latest_status, latest_days, latest_alpha.
    `latest_*` est None si aucune évaluation n'existe encore.
    """
    with session_scope() as session:
        # Sous-requête : dernier jalon évalué par thèse.
        sub = (
            select(
                Evaluation.thesis_id,
                func.max(Evaluation.days_since_thesis).label("max_days"),
            )
            .group_by(Evaluation.thesis_id)
            .subquery()
        )
        stmt = (
            select(
                Thesis.id,
                Thesis.asset_id,
                Thesis.sector_id,
                Thesis.recommendation,
                Thesis.score,
                Thesis.created_at,
                Thesis.entry_price,
                Evaluation.status,
                Evaluation.days_since_thesis,
                Evaluation.alpha_pct,
            )
            .outerjoin(sub, sub.c.thesis_id == Thesis.id)
            .outerjoin(
                Evaluation,
                (Evaluation.thesis_id == Thesis.id)
                & (Evaluation.days_since_thesis == sub.c.max_days),
            )
            .order_by(Thesis.created_at.desc())
        )
        if date_from is not None:
            stmt = stmt.where(Thesis.created_at >= date_from)
        if date_to is not None:
            stmt = stmt.where(Thesis.created_at <= date_to)
        rows = session.execute(stmt).all()

    cols = [
        "thesis_id", "asset_id", "sector_id", "recommendation", "score",
        "created_at", "entry_price", "latest_status",
        "latest_days", "latest_alpha",
    ]
    df = pd.DataFrame(rows, columns=cols)
    if status_filter:
        df = df[df["latest_status"].isin(status_filter)]
    return df
