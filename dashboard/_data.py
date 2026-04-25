"""Accès base pour le dashboard — **lecture seule**.

Helpers purs qui renvoient des `pandas.DataFrame` pour Streamlit /
Plotly. Aucun couplage à Streamlit ici : testable en isolation.

Les fonctions respectent la discipline point-in-time (on lit
systématiquement la dernière valeur valide à `as_of`, par défaut
`utc_now()`). Aucun calcul n'est fait — on lit simplement les valeurs
persistées par les features / scorers.
"""
from __future__ import annotations

from datetime import datetime
from typing import Any, Optional

import pandas as pd
from sqlalchemy import func, select

from config.sectors import SECTORS_BY_ID
from config.watchlists import STOCK_WATCHLIST
from memory.database import Feature, RawData, session_scope, utc_now


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
