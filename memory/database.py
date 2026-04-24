"""Schéma SQLite complet d'Alpha Radar (section 5 du cahier des charges).

Toutes les tables sont définies ici via SQLAlchemy déclaratif. Les données
brutes, features, thèses et évaluations y sont stockées en respectant les
principes **point-in-time** et **append-only** :

- `raw_data` : jamais modifiée. Déduplication par `(source, entity_id, hash)`.
- `features` : recalculables depuis `raw_data` à tout moment.
- `theses` : append-only, immuables une fois créées.
- `evaluations` : append-only, une ligne par jalon (30/90/180/365/540 j).

Point-in-time : chaque ligne porte un timestamp explicite de validité
(`content_at` ou `computed_at`). Un calcul à l'instant T ne doit jamais
utiliser des lignes dont le timestamp de validité est postérieur à T —
c'est la discipline qui empêche toute fuite d'information du futur.
"""
from __future__ import annotations

from contextlib import contextmanager
from datetime import datetime, timezone
from typing import Iterator


def utc_now() -> datetime:
    """Retourne l'instant courant UTC en datetime naïf.

    Convention du projet : tous les timestamps stockés en base sont
    naïfs et implicitement UTC. On évite `datetime.utcnow()` (déprécié
    en Python 3.12) au profit d'une construction tz-aware convertie.
    """
    return datetime.now(timezone.utc).replace(tzinfo=None)

from sqlalchemy import (
    Boolean,
    Column,
    DateTime,
    Float,
    ForeignKey,
    Index,
    Integer,
    String,
    Text,
    UniqueConstraint,
    create_engine,
    event,
)
from sqlalchemy.orm import DeclarativeBase, Session, sessionmaker

from config.settings import settings


class Base(DeclarativeBase):
    """Base déclarative SQLAlchemy."""


# --- Tables ---------------------------------------------------------------


class RawData(Base):
    """Données brutes horodatées, append-only.

    - `fetched_at` : instant où le collecteur a récupéré la donnée.
    - `content_at` : timestamp intrinsèque (date de publication d'un papier,
      clôture d'une séance de bourse, etc.). C'est **cette** date qui fait
      foi pour la reconstruction point-in-time des features.

    La contrainte UNIQUE (source, entity_id, hash) rend la collecte
    idempotente : re-collecter la même donnée est un no-op.
    """

    __tablename__ = "raw_data"

    id = Column(Integer, primary_key=True, autoincrement=True)
    source = Column(String, nullable=False)          # 'arxiv', 'yfinance'…
    entity_type = Column(String, nullable=False)     # 'paper', 'ohlcv_daily'…
    entity_id = Column(String, nullable=False)       # id unique côté source
    fetched_at = Column(DateTime, nullable=False, default=utc_now)
    content_at = Column(DateTime, nullable=True)
    payload_json = Column(Text, nullable=False)
    hash = Column(String, nullable=False)

    __table_args__ = (
        UniqueConstraint("source", "entity_id", "hash", name="uq_raw_dedupe"),
        Index("idx_raw_source_time", "source", "content_at"),
        Index("idx_raw_entity", "entity_type", "entity_id"),
    )


class Feature(Base):
    """Feature calculée point-in-time.

    `computed_at` est l'instant où la feature **était valide** — pas
    l'instant de calcul. Deux calculs de la même feature à la même date
    doivent produire la même valeur (d'où la contrainte UNIQUE).
    """

    __tablename__ = "features"

    id = Column(Integer, primary_key=True, autoincrement=True)
    feature_name = Column(String, nullable=False)
    target_type = Column(String, nullable=False)  # 'sector' ou 'asset'
    target_id = Column(String, nullable=False)
    computed_at = Column(DateTime, nullable=False)
    value = Column(Float, nullable=False)
    metadata_json = Column(Text, nullable=True)

    __table_args__ = (
        UniqueConstraint(
            "feature_name", "target_type", "target_id", "computed_at",
            name="uq_feat_unique",
        ),
        Index("idx_feat_target", "target_type", "target_id", "computed_at"),
        Index("idx_feat_name", "feature_name", "computed_at"),
    )


class Sector(Base):
    """Définition persistée d'un secteur suivi."""

    __tablename__ = "sectors"

    id = Column(String, primary_key=True)
    name = Column(String, nullable=False)
    category = Column(String, nullable=False)
    keywords_json = Column(Text, nullable=False)
    arxiv_categories_json = Column(Text, nullable=True)
    active = Column(Boolean, default=True, nullable=False)
    created_at = Column(DateTime, default=utc_now, nullable=False)


class Thesis(Base):
    """Thèse d'investissement générée par le système.

    **Immuable** — jamais modifiée après insertion. Un re-scoring crée une
    nouvelle ligne. Les poids utilisés sont snapshottés dans
    `weights_snapshot_json` pour garantir la reproductibilité.
    """

    __tablename__ = "theses"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    asset_type = Column(String, nullable=False)       # 'stock' | 'crypto' | 'startup'
    asset_id = Column(String, nullable=False)
    sector_id = Column(String, nullable=False)
    score = Column(Float, nullable=False)
    score_breakdown_json = Column(Text, nullable=False)
    recommendation = Column(String, nullable=False)   # 'BUY' | 'WATCH' | 'AVOID'
    horizon_days = Column(Integer, nullable=False)
    entry_price = Column(Float, nullable=True)
    entry_conditions_json = Column(Text, nullable=True)
    triggers_json = Column(Text, nullable=False)
    risks_json = Column(Text, nullable=False)
    catalysts_json = Column(Text, nullable=True)
    narrative = Column(Text, nullable=False)
    model_version = Column(String, nullable=False)
    weights_snapshot_json = Column(Text, nullable=False)

    __table_args__ = (
        Index("idx_theses_asset", "asset_type", "asset_id"),
        Index("idx_theses_created", "created_at"),
    )


class Evaluation(Base):
    """Évaluation périodique d'une thèse aux jalons définis (append-only)."""

    __tablename__ = "evaluations"

    id = Column(Integer, primary_key=True, autoincrement=True)
    thesis_id = Column(Integer, ForeignKey("theses.id"), nullable=False)
    evaluated_at = Column(DateTime, nullable=False)
    days_since_thesis = Column(Integer, nullable=False)  # 30, 90, 180, 365, 540…
    current_price = Column(Float, nullable=True)
    return_pct = Column(Float, nullable=True)
    benchmark_return_pct = Column(Float, nullable=True)
    alpha_pct = Column(Float, nullable=True)
    status = Column(String, nullable=False)  # 'active' | 'success' | 'failure' | 'partial'
    events_occurred_json = Column(Text, nullable=True)
    predicted_catalysts_hit = Column(Boolean, nullable=True)
    notes = Column(Text, nullable=True)

    __table_args__ = (
        Index("idx_eval_thesis", "thesis_id"),
        Index("idx_eval_date", "evaluated_at"),
    )


class SignalPerformance(Base):
    """Track record d'un signal, éventuellement croisé avec secteur/horizon."""

    __tablename__ = "signal_performance"

    id = Column(Integer, primary_key=True, autoincrement=True)
    signal_name = Column(String, nullable=False)
    sector_id = Column(String, nullable=True)  # NULL = tous secteurs confondus
    horizon_days = Column(Integer, nullable=False)
    n_predictions = Column(Integer, nullable=False)
    n_successes = Column(Integer, nullable=False)
    accuracy = Column(Float, nullable=False)
    avg_alpha = Column(Float, nullable=True)
    last_updated = Column(DateTime, default=utc_now, nullable=False)

    __table_args__ = (
        UniqueConstraint("signal_name", "sector_id", "horizon_days", name="uq_sigperf"),
    )


class Alert(Base):
    """Alerte déclenchée par le moteur de règles."""

    __tablename__ = "alerts"

    id = Column(Integer, primary_key=True, autoincrement=True)
    created_at = Column(DateTime, default=utc_now, nullable=False)
    rule_name = Column(String, nullable=False)
    severity = Column(String, nullable=False)  # 'info' | 'warning' | 'critical'
    asset_type = Column(String, nullable=True)
    asset_id = Column(String, nullable=True)
    sector_id = Column(String, nullable=True)
    message = Column(Text, nullable=False)
    data_json = Column(Text, nullable=True)
    thesis_id = Column(Integer, ForeignKey("theses.id"), nullable=True)
    acknowledged = Column(Boolean, default=False, nullable=False)


# --- Engine / session -----------------------------------------------------

_engine = None
_SessionFactory: sessionmaker | None = None


def get_engine(db_url: str | None = None):
    """Retourne l'engine SQLAlchemy (lazy singleton).

    Par défaut utilise `settings.db_url`. Un `db_url` explicite permet de
    cibler une base de test. Si l'URL change, l'engine est recréé.
    """
    global _engine, _SessionFactory
    url = db_url or settings.db_url
    if _engine is None or str(_engine.url) != url:
        _engine = create_engine(url, future=True)
        if url.startswith("sqlite"):
            # Active les foreign keys sur SQLite (désactivées par défaut).
            @event.listens_for(_engine, "connect")
            def _fk_on(dbapi_conn, _):  # pragma: no cover
                dbapi_conn.execute("PRAGMA foreign_keys=ON")

        _SessionFactory = sessionmaker(
            bind=_engine, future=True, expire_on_commit=False
        )
    return _engine


def init_db(db_url: str | None = None) -> None:
    """Crée toutes les tables si elles n'existent pas. Idempotent."""
    engine = get_engine(db_url)
    Base.metadata.create_all(engine)


@contextmanager
def session_scope(db_url: str | None = None) -> Iterator[Session]:
    """Context manager transactionnel.

    Commit automatique en sortie, rollback en cas d'exception, session
    toujours fermée proprement.
    """
    get_engine(db_url)
    assert _SessionFactory is not None, "Engine non initialisé"
    session = _SessionFactory()
    try:
        yield session
        session.commit()
    except Exception:
        session.rollback()
        raise
    finally:
        session.close()
