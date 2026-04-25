"""Helpers I/O partagés entre `thesis.generator` et `thesis.evaluator`.

`latest_close_at` est utilisé à deux endroits :
  - dans le générateur, pour fixer `entry_price` à la création de la thèse,
  - dans l'évaluateur, pour relever le prix à chaque jalon (et pour le
    return moyen des peers du secteur).

Garder ce helper isolé évite la duplication de la requête PIT et garantit
qu'une évolution du schéma `raw_data` se propage des deux côtés.
"""
from __future__ import annotations

import json
from datetime import datetime

from sqlalchemy import select

from memory.database import RawData, session_scope


def latest_close_at(ticker: str, at: datetime) -> float | None:
    """Dernier close yfinance ≤ `at` pour `ticker`, sous discipline PIT.

    Double filtre `content_at <= at` et `fetched_at <= at`. Le `close`
    est privilégié, on retombe sur `adj_close` si nécessaire. Les
    payloads mal formés sont silencieusement sautés.
    """
    stmt = (
        select(RawData.content_at, RawData.payload_json)
        .where(RawData.source == "yfinance")
        .where(RawData.entity_type == "ohlcv_daily")
        .where(RawData.content_at <= at)
        .where(RawData.fetched_at <= at)
        .order_by(RawData.content_at.desc())
    )
    with session_scope() as session:
        for _, payload_json in session.execute(stmt):
            try:
                payload = json.loads(payload_json)
            except (TypeError, ValueError):
                continue
            if payload.get("ticker") != ticker:
                continue
            close = payload.get("close")
            if close is None:
                close = payload.get("adj_close")
            if close is None:
                continue
            try:
                return float(close)
            except (TypeError, ValueError):
                continue
    return None
