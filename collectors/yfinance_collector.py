"""Collecteur yfinance.

Récupère les séances quotidiennes OHLCV pour les tickers de
`config.watchlists.WATCHLIST_TICKERS`.

Point-in-time
-------------
Chaque séance est stockée avec `content_at` = horodatage de clôture de
la séance (converti UTC, naïf). Une feature à l'instant T ne peut
utiliser que des séances dont la clôture est **strictement** antérieure
à T, sinon on voit la clôture du jour avant la fin du jour.

`entity_id = f"{ticker}:{YYYY-MM-DD}"` garantit un identifiant unique
par (actif, séance) et rend la déduplication triviale si on re-collecte
la même période.

Mode dégradé
------------
Yahoo Finance est rate-limité de façon opaque. Un ticker qui échoue est
loggé et skippé, les autres continuent. Le cycle ne plante jamais à
cause d'un ticker isolé.
"""
from __future__ import annotations

from datetime import datetime, timezone
from typing import Any

import yfinance as yf
from loguru import logger

from collectors.base import BaseCollector, NormalizedItem
from config.watchlists import WATCHLIST_TICKERS


class YFinanceCollector(BaseCollector):
    """Collecte les séances OHLCV quotidiennes pour une watchlist."""

    source_name = "yfinance"
    request_delay = 1.5

    def __init__(self, tickers: list[str] | None = None) -> None:
        super().__init__()
        self.tickers: list[str] = list(tickers) if tickers else list(WATCHLIST_TICKERS)

    # ------------------------------------------------------------ collect

    def collect(self, since: datetime, until: datetime) -> list[dict[str, Any]]:
        """Récupère les séances journalières dans [since, until].

        Retourne une liste de dicts (un par (ticker, séance)). yfinance
        attend des bornes en dates calendaires — on passe en format
        `YYYY-MM-DD` pour rester agnostique du fuseau de l'exchange.
        """
        start_str = since.strftime("%Y-%m-%d")
        end_str = until.strftime("%Y-%m-%d")

        items: list[dict[str, Any]] = []
        for ticker in self.tickers:
            try:
                tk = yf.Ticker(ticker)
                hist = tk.history(
                    start=start_str,
                    end=end_str,
                    interval="1d",
                    auto_adjust=False,
                    actions=False,
                )
            except Exception as exc:
                # Mode dégradé : on skippe ce ticker et on continue.
                logger.warning("[yfinance] échec history({}) : {}", ticker, exc)
                continue

            if hist is None or hist.empty:
                logger.debug("[yfinance] {} : aucune séance sur la période", ticker)
                self._throttle()
                continue

            for ts, row in hist.iterrows():
                py_ts = ts.to_pydatetime()
                items.append(
                    {
                        "ticker": ticker,
                        "timestamp": py_ts,
                        "open": _safe_float(row.get("Open")),
                        "high": _safe_float(row.get("High")),
                        "low": _safe_float(row.get("Low")),
                        "close": _safe_float(row.get("Close")),
                        "adj_close": _safe_float(row.get("Adj Close")),
                        "volume": _safe_int(row.get("Volume")),
                    }
                )

            self._throttle()

        return items

    # ---------------------------------------------------------- normalize

    def normalize(self, raw: dict[str, Any]) -> NormalizedItem | None:
        """Convertit une séance en item canonique.

        `content_at` = horodatage de clôture en UTC naïf. L'id composite
        `TICKER:YYYY-MM-DD` sert à la fois à l'idempotence et à
        l'inspection manuelle de la base.
        """
        ticker = raw.get("ticker")
        ts = raw.get("timestamp")
        if not ticker or ts is None:
            return None

        if hasattr(ts, "tzinfo") and ts.tzinfo is not None:
            ts_utc = ts.astimezone(timezone.utc).replace(tzinfo=None)
        else:
            ts_utc = ts

        entity_id = f"{ticker}:{ts_utc.strftime('%Y-%m-%d')}"

        payload = {
            "ticker": ticker,
            "session_date": ts_utc.strftime("%Y-%m-%d"),
            "open": raw.get("open"),
            "high": raw.get("high"),
            "low": raw.get("low"),
            "close": raw.get("close"),
            "adj_close": raw.get("adj_close"),
            "volume": raw.get("volume"),
        }

        return NormalizedItem(
            entity_type="ohlcv_daily",
            entity_id=entity_id,
            content_at=ts_utc,
            payload=payload,
        )


def _safe_float(v: Any) -> float | None:
    """Conversion robuste en float (None pour NaN et valeurs illisibles)."""
    try:
        if v is None:
            return None
        f = float(v)
        if f != f:  # NaN
            return None
        return f
    except (TypeError, ValueError):
        return None


def _safe_int(v: Any) -> int | None:
    f = _safe_float(v)
    return int(f) if f is not None else None
