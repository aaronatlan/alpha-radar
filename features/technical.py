"""Features techniques calculées sur les séries OHLCV yfinance.

Trois features, toutes `target_type='asset'` (clé = ticker) :

- `rsi_14` — Relative Strength Index 14j (Wilder). Indique la
  sur/sous-achat dans [0, 100].
- `momentum_30d` — rendement net sur 30 jours ouvrés (fraction, pas %).
- `volume_ratio_7_30` — volume moyen sur 7j / volume moyen sur 30j.

Point-in-time
-------------
Toutes les séances lues sont filtrées avec :

    content_at  <  as_of     (clôture strictement antérieure à l'instant T)
    fetched_at  <= as_of

La borne stricte sur `content_at` est essentielle : la clôture du jour
de `as_of` n'est connue qu'après `as_of`. L'utiliser serait une fuite.

Fenêtre de fetch
----------------
Chaque feature déclare une borne `_lookback_days` pour limiter la
quantité de rows SQL ramenées. Choisie large (lookback * 2) pour
absorber week-ends, fériés et trous de collecte sans casser le calcul.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta
from typing import Any

from sqlalchemy import select

from config.watchlists import STOCK_WATCHLIST, WATCHLIST_TICKERS
from features.base import BaseFeature
from memory.database import RawData, session_scope


# -------------------------------------------------------------------- helpers


def _load_closes(
    ticker: str, as_of: datetime, lookback_days: int
) -> list[tuple[datetime, float, int | None]]:
    """Retourne les séances `(date, close, volume)` pour `ticker`,
    triées chronologiquement, avec le double filtre PIT.

    `lookback_days` doit couvrir largement la plus longue fenêtre de la
    feature (on prend typiquement 2× la fenêtre nominale).
    """
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(RawData.content_at, RawData.payload_json)
        .where(RawData.source == "yfinance")
        .where(RawData.entity_type == "ohlcv_daily")
        .where(RawData.content_at >= start)
        .where(RawData.content_at < as_of)  # strict : clôture du jour as_of inconnue
        .where(RawData.fetched_at <= as_of)
    )

    rows: list[tuple[datetime, float, int | None]] = []
    with session_scope() as session:
        for content_at, payload_json in session.execute(stmt):
            try:
                payload = json.loads(payload_json)
            except (TypeError, ValueError):
                continue
            if payload.get("ticker") != ticker:
                continue
            close = payload.get("adj_close") or payload.get("close")
            if close is None:
                continue
            rows.append((content_at, float(close), payload.get("volume")))

    rows.sort(key=lambda r: r[0])
    return rows


# ------------------------------------------------------------- RSI (Wilder)


def _wilder_rsi(closes: list[float], window: int) -> float | None:
    """RSI de Wilder sur la série `closes`. Retourne `None` si < window+1 points.

    Convention : pas de gain/perte → RSI = 50. Aucune perte → RSI = 100.
    """
    if len(closes) < window + 1:
        return None

    gains: list[float] = []
    losses: list[float] = []
    for i in range(1, len(closes)):
        diff = closes[i] - closes[i - 1]
        gains.append(max(diff, 0.0))
        losses.append(-min(diff, 0.0))

    # Moyenne initiale (SMA sur les `window` premiers deltas).
    avg_gain = sum(gains[:window]) / window
    avg_loss = sum(losses[:window]) / window

    # Smoothing de Wilder pour les deltas suivants.
    for g, l in zip(gains[window:], losses[window:]):
        avg_gain = (avg_gain * (window - 1) + g) / window
        avg_loss = (avg_loss * (window - 1) + l) / window

    if avg_loss == 0.0 and avg_gain == 0.0:
        return 50.0
    if avg_loss == 0.0:
        return 100.0

    rs = avg_gain / avg_loss
    return 100.0 - 100.0 / (1.0 + rs)


# ---------------------------------------------------------------- base asset


class _AssetFeature(BaseFeature):
    """Base partagée par les features techniques : targets = watchlist tickers."""

    target_type = "asset"

    def __init__(self, tickers: list[str] | None = None) -> None:
        super().__init__()
        self._tickers = list(tickers) if tickers else list(WATCHLIST_TICKERS)

    def targets(self) -> list[str]:
        return list(self._tickers)


# ---------------------------------------------------------------- RSI feature


class RSI14Feature(_AssetFeature):
    """RSI 14j pour chaque ticker de la watchlist."""

    feature_name = "rsi_14"
    window = 14
    _lookback_days = 60  # marge pour week-ends et fériés

    def compute(self, target_id: str, as_of: datetime):
        rows = _load_closes(target_id, as_of, self._lookback_days)
        closes = [r[1] for r in rows]
        rsi = _wilder_rsi(closes, self.window)
        if rsi is None:
            return None
        metadata = {"n_sessions": len(closes), "window": self.window}
        return rsi, metadata


# ------------------------------------------------------------ momentum feature


class Momentum30DFeature(_AssetFeature):
    """Rendement net sur 30 séances (fraction, pas %).

    On prend la dernière clôture disponible dans [as_of - 30j séances,
    as_of) et on la compare à la 30e précédente. Si moins de 30 séances
    disponibles, retourne `None`.
    """

    feature_name = "momentum_30d"
    window = 30
    _lookback_days = 75

    def compute(self, target_id: str, as_of: datetime):
        rows = _load_closes(target_id, as_of, self._lookback_days)
        if len(rows) < self.window + 1:
            return None
        closes = [r[1] for r in rows]
        recent = closes[-1]
        prior = closes[-(self.window + 1)]
        if prior == 0:
            return None
        value = (recent / prior) - 1.0
        metadata = {"window": self.window, "n_sessions": len(closes)}
        return value, metadata


# ------------------------------------------------------------- volume feature


class VolumeRatio7_30Feature(_AssetFeature):
    """Ratio volume moyen 7 séances / volume moyen 30 séances.

    >1 : attention croissante. Valeurs typiques entre 0.5 et 3.
    """

    feature_name = "volume_ratio_7_30"
    short_window = 7
    long_window = 30
    _lookback_days = 75

    def compute(self, target_id: str, as_of: datetime):
        rows = _load_closes(target_id, as_of, self._lookback_days)
        if len(rows) < self.long_window:
            return None
        volumes = [r[2] for r in rows if r[2] is not None]
        if len(volumes) < self.long_window:
            return None

        recent = volumes[-self.short_window:]
        long_ = volumes[-self.long_window:]
        avg_recent = sum(recent) / len(recent)
        avg_long = sum(long_) / len(long_)
        if avg_long == 0:
            return None
        value = avg_recent / avg_long
        metadata = {
            "short_window": self.short_window,
            "long_window": self.long_window,
            "n_sessions": len(volumes),
        }
        return value, metadata


__all__ = [
    "RSI14Feature",
    "Momentum30DFeature",
    "VolumeRatio7_30Feature",
    "STOCK_WATCHLIST",  # ré-exporté pour pratique
]
