"""Tests des features techniques (RSI, momentum, volume ratio).

Seed direct de `raw_data` avec des séances OHLCV synthétiques —
aucune connexion réseau.
"""
from __future__ import annotations

import json
from datetime import datetime, timedelta

import pytest

from features.technical import (
    Momentum30DFeature,
    RSI14Feature,
    VolumeRatio7_30Feature,
    _wilder_rsi,
)
from memory.database import RawData, session_scope


def _seed_session(
    ticker: str,
    date: datetime,
    close: float,
    volume: int = 1_000_000,
    fetched_at: datetime | None = None,
) -> None:
    payload = {
        "ticker": ticker,
        "session_date": date.strftime("%Y-%m-%d"),
        "open": close,
        "high": close,
        "low": close,
        "close": close,
        "adj_close": close,
        "volume": volume,
    }
    with session_scope() as session:
        session.add(
            RawData(
                source="yfinance",
                entity_type="ohlcv_daily",
                entity_id=f"{ticker}:{date.strftime('%Y-%m-%d')}",
                fetched_at=fetched_at or date + timedelta(hours=1),
                content_at=date,
                payload_json=json.dumps(payload),
                hash=f"{ticker}:{date.strftime('%Y-%m-%d')}:{close}",
            )
        )


# ------------------------------------------------------------------- Wilder


def test_wilder_rsi_constant_series_is_fifty():
    """Aucun mouvement → gains et pertes nulles → RSI = 50 (convention)."""
    closes = [100.0] * 20
    assert _wilder_rsi(closes, window=14) == 50.0


def test_wilder_rsi_monotonic_up_is_hundred():
    """Série strictement croissante → pas de perte → RSI = 100."""
    closes = [100.0 + i for i in range(20)]
    assert _wilder_rsi(closes, window=14) == 100.0


def test_wilder_rsi_not_enough_data_returns_none():
    assert _wilder_rsi([1.0, 2.0, 3.0], window=14) is None


def test_wilder_rsi_range_stays_in_bounds():
    """Série oscillante : RSI reste strictement dans [0, 100]."""
    closes = [100 + ((-1) ** i) * 2 for i in range(30)]
    v = _wilder_rsi(closes, window=14)
    assert v is not None and 0 < v < 100


# ----------------------------------------------------------------- RSI14Feature


def test_rsi14_returns_none_without_data(tmp_db):
    feat = RSI14Feature(tickers=["NVDA"])
    assert feat.compute("NVDA", datetime(2026, 2, 1)) is None


def test_rsi14_nominal_on_monotonic_series(tmp_db):
    as_of = datetime(2026, 2, 15)
    # 20 séances consécutives avec closes croissants → RSI = 100
    for i in range(20):
        day = as_of - timedelta(days=20 - i)
        _seed_session("NVDA", day, close=100.0 + i)
    feat = RSI14Feature(tickers=["NVDA"])
    result = feat.compute("NVDA", as_of)
    assert result is not None
    value, meta = result
    assert value == 100.0
    assert meta["n_sessions"] == 20


def test_rsi14_respects_pit_strict(tmp_db):
    """La clôture de `as_of` elle-même ne doit jamais être utilisée."""
    as_of = datetime(2026, 2, 15)
    # Seed une séance AU `as_of` exact qui ne doit pas entrer en compte
    _seed_session("NVDA", as_of, close=1e9)
    # + 15 séances antérieures stables
    for i in range(15):
        _seed_session("NVDA", as_of - timedelta(days=i + 1), close=100.0)

    feat = RSI14Feature(tickers=["NVDA"])
    result = feat.compute("NVDA", as_of)
    assert result is not None
    value, _ = result
    assert value == 50.0  # la grosse valeur n'a pas été lue


def test_rsi14_ignores_sessions_fetched_after_as_of(tmp_db):
    as_of = datetime(2026, 2, 15)
    # Seed 20 séances dont le fetched_at est APRÈS as_of → PIT interdit
    for i in range(20):
        day = as_of - timedelta(days=20 - i)
        _seed_session(
            "NVDA", day, close=100.0 + i,
            fetched_at=as_of + timedelta(days=5),
        )
    feat = RSI14Feature(tickers=["NVDA"])
    assert feat.compute("NVDA", as_of) is None


# -------------------------------------------------------- Momentum30DFeature


def test_momentum_returns_none_when_too_few_sessions(tmp_db):
    as_of = datetime(2026, 2, 15)
    for i in range(10):
        _seed_session("NVDA", as_of - timedelta(days=i + 1), close=100.0)
    feat = Momentum30DFeature(tickers=["NVDA"])
    assert feat.compute("NVDA", as_of) is None


def test_momentum_nominal_20pct_gain(tmp_db):
    as_of = datetime(2026, 2, 15)
    # 31 séances strictement croissantes de 100 à 120 → momentum = 0.20
    for i in range(31):
        day = as_of - timedelta(days=31 - i)
        price = 100.0 + 20.0 * (i / 30)
        _seed_session("NVDA", day, close=price)
    feat = Momentum30DFeature(tickers=["NVDA"])
    value, meta = feat.compute("NVDA", as_of)
    assert value == pytest.approx(0.20, rel=1e-6)
    assert meta["window"] == 30


# ---------------------------------------------------- VolumeRatio7_30Feature


def test_volume_ratio_nominal(tmp_db):
    as_of = datetime(2026, 2, 15)
    # 30 séances : 23 à volume 1e6, 7 récentes à volume 3e6 → ratio théorique
    for i in range(30):
        day = as_of - timedelta(days=30 - i)
        vol = 3_000_000 if i >= 23 else 1_000_000
        _seed_session("NVDA", day, close=100.0, volume=vol)
    feat = VolumeRatio7_30Feature(tickers=["NVDA"])
    value, _ = feat.compute("NVDA", as_of)
    # recent avg = 3e6 ; long avg = (23*1e6 + 7*3e6)/30 = 44e6/30 ≈ 1.4667e6
    assert value == pytest.approx(3_000_000 / (44_000_000 / 30), rel=1e-6)


def test_volume_ratio_insufficient_data(tmp_db):
    as_of = datetime(2026, 2, 15)
    for i in range(10):
        _seed_session("NVDA", as_of - timedelta(days=i + 1), close=100.0)
    feat = VolumeRatio7_30Feature(tickers=["NVDA"])
    assert feat.compute("NVDA", as_of) is None


# --------------------------------------------------------------------- run


def test_run_stores_multiple_tickers(tmp_db):
    as_of = datetime(2026, 2, 15)
    for ticker in ("NVDA", "AMD"):
        for i in range(20):
            day = as_of - timedelta(days=20 - i)
            _seed_session(ticker, day, close=100.0 + i)

    feat = RSI14Feature(tickers=["NVDA", "AMD"])
    n = feat.run(as_of=as_of)
    assert n == 2
