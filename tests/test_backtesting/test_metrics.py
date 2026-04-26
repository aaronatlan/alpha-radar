"""Tests des métriques de performance (Phase 5 étape 1)."""
from __future__ import annotations

import math

import pytest

from backtesting.metrics import (
    TRADING_DAYS_PER_YEAR,
    alpha_vs_benchmark,
    cagr,
    daily_returns_from_equity,
    hit_rate,
    max_drawdown,
    sharpe_ratio,
    total_return,
)


# ---------------------------------------------------------- sharpe_ratio


def test_sharpe_returns_none_for_short_series():
    assert sharpe_ratio([]) is None
    assert sharpe_ratio([0.01]) is None


def test_sharpe_returns_none_when_zero_volatility():
    """Variance = 0 → division par zéro évitée → None."""
    assert sharpe_ratio([0.01, 0.01, 0.01, 0.01]) is None


def test_sharpe_positive_for_positive_returns():
    # Returns constants positifs → mean>0, vol bornée → Sharpe positif.
    daily = [0.001, 0.002, 0.0015, 0.0008, 0.0012]
    s = sharpe_ratio(daily)
    assert s is not None
    assert s > 0


def test_sharpe_annualises_with_252():
    """Sharpe annualisé = mean/std * sqrt(252)."""
    daily = [0.01, -0.005, 0.012, -0.008, 0.006]
    n = len(daily)
    mean = sum(daily) / n
    var = sum((x - mean) ** 2 for x in daily) / (n - 1)
    expected = (mean / math.sqrt(var)) * math.sqrt(TRADING_DAYS_PER_YEAR)
    assert sharpe_ratio(daily) == pytest.approx(expected, rel=1e-9)


def test_sharpe_subtracts_risk_free_rate():
    daily = [0.001, 0.002, 0.0015]
    s_rf0 = sharpe_ratio(daily, risk_free_rate=0.0)
    s_rf4 = sharpe_ratio(daily, risk_free_rate=0.04)
    # Sharpe avec rf > 0 doit être plus bas (mean réduit).
    assert s_rf4 < s_rf0


# ---------------------------------------------------------- max_drawdown


def test_max_drawdown_zero_for_monotonic_growth():
    curve = [1.0, 1.05, 1.10, 1.15]
    assert max_drawdown(curve) == 0.0


def test_max_drawdown_finds_largest_peak_to_trough():
    # Peak à 1.20, trough à 0.80 → drawdown -33.3%
    curve = [1.0, 1.20, 1.10, 0.80, 0.95, 1.50]
    dd = max_drawdown(curve)
    assert dd == pytest.approx(-1.0 / 3.0, rel=1e-6)


def test_max_drawdown_handles_empty():
    assert max_drawdown([]) is None


def test_max_drawdown_handles_single_point():
    """Une seule observation → drawdown nul."""
    assert max_drawdown([1.0]) == 0.0


# ---------------------------------------------------------- hit_rate


def test_hit_rate_returns_proportion():
    # 3 positifs sur 5 = 0.60
    returns = [0.10, -0.05, 0.20, -0.01, 0.05]
    assert hit_rate(returns) == pytest.approx(0.60, rel=1e-9)


def test_hit_rate_zero_strict_for_negative_or_zero():
    # 0 n'est pas positif strict.
    returns = [0.0, -0.05, 0.0]
    assert hit_rate(returns) == 0.0


def test_hit_rate_empty_returns_none():
    assert hit_rate([]) is None


# ---------------------------------------------------------- alpha vs benchmark


def test_alpha_vs_benchmark_simple_difference():
    p = [0.01, 0.02, 0.015]
    b = [0.005, 0.005, 0.005]
    # Mean p = 0.015, Mean b = 0.005 → alpha = 0.010.
    assert alpha_vs_benchmark(p, b) == pytest.approx(0.010, rel=1e-9)


def test_alpha_vs_benchmark_empty_returns_none():
    assert alpha_vs_benchmark([], []) is None


def test_alpha_vs_benchmark_length_mismatch():
    """Mismatch de longueur → None (signal d'usage incorrect)."""
    assert alpha_vs_benchmark([0.01, 0.02], [0.01]) is None


# ---------------------------------------------------------- aggregation


def test_total_return_basic():
    assert total_return([1.0, 1.20]) == pytest.approx(0.20)


def test_total_return_handles_loss():
    assert total_return([1.0, 0.80]) == pytest.approx(-0.20)


def test_total_return_empty_or_zero_initial():
    assert total_return([]) is None
    assert total_return([1.0]) is None
    assert total_return([0.0, 1.0]) is None


def test_cagr_consistent_with_total_return():
    """Pour 1 année (252 j), CAGR = total_return."""
    # 252 valeurs : 1 → 1.20 sur 251 step => ~252j de bourse.
    curve = [1.0] * 252
    curve[-1] = 1.20
    c = cagr(curve)
    assert c is not None
    # Avec ce curve constant à 1.0 sauf le dernier point, le calcul
    # CAGR sur (n-1)=251 step donne 1.20**(252/251) - 1 ≈ 0.2009
    assert c == pytest.approx(0.20, rel=1e-2)


def test_cagr_handles_empty():
    assert cagr([]) is None
    assert cagr([1.0]) is None
    assert cagr([0.0, 1.0]) is None


def test_cagr_handles_negative_terminal():
    """Equity finale ≤ 0 → CAGR indéfini."""
    assert cagr([1.0, -0.5]) is None


# ---------------------------------------------------------- daily_returns


def test_daily_returns_from_equity_basic():
    out = daily_returns_from_equity([1.0, 1.10, 1.20])
    assert out == pytest.approx([0.10, 1.20 / 1.10 - 1.0])


def test_daily_returns_from_equity_skips_zero_prev():
    out = daily_returns_from_equity([0.0, 1.0, 2.0])
    # Premier pair (0, 1) sauté ; (1, 2) → +1.0
    assert out == pytest.approx([1.0])


def test_daily_returns_from_equity_empty():
    assert daily_returns_from_equity([]) == []
    assert daily_returns_from_equity([1.0]) == []
