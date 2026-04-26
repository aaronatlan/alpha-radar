"""Métriques de performance — Phase 5 étape 1.

Pures fonctions sur listes / arrays. Aucun couplage à la base ou au
domaine — réutilisables pour le simulateur de portefeuille (`portfolio.py`)
ou pour reporting ad-hoc.

Conventions
-----------
- Returns en **fraction décimale** (ex : `+0.10` = +10%, pas 10.0).
- Daily returns : on suppose 252 jours de bourse par an pour
  l'annualisation du Sharpe.
- Toutes les fonctions tolèrent les listes vides ou trop courtes en
  retournant `None` plutôt qu'en levant — le caller reste responsable
  de gérer le cas dégradé.
"""
from __future__ import annotations

import math
from typing import Iterable, Sequence

#: Nombre de jours de bourse US par an. Convention standard.
TRADING_DAYS_PER_YEAR = 252


# ----------------------------------------------------------------- core


def sharpe_ratio(
    daily_returns: Sequence[float],
    *,
    risk_free_rate: float = 0.0,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Sharpe ratio annualisé d'une série de daily returns.

    `risk_free_rate` est le taux annualisé en fraction (ex : 0.04 = 4%).
    Retourne `None` si moins de 2 observations ou stdev nulle.
    """
    if len(daily_returns) < 2:
        return None
    rf_daily = risk_free_rate / periods_per_year
    excess = [r - rf_daily for r in daily_returns]
    n = len(excess)
    mean = sum(excess) / n
    var = sum((x - mean) ** 2 for x in excess) / (n - 1)   # sample variance
    std = math.sqrt(var)
    if std == 0:
        return None
    return (mean / std) * math.sqrt(periods_per_year)


def max_drawdown(equity_curve: Sequence[float]) -> float | None:
    """Maximum drawdown (en fraction décimale, négative) sur une courbe.

    `equity_curve` est la valeur du portefeuille indexée par jour. Le
    drawdown est mesuré par rapport au plus haut historique (peak).
    Retourne `None` si la courbe est vide.
    """
    if not equity_curve:
        return None
    peak = equity_curve[0]
    max_dd = 0.0
    for value in equity_curve:
        peak = max(peak, value)
        if peak <= 0:
            continue
        dd = (value - peak) / peak
        if dd < max_dd:
            max_dd = dd
    return max_dd


def hit_rate(returns: Iterable[float]) -> float | None:
    """Proportion de returns strictement positifs.

    Adapté pour des returns d'investissements terminaux (alpha à 180j),
    pas pour des séries quotidiennes. Retourne `None` si vide.
    """
    items = list(returns)
    if not items:
        return None
    n_positive = sum(1 for r in items if r > 0)
    return n_positive / len(items)


def alpha_vs_benchmark(
    portfolio_returns: Sequence[float],
    benchmark_returns: Sequence[float],
) -> float | None:
    """Différence des moyennes (alpha simple, pas du CAPM-alpha).

    Pour un alpha de Jensen complet (intercept de la régression), il
    faudrait un solveur OLS — ajout en Phase 6 ML. Cette version donne
    le **gap moyen** portfolio − benchmark, suffisant pour
    benchmarker rapidement une configuration.
    """
    if not portfolio_returns or not benchmark_returns:
        return None
    if len(portfolio_returns) != len(benchmark_returns):
        return None
    mean_p = sum(portfolio_returns) / len(portfolio_returns)
    mean_b = sum(benchmark_returns) / len(benchmark_returns)
    return mean_p - mean_b


# ----------------------------------------------------------- aggregation


def total_return(equity_curve: Sequence[float]) -> float | None:
    """Return total de la courbe (final / initial - 1)."""
    if len(equity_curve) < 2:
        return None
    if equity_curve[0] <= 0:
        return None
    return equity_curve[-1] / equity_curve[0] - 1.0


def cagr(
    equity_curve: Sequence[float],
    *,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float | None:
    """Compound Annual Growth Rate : return total annualisé."""
    if len(equity_curve) < 2 or equity_curve[0] <= 0:
        return None
    n_periods = len(equity_curve) - 1
    if n_periods <= 0:
        return None
    final_over_initial = equity_curve[-1] / equity_curve[0]
    if final_over_initial <= 0:
        return None
    return final_over_initial ** (periods_per_year / n_periods) - 1.0


def daily_returns_from_equity(equity_curve: Sequence[float]) -> list[float]:
    """Convertit une equity curve en série de daily returns (linéaires).

    Le 1er return est calculé entre `equity[0]` et `equity[1]`. Si une
    valeur est ≤0 (ne devrait pas arriver), on saute la paire pour
    éviter une division par zéro.
    """
    out: list[float] = []
    for prev, curr in zip(equity_curve, equity_curve[1:]):
        if prev <= 0:
            continue
        out.append(curr / prev - 1.0)
    return out
