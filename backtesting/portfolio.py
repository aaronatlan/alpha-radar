"""Simulateur de portefeuille théorique — Phase 5 étape 1.

Transforme une liste de thèses en `PortfolioResult` :
  - equity curve quotidienne (capital initial = 1.0)
  - métriques agrégées (Sharpe, max drawdown, hit rate, alpha vs benchmark)

Modèle simplifié v1
-------------------
- À la date `created_at` de chaque thèse BUY/WATCH, on alloue une
  position de poids égal (1/N entre toutes les positions ouvertes).
- La position se ferme automatiquement à `created_at + horizon_days`.
- Les thèses `AVOID` sont ignorées (pas de short en v1).
- Pour le P&L journalier d'une position, on lit le `close` PIT du
  ticker (yfinance ohlcv_daily) jour par jour.

Frais & slippage
----------------
À l'ouverture de chaque position, on déduit du capital une charge
round-trip = ``weight × 2 × (fee_bps + slippage_bps) / 1e4`` où
``weight = 1 / N_active`` au moment de l'ouverture. Cette approximation
modélise commissions et impact de marché à l'entrée + sortie comme un
coût ponctuel — sans frais, le Sharpe et le total return sont
sur-évalués de plusieurs centaines de bps sur un an.

Defaults : ``fee_bps=0`` (retail commission-free), ``slippage_bps=5``
par côté (≈ 10 bps round-trip, conservateur sur small/mid-caps liquides).
Les modes 'replay' / 'walk-forward' héritent de ces mêmes coûts.

Benchmark
---------
Optionnel. Si fourni (liste de tickers, ex: `["SPY"]`), la simulation
maintient en parallèle un portefeuille buy-and-hold equally-weighted
sur ces tickers, pour le calcul d'alpha.

PIT discipline
--------------
On lit les closes via `latest_close_at` qui filtre `content_at <= date`
ET `fetched_at <= date`. Aucune information du futur ne fuite.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Iterable, Sequence

from loguru import logger
from sqlalchemy import select

from backtesting import metrics as M
from memory.database import Thesis, session_scope
from thesis._io import latest_close_at


@dataclass
class Position:
    """Position théorique ouverte au sein du portefeuille."""

    ticker: str
    open_date: datetime
    close_date: datetime          # date de fermeture programmée
    open_price: float
    weight: float = 0.0           # fraction du portefeuille à l'ouverture
    last_price: float = 0.0       # dernier close lu pour mark-to-market

    def is_active(self, on: datetime) -> bool:
        return self.open_date <= on < self.close_date


@dataclass
class PortfolioResult:
    """Sortie du simulateur."""

    dates: list[datetime] = field(default_factory=list)
    equity_curve: list[float] = field(default_factory=list)
    benchmark_curve: list[float] = field(default_factory=list)
    positions_taken: int = 0
    metrics: dict[str, float | None] = field(default_factory=dict)

    @property
    def initial_capital(self) -> float:
        return self.equity_curve[0] if self.equity_curve else 0.0

    @property
    def final_capital(self) -> float:
        return self.equity_curve[-1] if self.equity_curve else 0.0


class PortfolioSimulator:
    """Simule un portefeuille équipondéré sur les recommandations BUY/WATCH.

    Paramètres
    ----------
    initial_capital
        Valeur de départ du portefeuille (par défaut 1.0 — l'equity curve
        est alors directement interprétable en fraction).
    accepted_recommendations
        Quelles recommandations ouvrent une position. Par défaut BUY+WATCH.
    benchmark_tickers
        Liste de tickers de benchmark (équipondéré). Si vide, l'alpha
        est laissé à `None`.
    """

    DEFAULT_RECOMMENDATIONS: tuple[str, ...] = ("BUY", "WATCH")

    #: Commission par côté en basis points (1 bp = 0.01%). Default retail
    #: commission-free post-2019. Mettre 5–10 pour brokers traditionnels.
    DEFAULT_FEE_BPS: float = 0.0

    #: Slippage par côté en basis points. 5 bps ≈ 0.05% — couvre le coût
    #: bid-ask + impact pour des petits notionnels sur tickers liquides.
    DEFAULT_SLIPPAGE_BPS: float = 5.0

    def __init__(
        self,
        *,
        initial_capital: float = 1.0,
        accepted_recommendations: Sequence[str] = DEFAULT_RECOMMENDATIONS,
        benchmark_tickers: Sequence[str] | None = None,
        fee_bps: float = DEFAULT_FEE_BPS,
        slippage_bps: float = DEFAULT_SLIPPAGE_BPS,
    ) -> None:
        if initial_capital <= 0:
            raise ValueError("initial_capital doit être strictement positif")
        if not accepted_recommendations:
            raise ValueError("accepted_recommendations ne peut pas être vide")
        if fee_bps < 0 or slippage_bps < 0:
            raise ValueError("fee_bps et slippage_bps doivent être ≥ 0")
        self._initial_capital = float(initial_capital)
        self._accepted = tuple(accepted_recommendations)
        self._benchmark = tuple(benchmark_tickers or [])
        # Conversion bps → fraction. `_cost_per_side` agrège commission
        # et slippage car ils s'appliquent aux mêmes moments (entrée/sortie).
        self._cost_per_side = (float(fee_bps) + float(slippage_bps)) / 10_000.0

    # --- API publique ----------------------------------------------------

    def run(
        self,
        *,
        start: datetime,
        end: datetime,
        theses: Iterable[Thesis] | None = None,
    ) -> PortfolioResult:
        """Exécute la simulation entre `start` et `end`.

        Si `theses` est `None`, charge toutes les thèses dont `created_at`
        est dans la fenêtre. Sinon utilise la liste fournie (utile en test).
        """
        if end <= start:
            raise ValueError("end doit être > start")
        thesis_list = self._load_theses(start, end) if theses is None else list(theses)
        thesis_list.sort(key=lambda t: t.created_at)

        # Génère une grille jour-par-jour (calendrier civil — on évalue
        # tous les jours, le close PIT le plus récent fait foi le week-end).
        dates = _daily_grid(start, end)
        if not dates:
            return PortfolioResult(metrics={})

        positions: list[Position] = []
        thesis_idx = 0
        equity = self._initial_capital
        equity_curve: list[float] = []
        n_taken = 0

        for current in dates:
            # 1) Ouverture de toutes les thèses dont created_at == current.
            opened_today: list[Position] = []
            while thesis_idx < len(thesis_list) and \
                    thesis_list[thesis_idx].created_at <= current:
                t = thesis_list[thesis_idx]
                thesis_idx += 1
                if t.recommendation not in self._accepted:
                    continue
                if t.entry_price is None or t.entry_price <= 0:
                    continue
                close_date = t.created_at + timedelta(days=t.horizon_days)
                pos = Position(
                    ticker=t.asset_id,
                    open_date=t.created_at,
                    close_date=close_date,
                    open_price=float(t.entry_price),
                    last_price=float(t.entry_price),
                )
                positions.append(pos)
                opened_today.append(pos)
                n_taken += 1

            # 1b) Frais round-trip à l'ouverture. On calcule N_active
            #     APRÈS ajout des nouvelles positions, donc le poids tient
            #     compte de la dilution. Charge = w × 2 × cost_per_side
            #     (entrée + sortie) appliquée immédiatement.
            if opened_today and self._cost_per_side > 0:
                active_after = [p for p in positions if p.is_active(current)]
                n_active = len(active_after)
                if n_active > 0:
                    weight = 1.0 / n_active
                    drag = weight * 2.0 * self._cost_per_side * len(opened_today)
                    equity *= max(0.0, 1.0 - drag)

            # 2) Mark-to-market : pour chaque position ouverte, lire le
            #    close PIT et calculer la valeur courante.
            active = [p for p in positions if p.is_active(current)]
            if active:
                # Réallocation équipondérée à chaque tick : avec frais nuls
                # ce n'est qu'un effet de comptabilité ; le rendement
                # journalier moyen des positions actives gouverne l'equity.
                weight = 1.0 / len(active)
                day_return = 0.0
                for p in active:
                    price = latest_close_at(p.ticker, current)
                    if price is not None and price > 0:
                        # Variation depuis le dernier close observé.
                        day_return += weight * (price / p.last_price - 1.0)
                        p.last_price = price
                equity *= 1.0 + day_return
            equity_curve.append(equity)

        # 3) Benchmark.
        benchmark_curve = self._benchmark_curve(dates) if self._benchmark else []

        # 4) Métriques.
        result_metrics = self._compute_metrics(equity_curve, benchmark_curve, thesis_list)

        return PortfolioResult(
            dates=list(dates),
            equity_curve=equity_curve,
            benchmark_curve=benchmark_curve,
            positions_taken=n_taken,
            metrics=result_metrics,
        )

    # --- internals -------------------------------------------------------

    def _load_theses(self, start: datetime, end: datetime) -> list[Thesis]:
        stmt = (
            select(Thesis)
            .where(Thesis.created_at >= start)
            .where(Thesis.created_at <= end)
            .order_by(Thesis.created_at.asc())
        )
        with session_scope() as session:
            rows = list(session.execute(stmt).scalars().all())
            for r in rows:
                session.expunge(r)
        return rows

    def _benchmark_curve(self, dates: list[datetime]) -> list[float]:
        """Buy-and-hold équipondéré sur `_benchmark` tickers."""
        if not self._benchmark or not dates:
            return []
        equity = self._initial_capital
        # Initialise les last_price au close à `dates[0]`.
        last_prices: dict[str, float] = {}
        for ticker in self._benchmark:
            p = latest_close_at(ticker, dates[0])
            if p is not None and p > 0:
                last_prices[ticker] = p

        curve: list[float] = []
        if not last_prices:
            # Pas de prix initial → benchmark plat à 1.0.
            return [equity] * len(dates)

        weight = 1.0 / len(last_prices)
        for current in dates:
            day_return = 0.0
            for ticker, last_p in last_prices.items():
                p = latest_close_at(ticker, current)
                if p is not None and p > 0:
                    day_return += weight * (p / last_p - 1.0)
                    last_prices[ticker] = p
            equity *= 1.0 + day_return
            curve.append(equity)
        return curve

    def _compute_metrics(
        self,
        equity_curve: list[float],
        benchmark_curve: list[float],
        theses: list[Thesis],
    ) -> dict[str, float | None]:
        daily_p = M.daily_returns_from_equity(equity_curve)
        out: dict[str, float | None] = {
            "total_return": M.total_return(equity_curve),
            "cagr": M.cagr(equity_curve),
            "sharpe": M.sharpe_ratio(daily_p),
            "max_drawdown": M.max_drawdown(equity_curve),
            "n_theses": float(len(theses)) if theses else 0.0,
        }
        # Hit rate basé sur les thèses : positif si entry_price < dernier
        # close lu (proxy simple). Pour un hit rate « propre », il faut
        # joindre avec `evaluations` — ajout en étape 2.
        out["hit_rate"] = None
        if benchmark_curve and len(benchmark_curve) == len(equity_curve):
            daily_b = M.daily_returns_from_equity(benchmark_curve)
            out["alpha_vs_benchmark"] = M.alpha_vs_benchmark(daily_p, daily_b)
            out["benchmark_total_return"] = M.total_return(benchmark_curve)
        else:
            out["alpha_vs_benchmark"] = None
            out["benchmark_total_return"] = None
        return out


# ------------------------------------------------------------------ helpers


def _daily_grid(start: datetime, end: datetime) -> list[datetime]:
    """Liste des dates jour par jour (incl. start, excl. end)."""
    out: list[datetime] = []
    cursor = start
    while cursor < end:
        out.append(cursor)
        cursor += timedelta(days=1)
    return out
