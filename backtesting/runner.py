"""Orchestrateur CLI du backtesting — Phase 5 étape 4.

Câble bout-en-bout :

    PortfolioSimulator + HistoricalReplay + walk_forward
        sur des thèses persistées dans la base.

Trois modes
-----------
1. **portfolio** — simule un portefeuille sur les thèses **déjà** en
   base entre `start` et `end`. C'est le mode courant pour évaluer le
   track record d'une production active.

2. **replay** — re-rejoue scoring + thèse generation jour par jour sur
   `[start, end[`, puis simule le portefeuille sur les thèses générées.
   Utile pour tester un nouveau jeu de poids sur l'historique
   disponible.

3. **walk-forward** — lance le walk-forward sur plusieurs candidats de
   poids (grid search). Pour chaque fold, replay sur la fenêtre, puis
   sélection out-of-sample.

Sortie
------
Rapport JSON (stdout par défaut, ou `--output`). Les courbes complètes
sont incluses pour permettre un re-rendering downstream.

Usage
-----
::

    python -m backtesting.runner portfolio --start 2026-01-01 --end 2026-04-01
    python -m backtesting.runner walk-forward --start 2024-01-01 --end 2026-01-01 \\
        --folds 4 --weights v2_mom_sigqual,v3_mom_sigqual_sent,v4_sectoral
"""
from __future__ import annotations

import argparse
import json
import sys
from dataclasses import asdict, is_dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable

from loguru import logger

from backtesting.portfolio import PortfolioResult, PortfolioSimulator
from backtesting.replay import HistoricalReplay
from backtesting.walk_forward import (
    SELECTION_METRICS,
    WalkForwardResult,
    run_walk_forward,
)
from config.settings import configure_logging
from memory.database import Thesis, init_db, session_scope
from scoring.sector_heat import SectorHeatScorer
from scoring.stock_scorer import StockScorer
from thesis.generator import ThesisGenerator


# ----------------------------------------------------------------- helpers


def _parse_date(value: str) -> datetime:
    """Parse YYYY-MM-DD ou YYYY-MM-DDTHH:MM:SS."""
    for fmt in ("%Y-%m-%dT%H:%M:%S", "%Y-%m-%d"):
        try:
            return datetime.strptime(value, fmt)
        except ValueError:
            continue
    raise argparse.ArgumentTypeError(
        f"Date invalide : {value!r}. Format attendu : YYYY-MM-DD."
    )


def _result_to_dict(obj: Any) -> Any:
    """Sérialise dataclasses + datetimes pour le JSON."""
    if is_dataclass(obj) and not isinstance(obj, type):
        return _result_to_dict(asdict(obj))
    if isinstance(obj, dict):
        return {k: _result_to_dict(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_result_to_dict(v) for v in obj]
    if isinstance(obj, datetime):
        return obj.isoformat()
    return obj


def _emit(report: dict[str, Any], output: Path | None) -> None:
    """Écrit le rapport en JSON sur stdout ou dans un fichier."""
    text = json.dumps(report, indent=2, default=str, ensure_ascii=False)
    if output is None:
        sys.stdout.write(text + "\n")
    else:
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text(text, encoding="utf-8")
        logger.info("[runner] Rapport écrit dans {}", output)


# ----------------------------------------------------------- mode: portfolio


def run_portfolio_mode(
    *,
    start: datetime,
    end: datetime,
    benchmark_tickers: list[str] | None = None,
    initial_capital: float = 1.0,
) -> dict[str, Any]:
    """Simule un portefeuille sur les thèses persistées entre start/end."""
    sim = PortfolioSimulator(
        initial_capital=initial_capital,
        benchmark_tickers=benchmark_tickers or [],
    )
    res = sim.run(start=start, end=end)
    return {
        "mode": "portfolio",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "benchmark_tickers": list(benchmark_tickers or []),
        "result": _result_to_dict(res),
    }


# ----------------------------------------------------------- mode: replay


def _build_default_scorers() -> list:
    """Pile par défaut : sector_heat puis stock_score (ordre important)."""
    return [SectorHeatScorer(), StockScorer()]


def run_replay_mode(
    *,
    start: datetime,
    end: datetime,
    step_days: int = 1,
    benchmark_tickers: list[str] | None = None,
    initial_capital: float = 1.0,
) -> dict[str, Any]:
    """Re-joue scoring + thèses sur la fenêtre, puis simule le portfolio."""
    replay = HistoricalReplay(
        scorers=_build_default_scorers(),
        thesis_generator=ThesisGenerator(),
    )
    replay_res = replay.run(start=start, end=end, step_days=step_days)

    sim = PortfolioSimulator(
        initial_capital=initial_capital,
        benchmark_tickers=benchmark_tickers or [],
    )
    portfolio_res = sim.run(start=start, end=end)
    return {
        "mode": "replay",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "step_days": step_days,
        "replay": _result_to_dict(replay_res),
        "portfolio": _result_to_dict(portfolio_res),
    }


# ----------------------------------------------------- mode: walk-forward


def _theses_in_window(
    weight: str, start: datetime, end: datetime,
) -> Iterable[Thesis]:
    """Lit en base les thèses générées avec ce `model_version` dans la fenêtre."""
    from sqlalchemy import select
    stmt = (
        select(Thesis)
        .where(Thesis.created_at >= start)
        .where(Thesis.created_at <= end)
        .where(Thesis.model_version == weight)
        .order_by(Thesis.created_at.asc())
    )
    with session_scope() as session:
        rows = list(session.execute(stmt).scalars().all())
        for r in rows:
            session.expunge(r)
    return rows


def run_walk_forward_mode(
    *,
    start: datetime,
    end: datetime,
    n_folds: int,
    weight_candidates: list[str],
    selection_metric: str = "sharpe",
    test_ratio: float = 0.2,
    benchmark_tickers: list[str] | None = None,
) -> dict[str, Any]:
    """Walk-forward sur la base : suppose que les thèses des candidats existent."""
    sim = PortfolioSimulator(benchmark_tickers=benchmark_tickers or [])
    wf_res: WalkForwardResult = run_walk_forward(
        start=start,
        end=end,
        n_folds=n_folds,
        weight_candidates=weight_candidates,
        build_theses=_theses_in_window,
        selection_metric=selection_metric,
        test_ratio=test_ratio,
        portfolio_simulator=sim,
    )
    return {
        "mode": "walk_forward",
        "start": start.isoformat(),
        "end": end.isoformat(),
        "n_folds": n_folds,
        "weight_candidates": list(weight_candidates),
        "selection_metric": selection_metric,
        "result": _result_to_dict(wf_res),
        "summary": {
            "best_per_fold": wf_res.best_per_fold(),
            "avg_test_sharpe": wf_res.average_test_metric("sharpe"),
            "avg_test_total_return": wf_res.average_test_metric("total_return"),
        },
    }


# --------------------------------------------------------------- CLI


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="python -m backtesting.runner",
        description="Backtest Alpha Radar (Phase 5).",
    )
    sub = p.add_subparsers(dest="mode", required=True)

    common = argparse.ArgumentParser(add_help=False)
    common.add_argument("--start", type=_parse_date, required=True,
                         help="Date de début (YYYY-MM-DD).")
    common.add_argument("--end", type=_parse_date, required=True,
                         help="Date de fin (exclue).")
    common.add_argument("--benchmark", default="",
                         help="Tickers benchmark, virgule (ex: 'SPY,QQQ').")
    common.add_argument("--initial-capital", type=float, default=1.0)
    common.add_argument("--output", type=Path, default=None,
                         help="Fichier JSON de sortie. stdout si omis.")

    p_portfolio = sub.add_parser(
        "portfolio", parents=[common],
        help="Simuler un portfolio sur les thèses persistées.",
    )

    p_replay = sub.add_parser(
        "replay", parents=[common],
        help="Re-jouer scoring+thèses puis simuler.",
    )
    p_replay.add_argument("--step-days", type=int, default=1)

    p_wf = sub.add_parser(
        "walk-forward", parents=[common],
        help="Walk-forward avec grid search sur des poids candidats.",
    )
    p_wf.add_argument("--folds", type=int, required=True,
                       help="Nombre de folds train/test glissants.")
    p_wf.add_argument("--weights", required=True,
                       help="Candidats de poids (model_version), virgule.")
    p_wf.add_argument("--selection-metric", default="sharpe",
                       choices=list(SELECTION_METRICS))
    p_wf.add_argument("--test-ratio", type=float, default=0.2)
    return p


def main(argv: list[str] | None = None) -> int:
    """Point d'entrée CLI."""
    args = build_parser().parse_args(argv)
    configure_logging()
    init_db()

    benchmark = [t.strip() for t in args.benchmark.split(",") if t.strip()]
    if args.mode == "portfolio":
        report = run_portfolio_mode(
            start=args.start, end=args.end,
            benchmark_tickers=benchmark,
            initial_capital=args.initial_capital,
        )
    elif args.mode == "replay":
        report = run_replay_mode(
            start=args.start, end=args.end,
            step_days=args.step_days,
            benchmark_tickers=benchmark,
            initial_capital=args.initial_capital,
        )
    elif args.mode == "walk-forward":
        weights = [w.strip() for w in args.weights.split(",") if w.strip()]
        if not weights:
            raise SystemExit("--weights ne peut pas être vide")
        report = run_walk_forward_mode(
            start=args.start, end=args.end,
            n_folds=args.folds,
            weight_candidates=weights,
            selection_metric=args.selection_metric,
            test_ratio=args.test_ratio,
            benchmark_tickers=benchmark,
        )
    else:   # pragma: no cover — argparse interdit
        raise SystemExit(f"Mode inconnu : {args.mode}")

    _emit(report, args.output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
