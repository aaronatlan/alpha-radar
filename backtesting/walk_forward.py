"""Walk-forward validation — Phase 5 étape 3.

Découpe une fenêtre historique en `n_folds` segments train/test
glissants, pour évaluer une configuration sur des périodes
**out-of-sample**. C'est la mitigation principale contre l'overfitting
dans le backtesting (cf. SPEC §10 — risque "Overfitting").

Schéma walk-forward
-------------------
::

    |─── Train fold 1 ────|─ Test 1 ─|
              |─── Train fold 2 ────|─ Test 2 ─|
                       |─── Train fold 3 ────|─ Test 3 ─|

Chaque fold :
  - Sélectionne le meilleur jeu de poids candidat sur la période train
    (par maximisation d'une métrique objective, par défaut le Sharpe).
  - Mesure les performances de ce gagnant sur la période test
    *strictement postérieure*.
  - Aucune information du futur ne fuite vers la décision de poids.

Sélection des poids
-------------------
On évalue chaque jeu candidat avec `PortfolioSimulator.run()` sur la
fenêtre train. Pour le rejouer, le caller doit fournir une fonction
`build_theses(weights, train_start, train_end)` qui retourne les thèses
correspondantes — typiquement via `HistoricalReplay` paramétré avec
le `model_version` de chaque jeu.

Le module reste **agnostique** de la base : il prend les thèses et
appelle `PortfolioSimulator`. Un wrapper plus haut niveau viendra dans
`runner.py` (étape 4) pour automatiser l'orchestration sur la base.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from typing import Callable, Iterable, Sequence

from loguru import logger

from backtesting.portfolio import PortfolioResult, PortfolioSimulator
from memory.database import Thesis


#: Métriques candidates pour la sélection. Doivent être des clés
#: présentes dans `PortfolioResult.metrics`.
SELECTION_METRICS: tuple[str, ...] = (
    "sharpe", "total_return", "cagr", "alpha_vs_benchmark",
)


@dataclass
class FoldResult:
    """Résultat d'un fold walk-forward."""

    fold_index: int
    train_start: datetime
    train_end: datetime
    test_start: datetime
    test_end: datetime
    best_weights: str | None = None
    train_metrics: dict[str, dict[str, float | None]] = field(default_factory=dict)
    test_metrics: dict[str, float | None] = field(default_factory=dict)


@dataclass
class WalkForwardResult:
    """Résumé d'un walk-forward complet."""

    folds: list[FoldResult] = field(default_factory=list)
    selection_metric: str = "sharpe"

    def best_per_fold(self) -> list[tuple[int, str | None]]:
        """Pour chaque fold, le poids gagnant retenu."""
        return [(f.fold_index, f.best_weights) for f in self.folds]

    def average_test_metric(self, name: str) -> float | None:
        """Moyenne d'une métrique test sur tous les folds."""
        values = [
            f.test_metrics.get(name) for f in self.folds
            if f.test_metrics.get(name) is not None
        ]
        return sum(values) / len(values) if values else None  # type: ignore[arg-type]


# --------------------------------------------------------------- splitter


def split_folds(
    *,
    start: datetime,
    end: datetime,
    n_folds: int,
    test_ratio: float = 0.2,
) -> list[tuple[datetime, datetime, datetime, datetime]]:
    """Découpe `[start, end[` en `n_folds` paires (train, test) glissantes.

    Schéma : la fenêtre globale est répartie en `n_folds` segments de
    durée égale ; chaque segment se termine par une fenêtre test
    représentant `test_ratio` de la durée du segment, le reste sert de
    train. Les folds avancent dans le temps — on respecte strictement
    la causalité (test toujours après train du même fold).

    Retourne une liste de tuples `(train_start, train_end, test_start, test_end)`.
    """
    if end <= start:
        raise ValueError("end doit être > start")
    if n_folds <= 0:
        raise ValueError("n_folds doit être strictement positif")
    if not 0.0 < test_ratio < 1.0:
        raise ValueError("test_ratio doit être dans ]0, 1[")

    total_seconds = (end - start).total_seconds()
    fold_seconds = total_seconds / n_folds
    if fold_seconds <= 0:
        raise ValueError("Fenêtre trop courte pour le nombre de folds demandé")

    out: list[tuple[datetime, datetime, datetime, datetime]] = []
    for i in range(n_folds):
        fold_start = start + timedelta(seconds=fold_seconds * i)
        fold_end = start + timedelta(seconds=fold_seconds * (i + 1))
        # Test = derniers `test_ratio` du segment.
        test_seconds = fold_seconds * test_ratio
        test_start = fold_end - timedelta(seconds=test_seconds)
        out.append((fold_start, test_start, test_start, fold_end))
    return out


# --------------------------------------------------------------- runner


def _select_best_weight(
    candidates: Sequence[str],
    metrics_by_weight: dict[str, dict[str, float | None]],
    selection_metric: str,
) -> str | None:
    """Sélectionne le candidat dont la `selection_metric` est maximale.

    `None` est traité comme `-inf` (les configs sans métrique calculable
    sont systématiquement battues par celles qui en ont une). Égalité →
    premier candidat dans `candidates`.
    """
    best: tuple[float, str] | None = None
    for w in candidates:
        value = metrics_by_weight.get(w, {}).get(selection_metric)
        score = float(value) if value is not None else float("-inf")
        if best is None or score > best[0]:
            best = (score, w)
    return best[1] if best is not None else None


def run_walk_forward(
    *,
    start: datetime,
    end: datetime,
    n_folds: int,
    weight_candidates: Sequence[str],
    build_theses: Callable[[str, datetime, datetime], Iterable[Thesis]],
    selection_metric: str = "sharpe",
    test_ratio: float = 0.2,
    portfolio_simulator: PortfolioSimulator | None = None,
) -> WalkForwardResult:
    """Lance le walk-forward.

    Paramètres
    ----------
    start, end
        Fenêtre globale.
    n_folds
        Nombre de plis. ≥1.
    weight_candidates
        Identifiants des jeux de poids à comparer (ex: `["v2_mom_sigqual",
        "v3_mom_sigqual_sent", "v4_sectoral"]`). Chaque candidat est
        exécuté sur la fenêtre **train** de chaque fold.
    build_theses
        Fonction `(weight, start, end) -> Iterable[Thesis]` qui produit
        les thèses correspondant à un jeu de poids sur une période. Le
        caller fournit cette fonction — typiquement en chaînant un
        `HistoricalReplay` paramétré, puis en re-lisant les thèses
        générées. Pour les tests unitaires, on injecte un fake.
    selection_metric
        Métrique à maximiser pour choisir le gagnant par fold. Doit être
        dans `SELECTION_METRICS` (Sharpe par défaut).
    test_ratio
        Fraction du fold dédiée au test (le reste est train).
    portfolio_simulator
        Simulateur à utiliser. Par défaut, instance avec capital 1.0.

    Retourne
    --------
    `WalkForwardResult` avec un `FoldResult` par fold.
    """
    if not weight_candidates:
        raise ValueError("weight_candidates ne peut pas être vide")
    if selection_metric not in SELECTION_METRICS:
        raise ValueError(
            f"selection_metric doit être dans {SELECTION_METRICS}, "
            f"reçu : {selection_metric}"
        )

    sim = portfolio_simulator or PortfolioSimulator()
    splits = split_folds(start=start, end=end, n_folds=n_folds, test_ratio=test_ratio)

    result = WalkForwardResult(selection_metric=selection_metric)
    for fold_idx, (tr_s, tr_e, te_s, te_e) in enumerate(splits):
        logger.info(
            "[walk_forward] Fold {}/{} : train [{}, {}] test [{}, {}]",
            fold_idx + 1, n_folds, tr_s, tr_e, te_s, te_e,
        )
        fold = FoldResult(
            fold_index=fold_idx,
            train_start=tr_s,
            train_end=tr_e,
            test_start=te_s,
            test_end=te_e,
        )

        # 1. Évaluer chaque jeu de poids sur le train.
        train_metrics: dict[str, dict[str, float | None]] = {}
        for weight in weight_candidates:
            try:
                theses = list(build_theses(weight, tr_s, tr_e))
                res = sim.run(start=tr_s, end=tr_e, theses=theses)
                train_metrics[weight] = res.metrics
            except Exception as exc:
                logger.warning(
                    "[walk_forward] fold {} train {} a échoué : {}",
                    fold_idx, weight, exc,
                )
                train_metrics[weight] = {}
        fold.train_metrics = train_metrics

        # 2. Sélection du gagnant.
        winner = _select_best_weight(
            weight_candidates, train_metrics, selection_metric,
        )
        fold.best_weights = winner

        # 3. Évaluer le gagnant sur le test (out-of-sample).
        if winner is not None:
            try:
                test_theses = list(build_theses(winner, te_s, te_e))
                test_res = sim.run(start=te_s, end=te_e, theses=test_theses)
                fold.test_metrics = test_res.metrics
            except Exception as exc:
                logger.warning(
                    "[walk_forward] fold {} test {} a échoué : {}",
                    fold_idx, winner, exc,
                )

        result.folds.append(fold)

    logger.info(
        "[walk_forward] Terminé : {} fold(s) ; gagnants = {}",
        len(result.folds),
        [(f.fold_index, f.best_weights) for f in result.folds],
    )
    return result
