"""Backtesting — Phase 5.

Évalue rétrospectivement la qualité du système (scoring + génération de
thèses + verdicts) sur des données historiques en respectant la
discipline point-in-time.

Modules
-------
- `metrics`   : Sharpe, max drawdown, hit rate, alpha vs benchmark.
- `portfolio` : `PortfolioSimulator` qui transforme une liste de thèses
                + leurs évaluations en equity curve théorique.
- `replay`    : (étape 2) re-rejoue features + scoring + thèses sur une
                fenêtre historique pour valider une configuration de poids.
- `walk_forward` : (étape 3) folds glissants train/test pour comparer
                des jeux de poids alternatifs.
"""
