# features/

Calcul point-in-time des métriques dérivées des données brutes.

## Principe

Chaque feature est une fonction pure `(target_id, as_of) → float | None`
héritant de `BaseFeature`. La valeur produite ne doit dépendre que des
données disponibles à `as_of` ou avant (discipline PIT). Deux calculs à
la même date produisent la même valeur — contrainte UNIQUE en base.

`BaseFeature.run(as_of)` itère sur `targets()`, appelle `compute()` sur
chacun et écrit dans `features` via `INSERT OR IGNORE`. Une exception
isolée n'interrompt pas le run (mode dégradé).

## Features Phase 2

- `velocity.ArxivVelocityFeature` — ratio taux récent / taux de
  référence des publications arXiv par secteur. `target_type='sector'`.
- (à venir) `technical` — RSI, momentum, volume ratio sur les OHLCV
  yfinance. `target_type='asset'`.
- (à venir) `sentiment` — score FinBERT agrégé sur titres/résumés news.

## Cas d'une valeur indéfinie

Une feature peut renvoyer `None` pour signaler « non calculable à
cette date » (ex. fenêtre vide). Aucune ligne n'est alors écrite —
plutôt qu'une valeur 0 qui serait interprétée comme un signal.
