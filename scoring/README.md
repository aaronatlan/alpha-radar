# scoring/

Agrégation des features en scores 0-100 interprétables.

## Principe

Un scoreur est une feature composite : il lit des features existantes
dans la table `features`, combine selon un jeu de poids versionné
(`scoring/weights.py`), et stocke le résultat dans la même table
`features` avec un nom parlant (`sector_heat_score`, `stock_score`, …).

Tous les scoreurs héritent de `BaseFeature` pour partager la mécanique
PIT et l'idempotence `INSERT OR IGNORE`.

## Poids

Les poids sont **versionnés** (`model_version`). Toute recalibration
crée une nouvelle entrée dans `weights.py` avec un nouveau nom ; on ne
modifie jamais un set existant. Ce choix garantit qu'on peut toujours
rejouer un score historique avec ses poids d'origine.

## Scoreurs

- `sector_heat.SectorHeatScorer` — Heat Score par secteur (v1 basée
  sur la seule vélocité arXiv, extensions multi-signaux prévues).
- (à venir) `stock_scorer.StockScorer` — score composite par action.
- (à venir) `crypto_scorer.CryptoScorer` — score composite crypto.

## Feature absente vs valeur nulle

Si un input est absent (feature jamais calculée à ou avant `as_of`),
le scoreur renvoie `None` plutôt que de substituer 0. C'est une règle
forte : un "signal muet" n'est pas un "signal négatif".
