"""Module thèses — génération et évaluation (Phase 3).

Une thèse est la synthèse narrative + structurée d'une opportunité
détectée par le scoring. Elle est **immuable** : une fois écrite, elle
ne bouge plus (append-only). Les évaluations ultérieures sont des
lignes séparées dans `evaluations`.

Deux sous-modules :
  - `thesis.generator` — produit des thèses à partir des scores courants.
  - `thesis.evaluator` — (étape 2) évalue les thèses aux jalons temporels.
"""
