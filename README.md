# Alpha Radar

Système d'analyse d'investissement multi-sources qui détecte les opportunités
avant que le marché ne les valorise pleinement.

Voir [`SPEC.md`](./SPEC.md) pour le cahier des charges complet (architecture,
schéma DB, plan de développement en 6 phases).

## État du projet

**Phase 1 — Fondations (en cours).** Collecte arXiv + yfinance, stockage
point-in-time, scheduler minimal.

## Installation

```bash
cd alpha-radar
python -m venv .venv
source .venv/bin/activate
pip install -e ".[dev]"
cp .env.example .env
```

## Utilisation

Initialiser la base et lancer le scheduler :

```bash
python -m scheduler.jobs
```

Exécuter les tests :

```bash
pytest
```

## Structure

- `config/` — settings, secteurs suivis, watchlists
- `collectors/` — un module par source de données, héritant de `BaseCollector`
- `memory/` — schéma SQLite et accès
- `scheduler/` — orchestration des jobs périodiques
- `tests/` — tests unitaires pytest
- `data/` — données brutes et caches (gitignore)

## Principes non négociables

1. **Gratuit** — APIs publiques uniquement.
2. **Point-in-time** — chaque donnée porte son timestamp de validité ; aucun
   calcul ne doit utiliser d'information du futur.
3. **Règles d'abord, ML ensuite** — poids experts en dur pour les 6-12 premiers
   mois, ML seulement après labellisation suffisante.
4. **Interprétable** — chaque score est décomposable en ses contributions.
5. **Mémoire immuable** — les thèses ne sont jamais modifiées, les évaluations
   sont ajoutées en append-only.
6. **Mode dégradé** — si une API tombe, le système continue sur les autres.
