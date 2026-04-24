# scheduler/

Orchestration des tâches périodiques avec APScheduler.

## Phase 1

Deux jobs récurrents :

| Job         | Fréquence                  | Fenêtre collectée       |
|-------------|----------------------------|-------------------------|
| arXiv       | quotidien, 06:00 Europe/Paris | 48 h glissantes        |
| yfinance    | mon–fri, 22:00 Europe/Paris  | 5 jours glissants      |

La fenêtre glissante crée un chevauchement volontaire d'exécution à
exécution : la déduplication `(source, entity_id, hash)` dans `raw_data`
garantit l'idempotence, donc le chevauchement n'introduit jamais de
doublons et sert de filet de sécurité contre les trous (scheduler arrêté
quelques heures, API indisponible…).

## Lancement

```bash
python -m scheduler.jobs
```

Le scheduler initialise la base si nécessaire, enregistre les jobs et
bloque. `Ctrl+C` arrête proprement.
