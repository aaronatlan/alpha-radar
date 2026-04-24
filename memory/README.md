# memory/

Schéma SQLite et accès à la mémoire centrale du système.

## Tables (section 5 du cahier des charges)

- `raw_data` — données brutes horodatées, append-only, dédupliquées par
  `(source, entity_id, hash)`.
- `features` — features calculées point-in-time, indexées sur
  `(target, computed_at)`.
- `sectors` — définition des secteurs suivis.
- `theses` — prédictions du système, **immuables** une fois créées.
- `evaluations` — évaluation périodique des thèses aux jalons
  (30/90/180/365/540 j), append-only.
- `signal_performance` — track record par signal / secteur / horizon.
- `alerts` — alertes déclenchées.

## Principes

Les données brutes ne sont jamais modifiées ni supprimées. Les features
sont toujours recalculables depuis `raw_data`. Les thèses et évaluations
sont append-only — on n'écrit jamais l'histoire après coup.

Phase 1 : seules `raw_data` et le reste du schéma sont mis en place.
Les modules de lecture spécialisés (`theses.py`, `evaluations.py`,
`performance.py`) arriveront en Phase 3.
