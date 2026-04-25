"""Moteur d'alertes — Phase 3 étape 4.

Convertit les événements observables (nouvelle thèse, verdict d'évaluation,
flambée du Heat Score sectoriel) en lignes `alerts` et notifie par email
si SMTP est configuré. Cf. SPEC §7.7.

API publique :
    from alerts.engine import AlertsEngine
    AlertsEngine().run()
"""
