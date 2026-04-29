"""Seed démo : peuple la base avec des données réalistes pour le dashboard.

Usage : `python -m scripts.seed_demo`

Génère :
  - OHLCV daily yfinance pour 8 tickers sur 60 jours (random walk)
  - Features (RSI, momentum, volume_ratio, sector_heat_score)
  - Stock scores composites
  - 12 thèses sur 3 mois avec recommandations variées
  - Évaluations à J+30 et J+90
  - 5 alertes (critical, warning, info)
  - Lignes signal_performance

Idempotent : un re-run écrase les données précédentes (safe pour démo).
"""
from __future__ import annotations

import json
import random
from datetime import datetime, timedelta

from memory.database import (
    Alert,
    Evaluation,
    Feature,
    RawData,
    SignalPerformance,
    Thesis,
    init_db,
    session_scope,
)


random.seed(42)

# Période de démo : 90 jours jusqu'à aujourd'hui.
TODAY = datetime.utcnow().replace(hour=12, minute=0, second=0, microsecond=0)
START = TODAY - timedelta(days=90)

DEMO_TICKERS = [
    ("NVDA", "NVIDIA", ["ai_ml", "semiconductors"], 800.0),
    ("AMD", "Advanced Micro Devices", ["ai_ml", "semiconductors"], 160.0),
    ("MRNA", "Moderna", ["biotech"], 90.0),
    ("CRSP", "CRISPR Therapeutics", ["biotech"], 60.0),
    ("LMT", "Lockheed Martin", ["space", "defense"], 450.0),
    ("RKLB", "Rocket Lab", ["space"], 14.0),
    ("CRWD", "CrowdStrike", ["cybersecurity"], 320.0),
    ("PLTR", "Palantir", ["ai_ml", "cybersecurity", "defense"], 25.0),
]

DEMO_SECTORS = ["ai_ml", "biotech", "cybersecurity", "defense",
                 "quantum_computing", "robotics", "semiconductors", "space"]


def reset_tables() -> None:
    """Vide les tables qu'on va re-peupler."""
    with session_scope() as s:
        s.query(Alert).delete()
        s.query(Evaluation).delete()
        s.query(SignalPerformance).delete()
        s.query(Thesis).delete()
        s.query(Feature).delete()
        s.query(RawData).delete()


def seed_yfinance_closes() -> None:
    """Random walk sur 90 jours pour chaque ticker."""
    rows = []
    for ticker, _, _, base in DEMO_TICKERS:
        price = base
        for d in range(90):
            current = START + timedelta(days=d)
            # Drift positif moyen + bruit.
            drift = random.gauss(0.001, 0.022)
            price = price * (1 + drift)
            payload = {
                "ticker": ticker,
                "session_date": current.strftime("%Y-%m-%d"),
                "close": round(price, 2),
                "open": round(price * (1 + random.gauss(0, 0.005)), 2),
                "high": round(price * (1 + abs(random.gauss(0, 0.01))), 2),
                "low": round(price * (1 - abs(random.gauss(0, 0.01))), 2),
                "volume": int(random.uniform(1e6, 5e7)),
            }
            rows.append(RawData(
                source="yfinance", entity_type="ohlcv_daily",
                entity_id=f"{ticker}:{current.strftime('%Y-%m-%d')}",
                fetched_at=current,
                content_at=current,
                payload_json=json.dumps(payload),
                hash=f"h-{ticker}-{current.strftime('%Y-%m-%d')}",
            ))
    with session_scope() as s:
        for r in rows:
            s.add(r)
    print(f"  raw_data yfinance : {len(rows)}")


def seed_features() -> None:
    """RSI / momentum / volume_ratio par ticker, sector_heat_score par secteur."""
    rows = []
    for ticker, _, sectors, _ in DEMO_TICKERS:
        for d in range(60, 90):
            ts = START + timedelta(days=d)
            rsi = 30 + random.random() * 50
            mom = random.gauss(0.05, 0.10)
            vol = 0.8 + random.random() * 1.5
            for name, val in [
                ("rsi_14", rsi),
                ("momentum_30d", mom),
                ("volume_ratio_7_30", vol),
            ]:
                rows.append(Feature(
                    feature_name=name, target_type="asset",
                    target_id=ticker, computed_at=ts, value=float(val),
                ))

    # Sector Heat Scores : plus chaud sur AI / biotech.
    base_heat = {
        "ai_ml": 75, "biotech": 60, "cybersecurity": 55, "defense": 50,
        "quantum_computing": 45, "robotics": 40, "semiconductors": 70, "space": 50,
    }
    for sid in DEMO_SECTORS:
        for d in range(60, 90):
            ts = START + timedelta(days=d)
            val = base_heat[sid] + random.gauss(0, 5)
            val = max(0, min(100, val))
            rows.append(Feature(
                feature_name="sector_heat_score", target_type="sector",
                target_id=sid, computed_at=ts, value=float(val),
                metadata_json=json.dumps({
                    "model_version": "v2_arxiv_github",
                    "weights": {"arxiv_velocity": 0.6, "github_stars_velocity": 0.4},
                }),
            ))

    # News sentiment sectoriel.
    for sid in DEMO_SECTORS:
        for d in range(70, 90):
            ts = START + timedelta(days=d)
            rows.append(Feature(
                feature_name="news_sentiment_sector", target_type="sector",
                target_id=sid, computed_at=ts,
                value=random.gauss(0.1, 0.25),
            ))

    # Stock scores composites.
    for ticker, _, sectors, _ in DEMO_TICKERS:
        for d in range(75, 90):
            ts = START + timedelta(days=d)
            score = max(0, min(100, random.gauss(70, 12)))
            dims = {
                "momentum": max(0, min(100, random.gauss(score, 10))),
                "signal_quality": max(0, min(100, random.gauss(score, 8))),
                "sentiment": max(0, min(100, random.gauss(score, 10))),
            }
            rows.append(Feature(
                feature_name="stock_score", target_type="asset",
                target_id=ticker, computed_at=ts, value=float(score),
                metadata_json=json.dumps({
                    "model_version": "v3_mom_sigqual_sent",
                    "weights": {"momentum": 0.4, "signal_quality": 0.3, "sentiment": 0.3},
                    "dimensions": dims,
                    "details": {},
                }),
            ))

    with session_scope() as s:
        for r in rows:
            s.add(r)
    print(f"  features          : {len(rows)}")


def seed_theses_and_evals() -> None:
    """12 thèses sur 3 mois avec évaluations."""
    rows_thesis = []
    rows_eval = []
    recommendations = ["BUY", "BUY", "WATCH", "BUY", "WATCH", "BUY",
                        "AVOID", "BUY", "WATCH", "BUY", "BUY", "WATCH"]
    for i, (ticker, name, sectors, base_price) in enumerate(
        DEMO_TICKERS + DEMO_TICKERS[:4]
    ):
        created = START + timedelta(days=int(i * 5.5))
        score = max(60, min(95, random.gauss(78, 8)))
        reco = recommendations[i]
        dims = {
            "momentum": random.gauss(score, 6),
            "signal_quality": random.gauss(score, 6),
            "sentiment": random.gauss(score, 6),
        }
        breakdown = {
            "dimensions": {k: float(max(0, min(100, v))) for k, v in dims.items()},
            "details": {},
        }
        # Catalyseurs sectoriels selon le secteur principal.
        catalysts: list[dict] = []
        if "biotech" in sectors:
            catalysts.append({
                "type": "phase3_trial", "phase": "PHASE3", "drug": f"Drug-{i}",
                "description": f"PHASE3 en cours sur Drug-{i} (NCT0{1000 + i})",
                "source": "clinicaltrials",
            })
        if "space" in sectors or "defense" in sectors:
            catalysts.append({
                "type": "gov_contract", "amount_usd": 250_000_000.0,
                "agency": "Department of Defense",
                "description": f"Contrat $250.0M de DoD ({created.date()})",
                "source": "usaspending",
            })

        narrative = (
            f"**{name} ({ticker}) — Score {score:.1f}/100**\n\n"
            f"**Pourquoi maintenant**\nLe score composite atteint "
            f"{score:.1f}/100. Secteur(s) : {', '.join(sectors)}.\n\n"
            f"**Score**\n  - momentum : {dims['momentum']:.1f}/100\n  - "
            f"signal_quality : {dims['signal_quality']:.1f}/100\n  - "
            f"sentiment : {dims['sentiment']:.1f}/100\n\n"
            f"**Catalyseurs**\n" +
            ("\n".join(f"  - {c['description']}" for c in catalysts)
             if catalysts else "  - Pas de catalyseur sectoriel.")
            + f"\n  - Horizon : 180 jours.\n\n"
            f"**Risques**\n  - [macro] Sensibilité aux taux réels.\n"
            f"**Entrée**\nDernier close : {base_price:.2f}."
        )

        thesis = Thesis(
            created_at=created,
            asset_type="stock", asset_id=ticker, sector_id=sectors[0],
            score=float(score),
            score_breakdown_json=json.dumps(breakdown),
            recommendation=reco,
            horizon_days=180,
            entry_price=base_price,
            entry_conditions_json=json.dumps({"band_pct": 0.02,
                                                "reference_close": base_price}),
            triggers_json=json.dumps([
                {"dimension": k, "sub_score": v}
                for k, v in breakdown["dimensions"].items()
            ]),
            risks_json=json.dumps([
                {"category": "macro",
                 "description": "Sensibilité aux taux réels."},
            ]),
            catalysts_json=json.dumps(catalysts),
            narrative=narrative,
            model_version="v3_mom_sigqual_sent",
            weights_snapshot_json=json.dumps({
                "momentum": 0.4, "signal_quality": 0.3, "sentiment": 0.3,
            }),
        )
        rows_thesis.append(thesis)

    with session_scope() as s:
        for t in rows_thesis:
            s.add(t)
        s.flush()
        thesis_ids = [t.id for t in rows_thesis]

    # Évaluations : J+30 (active) puis J+90 (encore active sauf qq cas terminaux
    # pour rendre le post-mortem visible dans le dashboard).
    n_eval = 0
    for tid, t in zip(thesis_ids, rows_thesis):
        for days, status_pool in [(30, ["active"]),
                                    (90, ["active", "active", "success", "failure", "partial"])]:
            t_eval = t.created_at + timedelta(days=days)
            if t_eval > TODAY:
                continue
            status = random.choice(status_pool)
            return_pct = random.gauss(0.08 if status == "success" else
                                       -0.08 if status == "failure" else 0.02, 0.05)
            bench_pct = random.gauss(0.04, 0.02)
            alpha = return_pct - bench_pct
            current_price = t.entry_price * (1 + return_pct)
            with session_scope() as s:
                s.add(Evaluation(
                    thesis_id=tid,
                    evaluated_at=t_eval,
                    days_since_thesis=days,
                    current_price=float(current_price),
                    return_pct=float(return_pct),
                    benchmark_return_pct=float(bench_pct),
                    alpha_pct=float(alpha),
                    status=status,
                    notes=None,
                ))
                n_eval += 1

    print(f"  theses            : {len(rows_thesis)}")
    print(f"  evaluations       : {n_eval}")


def seed_alerts() -> None:
    """5 alertes représentatives (info / warning / critical)."""
    alerts = [
        {
            "rule_name": "new_thesis", "severity": "info",
            "asset_id": "NVDA", "sector_id": "ai_ml",
            "thesis_id": 1,
            "message": "Nouvelle thèse BUY sur NVDA (ai_ml) — score 86.5/100, horizon 180j.",
            "created_at": TODAY - timedelta(days=2),
            "acknowledged": False,
            "data": {"dedupe_key": "thesis:1", "score": 86.5,
                      "recommendation": "BUY"},
        },
        {
            "rule_name": "sector_heat_surge", "severity": "critical",
            "asset_id": None, "sector_id": "ai_ml", "thesis_id": None,
            "message": "Heat Score ai_ml : +25.0 pts en 48h (de 60.0 à 85.0).",
            "created_at": TODAY - timedelta(days=4),
            "acknowledged": True,
            "data": {"dedupe_key": "heat_surge:ai_ml:" + (TODAY - timedelta(days=4)).date().isoformat(),
                      "current": 85.0, "previous": 60.0, "delta": 25.0,
                      "window_hours": 48},
        },
        {
            "rule_name": "large_gov_contract", "severity": "warning",
            "asset_id": "LMT", "sector_id": None, "thesis_id": None,
            "message": "Contrat gouv US : Lockheed Martin reçoit $250.0M de "
                        "Department of Defense — F-35 lot procurement (2026-04-15).",
            "created_at": TODAY - timedelta(days=10),
            "acknowledged": False,
            "data": {"dedupe_key": "contract:FA8625-21-C-0001",
                      "amount_usd": 250000000.0, "agency": "DoD"},
        },
        {
            "rule_name": "eval_verdict", "severity": "info",
            "asset_id": "MRNA", "sector_id": "biotech", "thesis_id": 3,
            "message": "Thèse #3 (MRNA) : verdict **success** à J+90j (alpha +6.8%).",
            "created_at": TODAY - timedelta(days=1),
            "acknowledged": False,
            "data": {"dedupe_key": "eval:3", "status": "success",
                      "alpha_pct": 0.068},
        },
        {
            "rule_name": "form_13d", "severity": "critical",
            "asset_id": "PLTR", "sector_id": None, "thesis_id": None,
            "message": "Form SC 13D sur Palantir (PLTR) déposé le 2026-04-22. "
                        "Franchissement >5% du capital — signal smart money.",
            "created_at": TODAY - timedelta(hours=18),
            "acknowledged": False,
            "data": {"dedupe_key": "13d:0001-26-001234",
                      "form": "SC 13D"},
        },
    ]
    with session_scope() as s:
        for a in alerts:
            s.add(Alert(
                created_at=a["created_at"],
                rule_name=a["rule_name"],
                severity=a["severity"],
                asset_type="stock" if a["asset_id"] else None,
                asset_id=a["asset_id"],
                sector_id=a["sector_id"],
                message=a["message"],
                data_json=json.dumps(a["data"]),
                thesis_id=a["thesis_id"],
                acknowledged=a["acknowledged"],
            ))
    print(f"  alerts            : {len(alerts)}")


def seed_signal_performance() -> None:
    """Quelques lignes pour la page Performance."""
    rows = []
    for signal in ["momentum", "signal_quality", "sentiment"]:
        for sector_id in [None, "ai_ml", "biotech"]:
            n_pred = random.randint(8, 25)
            n_succ = random.randint(0, n_pred)
            rows.append(SignalPerformance(
                signal_name=signal,
                sector_id=sector_id,
                horizon_days=180,
                n_predictions=n_pred,
                n_successes=n_succ,
                accuracy=n_succ / n_pred,
                avg_alpha=random.gauss(0.03, 0.05),
                last_updated=TODAY,
            ))
    with session_scope() as s:
        for r in rows:
            s.add(r)
    print(f"  signal_performance: {len(rows)}")


def main() -> None:
    init_db()
    print("[seed_demo] Reset tables…")
    reset_tables()
    print("[seed_demo] Peuplement…")
    seed_yfinance_closes()
    seed_features()
    seed_theses_and_evals()
    seed_alerts()
    seed_signal_performance()
    print("[seed_demo] Fait.")


if __name__ == "__main__":
    main()
