"""Générateur de thèses — Phase 3 étape 1.

Lit la dernière `stock_score` PIT de chaque ticker de la watchlist et,
si le score dépasse un seuil (configurable par secteur), crée une
`Thesis` immuable avec :

- une recommandation (BUY / WATCH / AVOID),
- les signaux déclencheurs (dimensions + sous-scores),
- une narrative templatée en 5 sections (cf. SPEC §7.4),
- des risques structurés (génériques ici, spécialisables en Phase 4),
- une zone d'entrée basée sur le dernier close OHLCV.

Idempotence
-----------
Une seule thèse par (asset_type, asset_id) par jour UTC : si une thèse
existe déjà pour la date d'aujourd'hui (UTC), on ne la recrée pas. On
évite ainsi la prolifération en cas de re-run du job sur la même
journée, sans avoir à contraindre la table (qui reste append-only pure).

Reproductibilité
----------------
`model_version` et `weights_snapshot_json` sont **recopiés** depuis la
feature `stock_score` source — permet de rejouer la décision avec les
poids exacts de l'époque, même si `scoring.weights` évolue plus tard.
"""
from __future__ import annotations

import json
from datetime import date, datetime, timedelta, time
from typing import Any

from loguru import logger
from sqlalchemy import select

from config.pdufa_calendar import upcoming_pdufas
from config.watchlists import STOCK_WATCHLIST, WATCHLIST_TICKERS
from memory.database import Feature, RawData, Thesis, session_scope, utc_now
from thesis._io import latest_close_at as _latest_entry_price


# Seuils par secteur — scores composites [0, 100]. `_default` s'applique
# à tout secteur non listé. Valeurs à affiner au fil des observations
# (Phase 6 pourra les apprendre). Un score ≥ seuil génère une thèse.
THESIS_SCORE_THRESHOLDS: dict[str, float] = {
    "_default": 70.0,
    "ai_ml": 75.0,          # secteur sur-suivi → barrière plus haute
    "quantum_computing": 65.0,  # bruit plus élevé, seuil plus permissif
}

# Horizon d'investissement par défaut (jours). SPEC §7.5 évalue aux
# jalons 30/90/180/365/540 ; 180 est le milieu, compatible avec le
# classement success/failure/partial.
DEFAULT_HORIZON_DAYS = 180

# Zone d'entrée : ±2% autour du dernier close, sauf override.
DEFAULT_ENTRY_BAND_PCT = 0.02


# ----------------------------------------------------------- narrative


def _make_recommendation(score: float) -> str:
    if score >= 75.0:
        return "BUY"
    if score >= 60.0:
        return "WATCH"
    return "AVOID"


def _dominant_dimension(dimensions: dict[str, float]) -> tuple[str, float] | None:
    """Dimension au plus haut sub-score — moteur principal de la thèse."""
    if not dimensions:
        return None
    name, val = max(dimensions.items(), key=lambda kv: kv[1])
    return name, float(val)


def _make_triggers(
    dimensions: dict[str, float], details: dict[str, Any]
) -> list[dict[str, Any]]:
    """Signaux déclencheurs : dimensions triées décroissant + inputs clés."""
    out: list[dict[str, Any]] = []
    for dim, score in sorted(dimensions.items(), key=lambda kv: -kv[1]):
        trig: dict[str, Any] = {"dimension": dim, "sub_score": float(score)}
        dim_detail = details.get(dim, {}) if isinstance(details, dict) else {}
        inputs = dim_detail.get("inputs") if isinstance(dim_detail, dict) else None
        if isinstance(inputs, dict):
            trig["inputs"] = {k: v for k, v in inputs.items() if v is not None}
        out.append(trig)
    return out


# Risques par secteur — liste extensible. On prend l'union des listes
# pertinentes pour les secteurs de l'action, plus des risques de base
# applicables à tout titre coté.
_SECTOR_RISKS: dict[str, list[dict[str, str]]] = {
    "ai_ml": [
        {"category": "concurrentiel", "description":
         "Marché AI/ML saturé, commoditisation rapide des LLM génériques."},
        {"category": "régulatoire", "description":
         "Cadre législatif IA (EU AI Act, US exec orders) en consolidation."},
    ],
    "quantum_computing": [
        {"category": "technologique", "description":
         "Horizon de rentabilité distant ; NISQ → fault-tolerant incertain."},
    ],
    "biotech": [
        {"category": "régulatoire", "description":
         "Approbations FDA/EMA binaires ; échec de Phase 3 = -50% typique."},
    ],
    "cybersecurity": [
        {"category": "concurrentiel", "description":
         "Consolidation via M&A ; pricing pressure sur les mid-caps."},
    ],
    "space": [
        {"category": "opérationnel", "description":
         "Lancements ratés à fort impact ; cycles longs, CAPEX lourd."},
    ],
    "robotics": [
        {"category": "concurrentiel", "description":
         "Cycles d'adoption longs sur l'industrie traditionnelle."},
    ],
}


_BASE_RISKS: list[dict[str, str]] = [
    {"category": "macro", "description":
     "Sensibilité aux taux réels ; compression multiple possible."},
    {"category": "value_trap", "description":
     "Score élevé sur momentum récent — reversal possible si la tendance casse."},
]


def _make_risks(sector_ids: list[str]) -> list[dict[str, str]]:
    risks: list[dict[str, str]] = list(_BASE_RISKS)
    seen: set[tuple[str, str]] = {(r["category"], r["description"]) for r in risks}
    for sid in sector_ids:
        for r in _SECTOR_RISKS.get(sid, []):
            key = (r["category"], r["description"])
            if key not in seen:
                risks.append(r)
                seen.add(key)
    return risks


# --------------------------------------------------------- catalyseurs


#: Fenêtre par défaut pour la chasse aux catalyseurs récents (jours).
_CATALYST_RECENT_WINDOW_DAYS = 90

#: Phases considérées comme "tardives" (catalyseur fort si en cours).
_LATE_PHASES = {"PHASE3", "PHASE2/PHASE3", "PHASE4"}

#: Statuts considérés comme "trial en cours".
_LATE_TRIAL_ACTIVE = {
    "RECRUITING", "ACTIVE_NOT_RECRUITING", "ENROLLING_BY_INVITATION",
}

#: Top N contrats gouv à mentionner dans les catalyseurs défense.
_TOP_GOV_CONTRACTS = 3


def _pdufa_catalysts(ticker: str, as_of: datetime) -> list[dict[str, Any]]:
    """PDUFAs futures pour ce ticker depuis `pdufa_calendar`."""
    out: list[dict[str, Any]] = []
    today = as_of.date()
    for entry in upcoming_pdufas(today):
        if entry.get("ticker") != ticker:
            continue
        try:
            d = date.fromisoformat(entry["target_action_date"])
        except (KeyError, ValueError):
            continue
        days_to = (d - today).days
        drug = entry.get("drug", "n/a")
        out.append({
            "type": "pdufa",
            "date": entry["target_action_date"],
            "days_to": days_to,
            "drug": drug,
            "indication": entry.get("indication"),
            "description": (
                f"PDUFA J+{days_to}j sur **{drug}**"
                + (f" ({entry['indication']})" if entry.get("indication") else "")
            ),
            "source": "pdufa_calendar",
        })
    return out


def _active_late_phase_trials(
    ticker: str, as_of: datetime, lookback_days: int = 365,
) -> list[dict[str, Any]]:
    """Essais Phase 3+ en cours pour ce ticker, observés sur `lookback_days`."""
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(RawData.payload_json)
        .where(RawData.source == "clinicaltrials")
        .where(RawData.entity_type == "clinical_trial")
        .where(RawData.content_at >= start)
        .where(RawData.content_at <= as_of)
        .where(RawData.fetched_at <= as_of)
    )
    seen_nct: set[str] = set()
    out: list[dict[str, Any]] = []
    with session_scope() as session:
        for (payload_json,) in session.execute(stmt):
            try:
                payload = json.loads(payload_json or "{}")
            except (TypeError, ValueError):
                continue
            if payload.get("ticker") != ticker:
                continue
            phase = payload.get("phase")
            status = payload.get("overall_status")
            if phase not in _LATE_PHASES or status not in _LATE_TRIAL_ACTIVE:
                continue
            nct_id = payload.get("nct_id")
            if not nct_id or nct_id in seen_nct:
                continue
            seen_nct.add(nct_id)
            interv = payload.get("interventions") or []
            drug = interv[0] if interv else "candidat"
            out.append({
                "type": "phase3_trial",
                "nct_id": nct_id,
                "phase": phase,
                "status": status,
                "drug": drug,
                "completion_date": payload.get("primary_completion_date"),
                "description": (
                    f"{phase} en cours sur **{drug}** "
                    f"(NCT={nct_id}, {status})"
                ),
                "source": "clinicaltrials",
            })
    return out


def _recent_fda_approvals(
    ticker: str, as_of: datetime, lookback_days: int = _CATALYST_RECENT_WINDOW_DAYS,
) -> list[dict[str, Any]]:
    """Approvals FDA AP récents pour ce ticker."""
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(RawData.payload_json, RawData.content_at)
        .where(RawData.source == "fda")
        .where(RawData.entity_type == "fda_approval")
        .where(RawData.content_at >= start)
        .where(RawData.content_at <= as_of)
        .where(RawData.fetched_at <= as_of)
        .order_by(RawData.content_at.desc())
    )
    out: list[dict[str, Any]] = []
    with session_scope() as session:
        for payload_json, content_at in session.execute(stmt):
            try:
                payload = json.loads(payload_json or "{}")
            except (TypeError, ValueError):
                continue
            if payload.get("ticker") != ticker:
                continue
            if payload.get("submission_status") != "AP":
                continue
            brand = payload.get("brand_name") or payload.get("application_number")
            out.append({
                "type": "fda_approval",
                "date": content_at.date().isoformat(),
                "drug": brand,
                "application_number": payload.get("application_number"),
                "description": (
                    f"Approval FDA récente : **{brand}** ({content_at.date()})"
                ),
                "source": "fda",
            })
    return out


def _recent_large_contracts(
    ticker: str, as_of: datetime,
    lookback_days: int = _CATALYST_RECENT_WINDOW_DAYS,
    top_n: int = _TOP_GOV_CONTRACTS,
) -> list[dict[str, Any]]:
    """Top contrats gouv US par montant pour ce ticker, fenêtre `lookback_days`."""
    start = as_of - timedelta(days=lookback_days)
    stmt = (
        select(RawData.payload_json, RawData.content_at)
        .where(RawData.source == "usaspending")
        .where(RawData.entity_type == "gov_contract")
        .where(RawData.content_at >= start)
        .where(RawData.content_at <= as_of)
        .where(RawData.fetched_at <= as_of)
    )
    rows: list[dict[str, Any]] = []
    with session_scope() as session:
        for payload_json, content_at in session.execute(stmt):
            try:
                payload = json.loads(payload_json or "{}")
            except (TypeError, ValueError):
                continue
            if payload.get("ticker") != ticker:
                continue
            amount = payload.get("award_amount")
            if amount is None:
                continue
            rows.append({
                "type": "gov_contract",
                "date": content_at.date().isoformat(),
                "amount_usd": float(amount),
                "agency": payload.get("awarding_agency"),
                "description_raw": (payload.get("description") or "").strip(),
                "award_id": payload.get("award_id"),
            })
    rows.sort(key=lambda r: r["amount_usd"], reverse=True)
    out: list[dict[str, Any]] = []
    for r in rows[:top_n]:
        desc_short = r["description_raw"][:80]
        msg = (
            f"Contrat ${r['amount_usd'] / 1e6:.1f}M de "
            f"{r['agency'] or '—'} ({r['date']})"
            + (f" — {desc_short}" if desc_short else "")
        )
        out.append({
            "type": r["type"],
            "date": r["date"],
            "amount_usd": r["amount_usd"],
            "agency": r["agency"],
            "award_id": r["award_id"],
            "description": msg,
            "source": "usaspending",
        })
    return out


def _make_catalysts(
    ticker: str, sectors: list[str], as_of: datetime,
) -> list[dict[str, Any]]:
    """Catalyseurs sectoriels datés pour la thèse.

    Les sources sont activées par secteur :
      - `biotech` → PDUFA calendar + Phase 3 actifs + approvals récents
      - `space` (et plus tard `defense`) → top contrats gouv récents

    Un ticker multi-secteurs (ex : ai_ml + biotech théorique) cumule.
    Les sources sans donnée renvoient `[]` silencieusement — la narrative
    affichera alors le placeholder générique.
    """
    catalysts: list[dict[str, Any]] = []
    if "biotech" in sectors:
        catalysts.extend(_pdufa_catalysts(ticker, as_of))
        catalysts.extend(_active_late_phase_trials(ticker, as_of))
        catalysts.extend(_recent_fda_approvals(ticker, as_of))
    if "space" in sectors:
        catalysts.extend(_recent_large_contracts(ticker, as_of))
    return catalysts


def _make_narrative(
    *,
    ticker: str,
    name: str,
    sectors: list[str],
    score: float,
    dimensions: dict[str, float],
    triggers: list[dict[str, Any]],
    risks: list[dict[str, str]],
    catalysts: list[dict[str, Any]] | None,
    entry_price: float | None,
    horizon_days: int,
) -> str:
    """Narrative structurée en 5 sections (Markdown)."""
    dom = _dominant_dimension(dimensions)
    dom_str = f"{dom[0]} ({dom[1]:.1f}/100)" if dom else "signal composite"
    sectors_str = ", ".join(sectors) if sectors else "secteur non classé"

    dims_lines = "\n".join(
        f"  - {dim} : {val:.1f}/100"
        for dim, val in sorted(dimensions.items(), key=lambda kv: -kv[1])
    )

    if catalysts:
        # Tri : PDUFA en premier (le plus actionable), puis trials, puis approvals,
        # puis contrats. À iso-type, dates les plus proches d'abord.
        order = {"pdufa": 0, "phase3_trial": 1, "fda_approval": 2, "gov_contract": 3}
        ordered = sorted(
            catalysts, key=lambda c: (order.get(c.get("type", ""), 9), c.get("date") or "")
        )
        catalysts_section = (
            "\n".join(f"  - {c['description']}" for c in ordered)
            + f"\n  - Horizon retenu : {horizon_days} jours."
        )
    else:
        catalysts_section = (
            "  - Pas de catalyseur daté collecté (PDUFA, contrats, "
            "approvals — voir Phase 4 pour les secteurs concernés).\n"
            f"  - Horizon retenu : {horizon_days} jours."
        )

    risks_lines = "\n".join(
        f"  - [{r['category']}] {r['description']}" for r in risks
    )

    if entry_price is not None:
        band_low = entry_price * (1 - DEFAULT_ENTRY_BAND_PCT)
        band_high = entry_price * (1 + DEFAULT_ENTRY_BAND_PCT)
        entry_str = (
            f"Dernier close : {entry_price:.2f}. Zone d'entrée suggérée : "
            f"{band_low:.2f}–{band_high:.2f} "
            f"(±{DEFAULT_ENTRY_BAND_PCT * 100:.0f}%)."
        )
    else:
        entry_str = (
            "Prix d'entrée non disponible — aucun close yfinance PIT. "
            "Différer l'ouverture de position tant que la donnée manque."
        )

    return (
        f"**{name} ({ticker}) — Score {score:.1f}/100**\n\n"
        f"**Pourquoi maintenant**\n"
        f"Le score composite atteint {score:.1f}/100, tiré par "
        f"{dom_str}. Secteur(s) : {sectors_str}.\n\n"
        f"**Score**\n{dims_lines}\n\n"
        f"**Catalyseurs**\n{catalysts_section}\n\n"
        f"**Risques**\n{risks_lines}\n\n"
        f"**Entrée**\n{entry_str}"
    )


# --------------------------------------------------- score lookup + dedupe


def _latest_score_row(ticker: str, as_of: datetime) -> Feature | None:
    """Dernière ligne `stock_score` PIT pour `ticker`."""
    stmt = (
        select(Feature)
        .where(Feature.feature_name == "stock_score")
        .where(Feature.target_type == "asset")
        .where(Feature.target_id == ticker)
        .where(Feature.computed_at <= as_of)
        .order_by(Feature.computed_at.desc())
        .limit(1)
    )
    with session_scope() as session:
        row = session.execute(stmt).scalar_one_or_none()
        if row is not None:
            session.expunge(row)
        return row


def _has_thesis_today(ticker: str, as_of: datetime) -> bool:
    """Vrai si une thèse stock existe déjà pour `ticker` sur la date UTC de `as_of`."""
    day_start = datetime.combine(as_of.date(), time.min)
    day_end = day_start + timedelta(days=1)
    stmt = (
        select(Thesis.id)
        .where(Thesis.asset_type == "stock")
        .where(Thesis.asset_id == ticker)
        .where(Thesis.created_at >= day_start)
        .where(Thesis.created_at < day_end)
        .limit(1)
    )
    with session_scope() as session:
        return session.execute(stmt).first() is not None


# --------------------------------------------------------------- generator


class ThesisGenerator:
    """Parcourt la watchlist et génère les thèses méritées par le score."""

    #: Type d'actif produit par ce générateur. Une classe dédiée sera ajoutée
    #: pour crypto/startups en Phase 4.
    asset_type = "stock"

    def __init__(
        self,
        tickers: list[str] | None = None,
        thresholds: dict[str, float] | None = None,
        horizon_days: int = DEFAULT_HORIZON_DAYS,
    ) -> None:
        self._tickers = list(tickers) if tickers else list(WATCHLIST_TICKERS)
        self._thresholds = dict(thresholds) if thresholds else dict(THESIS_SCORE_THRESHOLDS)
        if "_default" not in self._thresholds:
            self._thresholds["_default"] = 70.0
        self._horizon_days = int(horizon_days)
        self._sectors_by_ticker = {w["ticker"]: w["sectors"] for w in STOCK_WATCHLIST}
        self._names_by_ticker = {w["ticker"]: w["name"] for w in STOCK_WATCHLIST}

    # --- seuil ---------------------------------------------------------

    def _threshold_for(self, sectors: list[str]) -> float:
        """Seuil minimum pour le ticker : le plus **bas** parmi ses secteurs.

        Un ticker multi-secteurs bénéficie du seuil le plus permissif de
        ses secteurs — on ne veut pas qu'un secteur "dur" (AI) bloque
        une thèse qui aurait passé sur un secteur "permissif".
        """
        if not sectors:
            return float(self._thresholds["_default"])
        return min(
            float(self._thresholds.get(sid, self._thresholds["_default"]))
            for sid in sectors
        )

    # --- cycle ---------------------------------------------------------

    def run(self, as_of: datetime | None = None) -> int:
        """Traite tous les tickers configurés. Retourne le nb de thèses créées."""
        ts = as_of or utc_now()
        logger.info(
            "[thesis] Génération pour {} ticker(s) à {}",
            len(self._tickers), ts,
        )
        created = 0
        for ticker in self._tickers:
            try:
                if self._generate_one(ticker, ts):
                    created += 1
            except Exception as exc:
                logger.warning("[thesis] {} a échoué : {}", ticker, exc)
        logger.info("[thesis] Terminé : {} thèse(s) créée(s)", created)
        return created

    # --- unité ---------------------------------------------------------

    def _generate_one(self, ticker: str, as_of: datetime) -> bool:
        sectors = self._sectors_by_ticker.get(ticker, [])
        threshold = self._threshold_for(sectors)

        row = _latest_score_row(ticker, as_of)
        if row is None:
            return False
        score = float(row.value)
        if score < threshold:
            return False

        if _has_thesis_today(ticker, as_of):
            logger.debug("[thesis] {} déjà couvert aujourd'hui — skip", ticker)
            return False

        metadata: dict[str, Any] = {}
        if row.metadata_json:
            try:
                metadata = json.loads(row.metadata_json) or {}
            except (TypeError, ValueError):
                metadata = {}
        dimensions = metadata.get("dimensions") or {}
        details = metadata.get("details") or {}
        model_version = str(metadata.get("model_version") or "unknown")
        weights_snapshot = metadata.get("weights") or {}

        # `sector_id` canonique = premier secteur (historiquement le plus
        # central — AI pour NVDA, cyber pour CRWD). Les autres secteurs
        # sont conservés dans la narrative et les risques.
        sector_id = sectors[0] if sectors else "_unclassified"

        triggers = _make_triggers(dimensions, details)
        risks = _make_risks(sectors)
        catalysts = _make_catalysts(ticker, sectors, as_of)
        entry_price = _latest_entry_price(ticker, as_of)
        narrative = _make_narrative(
            ticker=ticker,
            name=self._names_by_ticker.get(ticker, ticker),
            sectors=sectors,
            score=score,
            dimensions={k: float(v) for k, v in dimensions.items()},
            triggers=triggers,
            risks=risks,
            catalysts=catalysts,
            entry_price=entry_price,
            horizon_days=self._horizon_days,
        )
        recommendation = _make_recommendation(score)

        entry_conditions = {
            "band_pct": DEFAULT_ENTRY_BAND_PCT,
            "reference_close": entry_price,
        }
        score_breakdown = {
            "dimensions": {k: float(v) for k, v in dimensions.items()},
            "details": details,
        }

        with session_scope() as session:
            session.add(Thesis(
                created_at=as_of,
                asset_type=self.asset_type,
                asset_id=ticker,
                sector_id=sector_id,
                score=score,
                score_breakdown_json=json.dumps(score_breakdown, default=str,
                                                ensure_ascii=False),
                recommendation=recommendation,
                horizon_days=self._horizon_days,
                entry_price=entry_price,
                entry_conditions_json=json.dumps(entry_conditions),
                triggers_json=json.dumps(triggers, default=str, ensure_ascii=False),
                risks_json=json.dumps(risks, ensure_ascii=False),
                catalysts_json=json.dumps(catalysts, default=str,
                                           ensure_ascii=False),
                narrative=narrative,
                model_version=model_version,
                weights_snapshot_json=json.dumps(weights_snapshot),
            ))
        logger.info(
            "[thesis] +1 pour {} ({:.1f}/100, {}, secteur={})",
            ticker, score, recommendation, sector_id,
        )
        return True
