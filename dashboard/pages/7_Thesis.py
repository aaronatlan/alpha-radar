"""Page 7 — Détail d'une thèse (SPEC §7.9 Page 3).

Vue détaillée d'une thèse : narrative, décomposition du score,
catalyseurs (calendrier), risques, graphique de prix avec zone d'entrée
et jalons d'évaluation.

Sélection
---------
- Query string `?thesis_id=42` (utilisable depuis la page Mémoire ou un
  partage de lien).
- Sinon, selectbox listant les thèses existantes (plus récente d'abord).

Discipline PIT
--------------
Le graphique de prix est lu via `get_price_history` qui filtre
`content_at` ET `fetched_at` — on ne trace que ce qui aurait été
observable à la date d'évaluation la plus récente.
"""
from __future__ import annotations

from datetime import datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from dashboard._data import (
    get_price_history,
    get_thesis_detail,
    list_thesis_ids,
)
from memory.database import init_db, utc_now


st.set_page_config(page_title="Thèse — Alpha Radar", page_icon="📝", layout="wide")
init_db()

st.title("📝 Thèse")
st.caption("Vue détaillée — narrative, score décomposé, catalyseurs, risques, prix.")

# --------------------------------------------------------------- sélecteur

available = list_thesis_ids()
if not available:
    st.info("Aucune thèse en base. Lancez `python -m scheduler.jobs` pour générer.")
    st.stop()

# Query param prioritaire (lien depuis page Mémoire / partage).
qp = st.query_params
preselected = None
if "thesis_id" in qp:
    try:
        preselected = int(qp["thesis_id"])
    except (TypeError, ValueError):
        preselected = None

ids = [tid for tid, _, _ in available]
default_idx = ids.index(preselected) if preselected in ids else 0


def _fmt(option: int) -> str:
    for tid, ticker, created in available:
        if tid == option:
            return f"#{tid} — {ticker} ({created.strftime('%Y-%m-%d')})"
    return str(option)


thesis_id = st.selectbox(
    "Thèse", options=ids, index=default_idx, format_func=_fmt,
)

detail = get_thesis_detail(int(thesis_id))
if detail is None:
    st.error(f"Thèse #{thesis_id} introuvable.")
    st.stop()

th = detail["thesis"]

# Synchronise le query param pour le partage de lien.
st.query_params["thesis_id"] = str(thesis_id)

# --------------------------------------------------------------- entête

_RECO_COLOR = {"BUY": "🟢", "WATCH": "🟡", "AVOID": "🔴"}
_STATUS_EMOJI = {
    "success": "✅", "failure": "❌", "partial": "🔶", "active": "🔵",
}

reco_badge = f"{_RECO_COLOR.get(th['recommendation'], '⚪')} {th['recommendation']}"
status_badge = (
    f"{_STATUS_EMOJI.get(detail['latest_status'], '—')} {detail['latest_status']}"
    if detail["latest_status"] else "— pas encore évaluée"
)

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Ticker", th["asset_id"])
c2.metric("Secteur", th["sector_id"])
c3.metric("Score", f"{th['score']:.1f}/100")
c4.metric("Reco", reco_badge)
if detail["latest_alpha"] is not None:
    c5.metric("Alpha (dernier jalon)",
              f"{detail['latest_alpha'] * 100:+.2f}%")
else:
    c5.metric("Statut", status_badge)

st.caption(
    f"Créée le {th['created_at'].strftime('%Y-%m-%d %H:%M')} UTC · "
    f"Horizon {th['horizon_days']}j · "
    f"Modèle {th['model_version']}"
)

st.divider()

# --------------------------------------------------------------- narrative

left, right = st.columns([3, 2])

with left:
    st.subheader("Narrative")
    st.markdown(th["narrative"] or "_(narrative vide)_")

with right:
    st.subheader("Décomposition du score")
    dims = detail["dimensions"] or {}
    if not dims:
        st.write("_(aucune dimension persistée)_")
    else:
        dims_df = pd.DataFrame(
            sorted(dims.items(), key=lambda kv: -kv[1]),
            columns=["Dimension", "Sous-score"],
        )
        fig_dims = go.Figure(go.Bar(
            x=dims_df["Sous-score"],
            y=dims_df["Dimension"],
            orientation="h",
            text=[f"{v:.1f}" for v in dims_df["Sous-score"]],
            textposition="outside",
            marker={"color": dims_df["Sous-score"], "colorscale": "RdYlGn",
                    "cmin": 0, "cmax": 100},
        ))
        fig_dims.update_layout(
            height=max(180, 40 + 35 * len(dims_df)),
            margin={"l": 10, "r": 10, "t": 10, "b": 10},
            xaxis={"range": [0, 105], "title": ""},
            yaxis={"title": ""},
            showlegend=False,
        )
        st.plotly_chart(fig_dims, use_container_width=True)

st.divider()

# --------------------------------------------------------------- catalyseurs

st.subheader("Catalyseurs")
catalysts = detail["catalysts"] or []
if not catalysts:
    st.caption(
        "Aucun catalyseur daté collecté. Les sources (PDUFA, contrats gouv, "
        "approvals FDA) ne s'activent que pour certains secteurs."
    )
else:
    rows = []
    for c in catalysts:
        if not isinstance(c, dict):
            continue
        rows.append({
            "Type": c.get("type", "—"),
            "Date": c.get("date", "—"),
            "Description": c.get("description", "—"),
            "Source": c.get("source", "—"),
        })
    if rows:
        st.dataframe(
            pd.DataFrame(rows),
            use_container_width=True,
            hide_index=True,
        )

st.divider()

# --------------------------------------------------------------- risques

st.subheader("Risques")
risks = detail["risks"] or []
if not risks:
    st.caption("_(aucun risque persisté)_")
else:
    for r in risks:
        if not isinstance(r, dict):
            continue
        cat = r.get("category", "—")
        desc = r.get("description", "—")
        st.markdown(f"- **[{cat}]** {desc}")

st.divider()

# --------------------------------------------------------------- prix + zone d'entrée

st.subheader("Prix & zone d'entrée")
ticker = th["asset_id"]
created_at = th["created_at"]
horizon = th["horizon_days"]

# Fenêtre : 60j avant la création → min(now, created + horizon).
window_start = created_at - timedelta(days=60)
window_end = min(utc_now(), created_at + timedelta(days=horizon))
prices = get_price_history(ticker, start=window_start, end=window_end)

if prices.empty:
    st.caption(
        f"Aucun close yfinance disponible pour **{ticker}** "
        f"sur la fenêtre [{window_start.date()}, {window_end.date()}]."
    )
else:
    entry_price = th["entry_price"]
    band_pct = (
        detail["entry_conditions"].get("band_pct")
        if isinstance(detail["entry_conditions"], dict) else None
    ) or 0.02

    fig_px = go.Figure()
    fig_px.add_trace(go.Scatter(
        x=prices["date"], y=prices["close"],
        mode="lines", name=f"{ticker} close", line={"width": 2},
    ))
    if entry_price is not None:
        band_low = entry_price * (1 - band_pct)
        band_high = entry_price * (1 + band_pct)
        fig_px.add_hrect(
            y0=band_low, y1=band_high,
            fillcolor="LightGreen", opacity=0.25, line_width=0,
            annotation_text=f"Zone d'entrée ±{band_pct * 100:.1f}%",
            annotation_position="top left",
        )
        fig_px.add_hline(
            y=entry_price, line_dash="dash", line_color="green",
            annotation_text=f"entry {entry_price:.2f}",
            annotation_position="bottom right",
        )

    # Marqueur "création thèse"
    fig_px.add_vline(
        x=created_at, line_dash="dot", line_color="gray",
        annotation_text="thèse",
        annotation_position="top",
    )

    # Marqueurs jalons d'évaluation
    for ev in detail["evaluations"]:
        ev_date = created_at + timedelta(days=int(ev["days_since_thesis"]))
        fig_px.add_vline(
            x=ev_date, line_dash="dot",
            line_color={"success": "green", "failure": "red",
                        "partial": "orange", "active": "blue"}.get(
                            ev["status"], "gray"),
            annotation_text=f"J+{ev['days_since_thesis']} {ev['status']}",
            annotation_position="bottom",
        )

    fig_px.update_layout(
        height=420,
        margin={"l": 40, "r": 20, "t": 20, "b": 40},
        xaxis_title="Date",
        yaxis_title="Close",
        legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
                "xanchor": "right", "x": 1},
    )
    st.plotly_chart(fig_px, use_container_width=True)

# --------------------------------------------------------------- jalons

st.subheader("Jalons d'évaluation")
evaluations = detail["evaluations"]
if not evaluations:
    st.caption(
        "Pas encore d'évaluation. Le premier jalon est calculé à J+30 par "
        "le scheduler."
    )
else:
    eval_rows = []
    for ev in evaluations:
        eval_rows.append({
            "Jalon": f"J+{ev['days_since_thesis']}",
            "Évalué le": ev["evaluated_at"].strftime("%Y-%m-%d")
                if ev["evaluated_at"] else "—",
            "Prix": f"{ev['current_price']:.2f}"
                if ev["current_price"] is not None else "—",
            "Return": f"{ev['return_pct'] * 100:+.2f}%"
                if ev["return_pct"] is not None else "—",
            "Bench": f"{ev['benchmark_return_pct'] * 100:+.2f}%"
                if ev["benchmark_return_pct"] is not None else "—",
            "Alpha": f"{ev['alpha_pct'] * 100:+.2f}%"
                if ev["alpha_pct"] is not None else "—",
            "Statut": f"{_STATUS_EMOJI.get(ev['status'], '—')} {ev['status']}",
            "Note": ev["notes"] or "",
        })
    st.dataframe(
        pd.DataFrame(eval_rows),
        use_container_width=True,
        hide_index=True,
    )
