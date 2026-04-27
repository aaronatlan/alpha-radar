"""Page 6 — Backtest (SPEC §7.8 + §7.9 Page 4 enrichie).

Lance un backtest portefeuille sur les thèses déjà persistées et
affiche :
  - métriques agrégées (total return, CAGR, Sharpe, max DD, alpha)
  - equity curve vs benchmark
  - tableau des thèses prises sur la fenêtre

Le mode ici est restreint à `portfolio` (simulation sur les thèses
déjà en base) — les modes `replay` et `walk-forward` se lancent en CLI
via `python -m backtesting.runner` (calcul plus long, sortie JSON).
"""
from __future__ import annotations

from datetime import date, datetime, timedelta

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from backtesting.portfolio import PortfolioSimulator
from memory.database import init_db

init_db()

st.set_page_config(page_title="Backtest — Alpha Radar", layout="wide")
st.title("📈 Backtest")
st.caption(
    "Simulation théorique d'un portefeuille équipondéré sur les "
    "recommandations BUY / WATCH déjà en base."
)

# ---------------------------------------------------------------- contrôles

col_from, col_to, col_bench = st.columns(3)
with col_from:
    d_from = st.date_input(
        "Depuis", value=date.today() - timedelta(days=180),
    )
with col_to:
    d_to = st.date_input("Jusqu'au", value=date.today())
with col_bench:
    bench_input = st.text_input(
        "Benchmark (tickers, virgule)", value="",
        help="Optionnel, ex : 'SPY' ou 'SPY,QQQ'.",
    )

if d_to <= d_from:
    st.error("La date de fin doit être strictement supérieure à la date de début.")
    st.stop()

start = datetime.combine(d_from, datetime.min.time())
end = datetime.combine(d_to, datetime.max.time())
benchmark = [t.strip() for t in bench_input.split(",") if t.strip()]

# ---------------------------------------------------------------- run

sim = PortfolioSimulator(benchmark_tickers=benchmark)
res = sim.run(start=start, end=end)

if res.positions_taken == 0:
    st.info(
        "Aucune position prise sur cette fenêtre — pas de thèse BUY/WATCH "
        "avec entry_price valide en base."
    )
    st.stop()

# ---------------------------------------------------------------- KPIs

m = res.metrics
c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Positions prises", res.positions_taken)


def _fmt_pct(v: float | None) -> str:
    return f"{v * 100:+.2f}%" if v is not None else "—"


def _fmt_num(v: float | None) -> str:
    return f"{v:.2f}" if v is not None else "—"


c2.metric("Total return", _fmt_pct(m.get("total_return")))
c3.metric("Sharpe (ann.)", _fmt_num(m.get("sharpe")))
c4.metric("Max drawdown", _fmt_pct(m.get("max_drawdown")))
if m.get("alpha_vs_benchmark") is not None:
    c5.metric("Alpha vs bench. (daily)", _fmt_pct(m.get("alpha_vs_benchmark")))
else:
    c5.metric("Alpha vs bench. (daily)", "—")

st.divider()

# ---------------------------------------------------------------- equity curve

st.subheader("Equity curve")
df_eq = pd.DataFrame({
    "date": res.dates,
    "portfolio": res.equity_curve,
})
fig = go.Figure()
fig.add_trace(go.Scatter(
    x=df_eq["date"], y=df_eq["portfolio"],
    mode="lines", name="Portefeuille", line={"width": 2.5},
))
if res.benchmark_curve and len(res.benchmark_curve) == len(res.dates):
    df_eq["benchmark"] = res.benchmark_curve
    fig.add_trace(go.Scatter(
        x=df_eq["date"], y=df_eq["benchmark"],
        mode="lines", name="Benchmark",
        line={"width": 1.8, "dash": "dot"},
    ))
fig.update_layout(
    yaxis_title="Equity (capital initial = 1.0)",
    xaxis_title="Date",
    height=420,
    margin={"l": 40, "r": 20, "t": 20, "b": 40},
    legend={"orientation": "h", "yanchor": "bottom", "y": 1.02,
            "xanchor": "right", "x": 1},
)
st.plotly_chart(fig, use_container_width=True)

st.divider()

# ---------------------------------------------------------------- table métriques

st.subheader("Métriques détaillées")
metrics_rows = [
    ("Total return", _fmt_pct(m.get("total_return"))),
    ("CAGR", _fmt_pct(m.get("cagr"))),
    ("Sharpe (annualisé)", _fmt_num(m.get("sharpe"))),
    ("Max drawdown", _fmt_pct(m.get("max_drawdown"))),
    ("Alpha vs benchmark (daily)",
     _fmt_pct(m.get("alpha_vs_benchmark"))),
    ("Benchmark total return",
     _fmt_pct(m.get("benchmark_total_return"))),
    ("Nb thèses dans la fenêtre", int(m.get("n_theses") or 0)),
]
st.table(pd.DataFrame(metrics_rows, columns=["Métrique", "Valeur"]))

st.caption(
    "ℹ️ Les modes `replay` et `walk-forward` (calcul plus lourd) sont "
    "disponibles en CLI : `python -m backtesting.runner walk-forward "
    "--start ... --end ... --folds N --weights ...`"
)
