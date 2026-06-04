"""
v15_visual_paper_dashboard.py

Visual dashboard for the paper trading system.

Run locally:
    streamlit run v15_visual_paper_dashboard.py

Run in Colab:
    !pip install streamlit pyngrok
    !streamlit run v15_visual_paper_dashboard.py --server.port 8501 &

Expected input files:
    paper_dashboard/portfolio_state.csv
    paper_dashboard/daily_log.csv
    paper_dashboard/dashboard_summary.csv
    paper_trading/v12_today_portfolio.csv
    paper_trading/v13_today_portfolio.csv
"""

import os
import pandas as pd
import numpy as np
import streamlit as st
import matplotlib.pyplot as plt

DASH_DIR = "paper_dashboard"
PAPER_DIR = "paper_trading"

STATE_PATH = os.path.join(DASH_DIR, "portfolio_state.csv")
LOG_PATH = os.path.join(DASH_DIR, "daily_log.csv")
SUMMARY_PATH = os.path.join(DASH_DIR, "dashboard_summary.csv")
V12_PATH = os.path.join(PAPER_DIR, "v12_today_portfolio.csv")
V13_PATH = os.path.join(PAPER_DIR, "v13_today_portfolio.csv")


st.set_page_config(
    page_title="Paper Trading Dashboard",
    layout="wide",
)

st.title("📈 Paper Trading Dashboard")
st.caption("Tracks V12 Champion, V13 Defensive, and SPY benchmarks.")


def load_csv(path):
    if not os.path.exists(path):
        return None
    return pd.read_csv(path)


state = load_csv(STATE_PATH)
log = load_csv(LOG_PATH)
summary = load_csv(SUMMARY_PATH)
v12 = load_csv(V12_PATH)
v13 = load_csv(V13_PATH)

if state is None or log is None:
    st.error(
        "Missing dashboard files. Run:\n\n"
        "`python v14_paper_trading_dashboard.py --init 10000`\n\n"
        "or\n\n"
        "`python v14_paper_trading_dashboard.py --update`"
    )
    st.stop()

log["date"] = pd.to_datetime(log["date"])

latest = log.sort_values("date").groupby("strategy").tail(1)

st.subheader("Current Overview")

cols = st.columns(len(latest))

for col, (_, row) in zip(cols, latest.iterrows()):
    col.metric(
        label=row["strategy"],
        value=f"${row['current_value']:,.2f}",
        delta=f"{row['return'] * 100:.2f}%"
    )

st.divider()

st.subheader("Equity Curve")

fig, ax = plt.subplots(figsize=(12, 5))

for strategy, group in log.groupby("strategy"):
    group = group.sort_values("date")
    ax.plot(group["date"], group["current_value"], label=strategy)

ax.set_title("Portfolio Value Over Time")
ax.set_xlabel("Date")
ax.set_ylabel("Portfolio Value")
ax.legend()
ax.grid(True)

st.pyplot(fig)

st.divider()

left, right = st.columns(2)

with left:
    st.subheader("Dashboard Summary")
    if summary is not None:
        st.dataframe(summary, use_container_width=True)
    else:
        st.info("No dashboard_summary.csv yet.")

with right:
    st.subheader("Latest Daily Log")
    st.dataframe(log.sort_values(["date", "strategy"], ascending=[False, True]), use_container_width=True)

st.divider()

st.subheader("Current Positions")

strategy_options = sorted(state["strategy"].unique())
selected_strategy = st.selectbox("Choose strategy", strategy_options)

positions = state[state["strategy"] == selected_strategy].copy()
positions["weight_now"] = positions["last_value"] / positions["last_value"].sum()
positions["gain_loss"] = positions["last_value"] / positions["entry_value"] - 1

c1, c2 = st.columns([2, 1])

with c1:
    st.dataframe(
        positions[
            [
                "ticker",
                "target_weight",
                "weight_now",
                "entry_price",
                "last_price",
                "shares",
                "entry_value",
                "last_value",
                "gain_loss",
            ]
        ].sort_values("last_value", ascending=False),
        use_container_width=True,
    )

with c2:
    fig2, ax2 = plt.subplots(figsize=(5, 5))
    ax2.pie(
        positions["last_value"],
        labels=positions["ticker"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax2.set_title(f"{selected_strategy} Allocation")
    st.pyplot(fig2)

st.divider()

st.subheader("V12 vs V13 Target Portfolios")

col_a, col_b = st.columns(2)

with col_a:
    st.markdown("### V12 Champion")
    if v12 is not None:
        st.dataframe(v12, use_container_width=True)
    else:
        st.info("Missing paper_trading/v12_today_portfolio.csv")

with col_b:
    st.markdown("### V13 Defensive")
    if v13 is not None:
        st.dataframe(v13, use_container_width=True)
    else:
        st.info("Missing paper_trading/v13_today_portfolio.csv")

st.divider()

st.subheader("Drawdown")

drawdown_rows = []

for strategy, group in log.groupby("strategy"):
    group = group.sort_values("date").copy()
    group["peak"] = group["current_value"].cummax()
    group["drawdown"] = group["current_value"] / group["peak"] - 1
    group["strategy"] = strategy
    drawdown_rows.append(group[["date", "strategy", "drawdown"]])

dd = pd.concat(drawdown_rows)

fig3, ax3 = plt.subplots(figsize=(12, 4))

for strategy, group in dd.groupby("strategy"):
    ax3.plot(group["date"], group["drawdown"] * 100, label=strategy)

ax3.set_title("Drawdown (%)")
ax3.set_xlabel("Date")
ax3.set_ylabel("Drawdown %")
ax3.legend()
ax3.grid(True)

st.pyplot(fig3)

st.divider()

st.subheader("Instructions")

st.markdown(
    """
    **Daily or weekly update:**
    ```bash
    python v14_paper_trading_dashboard.py --update
    ```

    **Monthly rebalance:**
    ```bash
    python v12_champion_stress_tests.py
    python v13_defensive_momentum_bayesian.py
    python v14_paper_trading_dashboard.py --rebalance
    ```

    **Run this visual dashboard:**
    ```bash
    streamlit run v15_visual_paper_dashboard.py
    ```
    """
)