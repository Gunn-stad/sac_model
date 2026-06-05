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

SAFE_ASSETS = {"IEF", "TLT", "BND", "GLD", "SHY", "SGOV", "BIL"}

st.set_page_config(page_title="V16 Paper Trading Dashboard", layout="wide")
st.title("📊 V16 Paper Trading Dashboard")
st.caption("V12 Champion vs V13 Defensive vs SPY benchmark")


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
        "Missing paper dashboard files. Run first:\n\n"
        "python v14_paper_trading_dashboard.py --init 10000\n\n"
        "or:\n\n"
        "python v14_paper_trading_dashboard.py --update"
    )
    st.stop()

log["date"] = pd.to_datetime(log["date"])
state["gain_loss"] = state["last_value"] / state["entry_value"] - 1
state["weight_now"] = state.groupby("strategy")["last_value"].transform(lambda x: x / x.sum())

latest = log.sort_values("date").groupby("strategy").tail(1).copy()

st.subheader("Overview")
card_cols = st.columns(len(latest))

for col, (_, row) in zip(card_cols, latest.iterrows()):
    col.metric(
        label=row["strategy"],
        value=f"${row['current_value']:,.2f}",
        delta=f"{row['return'] * 100:.2f}%"
    )

st.subheader("Alpha vs SPY")
latest_map = latest.set_index("strategy")
alpha_cols = st.columns(2)

if "V12_champion" in latest_map.index and "SPY_benchmark_for_V12" in latest_map.index:
    alpha = latest_map.loc["V12_champion", "return"] - latest_map.loc["SPY_benchmark_for_V12", "return"]
    alpha_cols[0].metric("V12 Alpha vs SPY", f"{alpha * 100:.2f}%")

if "V13_defensive" in latest_map.index and "SPY_benchmark_for_V13" in latest_map.index:
    alpha = latest_map.loc["V13_defensive", "return"] - latest_map.loc["SPY_benchmark_for_V13", "return"]
    alpha_cols[1].metric("V13 Alpha vs SPY", f"{alpha * 100:.2f}%")

st.divider()

st.subheader("Equity Curve")
fig, ax = plt.subplots(figsize=(12, 5))
for strategy, group in log.groupby("strategy"):
    group = group.sort_values("date")
    ax.plot(group["date"], group["current_value"], label=strategy)
ax.set_title("Portfolio Value")
ax.set_xlabel("Date")
ax.set_ylabel("Value")
ax.legend()
ax.grid(True)
st.pyplot(fig)

st.subheader("Drawdown")
dd_rows = []
for strategy, group in log.groupby("strategy"):
    g = group.sort_values("date").copy()
    g["peak"] = g["current_value"].cummax()
    g["drawdown"] = g["current_value"] / g["peak"] - 1
    g["strategy"] = strategy
    dd_rows.append(g[["date", "strategy", "drawdown"]])

dd = pd.concat(dd_rows)
fig_dd, ax_dd = plt.subplots(figsize=(12, 4))
for strategy, group in dd.groupby("strategy"):
    ax_dd.plot(group["date"], group["drawdown"] * 100, label=strategy)
ax_dd.set_title("Drawdown (%)")
ax_dd.set_xlabel("Date")
ax_dd.set_ylabel("Drawdown %")
ax_dd.legend()
ax_dd.grid(True)
st.pyplot(fig_dd)

st.divider()

st.subheader("Current Positions")
strategy_options = sorted(state["strategy"].unique())
selected_strategy = st.selectbox("Choose strategy", strategy_options)

positions = state[state["strategy"] == selected_strategy].copy()
positions = positions.sort_values("last_value", ascending=False)

pcol1, pcol2 = st.columns([2, 1])

with pcol1:
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
        ],
        use_container_width=True,
    )

with pcol2:
    fig_pie, ax_pie = plt.subplots(figsize=(5, 5))
    ax_pie.pie(
        positions["last_value"],
        labels=positions["ticker"],
        autopct="%1.1f%%",
        startangle=90,
    )
    ax_pie.set_title(f"{selected_strategy} Allocation")
    st.pyplot(fig_pie)

st.subheader("Top Winners and Losers")
wl = positions.copy()
wl = wl[~wl["strategy"].str.contains("SPY_benchmark", na=False)]
wcol, lcol = st.columns(2)

with wcol:
    st.markdown("### Winners")
    st.dataframe(
        wl.sort_values("gain_loss", ascending=False)[["ticker", "gain_loss", "last_value"]].head(10),
        use_container_width=True,
    )

with lcol:
    st.markdown("### Losers")
    st.dataframe(
        wl.sort_values("gain_loss", ascending=True)[["ticker", "gain_loss", "last_value"]].head(10),
        use_container_width=True,
    )

st.divider()

st.subheader("Sector and Safe-Asset Exposure")

def infer_asset_type(ticker):
    if ticker in SAFE_ASSETS:
        if ticker == "GLD":
            return "Gold"
        if ticker in {"SHY", "SGOV", "BIL"}:
            return "Cash-like"
        return "Bonds"
    return "Stock"

positions["asset_bucket"] = positions["ticker"].apply(infer_asset_type)
bucket_exp = positions.groupby("asset_bucket")["last_value"].sum()
bucket_exp = bucket_exp / bucket_exp.sum()

bcol1, bcol2 = st.columns(2)

with bcol1:
    st.markdown("### Asset Bucket Exposure")
    st.dataframe(bucket_exp.reset_index().rename(columns={"last_value": "weight"}), use_container_width=True)

with bcol2:
    fig_bucket, ax_bucket = plt.subplots(figsize=(5, 5))
    ax_bucket.pie(bucket_exp.values, labels=bucket_exp.index, autopct="%1.1f%%", startangle=90)
    ax_bucket.set_title("Stocks / Bonds / Gold / Cash")
    st.pyplot(fig_bucket)

if selected_strategy == "V13_defensive" and v13 is not None and "sector" in v13.columns:
    st.markdown("### V13 Sector Exposure")
    v13_temp = v13.copy()
    v13_temp["target_weight"] = v13_temp["target_weight"].astype(float)
    sector_exp = v13_temp.groupby("sector")["target_weight"].sum().sort_values(ascending=False)
    st.dataframe(sector_exp.reset_index(), use_container_width=True)

    fig_sector, ax_sector = plt.subplots(figsize=(8, 4))
    sector_exp.plot(kind="bar", ax=ax_sector)
    ax_sector.set_title("V13 Target Sector Exposure")
    ax_sector.set_ylabel("Weight")
    ax_sector.grid(True)
    st.pyplot(fig_sector)

st.divider()

st.subheader("Data Tables")
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "Dashboard Summary", "Daily Log", "State", "V12 Portfolio", "V13 Portfolio"
])

with tab1:
    if summary is not None:
        st.dataframe(summary, use_container_width=True)
        st.download_button("Download dashboard_summary.csv", summary.to_csv(index=False), file_name="dashboard_summary.csv")
    else:
        st.info("No dashboard summary yet.")

with tab2:
    st.dataframe(log.sort_values(["date", "strategy"], ascending=[False, True]), use_container_width=True)
    st.download_button("Download daily_log.csv", log.to_csv(index=False), file_name="daily_log.csv")

with tab3:
    st.dataframe(state, use_container_width=True)
    st.download_button("Download portfolio_state.csv", state.to_csv(index=False), file_name="portfolio_state.csv")

with tab4:
    if v12 is not None:
        st.dataframe(v12, use_container_width=True)

with tab5:
    if v13 is not None:
        st.dataframe(v13, use_container_width=True)

st.divider()

st.subheader("Operating Rules")
st.markdown(
    """
    **Hourly update during market hours:**
    ```bash
    python v16_hourly_updater.py
    ```

    **Manual update:**
    ```bash
    python v14_paper_trading_dashboard.py --update
    ```

    **Monthly rebalance:**
    ```bash
    python v12_champion_stress_tests.py
    python v13_defensive_momentum_bayesian.py
    python v14_paper_trading_dashboard.py --rebalance
    ```
    """
)