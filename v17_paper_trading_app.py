import os
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DASH_DIR = "paper_dashboard"
PAPER_DIR = "paper_trading"

STATE_PATH = os.path.join(DASH_DIR, "portfolio_state.csv")
LOG_PATH = os.path.join(DASH_DIR, "daily_log.csv")
SUMMARY_PATH = os.path.join(DASH_DIR, "dashboard_summary.csv")
V13_PATH = os.path.join(PAPER_DIR, "v13_today_portfolio.csv")

SAFE_ASSETS = {"IEF", "TLT", "BND", "GLD", "SHY", "SGOV", "BIL"}

st.set_page_config(page_title="V17 Paper Trading App", layout="wide")
st.title("📊 V17 Paper Trading App")

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def pct(x):
    return "N/A" if pd.isna(x) else f"{100*x:.2f}%"

def money(x):
    return "N/A" if pd.isna(x) else f"${x:,.2f}"

state = load_csv(STATE_PATH)
log = load_csv(LOG_PATH)
summary = load_csv(SUMMARY_PATH)
v13 = load_csv(V13_PATH)

if state is None or log is None:
    st.error("Missing dashboard data. Run: python v14_paper_trading_dashboard.py --init 10000")
    st.stop()

log["date"] = pd.to_datetime(log["date"])
state["gain_loss"] = state["last_value"] / state["entry_value"] - 1
state["weight_now"] = state.groupby("strategy")["last_value"].transform(lambda x: x / x.sum())
latest = log.sort_values("date").groupby("strategy").tail(1).copy()

page = st.sidebar.radio(
    "Choose page",
    [
        "1. Live Performance",
        "2. Current Holdings",
        "3. Sector Exposure",
        "4. Leaderboard",
        "5. Monthly / Hourly History",
    ],
)

def plot_equity():
    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy, group in log.groupby("strategy"):
        group = group.sort_values("date")
        ax.plot(group["date"], group["current_value"], label=strategy)
    ax.set_title("Portfolio Value Over Time")
    ax.set_xlabel("Date")
    ax.set_ylabel("Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

def plot_drawdown():
    fig, ax = plt.subplots(figsize=(12, 4))
    for strategy, group in log.groupby("strategy"):
        g = group.sort_values("date").copy()
        g["peak"] = g["current_value"].cummax()
        g["drawdown"] = g["current_value"] / g["peak"] - 1
        ax.plot(g["date"], g["drawdown"] * 100, label=strategy)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

if page == "1. Live Performance":
    st.header("1. Live Performance")
    cols = st.columns(len(latest))
    for col, (_, row) in zip(cols, latest.iterrows()):
        col.metric(row["strategy"], money(row["current_value"]), pct(row["return"]))

    st.subheader("Alpha vs SPY")
    latest_map = latest.set_index("strategy")
    a1, a2 = st.columns(2)
    if "V12_champion" in latest_map.index and "SPY_benchmark_for_V12" in latest_map.index:
        alpha = latest_map.loc["V12_champion", "return"] - latest_map.loc["SPY_benchmark_for_V12", "return"]
        a1.metric("V12 Alpha vs SPY", pct(alpha))
    if "V13_defensive" in latest_map.index and "SPY_benchmark_for_V13" in latest_map.index:
        alpha = latest_map.loc["V13_defensive", "return"] - latest_map.loc["SPY_benchmark_for_V13", "return"]
        a2.metric("V13 Alpha vs SPY", pct(alpha))

    st.subheader("Equity Curve")
    plot_equity()
    st.subheader("Drawdown")
    plot_drawdown()

elif page == "2. Current Holdings":
    st.header("2. Current Holdings")
    strategy = st.selectbox("Choose strategy", sorted(state["strategy"].unique()))
    positions = state[state["strategy"] == strategy].copy().sort_values("last_value", ascending=False)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(
            positions[["ticker","target_weight","weight_now","entry_price","last_price","shares","entry_value","last_value","gain_loss"]],
            use_container_width=True,
        )
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(positions["last_value"], labels=positions["ticker"], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{strategy} Allocation")
        st.pyplot(fig)

    st.subheader("Top Winners and Losers")
    wl = positions[~positions["strategy"].str.contains("SPY_benchmark", na=False)]
    wcol, lcol = st.columns(2)
    with wcol:
        st.markdown("### Winners")
        st.dataframe(wl.sort_values("gain_loss", ascending=False)[["ticker","gain_loss","last_value"]].head(10), use_container_width=True)
    with lcol:
        st.markdown("### Losers")
        st.dataframe(wl.sort_values("gain_loss", ascending=True)[["ticker","gain_loss","last_value"]].head(10), use_container_width=True)

elif page == "3. Sector Exposure":
    st.header("3. Sector Exposure")
    strategy = st.selectbox("Choose strategy", sorted(state["strategy"].unique()))
    positions = state[state["strategy"] == strategy].copy()

    def bucket(ticker):
        if ticker == "GLD":
            return "Gold"
        if ticker in {"IEF", "TLT", "BND"}:
            return "Bonds"
        if ticker in {"SHY", "SGOV", "BIL"}:
            return "Cash-like"
        return "Stocks"

    positions["asset_bucket"] = positions["ticker"].apply(bucket)
    bucket_exp = positions.groupby("asset_bucket")["last_value"].sum()
    bucket_exp = bucket_exp / bucket_exp.sum()

    c1, c2 = st.columns(2)
    with c1:
        st.subheader("Asset Bucket Exposure")
        st.dataframe(bucket_exp.reset_index().rename(columns={"last_value": "weight"}), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(bucket_exp.values, labels=bucket_exp.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Stocks / Bonds / Gold / Cash")
        st.pyplot(fig)

    if strategy == "V13_defensive" and v13 is not None and "sector" in v13.columns:
        st.subheader("V13 Sector Exposure")
        sector = v13.copy()
        sector["target_weight"] = sector["target_weight"].astype(float)
        sector_exp = sector.groupby("sector")["target_weight"].sum().sort_values(ascending=False)
        st.dataframe(sector_exp.reset_index(), use_container_width=True)
        fig, ax = plt.subplots(figsize=(10, 4))
        sector_exp.plot(kind="bar", ax=ax)
        ax.set_title("V13 Sector Exposure")
        ax.set_ylabel("Weight")
        ax.grid(True)
        st.pyplot(fig)

elif page == "4. Leaderboard":
    st.header("4. Leaderboard")
    board = latest.sort_values("return", ascending=False).copy()
    board["return_pct"] = board["return"] * 100
    st.dataframe(board[["strategy", "current_value", "return_pct", "num_positions"]], use_container_width=True)
    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(board["strategy"], board["return_pct"])
    ax.set_title("Strategy Return Leaderboard")
    ax.set_ylabel("Return %")
    ax.grid(True)
    st.pyplot(fig)
    if summary is not None:
        st.subheader("Risk Summary")
        st.dataframe(summary, use_container_width=True)

elif page == "5. Monthly / Hourly History":
    st.header("5. Monthly / Hourly History")
    st.subheader("Full Log")
    st.dataframe(log.sort_values(["date","strategy"], ascending=[False, True]), use_container_width=True)
    st.subheader("Equity Curve")
    plot_equity()
    st.download_button("Download daily_log.csv", log.to_csv(index=False), "daily_log.csv")
    if summary is not None:
        st.download_button("Download dashboard_summary.csv", summary.to_csv(index=False), "dashboard_summary.csv")
    st.subheader("Raw State")
    st.dataframe(state, use_container_width=True)
    st.download_button("Download portfolio_state.csv", state.to_csv(index=False), "portfolio_state.csv")