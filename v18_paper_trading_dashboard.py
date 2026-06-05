import os
from datetime import datetime
import numpy as np
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt

DASH_DIR = "paper_dashboard"
PAPER_DIR = "paper_trading"

STATE_PATH = os.path.join(DASH_DIR, "portfolio_state.csv")
LOG_PATH = os.path.join(DASH_DIR, "daily_log.csv")
SUMMARY_PATH = os.path.join(DASH_DIR, "dashboard_summary.csv")
TRADES_PATH = os.path.join(DASH_DIR, "trades_log.csv")
V13_PATH = os.path.join(PAPER_DIR, "v13_today_portfolio.csv")
V18_ICELAND_PATH = os.path.join(PAPER_DIR, "v18_iceland_today_portfolio.csv")

SAFE_ASSETS = {"IEF", "TLT", "BND", "GLD", "SHY", "SGOV", "BIL"}

st.set_page_config(page_title="Paper Trading Dashboard", layout="wide")
st.title("📊 Paper Trading Dashboard")

def load_csv(path):
    return pd.read_csv(path) if os.path.exists(path) else None

def pct(x):
    return "N/A" if pd.isna(x) else f"{100 * x:.2f}%"

def money(x):
    return "N/A" if pd.isna(x) else f"${x:,.2f}"

def calc_drawdown(values):
    values = pd.Series(values).astype(float)
    return values / values.cummax() - 1.0

def calculate_metrics(values):
    values = pd.Series(values).astype(float).dropna()
    if len(values) < 2:
        return {"return": 0.0, "volatility": np.nan, "sharpe": np.nan, "max_drawdown": 0.0}
    rets = values.pct_change().dropna()
    total = values.iloc[-1] / values.iloc[0] - 1.0
    dd = calc_drawdown(values)
    vol = rets.std() * np.sqrt(252) if len(rets) > 1 else np.nan
    ann = (1 + total) ** (252 / max(len(values), 1)) - 1
    sharpe = ann / vol if pd.notna(vol) and vol > 0 else np.nan
    return {"return": total, "volatility": vol, "sharpe": sharpe, "max_drawdown": dd.min()}

def infer_asset_bucket(ticker):
    ticker = str(ticker)
    if ticker == "GLD":
        return "Gold"
    if ticker in {"IEF", "TLT", "BND"}:
        return "Bonds"
    if ticker in {"SHY", "SGOV", "BIL"}:
        return "Cash-like"
    if ticker.endswith(".IC"):
        return "Iceland Stock"
    return "US Stock"

state = load_csv(STATE_PATH)
log = load_csv(LOG_PATH)
summary = load_csv(SUMMARY_PATH)
trades = load_csv(TRADES_PATH)
v13 = load_csv(V13_PATH)
v18_iceland = load_csv(V18_ICELAND_PATH)

if state is None or log is None:
    st.error("Missing data. Run: python v14_paper_trading_dashboard.py --init 10000")
    st.stop()

log["date"] = pd.to_datetime(log["date"])
state["gain_loss"] = state["last_value"] / state["entry_value"] - 1
state["weight_now"] = state.groupby("strategy")["last_value"].transform(lambda x: x / x.sum())
latest = log.sort_values("date").groupby("strategy").tail(1).copy()

wide = log.pivot_table(index="date", columns="strategy", values="current_value", aggfunc="last").sort_index()

page = st.sidebar.radio("Choose page", [
    "1. Portfolio / Holdings",
    "2. Performance",
    "3. Drawdowns",
    "4. Trades",
    "5. Benchmarks",
    "6. Sector / Safe Assets",
])

if page == "1. Portfolio / Holdings":
    st.header("1. Portfolio / Current Holdings")
    cols = st.columns(len(latest))
    for col, (_, row) in zip(cols, latest.iterrows()):
        col.metric(row["strategy"], money(row["current_value"]), pct(row["return"]))

    strategy = st.selectbox("Choose strategy", sorted(state["strategy"].unique()))
    positions = state[state["strategy"] == strategy].copy().sort_values("last_value", ascending=False)

    c1, c2 = st.columns([2, 1])
    with c1:
        st.dataframe(positions[["ticker","target_weight","weight_now","entry_price","last_price","shares","entry_value","last_value","gain_loss"]], use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(positions["last_value"], labels=positions["ticker"], autopct="%1.1f%%", startangle=90)
        ax.set_title(f"{strategy} Allocation")
        st.pyplot(fig)

    st.subheader("Top Winners / Losers")
    wl = positions[~positions["strategy"].str.contains("SPY_benchmark", na=False)]
    wcol, lcol = st.columns(2)
    with wcol:
        st.markdown("### Winners")
        st.dataframe(wl.sort_values("gain_loss", ascending=False)[["ticker","gain_loss","last_value"]].head(10), use_container_width=True)
    with lcol:
        st.markdown("### Losers")
        st.dataframe(wl.sort_values("gain_loss")[["ticker","gain_loss","last_value"]].head(10), use_container_width=True)

elif page == "2. Performance":
    st.header("2. Performance")
    cols = st.columns(len(latest))
    for col, (_, row) in zip(cols, latest.iterrows()):
        col.metric(row["strategy"], money(row["current_value"]), pct(row["return"]))

    st.subheader("Portfolio Value Over Time")
    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in wide.columns:
        ax.plot(wide.index, wide[strategy], label=strategy)
    ax.set_title("Equity Curve")
    ax.set_xlabel("Date")
    ax.set_ylabel("Portfolio Value")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

    st.subheader("Alpha vs SPY")
    latest_map = latest.set_index("strategy")
    alpha_rows = []
    pairs = [("V12_champion", "SPY_benchmark_for_V12"), ("V13_defensive", "SPY_benchmark_for_V13")]
    for strat, bench in pairs:
        if strat in latest_map.index and bench in latest_map.index:
            alpha_rows.append({
                "strategy": strat,
                "benchmark": bench,
                "strategy_return": latest_map.loc[strat, "return"],
                "benchmark_return": latest_map.loc[bench, "return"],
                "alpha": latest_map.loc[strat, "return"] - latest_map.loc[bench, "return"],
            })
    if alpha_rows:
        alpha_df = pd.DataFrame(alpha_rows)
        st.dataframe(alpha_df, use_container_width=True)
        fig2, ax2 = plt.subplots(figsize=(8, 4))
        ax2.bar(alpha_df["strategy"], alpha_df["alpha"] * 100)
        ax2.set_ylabel("Alpha %")
        ax2.grid(True)
        st.pyplot(fig2)
    else:
        st.info("Need V12/V13 and SPY benchmark rows to calculate alpha.")

elif page == "3. Drawdowns":
    st.header("3. Drawdowns")
    rows = []
    fig, ax = plt.subplots(figsize=(12, 5))
    for strategy in wide.columns:
        values = wide[strategy].dropna()
        dd = calc_drawdown(values)
        rows.append({"strategy": strategy, "current_drawdown": dd.iloc[-1], "max_drawdown": dd.min()})
        ax.plot(dd.index, dd.values * 100, label=strategy)
    st.dataframe(pd.DataFrame(rows), use_container_width=True)
    ax.set_title("Drawdown (%)")
    ax.set_xlabel("Date")
    ax.set_ylabel("Drawdown %")
    ax.legend()
    ax.grid(True)
    st.pyplot(fig)

elif page == "4. Trades":
    st.header("4. Trades")
    if trades is None:
        st.warning("No trades_log.csv yet. Showing synthetic initial positions.")
        synthetic = state.copy()
        synthetic["date"] = synthetic["created_at"] if "created_at" in synthetic.columns else datetime.utcnow().strftime("%Y-%m-%d")
        synthetic["action"] = "INITIAL_POSITION"
        synthetic["old_weight"] = 0.0
        synthetic["new_weight"] = synthetic["target_weight"]
        synthetic["trade_weight"] = synthetic["target_weight"]
        st.dataframe(synthetic[["date","action","strategy","ticker","old_weight","new_weight","trade_weight"]], use_container_width=True)
    else:
        st.dataframe(trades.sort_values("date", ascending=False), use_container_width=True)
        st.download_button("Download trades_log.csv", trades.to_csv(index=False), "trades_log.csv")

elif page == "5. Benchmarks":
    st.header("5. Benchmarks")
    rows = []
    for strategy in wide.columns:
        metrics = calculate_metrics(wide[strategy])
        rows.append({
            "strategy": strategy,
            "return": metrics["return"],
            "volatility_est": metrics["volatility"],
            "sharpe_est": metrics["sharpe"],
            "max_drawdown": metrics["max_drawdown"],
            "current_value": wide[strategy].dropna().iloc[-1],
        })
    bench = pd.DataFrame(rows).sort_values("return", ascending=False)
    st.dataframe(bench, use_container_width=True)

    fig, ax = plt.subplots(figsize=(10, 4))
    ax.bar(bench["strategy"], bench["return"] * 100)
    ax.set_title("Total Return")
    ax.set_ylabel("Return %")
    ax.grid(True)
    st.pyplot(fig)

elif page == "6. Sector / Safe Assets":
    st.header("6. Sector / Safe Asset Exposure")
    strategy = st.selectbox("Choose strategy", sorted(state["strategy"].unique()))
    positions = state[state["strategy"] == strategy].copy()
    positions["asset_bucket"] = positions["ticker"].apply(infer_asset_bucket)
    bucket_exp = positions.groupby("asset_bucket")["last_value"].sum()
    bucket_exp = bucket_exp / bucket_exp.sum()

    c1, c2 = st.columns(2)
    with c1:
        st.dataframe(bucket_exp.reset_index().rename(columns={"last_value": "weight"}), use_container_width=True)
    with c2:
        fig, ax = plt.subplots(figsize=(5, 5))
        ax.pie(bucket_exp.values, labels=bucket_exp.index, autopct="%1.1f%%", startangle=90)
        ax.set_title("Asset Buckets")
        st.pyplot(fig)

    if strategy == "V13_defensive" and v13 is not None and "sector" in v13.columns:
        st.subheader("V13 Sector Exposure")
        sector_exp = v13.groupby("sector")["target_weight"].sum().sort_values(ascending=False)
        st.dataframe(sector_exp.reset_index(), use_container_width=True)
        fig2, ax2 = plt.subplots(figsize=(10, 4))
        sector_exp.plot(kind="bar", ax=ax2)
        ax2.set_ylabel("Weight")
        ax2.grid(True)
        st.pyplot(fig2)

    if v18_iceland is not None:
        st.subheader("Iceland Portfolio")
        st.dataframe(v18_iceland, use_container_width=True)