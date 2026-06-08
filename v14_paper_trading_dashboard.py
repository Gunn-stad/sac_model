"""
v14_paper_trading_dashboard.py

V14 Paper Trading Dashboard

Goal:
    Track paper trading for 30-90 days.

It tracks:
    - V12 champion portfolio
    - V13 defensive portfolio
    - SPY benchmark

It creates:
    paper_dashboard/portfolio_state.csv
    paper_dashboard/daily_log.csv
    paper_dashboard/dashboard_summary.csv
    paper_dashboard/equity_curve.png

How to use:
    1. Run V12 and V13 first so these files exist:
        paper_trading/v12_today_portfolio.csv
        paper_trading/v13_today_portfolio.csv

    2. Run:
        python v14_paper_trading_dashboard.py --init 10000

    3. Every day or week run:
        python v14_paper_trading_dashboard.py --update

    4. Every month, after new V12/V13 portfolio files are generated:
        python v14_paper_trading_dashboard.py --rebalance
"""

import argparse
import os
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf
import matplotlib.pyplot as plt

DASH_DIR = "paper_dashboard"
PAPER_DIR = "paper_trading"

STATE_PATH = os.path.join(DASH_DIR, "portfolio_state.csv")
LOG_PATH = os.path.join(DASH_DIR, "daily_log.csv")
SUMMARY_PATH = os.path.join(DASH_DIR, "dashboard_summary.csv")
CHART_PATH = os.path.join(DASH_DIR, "equity_curve.png")

V12_PORTFOLIO = os.path.join(PAPER_DIR, "v12_today_portfolio.csv")
V13_PORTFOLIO = os.path.join(PAPER_DIR, "v13_today_portfolio.csv")

BENCHMARK = "SPY"

os.makedirs(DASH_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def today_str():
    return datetime.utcnow().strftime("%Y-%m-%d")


def now_str():
    return datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")


def read_portfolio(path, name):
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"Missing {path}. Run the model that creates this portfolio first."
        )

    df = pd.read_csv(path)

    if "ticker" not in df.columns:
        raise ValueError(f"{path} needs a ticker column.")

    if "target_weight" not in df.columns:
        raise ValueError(f"{path} needs a target_weight column.")

    out = df[["ticker", "target_weight"]].copy()
    out["strategy"] = name
    out["target_weight"] = out["target_weight"].astype(float)
    out["target_weight"] = out["target_weight"] / out["target_weight"].sum()

    return out


def latest_prices(tickers):
    tickers = sorted(set(tickers))
    data = yf.download(
        tickers,
        period="10d",
        interval="1d",
        auto_adjust=True,
        progress=False,
        group_by="column",
        threads=True,
    )

    if len(tickers) == 1:
        close = pd.Series({tickers[0]: float(data["Close"].dropna().iloc[-1])})
        return close

    close = data["Close"].ffill().dropna(how="all")
    return close.iloc[-1].dropna()


def create_positions(portfolio, capital):
    tickers = portfolio["ticker"].tolist()
    prices = latest_prices(tickers)

    rows = []
    for _, row in portfolio.iterrows():
        ticker = row["ticker"]
        weight = float(row["target_weight"])
        if ticker not in prices.index:
            print(f"[warning] no price for {ticker}, skipping")
            continue

        price = float(prices.loc[ticker])
        dollars = capital * weight
        shares = dollars / price

        rows.append({
            "strategy": row["strategy"],
            "ticker": ticker,
            "target_weight": weight,
            "entry_price": price,
            "shares": shares,
            "entry_value": dollars,
            "last_price": price,
            "last_value": dollars,
            "created_at": now_str(),
            "updated_at": now_str(),
        })

    return pd.DataFrame(rows)


def init_dashboard(capital):
    print("Initializing paper dashboard...")

    v12 = read_portfolio(V12_PORTFOLIO, "V12_champion")
    v13 = read_portfolio(V13_PORTFOLIO, "V13_defensive")

    state = pd.concat([
        create_positions(v12, capital),
        create_positions(v13, capital),
    ], ignore_index=True)

    # Add SPY benchmark positions for both strategies.
    spy_price = float(latest_prices([BENCHMARK]).loc[BENCHMARK])

    bench_rows = []
    for strategy in ["SPY_benchmark_for_V12", "SPY_benchmark_for_V13"]:
        bench_rows.append({
            "strategy": strategy,
            "ticker": BENCHMARK,
            "target_weight": 1.0,
            "entry_price": spy_price,
            "shares": capital / spy_price,
            "entry_value": capital,
            "last_price": spy_price,
            "last_value": capital,
            "created_at": now_str(),
            "updated_at": now_str(),
        })

    state = pd.concat([state, pd.DataFrame(bench_rows)], ignore_index=True)
    state.to_csv(STATE_PATH, index=False)

    update_dashboard()
    print(f"Created {STATE_PATH}")


def update_prices_in_state(state):
    tickers = state["ticker"].unique().tolist()
    prices = latest_prices(tickers)

    state = state.copy()
    for i, row in state.iterrows():
        ticker = row["ticker"]
        if ticker in prices.index:
            price = float(prices.loc[ticker])
            state.loc[i, "last_price"] = price
            state.loc[i, "last_value"] = price * float(row["shares"])

    state["updated_at"] = now_str()
    return state


def compute_summary(state):
    rows = []
    for strategy, group in state.groupby("strategy"):
        value = float(group["last_value"].sum())
        initial = float(group["entry_value"].sum())
        ret = value / initial - 1.0

        rows.append({
            "date": today_str(),
            "timestamp": now_str(),
            "strategy": strategy,
            "initial_value": initial,
            "current_value": value,
            "return": ret,
            "num_positions": len(group),
        })

    return pd.DataFrame(rows)


def append_log(summary):
    if os.path.exists(LOG_PATH):
        old = pd.read_csv(LOG_PATH)
        combined = pd.concat([old, summary], ignore_index=True)
    else:
        combined = summary.copy()


    combined.to_csv(LOG_PATH, index=False)


def make_dashboard_summary():
    if not os.path.exists(LOG_PATH):
        return pd.DataFrame()

    log = pd.read_csv(LOG_PATH)
    rows = []

    for strategy, group in log.groupby("strategy"):
        group = group.sort_values("date")
        values = group["current_value"].astype(float)
        returns = values.pct_change().dropna()

        total_return = values.iloc[-1] / values.iloc[0] - 1.0 if len(values) > 1 else group["return"].iloc[-1]
        max_drawdown = (values / values.cummax() - 1.0).min()
        vol = returns.std() * np.sqrt(252) if len(returns) > 1 else np.nan
        ann_return = (1 + total_return) ** (252 / max(len(group), 1)) - 1 if len(group) > 1 else np.nan
        sharpe = ann_return / vol if vol and vol > 0 else np.nan

        rows.append({
            "strategy": strategy,
            "days_tracked": len(group),
            "initial_value": values.iloc[0],
            "current_value": values.iloc[-1],
            "total_return": total_return,
            "max_drawdown": max_drawdown,
            "annualized_vol": vol,
            "annualized_return": ann_return,
            "sharpe_estimate": sharpe,
        })

    out = pd.DataFrame(rows)
    out.to_csv(SUMMARY_PATH, index=False)
    return out


def plot_equity_curve():
    if not os.path.exists(LOG_PATH):
        return

    log = pd.read_csv(LOG_PATH)
    log["date"] = pd.to_datetime(log["date"])

    plt.figure(figsize=(10, 5))
    for strategy, group in log.groupby("strategy"):
        group = group.sort_values("date")
        plt.plot(group["date"], group["current_value"], label=strategy)

    plt.title("Paper Trading Dashboard: Portfolio Value")
    plt.xlabel("Date")
    plt.ylabel("Value")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(CHART_PATH)
    plt.close()


def update_dashboard():
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError("No portfolio_state.csv found. Run --init first.")

    state = pd.read_csv(STATE_PATH)
    state = update_prices_in_state(state)
    state.to_csv(STATE_PATH, index=False)

    summary = compute_summary(state)
    append_log(summary)

    dashboard = make_dashboard_summary()
    plot_equity_curve()

    print("\n========== TODAY SUMMARY ==========")
    print(summary)

    print("\n========== DASHBOARD SUMMARY ==========")
    print(dashboard)

    print("\nSaved:")
    print(STATE_PATH)
    print(LOG_PATH)
    print(SUMMARY_PATH)
    print(CHART_PATH)


def rebalance_strategy(strategy_name, portfolio_path, capital_source_strategy):
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError("No state file. Run --init first.")

    state = pd.read_csv(STATE_PATH)
    state = update_prices_in_state(state)

    current_value = float(state[state["strategy"] == capital_source_strategy]["last_value"].sum())
    portfolio = read_portfolio(portfolio_path, strategy_name)
    new_positions = create_positions(portfolio, current_value)

    state = state[state["strategy"] != strategy_name]
    state = pd.concat([state, new_positions], ignore_index=True)
    state.to_csv(STATE_PATH, index=False)


def rebalance_dashboard():
    print("Rebalancing V12 and V13 from latest paper_trading portfolio files...")

    rebalance_strategy("V12_champion", V12_PORTFOLIO, "V12_champion")
    rebalance_strategy("V13_defensive", V13_PORTFOLIO, "V13_defensive")

    update_dashboard()


def show_positions():
    if not os.path.exists(STATE_PATH):
        raise FileNotFoundError("No state file. Run --init first.")
    state = pd.read_csv(STATE_PATH)
    print(state.sort_values(["strategy", "ticker"]))


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--init", type=float, default=None, help="Initialize dashboard with fake capital per strategy")
    parser.add_argument("--update", action="store_true", help="Update current prices and logs")
    parser.add_argument("--rebalance", action="store_true", help="Rebalance V12 and V13 using latest model outputs")
    parser.add_argument("--positions", action="store_true", help="Show current positions")

    args = parser.parse_args()

    if args.init is not None:
        init_dashboard(args.init)
    elif args.update:
        update_dashboard()
    elif args.rebalance:
        rebalance_dashboard()
    elif args.positions:
        show_positions()
    else:
        print("Use one of:")
        print("  python v14_paper_trading_dashboard.py --init 10000")
        print("  python v14_paper_trading_dashboard.py --update")
        print("  python v14_paper_trading_dashboard.py --rebalance")
        print("  python v14_paper_trading_dashboard.py --positions")


if __name__ == "__main__":
    main()