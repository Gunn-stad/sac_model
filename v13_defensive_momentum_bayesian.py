"""
v13_defensive_momentum_bayesian.py

V13 Defensive Momentum-Bayesian

Purpose:
    Take the champion V12 alpha:
        score = 60% momentum + 40% Bayesian confidence

    Then add:
        1. Sector diversification
        2. Safe assets: bonds, gold, cash-like ETF
        3. Crash/stress tests
        4. Live/paper-trading portfolio export

Tests:
    test_2_sector_diversification
    test_3_safe_asset_allocations
    test_4_crash_safe_assets
    test_5_live_portfolio_construction

Outputs:
    v13_results/v13_summary.csv
    v13_results/v13_sector_test.csv
    v13_results/v13_safe_asset_test.csv
    v13_results/v13_crash_test.csv
    v13_results/v13_all_backtests.csv
    paper_trading/v13_today_portfolio.csv
"""

import os
import warnings
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf

warnings.filterwarnings("ignore")

START = "2015-01-01"
BENCHMARK = "SPY"
VIX = "^VIX"

OUT_DIR = "v13_results"
PAPER_DIR = "paper_trading"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

SAFE_ASSETS = ["IEF", "TLT", "BND", "GLD", "SHY"]
BOND_ASSETS = ["IEF", "TLT", "BND"]
GOLD_ASSETS = ["GLD"]
CASH_ASSETS = ["SHY"]

# Fallback sector map for common names. If Wikipedia sector data is available,
# the script will use that instead.
FALLBACK_SECTORS = {
    "AAPL": "Information Technology", "MSFT": "Information Technology", "NVDA": "Information Technology",
    "AMD": "Information Technology", "INTC": "Information Technology", "MU": "Information Technology",
    "WDC": "Information Technology", "STX": "Information Technology", "DELL": "Information Technology",
    "ON": "Information Technology", "TXN": "Information Technology", "AMAT": "Information Technology",
    "LRCX": "Information Technology", "TER": "Information Technology", "GLW": "Information Technology",
    "CIEN": "Information Technology", "MPWR": "Information Technology", "KEYS": "Information Technology",
    "COHR": "Information Technology", "LITE": "Information Technology", "HPE": "Information Technology",
    "JBL": "Information Technology",

    "JPM": "Financials", "BAC": "Financials", "V": "Financials", "MA": "Financials",
    "GS": "Financials", "AXP": "Financials", "BLK": "Financials",

    "UNH": "Health Care", "LLY": "Health Care", "JNJ": "Health Care", "ABBV": "Health Care",
    "TMO": "Health Care", "ABT": "Health Care", "ISRG": "Health Care", "PFE": "Health Care",
    "ELV": "Health Care", "SYK": "Health Care", "VRTX": "Health Care",

    "XOM": "Energy",
    "COST": "Consumer Staples", "PG": "Consumer Staples", "KO": "Consumer Staples",
    "PEP": "Consumer Staples", "WMT": "Consumer Staples",
    "AMZN": "Consumer Discretionary", "TSLA": "Consumer Discretionary", "HD": "Consumer Discretionary",
    "MCD": "Consumer Discretionary", "LOW": "Consumer Discretionary", "TJX": "Consumer Discretionary",
    "BKNG": "Consumer Discretionary",
    "META": "Communication Services", "GOOGL": "Communication Services", "GOOG": "Communication Services",
    "NFLX": "Communication Services", "DIS": "Communication Services",
    "CAT": "Industrials", "GE": "Industrials", "HON": "Industrials", "RTX": "Industrials",
    "LMT": "Industrials", "NUE": "Materials", "GNRC": "Industrials",
    "NEE": "Utilities",
}


def get_sp500_tickers_and_sectors():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        table = pd.read_html(StringIO(html))[0]
        table["Symbol"] = table["Symbol"].str.replace(".", "-", regex=False)
        tickers = table["Symbol"].tolist()
        sector_map = dict(zip(table["Symbol"], table["GICS Sector"]))
        return tickers, sector_map
    except Exception as e:
        print("Could not fetch S&P500 table. Using fallback.")
        print(e)
        tickers = sorted(set(FALLBACK_SECTORS.keys()))
        return tickers, FALLBACK_SECTORS.copy()


def download_prices(tickers):
    all_tickers = sorted(set(tickers + [BENCHMARK, VIX] + SAFE_ASSETS))
    data = yf.download(
        all_tickers,
        start=START,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=True,
    )
    close = data["Close"].dropna(axis=1, how="all")
    close = close.ffill().dropna(axis=1, thresh=int(len(close) * 0.80))
    return close


def make_panel(close, sector_map):
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    daily_ret = close.pct_change()

    rows = []
    for ticker in close.columns:
        if ticker in [BENCHMARK, VIX] + SAFE_ASSETS:
            continue

        df = pd.DataFrame(index=monthly_close.index)
        df["ticker"] = ticker
        df["date"] = df.index
        df["sector"] = sector_map.get(ticker, FALLBACK_SECTORS.get(ticker, "Unknown"))

        df["ret_1m"] = monthly_close[ticker].pct_change(1)
        df["ret_3m"] = monthly_close[ticker].pct_change(3)
        df["ret_6m"] = monthly_close[ticker].pct_change(6)
        df["ret_12m"] = monthly_close[ticker].pct_change(12)
        df["vol_3m"] = daily_ret[ticker].rolling(63).std().resample("ME").last()
        df["vol_6m"] = daily_ret[ticker].rolling(126).std().resample("ME").last()
        df["next_1m_return"] = monthly_ret[ticker].shift(-1)

        rows.append(df)

    panel = pd.concat(rows).dropna()

    scored = []
    for date, group in panel.groupby("date"):
        g = group.copy()

        for col in ["ret_1m", "ret_3m", "ret_6m", "ret_12m", "vol_3m", "vol_6m"]:
            g[col + "_rank"] = g[col].rank(pct=True)

        g["score_momentum"] = (
            0.40 * g["ret_6m_rank"]
            + 0.30 * g["ret_12m_rank"]
            + 0.20 * g["ret_3m_rank"]
            - 0.10 * g["vol_3m_rank"]
        )

        market_mean = g["ret_6m"].mean()
        confidence = 1 / (1 + 30 * g["vol_6m"].fillna(g["vol_6m"].median()))
        bayes_return = confidence * g["ret_6m"] + (1 - confidence) * market_mean
        g["score_bayesian"] = bayes_return.rank(pct=True)

        g["score_champion"] = 0.60 * g["score_momentum"] + 0.40 * g["score_bayesian"]

        scored.append(g)

    return pd.concat(scored), monthly_ret


def select_with_sector_cap(group, top_n=15, max_per_sector=3):
    selected_rows = []
    sector_counts = {}

    ranked = group.sort_values("score_champion", ascending=False)

    for _, row in ranked.iterrows():
        sector = row["sector"]
        count = sector_counts.get(sector, 0)

        if count < max_per_sector:
            selected_rows.append(row)
            sector_counts[sector] = count + 1

        if len(selected_rows) >= top_n:
            break

    # If sector cap is too strict and we do not fill enough names, fill from remaining.
    if len(selected_rows) < top_n:
        selected_tickers = {r["ticker"] for r in selected_rows}
        for _, row in ranked.iterrows():
            if row["ticker"] not in selected_tickers:
                selected_rows.append(row)
                selected_tickers.add(row["ticker"])
            if len(selected_rows) >= top_n:
                break

    return pd.DataFrame(selected_rows)


def safe_asset_weights(mode):
    """
    Returns total allocation to stock bucket and safe bucket composition.
    """
    if mode == "pure_stock":
        return 1.00, {}

    if mode == "aggressive_90_10":
        return 0.90, {"IEF": 0.04, "GLD": 0.03, "SHY": 0.03}

    if mode == "balanced_80_20":
        return 0.80, {"IEF": 0.07, "TLT": 0.04, "GLD": 0.05, "SHY": 0.04}

    if mode == "defensive_60_40":
        return 0.60, {"IEF": 0.12, "TLT": 0.08, "BND": 0.08, "GLD": 0.07, "SHY": 0.05}

    raise ValueError(mode)


def build_portfolio(selected_stocks, mode, weighting="equal_weight"):
    stock_alloc, safe_w = safe_asset_weights(mode)

    tickers = selected_stocks["ticker"].tolist()

    if weighting == "equal_weight":
        stock_weights = pd.Series(stock_alloc / len(tickers), index=tickers)

    elif weighting == "bayesian_weight":
        raw = selected_stocks.set_index("ticker")["score_bayesian"].clip(lower=0.0) ** 2
        raw = raw / raw.sum()
        stock_weights = raw * stock_alloc

    elif weighting == "risk_parity":
        raw = 1.0 / (selected_stocks.set_index("ticker")["vol_6m"].abs() + 1e-6)
        raw = raw / raw.sum()
        stock_weights = raw * stock_alloc

    else:
        raise ValueError(weighting)

    safe_weights = pd.Series(safe_w, dtype=float)
    weights = pd.concat([stock_weights, safe_weights])
    weights = weights[weights > 0]
    weights = weights / weights.sum()
    return weights


def backtest(panel, monthly_ret, mode, top_n=15, max_per_sector=3, weighting="equal_weight", cost_rate=0.001, start_date=None, end_date=None):
    p = panel.copy()

    if start_date is not None:
        p = p[p["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        p = p[p["date"] <= pd.Timestamp(end_date)]

    rows = []
    prev_weights = None

    for date, group in p.groupby("date"):
        if date not in monthly_ret.index:
            continue

        idx = monthly_ret.index.get_loc(date) + 1
        if idx >= len(monthly_ret.index):
            continue

        next_date = monthly_ret.index[idx]

        selected = select_with_sector_cap(group, top_n=top_n, max_per_sector=max_per_sector)
        selected = selected[selected["ticker"].isin(monthly_ret.columns)]

        weights = build_portfolio(selected, mode=mode, weighting=weighting)
        valid = [t for t in weights.index if t in monthly_ret.columns]

        if len(valid) == 0:
            continue

        if prev_weights is None:
            turnover = 1.0
        else:
            all_names = sorted(set(prev_weights.index).union(weights.index))
            old = prev_weights.reindex(all_names).fillna(0.0)
            new = weights.reindex(all_names).fillna(0.0)
            turnover = float(np.sum(np.abs(new - old)))

        port_ret_before_cost = float(np.sum(monthly_ret.loc[next_date, valid].values * weights.loc[valid].values))
        cost = cost_rate * turnover
        port_ret = port_ret_before_cost - cost
        spy_ret = float(monthly_ret.loc[next_date, BENCHMARK])

        sector_exposure = {}
        for _, row in selected.iterrows():
            t = row["ticker"]
            sec = row["sector"]
            sector_exposure[sec] = sector_exposure.get(sec, 0.0) + float(weights.get(t, 0.0))

        rows.append({
            "date": next_date,
            "strategy_return": port_ret,
            "spy_return": spy_ret,
            "turnover": turnover,
            "cost": cost,
            "picks": ",".join(valid),
            "weights": ";".join([f"{t}:{weights.loc[t]:.4f}" for t in valid]),
            "sector_exposure": ";".join([f"{k}:{v:.4f}" for k, v in sector_exposure.items()]),
        })

        prev_weights = weights.copy()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.set_index("date")
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["spy_equity"] = (1 + out["spy_return"]).cumprod()
    return out


def performance(bt):
    if bt.empty:
        return {k: np.nan for k in [
            "strategy_total", "strategy_annual", "strategy_vol", "strategy_sharpe",
            "strategy_mdd", "spy_total", "spy_annual", "spy_vol", "spy_sharpe",
            "spy_mdd", "avg_turnover", "avg_cost"
        ]}

    def stats(r):
        r = r.dropna()
        total = (1 + r).prod() - 1
        ann = (1 + total) ** (12 / len(r)) - 1
        vol = r.std() * np.sqrt(12)
        sharpe = ann / vol if vol > 0 else np.nan
        eq = (1 + r).cumprod()
        mdd = (eq / eq.cummax() - 1).min()
        return total, ann, vol, sharpe, mdd

    s = stats(bt["strategy_return"])
    b = stats(bt["spy_return"])

    return {
        "strategy_total": s[0],
        "strategy_annual": s[1],
        "strategy_vol": s[2],
        "strategy_sharpe": s[3],
        "strategy_mdd": s[4],
        "spy_total": b[0],
        "spy_annual": b[1],
        "spy_vol": b[2],
        "spy_sharpe": b[3],
        "spy_mdd": b[4],
        "avg_turnover": bt["turnover"].mean() if "turnover" in bt else np.nan,
        "avg_cost": bt["cost"].mean() if "cost" in bt else np.nan,
    }


def run_experiment(panel, monthly_ret, name, mode, top_n=15, max_per_sector=3, weighting="equal_weight", cost_rate=0.001, start_date=None, end_date=None):
    bt = backtest(
        panel,
        monthly_ret,
        mode=mode,
        top_n=top_n,
        max_per_sector=max_per_sector,
        weighting=weighting,
        cost_rate=cost_rate,
        start_date=start_date,
        end_date=end_date,
    )
    stats = performance(bt)
    stats.update({
        "name": name,
        "mode": mode,
        "top_n": top_n,
        "max_per_sector": max_per_sector,
        "weighting": weighting,
        "cost_rate": cost_rate,
        "start_date": start_date,
        "end_date": end_date,
    })

    if not bt.empty:
        bt = bt.copy()
        bt["name"] = name

    return stats, bt


def make_paper_portfolio(panel, monthly_ret, mode="balanced_80_20", top_n=15, max_per_sector=3, weighting="equal_weight"):
    last_date = panel["date"].max()
    latest = panel[panel["date"] == last_date].copy()
    selected = select_with_sector_cap(latest, top_n=top_n, max_per_sector=max_per_sector)
    weights = build_portfolio(selected, mode=mode, weighting=weighting)

    sector_lookup = selected.set_index("ticker")["sector"].to_dict()
    asset_type = {}
    for t in weights.index:
        if t in SAFE_ASSETS:
            asset_type[t] = "safe_asset"
        else:
            asset_type[t] = "stock"

    out = pd.DataFrame({
        "ticker": weights.index,
        "target_weight": weights.values,
        "asset_type": [asset_type[t] for t in weights.index],
        "sector": [sector_lookup.get(t, "Safe Asset") for t in weights.index],
    }).sort_values("target_weight", ascending=False)

    out["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    out["model"] = "v13_defensive_momentum_bayesian"
    out.to_csv(os.path.join(PAPER_DIR, "v13_today_portfolio.csv"), index=False)
    return out


def main():
    print("Getting S&P 500 tickers and sectors...")
    tickers, sector_map = get_sp500_tickers_and_sectors()
    print("Universe:", len(tickers))

    close = download_prices(tickers)
    print("Downloaded columns:", len(close.columns))

    panel, monthly_ret = make_panel(close, sector_map)
    panel.to_csv(os.path.join(OUT_DIR, "v13_panel.csv"), index=False)

    dates = sorted(panel["date"].unique())
    holdout_start = dates[int(len(dates) * 0.70)]
    print("Holdout start:", holdout_start)

    summary_rows = []
    backtests = []

    # Test 2: Sector diversification
    print("\\n=== Test 2: Sector Diversification ===")
    for max_sector in [2, 3, 4, 99]:
        name = f"test2_sector_cap_{max_sector if max_sector < 99 else 'none'}"
        stats, bt = run_experiment(
            panel,
            monthly_ret,
            name=name,
            mode="pure_stock",
            top_n=20,
            max_per_sector=max_sector,
            weighting="equal_weight",
            cost_rate=0.001,
            start_date=holdout_start,
        )
        summary_rows.append(stats)
        if not bt.empty:
            backtests.append(bt)

    # Test 3: Safe asset allocations
    print("\\n=== Test 3: Safe Asset Allocations ===")
    for mode in ["pure_stock", "aggressive_90_10", "balanced_80_20", "defensive_60_40"]:
        stats, bt = run_experiment(
            panel,
            monthly_ret,
            name=f"test3_{mode}",
            mode=mode,
            top_n=15,
            max_per_sector=3,
            weighting="equal_weight",
            cost_rate=0.001,
            start_date=holdout_start,
        )
        summary_rows.append(stats)
        if not bt.empty:
            backtests.append(bt)

    # Test 4: Crash/stress with safe assets
    print("\\n=== Test 4: Crash Tests with Safe Assets ===")
    crash_windows = [
        ("covid_crash_2020", "2020-02-01", "2020-05-31"),
        ("bear_market_2022", "2022-01-01", "2022-12-31"),
        ("recent_period_2023_now", "2023-01-01", None),
    ]

    for win_name, start, end in crash_windows:
        for mode in ["pure_stock", "balanced_80_20", "defensive_60_40"]:
            stats, bt = run_experiment(
                panel,
                monthly_ret,
                name=f"test4_{win_name}_{mode}",
                mode=mode,
                top_n=15,
                max_per_sector=3,
                weighting="equal_weight",
                cost_rate=0.001,
                start_date=start,
                end_date=end,
            )
            summary_rows.append(stats)
            if not bt.empty:
                backtests.append(bt)

    # Test 5: Live construction candidates
    print("\\n=== Test 5: Live Portfolio Construction ===")
    live_candidates = [
        ("live_aggressive", "aggressive_90_10", 15, 3, "equal_weight"),
        ("live_balanced", "balanced_80_20", 15, 3, "equal_weight"),
        ("live_defensive", "defensive_60_40", 15, 3, "equal_weight"),
        ("live_balanced_bayesian_weight", "balanced_80_20", 15, 3, "bayesian_weight"),
    ]

    for name, mode, top_n, max_sector, weighting in live_candidates:
        stats, bt = run_experiment(
            panel,
            monthly_ret,
            name=f"test5_{name}",
            mode=mode,
            top_n=top_n,
            max_per_sector=max_sector,
            weighting=weighting,
            cost_rate=0.001,
            start_date=holdout_start,
        )
        summary_rows.append(stats)
        if not bt.empty:
            backtests.append(bt)

    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "v13_summary.csv"), index=False)

    summary[summary["name"].str.startswith("test2")].to_csv(os.path.join(OUT_DIR, "v13_sector_test.csv"), index=False)
    summary[summary["name"].str.startswith("test3")].to_csv(os.path.join(OUT_DIR, "v13_safe_asset_test.csv"), index=False)
    summary[summary["name"].str.startswith("test4")].to_csv(os.path.join(OUT_DIR, "v13_crash_test.csv"), index=False)
    summary[summary["name"].str.startswith("test5")].to_csv(os.path.join(OUT_DIR, "v13_live_construction_test.csv"), index=False)

    if backtests:
        pd.concat(backtests).to_csv(os.path.join(OUT_DIR, "v13_all_backtests.csv"))

    paper = make_paper_portfolio(
        panel,
        monthly_ret,
        mode="balanced_80_20",
        top_n=15,
        max_per_sector=3,
        weighting="equal_weight",
    )

    print("\\n========== V13 SUMMARY ==========")
    print(summary)

    print("\\n========== V13 PAPER PORTFOLIO ==========")
    print(paper)

    print("\\nSaved outputs in:", OUT_DIR)
    print("Paper portfolio:", os.path.join(PAPER_DIR, "v13_today_portfolio.csv"))


if __name__ == "__main__":
    main()