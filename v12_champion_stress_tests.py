"""
v12_champion_stress_tests.py

Stress test for the current champion model:
    score = 60% momentum + 40% Bayesian confidence
    select top N stocks monthly
    test different weighting methods and transaction costs

Tests:
    A. Top 10 vs Top 20 vs Top 30
    B. Equal weight vs Risk parity vs Minimum variance vs Bayesian weight
    C. Transaction costs: 0.05%, 0.10%, 0.20%
    D. Year-by-year performance: 2020-2026
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
OUT_DIR = "v12_results"
PAPER_DIR = "paper_trading"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        table = pd.read_html(StringIO(html))[0]
        return [x.replace(".", "-") for x in table["Symbol"].tolist()]
    except Exception as e:
        print("Could not fetch S&P 500 list. Using fallback universe.")
        print(e)
        return [
            "AAPL","MSFT","NVDA","AMZN","META","GOOGL","GOOG","BRK-B","JPM","LLY",
            "AVGO","XOM","UNH","V","MA","COST","PG","JNJ","HD","ABBV","BAC","KO",
            "NFLX","CRM","AMD","PEP","TMO","WMT","CSCO","MCD","ABT","DIS","INTU",
            "QCOM","IBM","GE","CAT","AMAT","TXN","NOW","GS","ISRG","UBER","BKNG",
            "PFE","RTX","SPGI","LOW","HON","AXP","NEE","BLK","ELV","SYK","LMT",
            "TJX","VRTX","ADBE","DE","PANW","MU","INTC","ON","STX","WDC","GLW"
        ]


def download_prices(tickers):
    all_tickers = sorted(set(tickers + [BENCHMARK, VIX]))
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


def make_panel(close):
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    daily_ret = close.pct_change()

    rows = []
    for ticker in close.columns:
        if ticker in [BENCHMARK, VIX]:
            continue
        df = pd.DataFrame(index=monthly_close.index)
        df["ticker"] = ticker
        df["date"] = df.index
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


def cap_and_normalize(w, max_weight):
    w = pd.Series(w, dtype=float).clip(lower=0.0)
    if w.sum() <= 1e-12:
        w[:] = 1.0 / len(w)
    w = w / w.sum()
    for _ in range(30):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if under.any() and w[under].sum() > 1e-12:
            w[under] += excess * w[under] / w[under].sum()
    w = w.clip(lower=0.0, upper=max_weight)
    return w / w.sum()


def min_variance_weights(returns_window, max_weight):
    tickers = returns_window.columns
    cov = returns_window.cov().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    try:
        cov_mat = cov.values + np.eye(len(tickers)) * 1e-6
        inv = np.linalg.pinv(cov_mat)
        ones = np.ones(len(tickers))
        raw = inv @ ones
        raw = np.maximum(raw, 0.0)
        if raw.sum() <= 1e-12:
            raise ValueError("bad minvar")
        w = pd.Series(raw / raw.sum(), index=tickers)
    except Exception:
        var = returns_window.var().replace(0, np.nan)
        inv_var = 1.0 / var
        w = inv_var / inv_var.sum()
    return cap_and_normalize(w, max_weight)


def get_weights(selected, monthly_ret, date, method, top_n):
    tickers = selected["ticker"].tolist()
    max_weight = min(0.20, max(0.10, 2.0 / top_n))
    if method == "equal_weight":
        return pd.Series(1.0 / len(tickers), index=tickers)
    if method == "risk_parity":
        inv_vol = 1.0 / (selected.set_index("ticker")["vol_6m"].abs() + 1e-6)
        return cap_and_normalize(inv_vol, max_weight)
    if method == "bayesian_weight":
        raw = selected.set_index("ticker")["score_bayesian"].clip(lower=0.0) ** 2
        return cap_and_normalize(raw, max_weight)
    if method == "min_variance":
        idx = monthly_ret.index.get_loc(date)
        start = max(0, idx - 12)
        window = monthly_ret.iloc[start:idx][tickers].dropna()
        if len(window) < 6:
            return pd.Series(1.0 / len(tickers), index=tickers)
        return min_variance_weights(window, max_weight)
    raise ValueError(method)


def backtest(panel, monthly_ret, top_n=20, weight_method="equal_weight", cost_rate=0.001, start_date=None, end_date=None):
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
        selected = group.sort_values("score_champion", ascending=False).head(top_n).copy()
        selected = selected[selected["ticker"].isin(monthly_ret.columns)]
        if selected.empty:
            continue
        weights = get_weights(selected, monthly_ret, date, weight_method, top_n)
        valid = [t for t in weights.index if t in monthly_ret.columns]
        if prev_weights is None:
            turnover = 1.0
        else:
            all_names = sorted(set(prev_weights.index).union(set(weights.index)))
            old = prev_weights.reindex(all_names).fillna(0.0)
            new = weights.reindex(all_names).fillna(0.0)
            turnover = float(np.sum(np.abs(new - old)))
        port_ret_before_cost = float(np.sum(monthly_ret.loc[next_date, valid].values * weights.loc[valid].values))
        cost = cost_rate * turnover
        port_ret = port_ret_before_cost - cost
        spy_ret = float(monthly_ret.loc[next_date, BENCHMARK])
        rows.append({
            "date": next_date,
            "strategy_return": port_ret,
            "spy_return": spy_ret,
            "turnover": turnover,
            "cost": cost,
            "picks": ",".join(valid),
            "weights": ";".join([f"{t}:{weights.loc[t]:.4f}" for t in valid]),
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
        return {k: np.nan for k in ["strategy_total","strategy_annual","strategy_vol","strategy_sharpe","strategy_mdd","spy_total","spy_annual","spy_vol","spy_sharpe","spy_mdd","avg_turnover","avg_cost"]}
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
        "strategy_total": s[0], "strategy_annual": s[1], "strategy_vol": s[2], "strategy_sharpe": s[3], "strategy_mdd": s[4],
        "spy_total": b[0], "spy_annual": b[1], "spy_vol": b[2], "spy_sharpe": b[3], "spy_mdd": b[4],
        "avg_turnover": bt["turnover"].mean(), "avg_cost": bt["cost"].mean(),
    }


def run_experiment(panel, monthly_ret, name, top_n, weight_method, cost_rate, start_date=None, end_date=None):
    bt = backtest(panel, monthly_ret, top_n=top_n, weight_method=weight_method, cost_rate=cost_rate, start_date=start_date, end_date=end_date)
    stats = performance(bt)
    stats.update({"name": name, "top_n": top_n, "weight_method": weight_method, "cost_rate": cost_rate, "start_date": start_date, "end_date": end_date})
    temp = bt.copy() if not bt.empty else pd.DataFrame()
    if not temp.empty:
        temp["name"] = name
    return stats, temp


def make_paper_portfolio(panel, monthly_ret):
    latest_date = panel["date"].max()
    latest = panel[panel["date"] == latest_date].sort_values("score_champion", ascending=False).head(20).copy()
    weights = get_weights(latest, monthly_ret, latest_date, "equal_weight", 20)
    out = pd.DataFrame({"ticker": weights.index, "target_weight": weights.values}).sort_values("target_weight", ascending=False)
    out["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    out["model"] = "v12_champion_momentum_bayesian_equal_weight"
    out.to_csv(os.path.join(PAPER_DIR, "v12_today_portfolio.csv"), index=False)
    return out


def main():
    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print("Universe:", len(tickers))
    close = download_prices(tickers)
    print("Downloaded columns:", len(close.columns))
    panel, monthly_ret = make_panel(close)
    panel.to_csv(os.path.join(OUT_DIR, "v12_panel.csv"), index=False)
    dates = sorted(panel["date"].unique())
    holdout_start = dates[int(len(dates) * 0.70)]
    print("Holdout start:", holdout_start)
    summary_rows = []
    all_backtests = []
    print("\n=== Test A: Top N ===")
    for top_n in [10, 20, 30]:
        stats, bt = run_experiment(panel, monthly_ret, f"test_a_top_{top_n}", top_n, "equal_weight", 0.001, holdout_start)
        summary_rows.append(stats)
        if not bt.empty: all_backtests.append(bt)
    print("\n=== Test B: Weighting methods ===")
    for method in ["equal_weight", "risk_parity", "min_variance", "bayesian_weight"]:
        stats, bt = run_experiment(panel, monthly_ret, f"test_b_{method}", 20, method, 0.001, holdout_start)
        summary_rows.append(stats)
        if not bt.empty: all_backtests.append(bt)
    print("\n=== Test C: Transaction costs ===")
    for cost in [0.0005, 0.0010, 0.0020]:
        stats, bt = run_experiment(panel, monthly_ret, f"test_c_cost_{cost}", 20, "equal_weight", cost, holdout_start)
        summary_rows.append(stats)
        if not bt.empty: all_backtests.append(bt)
    print("\n=== Test D: Year by year ===")
    for year in range(2020, 2027):
        stats, bt = run_experiment(panel, monthly_ret, f"test_d_year_{year}", 20, "equal_weight", 0.001, f"{year}-01-01", f"{year}-12-31")
        summary_rows.append(stats)
        if not bt.empty: all_backtests.append(bt)
    summary = pd.DataFrame(summary_rows)
    summary.to_csv(os.path.join(OUT_DIR, "v12_summary.csv"), index=False)
    summary[summary["name"].str.startswith("test_a")].to_csv(os.path.join(OUT_DIR, "v12_test_a_topn.csv"), index=False)
    summary[summary["name"].str.startswith("test_b")].to_csv(os.path.join(OUT_DIR, "v12_test_b_weights.csv"), index=False)
    summary[summary["name"].str.startswith("test_c")].to_csv(os.path.join(OUT_DIR, "v12_test_c_costs.csv"), index=False)
    summary[summary["name"].str.startswith("test_d")].to_csv(os.path.join(OUT_DIR, "v12_test_d_year_by_year.csv"), index=False)
    if all_backtests:
        pd.concat(all_backtests).to_csv(os.path.join(OUT_DIR, "v12_all_backtests.csv"))
    paper = make_paper_portfolio(panel, monthly_ret)
    print("\n========== V12 STRESS TEST SUMMARY ==========")
    print(summary)
    print("\n========== CURRENT PAPER PORTFOLIO ==========")
    print(paper)
    print("\nSaved outputs in:", OUT_DIR)
    print("Paper portfolio:", os.path.join(PAPER_DIR, "v12_today_portfolio.csv"))


if __name__ == "__main__":
    main()
