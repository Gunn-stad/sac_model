import os
import warnings
from datetime import datetime

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")

START = "2015-01-01"
OUT_DIR = "v18_iceland_results"
PAPER_DIR = "paper_trading"

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)

ICELAND_TICKERS = [
    "ALVO.IC", "AMRQ.IC", "ARION.IC", "BRIM.IC", "EIK.IC", "EIM.IC",
    "FESTI.IC", "HAGA.IC", "HAMP.IC", "HEIMAR.IC", "ICEAIR.IC",
    "ICESEA.IC", "ISB.IC", "ISF.IC", "JBTM.IC", "KALD.IC",
    "KVIKA.IC", "NOVA.IC", "OCS.IC", "OLGERD.IC", "REITIR.IC",
    "SJOVA.IC", "SKAGI.IC", "SKEL.IC", "SIMINN.IC", "SVN.IC", "SYN.IC",
]

SECTOR_MAP = {
    "ALVO.IC": "Health Care",
    "AMRQ.IC": "Materials",
    "ARION.IC": "Financials",
    "BRIM.IC": "Consumer Staples",
    "EIK.IC": "Real Estate",
    "EIM.IC": "Industrials",
    "FESTI.IC": "Consumer Discretionary",
    "HAGA.IC": "Consumer Staples",
    "HAMP.IC": "Consumer Discretionary",
    "HEIMAR.IC": "Real Estate",
    "ICEAIR.IC": "Industrials",
    "ICESEA.IC": "Consumer Staples",
    "ISB.IC": "Financials",
    "ISF.IC": "Consumer Staples",
    "JBTM.IC": "Industrials",
    "KALD.IC": "Real Estate",
    "KVIKA.IC": "Financials",
    "NOVA.IC": "Communication Services",
    "OCS.IC": "Health Care",
    "OLGERD.IC": "Consumer Staples",
    "REITIR.IC": "Real Estate",
    "SJOVA.IC": "Financials",
    "SKAGI.IC": "Financials",
    "SKEL.IC": "Energy",
    "SIMINN.IC": "Communication Services",
    "SVN.IC": "Consumer Staples",
    "SYN.IC": "Communication Services",
}


def download_prices(tickers):
    print("Downloading Icelandic stocks from Yahoo Finance...")
    data = yf.download(
        tickers,
        start=START,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=True,
    )

    if "Close" not in data:
        raise RuntimeError("No close price data downloaded.")

    close = data["Close"].dropna(axis=1, how="all")
    close = close.ffill()

    min_obs = min(500, max(50, int(len(close) * 0.25)))
    close = close.dropna(axis=1, thresh=min_obs)

    print("Downloaded usable tickers:", list(close.columns))
    print("Shape:", close.shape)
    return close


def make_panel(close):
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    daily_ret = close.pct_change()

    rows = []
    for ticker in close.columns:
        df = pd.DataFrame(index=monthly_close.index)
        df["ticker"] = ticker
        df["date"] = df.index
        df["sector"] = SECTOR_MAP.get(ticker, "Unknown")

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

        g["score_iceland_champion"] = 0.60 * g["score_momentum"] + 0.40 * g["score_bayesian"]
        scored.append(g)

    return pd.concat(scored), monthly_ret


def select_with_sector_cap(group, top_n=8, max_per_sector=2):
    ranked = group.sort_values("score_iceland_champion", ascending=False)
    selected = []
    counts = {}

    for _, row in ranked.iterrows():
        sec = row["sector"]
        if counts.get(sec, 0) < max_per_sector:
            selected.append(row)
            counts[sec] = counts.get(sec, 0) + 1
        if len(selected) >= top_n:
            break

    if len(selected) < top_n:
        selected_tickers = {x["ticker"] for x in selected}
        for _, row in ranked.iterrows():
            if row["ticker"] not in selected_tickers:
                selected.append(row)
                selected_tickers.add(row["ticker"])
            if len(selected) >= top_n:
                break

    return pd.DataFrame(selected)


def get_weights(selected, method="equal_weight"):
    tickers = selected["ticker"].tolist()

    if method == "equal_weight":
        return pd.Series(1 / len(tickers), index=tickers)

    if method == "bayesian_weight":
        raw = selected.set_index("ticker")["score_bayesian"].clip(lower=0) ** 2
        if raw.sum() <= 1e-12:
            return pd.Series(1 / len(tickers), index=tickers)
        return raw / raw.sum()

    if method == "risk_parity":
        raw = 1 / (selected.set_index("ticker")["vol_6m"].abs() + 1e-6)
        return raw / raw.sum()

    raise ValueError(method)


def backtest(panel, monthly_ret, top_n=8, max_per_sector=2, method="equal_weight", start_date=None, end_date=None, cost_rate=0.002):
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

        if selected.empty:
            continue

        weights = get_weights(selected, method=method)
        valid = [t for t in weights.index if t in monthly_ret.columns]

        if prev_weights is None:
            turnover = 1.0
        else:
            names = sorted(set(prev_weights.index).union(weights.index))
            old = prev_weights.reindex(names).fillna(0.0)
            new = weights.reindex(names).fillna(0.0)
            turnover = float(np.sum(np.abs(new - old)))

        strategy_return = float(np.sum(monthly_ret.loc[next_date, valid].values * weights.loc[valid].values))
        strategy_return -= cost_rate * turnover

        market_return = float(monthly_ret.loc[next_date, monthly_ret.columns].dropna().mean())

        rows.append({
            "date": next_date,
            "strategy_return": strategy_return,
            "iceland_equal_weight_return": market_return,
            "turnover": turnover,
            "picks": ",".join(valid),
            "weights": ";".join([f"{t}:{weights.loc[t]:.4f}" for t in valid]),
        })

        prev_weights = weights.copy()

    out = pd.DataFrame(rows)
    if out.empty:
        return out

    out = out.set_index("date")
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["iceland_equal_weight_equity"] = (1 + out["iceland_equal_weight_return"]).cumprod()
    return out


def performance(bt):
    if bt.empty:
        return {
            "strategy_total": np.nan,
            "strategy_annual": np.nan,
            "strategy_vol": np.nan,
            "strategy_sharpe": np.nan,
            "strategy_mdd": np.nan,
            "benchmark_total": np.nan,
            "benchmark_annual": np.nan,
            "benchmark_vol": np.nan,
            "benchmark_sharpe": np.nan,
            "benchmark_mdd": np.nan,
            "avg_turnover": np.nan,
        }

    def stats(r):
        r = r.dropna()
        total = (1 + r).prod() - 1
        ann = (1 + total) ** (12 / len(r)) - 1 if len(r) > 0 else np.nan
        vol = r.std() * np.sqrt(12)
        sharpe = ann / vol if vol and vol > 0 else np.nan
        eq = (1 + r).cumprod()
        mdd = (eq / eq.cummax() - 1).min()
        return total, ann, vol, sharpe, mdd

    s = stats(bt["strategy_return"])
    b = stats(bt["iceland_equal_weight_return"])

    return {
        "strategy_total": s[0],
        "strategy_annual": s[1],
        "strategy_vol": s[2],
        "strategy_sharpe": s[3],
        "strategy_mdd": s[4],
        "benchmark_total": b[0],
        "benchmark_annual": b[1],
        "benchmark_vol": b[2],
        "benchmark_sharpe": b[3],
        "benchmark_mdd": b[4],
        "avg_turnover": bt["turnover"].mean(),
    }


def make_today_portfolio(panel):
    last_date = panel["date"].max()
    latest = panel[panel["date"] == last_date].copy()
    selected = select_with_sector_cap(latest, top_n=8, max_per_sector=2)
    weights = get_weights(selected, method="equal_weight")

    out = pd.DataFrame({
        "ticker": weights.index,
        "target_weight": weights.values,
        "sector": [SECTOR_MAP.get(t, "Unknown") for t in weights.index],
        "model": "v18_iceland_momentum_bayesian",
        "created_at": datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC"),
    }).sort_values("target_weight", ascending=False)

    out.to_csv(os.path.join(PAPER_DIR, "v18_iceland_today_portfolio.csv"), index=False)
    return out


def main():
    close = download_prices(ICELAND_TICKERS)
    panel, monthly_ret = make_panel(close)

    panel.to_csv(os.path.join(OUT_DIR, "v18_iceland_panel.csv"), index=False)

    dates = sorted(panel["date"].unique())
    holdout_start = dates[int(len(dates) * 0.70)]
    print("Holdout start:", holdout_start)

    experiments = [
        ("iceland_top5_equal", 5, 2, "equal_weight"),
        ("iceland_top8_equal", 8, 2, "equal_weight"),
        ("iceland_top10_equal", 10, 3, "equal_weight"),
        ("iceland_top8_bayesian", 8, 2, "bayesian_weight"),
        ("iceland_top8_risk_parity", 8, 2, "risk_parity"),
    ]

    rows = []
    backtests = []

    for name, top_n, sector_cap, method in experiments:
        print("Running:", name)
        bt = backtest(
            panel,
            monthly_ret,
            top_n=top_n,
            max_per_sector=sector_cap,
            method=method,
            start_date=holdout_start,
            cost_rate=0.002,
        )
        stats = performance(bt)
        stats.update({"name": name, "top_n": top_n, "sector_cap": sector_cap, "method": method})
        rows.append(stats)

        if not bt.empty:
            temp = bt.copy()
            temp["name"] = name
            backtests.append(temp)

    summary = pd.DataFrame(rows)
    summary.to_csv(os.path.join(OUT_DIR, "v18_iceland_summary.csv"), index=False)

    if backtests:
        pd.concat(backtests).to_csv(os.path.join(OUT_DIR, "v18_iceland_backtests.csv"))

    today = make_today_portfolio(panel)

    print("\n========== V18 ICELAND SUMMARY ==========")
    print(summary)
    print("\n========== V18 ICELAND PAPER PORTFOLIO ==========")
    print(today)
    print("\nSaved outputs in:", OUT_DIR)
    print("Paper portfolio:", os.path.join(PAPER_DIR, "v18_iceland_today_portfolio.csv"))


if __name__ == "__main__":
    main()