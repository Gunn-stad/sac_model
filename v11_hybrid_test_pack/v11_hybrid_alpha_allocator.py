import os
import warnings
from io import StringIO
from datetime import datetime

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

START = "2015-01-01"
BENCHMARK = "SPY"
VIX = "^VIX"
TOP_N = 20
MAX_WEIGHT = 0.10
OUT_DIR = "v11_results"
PAPER_DIR = "paper_trading"
MAX_SENTIMENT_TICKERS = 180

os.makedirs(OUT_DIR, exist_ok=True)
os.makedirs(PAPER_DIR, exist_ok=True)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    try:
        html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
        table = pd.read_html(StringIO(html))[0]
        return [x.replace(".", "-") for x in table["Symbol"].tolist()]
    except Exception as e:
        print("Could not fetch S&P 500 list. Using fallback.")
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
    volume = data["Volume"].reindex(columns=close.columns).ffill()
    close = close.ffill().dropna(axis=1, thresh=int(len(close) * 0.80))
    volume = volume.reindex(columns=close.columns).ffill()
    return close, volume


def collect_sentiment(tickers):
    analyzer = SentimentIntensityAnalyzer()
    rows = []
    print("Collecting Yahoo news sentiment...")
    for i, ticker in enumerate(tickers[:MAX_SENTIMENT_TICKERS]):
        try:
            news = yf.Ticker(ticker).news or []
            scores = []
            for item in news:
                text = f"{item.get('title', '')} {item.get('publisher', '')}".strip()
                if text:
                    scores.append(analyzer.polarity_scores(text)["compound"])
            rows.append({
                "ticker": ticker,
                "sentiment": float(np.mean(scores)) if scores else 0.0,
                "sentiment_abs": float(np.mean(np.abs(scores))) if scores else 0.0,
                "attention": len(scores),
            })
        except Exception:
            rows.append({"ticker": ticker, "sentiment": 0.0, "sentiment_abs": 0.0, "attention": 0})
        if (i + 1) % 25 == 0:
            print(f"Sentiment done: {i + 1}/{min(len(tickers), MAX_SENTIMENT_TICKERS)}")
    return pd.DataFrame(rows)


def compute_regime(monthly_close):
    spy = monthly_close[BENCHMARK]
    spy_ret = spy.pct_change()
    regime = pd.DataFrame(index=monthly_close.index)
    regime["spy_ret_6m"] = spy.pct_change(6)
    regime["spy_ma_10"] = spy.rolling(10).mean()
    regime["spy_above_ma"] = (spy > regime["spy_ma_10"]).astype(float)
    regime["spy_vol_6m"] = spy_ret.rolling(6).std()
    if VIX in monthly_close.columns:
        regime["vix"] = monthly_close[VIX]
        regime["vix_rank"] = regime["vix"].rolling(36).rank(pct=True)
    else:
        regime["vix_rank"] = 0.5
    regime["bull_regime"] = (
        (regime["spy_ret_6m"] > 0).astype(float)
        + regime["spy_above_ma"]
        + (regime["vix_rank"].fillna(0.5) < 0.70).astype(float)
    ) / 3.0
    regime["stress_regime"] = 1.0 - regime["bull_regime"]
    return regime


def make_panel(close, volume):
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    daily_ret = close.pct_change()
    monthly_volume = volume.resample("ME").mean()
    regime = compute_regime(monthly_close)
    spy_ret = monthly_ret[BENCHMARK]
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
        rolling_max = close[ticker].rolling(126).max().resample("ME").last()
        df["drawdown_6m"] = monthly_close[ticker] / rolling_max - 1
        df["volume_z"] = (monthly_volume[ticker] - monthly_volume[ticker].rolling(12).mean()) / monthly_volume[ticker].rolling(12).std()
        df["beta_12m"] = monthly_ret[ticker].rolling(12).cov(spy_ret) / spy_ret.rolling(12).var()
        df["next_1m_return"] = monthly_ret[ticker].shift(-1)
        for col in ["bull_regime", "stress_regime", "spy_ret_6m", "spy_vol_6m", "vix_rank"]:
            df[col] = regime[col]
        rows.append(df)
    panel = pd.concat(rows).dropna()
    return add_alpha_scores(panel), monthly_ret


def add_alpha_scores(panel):
    out = []
    for date, group in panel.groupby("date"):
        g = group.copy()
        rank_cols = ["ret_1m","ret_3m","ret_6m","ret_12m","vol_3m","vol_6m","drawdown_6m","volume_z","beta_12m"]
        for col in rank_cols:
            g[col + "_rank"] = g[col].rank(pct=True)
        g["score_momentum"] = 0.40*g["ret_6m_rank"] + 0.30*g["ret_12m_rank"] + 0.20*g["ret_3m_rank"] - 0.10*g["vol_3m_rank"]
        market_mean = g["ret_6m"].mean()
        confidence = 1 / (1 + 30*g["vol_6m"].fillna(g["vol_6m"].median()))
        bayes = confidence*g["ret_6m"] + (1-confidence)*market_mean
        g["score_bayesian"] = bayes.rank(pct=True)
        g["score_behavioral"] = 0.35*g["ret_6m_rank"] + 0.25*g["ret_12m_rank"] - 0.25*g["ret_1m_rank"] - 0.15*g["drawdown_6m_rank"] + 0.10*g["volume_z_rank"]
        out.append(g)
    return pd.concat(out)


def add_sentiment_scores(panel, sentiment_df):
    panel = panel.merge(sentiment_df, on="ticker", how="left")
    panel["sentiment"] = panel["sentiment"].fillna(0.0)
    panel["sentiment_abs"] = panel["sentiment_abs"].fillna(0.0)
    panel["attention"] = panel["attention"].fillna(0.0)
    out = []
    for date, group in panel.groupby("date"):
        g = group.copy()
        g["sentiment_rank"] = g["sentiment"].rank(pct=True)
        g["attention_rank"] = g["attention"].rank(pct=True)
        g["sentiment_abs_rank"] = g["sentiment_abs"].rank(pct=True)
        g["score_sentiment"] = 0.70*g["sentiment_rank"] + 0.20*g["attention_rank"] + 0.10*g["sentiment_abs_rank"]
        bull = float(g["bull_regime"].iloc[0])
        stress = 1.0 - bull
        g["score_v11_alpha"] = (
            0.42*g["score_momentum"]
            + 0.33*g["score_bayesian"]
            + 0.10*g["score_sentiment"]
            + 0.10*g["score_behavioral"]
            - 0.10*stress*g["vol_6m_rank"]
            + 0.05*bull*g["ret_12m_rank"]
        )
        out.append(g)
    return pd.concat(out)


def cap_and_normalize(weights, max_weight=MAX_WEIGHT):
    w = pd.Series(weights, dtype=float).clip(lower=0.0)
    if w.sum() <= 1e-12:
        w[:] = 1.0 / len(w)
    w = w / w.sum()
    for _ in range(20):
        over = w > max_weight
        if not over.any():
            break
        excess = (w[over] - max_weight).sum()
        w[over] = max_weight
        under = ~over
        if under.any() and w[under].sum() > 0:
            w[under] += excess * w[under] / w[under].sum()
    w = w.clip(lower=0.0, upper=max_weight)
    return w / w.sum()


def get_weights(group, method):
    g = group.copy()
    if method == "equal_weight":
        return pd.Series(1.0 / len(g), index=g["ticker"])
    if method == "volatility_weight":
        return cap_and_normalize(pd.Series((1.0/(g["vol_6m"].abs()+1e-6)).values, index=g["ticker"]))
    if method == "bayesian_weight":
        return cap_and_normalize(pd.Series((g["score_bayesian"].clip(lower=0.0)**2).values, index=g["ticker"]))
    if method == "v11_hybrid_weight":
        base = pd.Series(1.0 / len(g), index=g["ticker"])
        alpha = g["score_v11_alpha"].rank(pct=True)
        vol_penalty = g["vol_6m"].rank(pct=True)
        tilt = alpha - alpha.mean()
        raw = base.values * (1.0 + 0.75*tilt.values - 0.25*vol_penalty.values)
        return cap_and_normalize(pd.Series(raw, index=g["ticker"]))
    raise ValueError(method)


def backtest(panel, monthly_ret, score_col, allocation_method, start_date=None, end_date=None, top_n=TOP_N):
    p = panel.copy()
    if start_date is not None:
        p = p[p["date"] >= pd.Timestamp(start_date)]
    if end_date is not None:
        p = p[p["date"] <= pd.Timestamp(end_date)]
    rows = []
    for date, group in p.groupby("date"):
        if date not in monthly_ret.index:
            continue
        idx = monthly_ret.index.get_loc(date) + 1
        if idx >= len(monthly_ret.index):
            continue
        next_date = monthly_ret.index[idx]
        selected = group.sort_values(score_col, ascending=False).head(top_n).copy()
        selected = selected[selected["ticker"].isin(monthly_ret.columns)]
        if len(selected) == 0:
            continue
        weights = get_weights(selected, allocation_method)
        valid = [t for t in weights.index if t in monthly_ret.columns]
        port_ret = float(np.sum(monthly_ret.loc[next_date, valid].values * weights.loc[valid].values))
        port_ret -= 0.001 * (1.0 / 12.0)
        spy_ret = float(monthly_ret.loc[next_date, BENCHMARK])
        rows.append({
            "date": next_date,
            "strategy_return": port_ret,
            "spy_return": spy_ret,
            "picks": ",".join(valid),
            "weights": ";".join([f"{t}:{weights.loc[t]:.4f}" for t in valid]),
        })
    out = pd.DataFrame(rows)
    if out.empty:
        return out
    out = out.set_index("date")
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["spy_equity"] = (1 + out["spy_return"]).cumprod()
    return out


def performance(bt):
    if bt.empty:
        return {k: np.nan for k in ["strategy_total","strategy_annual","strategy_vol","strategy_sharpe","strategy_mdd","spy_total","spy_annual","spy_vol","spy_sharpe","spy_mdd"]}
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
    }


def test_1_and_2(panel, monthly_ret):
    dates = sorted(panel["date"].unique())
    start_test = dates[int(len(dates) * 0.70)]
    experiments = [
        ("test1_v9_momentum_equal", "score_momentum", "equal_weight"),
        ("test1_v9_bayesian_equal", "score_bayesian", "equal_weight"),
        ("test1_v11_alpha_equal", "score_v11_alpha", "equal_weight"),
        ("test2_v11_alpha_vol_weight", "score_v11_alpha", "volatility_weight"),
        ("test2_v11_alpha_bayes_weight", "score_v11_alpha", "bayesian_weight"),
        ("test2_v11_hybrid_weight", "score_v11_alpha", "v11_hybrid_weight"),
    ]
    rows, backtests = [], []
    for name, score_col, alloc in experiments:
        print("Running", name)
        bt = backtest(panel, monthly_ret, score_col, alloc, start_date=start_test)
        stats = performance(bt)
        stats["name"] = name
        stats["test"] = "test_1_2"
        rows.append(stats)
        temp = bt.copy()
        temp["name"] = name
        backtests.append(temp)
    return pd.DataFrame(rows), pd.concat(backtests)


def test_3_walk_forward(panel, monthly_ret):
    windows = [
        ("wf_2020", "2020-01-01", "2020-12-31"),
        ("wf_2021", "2021-01-01", "2021-12-31"),
        ("wf_2022", "2022-01-01", "2022-12-31"),
        ("wf_2023", "2023-01-01", "2023-12-31"),
        ("wf_2024", "2024-01-01", "2024-12-31"),
        ("wf_2025_2026", "2025-01-01", None),
    ]
    rows = []
    for name, start, end in windows:
        print("Running walk-forward", name)
        bt = backtest(panel, monthly_ret, "score_v11_alpha", "v11_hybrid_weight", start_date=start, end_date=end)
        stats = performance(bt)
        stats["name"] = name
        stats["test"] = "walk_forward"
        rows.append(stats)
    return pd.DataFrame(rows)


def test_4_crash_tests(panel, monthly_ret):
    windows = [
        ("covid_crash_2020", "2020-02-01", "2020-05-31"),
        ("inflation_bear_2022", "2022-01-01", "2022-12-31"),
        ("recent_period", "2023-01-01", None),
    ]
    rows = []
    for name, start, end in windows:
        print("Running crash/stress", name)
        bt = backtest(panel, monthly_ret, "score_v11_alpha", "v11_hybrid_weight", start_date=start, end_date=end)
        stats = performance(bt)
        stats["name"] = name
        stats["test"] = "crash_stress"
        rows.append(stats)
    return pd.DataFrame(rows)


def test_5_paper_portfolio(panel):
    last_date = panel["date"].max()
    latest = panel[panel["date"] == last_date].copy()
    selected = latest.sort_values("score_v11_alpha", ascending=False).head(TOP_N)
    weights = get_weights(selected, "v11_hybrid_weight")
    out = pd.DataFrame({"ticker": weights.index, "target_weight": weights.values}).sort_values("target_weight", ascending=False)
    out["created_at"] = datetime.utcnow().strftime("%Y-%m-%d %H:%M:%S UTC")
    out["model"] = "v11_hybrid_alpha_allocator"
    out.to_csv(os.path.join(PAPER_DIR, "v11_today_portfolio.csv"), index=False)
    return out


def main():
    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print("Universe:", len(tickers))
    close, volume = download_prices(tickers)
    usable = [t for t in close.columns if t not in [BENCHMARK, VIX]]
    print("Usable stocks:", len(usable))
    panel, monthly_ret = make_panel(close, volume)
    sentiment = collect_sentiment(usable)
    sentiment.to_csv(os.path.join(OUT_DIR, "v11_sentiment_snapshot.csv"), index=False)
    panel = add_sentiment_scores(panel, sentiment)
    panel.to_csv(os.path.join(OUT_DIR, "v11_panel.csv"), index=False)
    summary_12, backtests = test_1_and_2(panel, monthly_ret)
    walk_forward = test_3_walk_forward(panel, monthly_ret)
    crash_tests = test_4_crash_tests(panel, monthly_ret)
    paper = test_5_paper_portfolio(panel)
    summary = pd.concat([summary_12, walk_forward, crash_tests], ignore_index=True)
    summary.to_csv(os.path.join(OUT_DIR, "v11_summary.csv"), index=False)
    backtests.to_csv(os.path.join(OUT_DIR, "v11_backtests.csv"))
    walk_forward.to_csv(os.path.join(OUT_DIR, "v11_walk_forward.csv"), index=False)
    crash_tests.to_csv(os.path.join(OUT_DIR, "v11_crash_tests.csv"), index=False)
    print("\n========== V11 SUMMARY ==========")
    print(summary)
    print("\n========== PAPER PORTFOLIO ==========")
    print(paper)
    print("\nSaved outputs:")
    print(os.path.join(OUT_DIR, "v11_summary.csv"))
    print(os.path.join(PAPER_DIR, "v11_today_portfolio.csv"))


if __name__ == "__main__":
    main()