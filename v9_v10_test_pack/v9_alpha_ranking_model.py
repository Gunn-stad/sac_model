"""
v9_alpha_ranking_model.py

V9 = stock ranking alpha model:
- Momentum
- Bayesian confidence
- Behavioral features
- Real Yahoo news sentiment using VADER
- Market regime detection using SPY and optional VIX
- Multi-agent voting score

Outputs:
    v9_results/v9_summary.csv
    v9_results/v9_panel.csv
    v9_results/v9_best_score_backtest.csv
    v9_results/v9_sentiment_snapshot.csv
"""

import os
import warnings
from io import StringIO

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
OUT_DIR = "v9_results"
MAX_SENTIMENT_TICKERS = 180

os.makedirs(OUT_DIR, exist_ok=True)


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
            "AAPL", "MSFT", "NVDA", "AMZN", "META", "GOOGL", "GOOG", "BRK-B", "JPM", "LLY",
            "AVGO", "XOM", "UNH", "V", "MA", "COST", "PG", "JNJ", "HD", "ABBV", "BAC",
            "NFLX", "CRM", "AMD", "PEP", "TMO", "WMT", "CSCO", "MCD", "ABT", "DIS",
            "QCOM", "IBM", "GE", "CAT", "AMAT", "TXN", "NOW", "GS", "ISRG", "UBER",
            "BKNG", "PFE", "RTX", "SPGI", "LOW", "HON", "AXP", "NEE", "BLK", "ELV",
            "SYK", "LMT", "TJX", "VRTX", "ADBE", "DE", "PANW"
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


def collect_real_sentiment(tickers, max_tickers=MAX_SENTIMENT_TICKERS):
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    print("Collecting real Yahoo news sentiment...")

    for i, ticker in enumerate(tickers[:max_tickers]):
        try:
            news = yf.Ticker(ticker).news or []
            scores = []
            timestamps = []

            for item in news:
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                text = f"{title} {publisher}".strip()

                if not text:
                    continue

                score = analyzer.polarity_scores(text)["compound"]
                scores.append(score)

                ts = item.get("providerPublishTime", None)
                if ts:
                    timestamps.append(ts)

            sentiment = float(np.mean(scores)) if scores else 0.0
            sentiment_abs = float(np.mean(np.abs(scores))) if scores else 0.0
            attention = len(scores)

            rows.append({
                "ticker": ticker,
                "sentiment": sentiment,
                "sentiment_abs": sentiment_abs,
                "attention": attention,
            })

        except Exception:
            rows.append({
                "ticker": ticker,
                "sentiment": 0.0,
                "sentiment_abs": 0.0,
                "attention": 0,
            })

        if (i + 1) % 25 == 0:
            print(f"Sentiment done: {i + 1}/{min(len(tickers), max_tickers)}")

    return pd.DataFrame(rows)


def compute_market_regime(monthly_close):
    spy = monthly_close[BENCHMARK]
    spy_ret = spy.pct_change()

    regime = pd.DataFrame(index=monthly_close.index)
    regime["spy_ret_1m"] = spy.pct_change(1)
    regime["spy_ret_3m"] = spy.pct_change(3)
    regime["spy_ret_6m"] = spy.pct_change(6)
    regime["spy_ma_10"] = spy.rolling(10).mean()
    regime["spy_above_ma"] = (spy > regime["spy_ma_10"]).astype(float)
    regime["spy_vol_6m"] = spy_ret.rolling(6).std()

    if VIX in monthly_close.columns:
        regime["vix"] = monthly_close[VIX]
        regime["vix_rank"] = regime["vix"].rolling(36).rank(pct=True)
    else:
        regime["vix"] = np.nan
        regime["vix_rank"] = 0.5

    regime["bull_regime"] = (
        (regime["spy_ret_6m"] > 0).astype(float)
        + regime["spy_above_ma"]
        + (regime["vix_rank"].fillna(0.5) < 0.70).astype(float)
    ) / 3.0

    regime["bear_or_stress_regime"] = 1.0 - regime["bull_regime"]
    return regime


def make_panel(close, volume):
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    daily_ret = close.pct_change()
    monthly_volume = volume.resample("ME").mean()

    regime = compute_market_regime(monthly_close)

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

        df["volume_z"] = (
            monthly_volume[ticker] - monthly_volume[ticker].rolling(12).mean()
        ) / monthly_volume[ticker].rolling(12).std()

        df["beta_12m"] = monthly_ret[ticker].rolling(12).cov(spy_ret) / spy_ret.rolling(12).var()
        df["next_1m_return"] = monthly_ret[ticker].shift(-1)

        for col in ["bull_regime", "bear_or_stress_regime", "spy_ret_6m", "spy_vol_6m", "vix_rank"]:
            df[col] = regime[col]

        rows.append(df)

    panel = pd.concat(rows).dropna()

    scored = []
    for date, group in panel.groupby("date"):
        g = group.copy()

        rank_cols = [
            "ret_1m", "ret_3m", "ret_6m", "ret_12m",
            "vol_3m", "vol_6m", "drawdown_6m",
            "volume_z", "beta_12m"
        ]

        for col in rank_cols:
            g[col + "_rank"] = g[col].rank(pct=True)

        g["score_momentum"] = (
            0.40 * g["ret_6m_rank"]
            + 0.30 * g["ret_12m_rank"]
            + 0.20 * g["ret_3m_rank"]
            - 0.10 * g["vol_3m_rank"]
        )

        g["score_behavioral"] = (
            0.35 * g["ret_6m_rank"]
            + 0.25 * g["ret_12m_rank"]
            - 0.25 * g["ret_1m_rank"]
            - 0.15 * g["drawdown_6m_rank"]
            + 0.10 * g["volume_z_rank"]
        )

        market_mean = g["ret_6m"].mean()
        confidence = 1 / (1 + 30 * g["vol_6m"].fillna(g["vol_6m"].median()))
        bayes = confidence * g["ret_6m"] + (1 - confidence) * market_mean
        g["score_bayesian"] = bayes.rank(pct=True)

        scored.append(g)

    return pd.concat(scored), monthly_ret


def add_sentiment_and_v9_scores(panel, sentiment_df):
    panel = panel.merge(sentiment_df, on="ticker", how="left")
    panel["sentiment"] = panel["sentiment"].fillna(0.0)
    panel["sentiment_abs"] = panel["sentiment_abs"].fillna(0.0)
    panel["attention"] = panel["attention"].fillna(0.0)

    out = []
    for date, group in panel.groupby("date"):
        g = group.copy()

        g["sentiment_rank"] = g["sentiment"].rank(pct=True)
        g["sentiment_abs_rank"] = g["sentiment_abs"].rank(pct=True)
        g["attention_rank"] = g["attention"].rank(pct=True)

        # In bull regimes, give more weight to momentum.
        # In stress regimes, give more weight to Bayesian confidence and lower volatility.
        bull = float(g["bull_regime"].iloc[0])
        stress = 1.0 - bull

        g["score_sentiment"] = (
            0.70 * g["sentiment_rank"]
            + 0.20 * g["attention_rank"]
            + 0.10 * g["sentiment_abs_rank"]
        )

        g["score_regime_adjusted"] = (
            (0.35 + 0.15 * bull) * g["score_momentum"]
            + (0.30 + 0.15 * stress) * g["score_bayesian"]
            + 0.15 * g["score_behavioral"]
            + 0.15 * g["score_sentiment"]
            - 0.10 * stress * g["vol_6m_rank"]
        )

        # Multi-agent voting: a stock gets votes if it is strong in several systems.
        g["vote_momentum"] = (g["score_momentum"] >= g["score_momentum"].quantile(0.80)).astype(float)
        g["vote_bayesian"] = (g["score_bayesian"] >= g["score_bayesian"].quantile(0.80)).astype(float)
        g["vote_behavioral"] = (g["score_behavioral"] >= g["score_behavioral"].quantile(0.80)).astype(float)
        g["vote_sentiment"] = (g["score_sentiment"] >= g["score_sentiment"].quantile(0.80)).astype(float)

        g["score_multi_agent_vote"] = (
            0.30 * g["vote_momentum"]
            + 0.30 * g["vote_bayesian"]
            + 0.20 * g["vote_behavioral"]
            + 0.20 * g["vote_sentiment"]
            + 0.10 * g["score_regime_adjusted"]
        )

        out.append(g)

    return pd.concat(out)


def backtest(panel, monthly_ret, score_col, start_test=None, top_n=TOP_N):
    results = []

    if start_test is not None:
        panel = panel[panel["date"] >= start_test]

    for date, group in panel.groupby("date"):
        if date not in monthly_ret.index:
            continue

        idx = monthly_ret.index.get_loc(date) + 1
        if idx >= len(monthly_ret.index):
            continue

        next_date = monthly_ret.index[idx]
        picks = group.sort_values(score_col, ascending=False).head(top_n)["ticker"].tolist()

        valid_picks = [p for p in picks if p in monthly_ret.columns]
        if not valid_picks:
            continue

        r = monthly_ret.loc[next_date, valid_picks].mean()
        spy = monthly_ret.loc[next_date, BENCHMARK]

        results.append({
            "date": next_date,
            "strategy_return": r,
            "spy_return": spy,
            "picks": ",".join(valid_picks),
        })

    out = pd.DataFrame(results).set_index("date")
    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["spy_equity"] = (1 + out["spy_return"]).cumprod()
    return out


def performance(bt):
    def stats(r):
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
    }


def main():
    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print("Universe:", len(tickers))

    close, volume = download_prices(tickers)
    usable = [t for t in close.columns if t not in [BENCHMARK, VIX]]
    print("Downloaded usable stocks:", len(usable))

    panel, monthly_ret = make_panel(close, volume)
    print("Panel before sentiment:", panel.shape)

    dates = sorted(panel["date"].unique())
    start_test = dates[int(len(dates) * 0.70)]
    print("Train/test split date:", start_test)

    sentiment_df = collect_real_sentiment(usable, max_tickers=MAX_SENTIMENT_TICKERS)
    sentiment_df.to_csv(os.path.join(OUT_DIR, "v9_sentiment_snapshot.csv"), index=False)

    panel = add_sentiment_and_v9_scores(panel, sentiment_df)
    panel.to_csv(os.path.join(OUT_DIR, "v9_panel.csv"), index=False)

    tests = {
        "momentum": "score_momentum",
        "bayesian": "score_bayesian",
        "behavioral": "score_behavioral",
        "sentiment": "score_sentiment",
        "regime_adjusted": "score_regime_adjusted",
        "multi_agent_vote": "score_multi_agent_vote",
    }

    summary_rows = []

    for name, score_col in tests.items():
        print("\nRunning:", name)
        bt = backtest(panel, monthly_ret, score_col, start_test=start_test)
        bt.to_csv(os.path.join(OUT_DIR, f"v9_{name}_backtest.csv"))

        stats = performance(bt)
        stats["name"] = name
        summary_rows.append(stats)

        print(pd.Series(stats))

    summary = pd.DataFrame(summary_rows)
    summary = summary[["name"] + [c for c in summary.columns if c != "name"]]
    summary.to_csv(os.path.join(OUT_DIR, "v9_summary.csv"), index=False)

    best = summary.sort_values("strategy_sharpe", ascending=False).iloc[0]
    print("\n========== V9 FINAL SUMMARY ==========")
    print(summary)
    print("\nBest strategy by Sharpe:")
    print(best)

    best_name = best["name"]
    best_score_col = tests[best_name]
    best_bt = backtest(panel, monthly_ret, best_score_col, start_test=start_test)
    best_bt.to_csv(os.path.join(OUT_DIR, "v9_best_score_backtest.csv"))

    print("\nSaved V9 outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()