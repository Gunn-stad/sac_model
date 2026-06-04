import os
import warnings
from io import StringIO

import numpy as np
import pandas as pd
import requests
import yfinance as yf
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
from sklearn.ensemble import RandomForestRegressor
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

START = "2015-01-01"
BENCHMARK = "SPY"
TOP_N = 20
OUT_DIR = "v8_real_results"
os.makedirs(OUT_DIR, exist_ok=True)


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    html = requests.get(url, headers={"User-Agent": "Mozilla/5.0"}, timeout=20).text
    table = pd.read_html(StringIO(html))[0]
    return [x.replace(".", "-") for x in table["Symbol"].tolist()]


def download_prices(tickers):
    all_tickers = sorted(set(tickers + [BENCHMARK]))
    data = yf.download(
        all_tickers,
        start=START,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=True,
    )

    close = data["Close"].dropna(axis=1, how="all")
    volume = data["Volume"].reindex(columns=close.columns)

    close = close.ffill().dropna(axis=1, thresh=int(len(close) * 0.80))
    volume = volume.reindex(columns=close.columns).ffill()

    return close, volume


def get_yahoo_sentiment(tickers, max_tickers=120):
    analyzer = SentimentIntensityAnalyzer()
    rows = []

    print("Collecting real Yahoo news sentiment...")

    for i, ticker in enumerate(tickers[:max_tickers]):
        try:
            news = yf.Ticker(ticker).news or []
            scores = []

            for item in news:
                title = item.get("title", "")
                publisher = item.get("publisher", "")
                text = f"{title} {publisher}"
                if text.strip():
                    scores.append(analyzer.polarity_scores(text)["compound"])

            sentiment = float(np.mean(scores)) if scores else 0.0
            attention = len(scores)

            rows.append({
                "ticker": ticker,
                "sentiment": sentiment,
                "attention": attention,
            })

        except Exception:
            rows.append({
                "ticker": ticker,
                "sentiment": 0.0,
                "attention": 0,
            })

        if (i + 1) % 25 == 0:
            print(f"Sentiment done: {i + 1}/{min(len(tickers), max_tickers)}")

    return pd.DataFrame(rows)


def make_panel(close, volume):
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change()
    daily_ret = close.pct_change()
    monthly_volume = volume.resample("ME").mean()

    spy_ret = monthly_ret[BENCHMARK]
    rows = []

    for ticker in close.columns:
        if ticker == BENCHMARK:
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


def add_real_sentiment(panel, sentiment_df):
    panel = panel.merge(sentiment_df, on="ticker", how="left")
    panel["sentiment"] = panel["sentiment"].fillna(0.0)
    panel["attention"] = panel["attention"].fillna(0.0)

    scored = []
    for date, group in panel.groupby("date"):
        g = group.copy()
        g["sentiment_rank"] = g["sentiment"].rank(pct=True)
        g["attention_rank"] = g["attention"].rank(pct=True)

        g["score_momentum_behavioral_sentiment"] = (
            0.45 * g["score_momentum"]
            + 0.25 * g["score_behavioral"]
            + 0.20 * g["sentiment_rank"]
            + 0.10 * g["attention_rank"]
        )

        scored.append(g)

    return pd.concat(scored)


def backtest(panel, monthly_ret, score_col, start_test=None):
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
        picks = group.sort_values(score_col, ascending=False).head(TOP_N)["ticker"].tolist()

        r = monthly_ret.loc[next_date, picks].mean()
        spy = monthly_ret.loc[next_date, BENCHMARK]

        results.append({
            "date": next_date,
            "strategy_return": r,
            "spy_return": spy,
            "picks": ",".join(picks),
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


def train_ml(panel, monthly_ret, start_test):
    feature_cols = [
        "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "vol_3m", "vol_6m", "drawdown_6m",
        "volume_z", "beta_12m",
        "score_momentum", "score_behavioral", "score_bayesian",
        "sentiment", "attention",
    ]

    train = panel[panel["date"] < start_test].dropna()
    test = panel[panel["date"] >= start_test].dropna()

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("rf", RandomForestRegressor(
            n_estimators=400,
            max_depth=6,
            random_state=42,
            n_jobs=-1,
        )),
    ])

    model.fit(train[feature_cols], train["next_1m_return"])

    test = test.copy()
    test["score_ml_real_sentiment"] = model.predict(test[feature_cols])

    return backtest(test, monthly_ret, "score_ml_real_sentiment")


def compare_sac_if_exists(summary_rows):
    possible_files = [
        "v6_summary.csv",
        "v7_summary.csv",
        "v8_summary.csv",
        "results/summary.csv",
        "v6_real_results/summary.csv",
    ]

    for path in possible_files:
        if os.path.exists(path):
            try:
                df = pd.read_csv(path)
                df.to_csv(os.path.join(OUT_DIR, "sac_comparison_input.csv"), index=False)
                print(f"Found possible SAC summary file: {path}")
                return
            except Exception:
                pass

    print("No SAC summary file found. We will compare manually after you send the SAC result CSV.")


def main():
    print("Getting S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print("Universe:", len(tickers))

    close, volume = download_prices(tickers)
    tickers = [t for t in close.columns if t != BENCHMARK]
    print("Downloaded usable stocks:", len(tickers))

    panel, monthly_ret = make_panel(close, volume)
    print("Panel:", panel.shape)

    dates = sorted(panel["date"].unique())
    start_test = dates[int(len(dates) * 0.70)]
    print("Train/test split date:", start_test)

    sentiment_df = get_yahoo_sentiment(tickers, max_tickers=150)
    sentiment_df.to_csv(os.path.join(OUT_DIR, "real_sentiment_snapshot.csv"), index=False)

    panel = add_real_sentiment(panel, sentiment_df)
    panel.to_csv(os.path.join(OUT_DIR, "v8_full_panel.csv"), index=False)

    tests = {
        "test1_momentum_train_test": "score_momentum",
        "test1_bayesian_train_test": "score_bayesian",
        "test2_behavioral_real_sentiment": "score_momentum_behavioral_sentiment",
        "test2_behavioral_no_sentiment": "score_behavioral",
    }

    summary_rows = []

    for name, score_col in tests.items():
        print("\nRunning:", name)
        bt = backtest(panel, monthly_ret, score_col, start_test=start_test)
        bt.to_csv(os.path.join(OUT_DIR, f"{name}_backtest.csv"))

        stats = performance(bt)
        stats["name"] = name
        summary_rows.append(stats)

        print(pd.Series(stats))

    print("\nRunning ML + real sentiment...")
    bt_ml = train_ml(panel, monthly_ret, start_test)
    bt_ml.to_csv(os.path.join(OUT_DIR, "test2_ml_real_sentiment_backtest.csv"))

    stats = performance(bt_ml)
    stats["name"] = "test2_ml_real_sentiment"
    summary_rows.append(stats)
    print(pd.Series(stats))

    summary = pd.DataFrame(summary_rows)
    cols = ["name"] + [c for c in summary.columns if c != "name"]
    summary = summary[cols]

    summary.to_csv(os.path.join(OUT_DIR, "v8_real_test_summary.csv"), index=False)

    print("\n========== FINAL SUMMARY ==========")
    print(summary)

    compare_sac_if_exists(summary_rows)


if __name__ == "__main__":
    main()