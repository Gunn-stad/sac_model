import warnings
warnings.filterwarnings("ignore")

from pathlib import Path
import pandas as pd
import numpy as np
import yfinance as yf
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error

BASE_DIR = Path(".")
RESULTS_DIR = BASE_DIR / "results" / "sp500_top20_ai"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

START = "2016-01-01"
END = None
TOP_N = 20
REBALANCE_DAYS = 21


def get_sp500_tickers():
    url = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
    table = pd.read_html(url)[0]
    tickers = table["Symbol"].str.replace(".", "-", regex=False).tolist()
    return tickers


def download_prices(tickers):
    data = yf.download(
        tickers,
        start=START,
        end=END,
        auto_adjust=True,
        progress=False,
        group_by="ticker",
        threads=True,
    )

    closes = {}
    for t in tickers:
        try:
            if isinstance(data.columns, pd.MultiIndex):
                closes[t] = data[t]["Close"]
            else:
                closes[t] = data["Close"]
        except Exception:
            pass

    close = pd.DataFrame(closes).dropna(axis=1, thresh=500)
    close = close.ffill().dropna(axis=1)
    return close


def make_features(close):
    returns = close.pct_change()

    feat = {}
    feat["ret_21"] = close.pct_change(21)
    feat["ret_63"] = close.pct_change(63)
    feat["ret_126"] = close.pct_change(126)
    feat["vol_21"] = returns.rolling(21).std()
    feat["vol_63"] = returns.rolling(63).std()
    feat["ma_ratio_21"] = close / close.rolling(21).mean() - 1
    feat["ma_ratio_63"] = close / close.rolling(63).mean() - 1

    # Bayesian-style shrinkage expected return
    rolling_mu = returns.rolling(63).mean()
    rolling_std = returns.rolling(63).std()
    prior_strength = 20
    bayes_mu = (63 * rolling_mu) / (63 + prior_strength)
    bayes_uncertainty = rolling_std / np.sqrt(63 + prior_strength)

    feat["bayes_mu"] = bayes_mu
    feat["bayes_uncertainty"] = bayes_uncertainty
    feat["bayes_score"] = bayes_mu / (bayes_uncertainty + 1e-8)

    # Target: next 21-day return
    target = close.shift(-21) / close - 1

    rows = []
    for date in close.index:
        for ticker in close.columns:
            row = {"date": date, "ticker": ticker}
            ok = True

            for name, df in feat.items():
                val = df.loc[date, ticker]
                if pd.isna(val) or np.isinf(val):
                    ok = False
                    break
                row[name] = val

            y = target.loc[date, ticker]
            if pd.isna(y) or np.isinf(y):
                ok = False

            if ok:
                row["target_next_21d"] = y
                rows.append(row)

    return pd.DataFrame(rows)


def backtest_top20(df, close):
    feature_cols = [
        "ret_21",
        "ret_63",
        "ret_126",
        "vol_21",
        "vol_63",
        "ma_ratio_21",
        "ma_ratio_63",
        "bayes_mu",
        "bayes_uncertainty",
        "bayes_score",
    ]

    dates = sorted(df["date"].unique())
    split = int(len(dates) * 0.7)

    train_dates = dates[:split]
    test_dates = dates[split:]

    train = df[df["date"].isin(train_dates)]

    model = RandomForestRegressor(
        n_estimators=300,
        max_depth=6,
        min_samples_leaf=20,
        random_state=1,
        n_jobs=-1,
    )

    model.fit(train[feature_cols], train["target_next_21d"])

    equity = 1.0
    spy_equity = 1.0
    log = []

    spy = yf.download("SPY", start=START, auto_adjust=True, progress=False)["Close"]
    spy = spy.reindex(close.index).ffill()

    rebalance_dates = test_dates[::REBALANCE_DAYS]

    for i in range(len(rebalance_dates) - 1):
        d0 = rebalance_dates[i]
        d1 = rebalance_dates[i + 1]

        current = df[df["date"] == d0].copy()
        if len(current) < TOP_N:
            continue

        current["pred"] = model.predict(current[feature_cols])
        picks = current.sort_values("pred", ascending=False).head(TOP_N)["ticker"].tolist()

        valid = [t for t in picks if t in close.columns and d0 in close.index and d1 in close.index]
        if len(valid) == 0:
            continue

        r = (close.loc[d1, valid] / close.loc[d0, valid] - 1).mean()
        spy_r = float(spy.loc[d1] / spy.loc[d0] - 1)

        equity *= 1 + r
        spy_equity *= 1 + spy_r

        log.append({
            "date": str(d0.date()),
            "next_date": str(d1.date()),
            "portfolio_return": float(r),
            "spy_return": float(spy_r),
            "equity": float(equity),
            "spy_equity": float(spy_equity),
            "picks": ",".join(valid),
        })

    result = pd.DataFrame(log)
    return result, model


def sharpe(returns):
    returns = pd.Series(returns).dropna()
    if returns.std() == 0:
        return 0
    return np.sqrt(12) * returns.mean() / returns.std()


def main():
    print("Downloading S&P 500 tickers...")
    tickers = get_sp500_tickers()
    print("Tickers:", len(tickers))

    print("Downloading prices...")
    close = download_prices(tickers)
    print("Usable stocks:", close.shape[1])

    print("Building features...")
    df = make_features(close)
    print("Rows:", len(df))

    print("Running backtest...")
    result, model = backtest_top20(df, close)

    result.to_csv(RESULTS_DIR / "sp500_top20_backtest.csv", index=False)

    total_return = result["equity"].iloc[-1] - 1
    spy_return = result["spy_equity"].iloc[-1] - 1

    summary = {
        "strategy_total_return": float(total_return),
        "spy_total_return": float(spy_return),
        "excess_return": float(total_return - spy_return),
        "strategy_sharpe": float(sharpe(result["portfolio_return"])),
        "spy_sharpe": float(sharpe(result["spy_return"])),
        "periods": int(len(result)),
    }

    pd.Series(summary).to_json(RESULTS_DIR / "summary.json", indent=2)

    print("\n=== SUMMARY ===")
    for k, v in summary.items():
        print(k, ":", v)

    print("\nLatest picks:")
    print(result.tail(1)["picks"].iloc[0])


if __name__ == "__main__":
    main()