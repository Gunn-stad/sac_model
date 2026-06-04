"""
v10_sac_portfolio_manager.py

V10 = SAC portfolio manager using V9 signals.

Important:
- V9 is the alpha/stock-picker layer.
- V10 is the reinforcement-learning portfolio layer.
- The SAC agent receives V9-style features and learns portfolio allocation weights.

Outputs:
    v10_results/v10_sac_summary.csv
    v10_results/v10_sac_backtest.csv
    v10_results/v10_selected_assets.csv

Run after v9_alpha_ranking_model.py or run standalone.
"""

import os
import warnings
from io import StringIO

import gymnasium as gym
import numpy as np
import pandas as pd
import requests
import yfinance as yf
from gymnasium import spaces
from stable_baselines3 import SAC
from stable_baselines3.common.noise import NormalActionNoise
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

warnings.filterwarnings("ignore")

START = "2015-01-01"
BENCHMARK = "SPY"
VIX = "^VIX"
TOP_N = 20
MAX_SENTIMENT_TICKERS = 180
OUT_DIR = "v10_results"
V9_DIR = "v9_results"

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


def make_v9_panel(close, volume):
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


def add_sentiment_and_scores(panel, sentiment_df):
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

        out.append(g)

    return pd.concat(out)


def build_or_load_panel():
    if os.path.exists(os.path.join(V9_DIR, "v9_panel.csv")):
        print("Loading existing V9 panel...")
        panel = pd.read_csv(os.path.join(V9_DIR, "v9_panel.csv"))
        panel["date"] = pd.to_datetime(panel["date"])
        return panel

    print("No V9 panel found. Building it now...")
    tickers = get_sp500_tickers()
    close, volume = download_prices(tickers)
    usable = [t for t in close.columns if t not in [BENCHMARK, VIX]]

    panel, monthly_ret = make_v9_panel(close, volume)
    sentiment = collect_real_sentiment(usable, max_tickers=MAX_SENTIMENT_TICKERS)
    panel = add_sentiment_and_scores(panel, sentiment)

    os.makedirs(V9_DIR, exist_ok=True)
    panel.to_csv(os.path.join(V9_DIR, "v9_panel.csv"), index=False)
    return panel


def get_monthly_returns_for_assets(tickers):
    data = yf.download(
        sorted(set(tickers + [BENCHMARK])),
        start=START,
        auto_adjust=True,
        progress=True,
        group_by="column",
        threads=True,
    )

    close = data["Close"].ffill().dropna(axis=1, how="all")
    monthly_close = close.resample("ME").last()
    monthly_ret = monthly_close.pct_change().dropna()
    return monthly_ret


class V10SACPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feature_tensor,
        return_matrix,
        benchmark_returns,
        transaction_cost=0.001,
        risk_lambda=0.10,
        turnover_lambda=0.001,
        max_weight=0.20,
    ):
        super().__init__()

        self.X = feature_tensor.astype(np.float32)
        self.R = return_matrix.astype(np.float32)
        self.B = benchmark_returns.astype(np.float32)

        self.T, self.N, self.F = self.X.shape
        self.transaction_cost = transaction_cost
        self.risk_lambda = risk_lambda
        self.turnover_lambda = turnover_lambda
        self.max_weight = max_weight

        obs_dim = self.N * self.F + self.N
        self.observation_space = spaces.Box(
            low=-10.0,
            high=10.0,
            shape=(obs_dim,),
            dtype=np.float32,
        )

        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(self.N,),
            dtype=np.float32,
        )

        self.t = 0
        self.weights = np.ones(self.N, dtype=np.float32) / self.N

    def _normalize_action(self, action):
        # Convert arbitrary SAC action to long-only portfolio weights.
        a = np.asarray(action, dtype=np.float32)
        exp_a = np.exp(a - np.max(a))
        w = exp_a / (np.sum(exp_a) + 1e-8)

        # Cap max single-name weight, then renormalize.
        w = np.minimum(w, self.max_weight)
        w = w / (np.sum(w) + 1e-8)
        return w.astype(np.float32)

    def _get_obs(self):
        x = self.X[self.t].reshape(-1)
        obs = np.concatenate([x, self.weights]).astype(np.float32)
        return obs

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.t = 0
        self.weights = np.ones(self.N, dtype=np.float32) / self.N
        return self._get_obs(), {}

    def step(self, action):
        new_weights = self._normalize_action(action)

        asset_ret = self.R[self.t]
        port_ret = float(np.dot(new_weights, asset_ret))
        bench_ret = float(self.B[self.t])

        turnover = float(np.sum(np.abs(new_weights - self.weights)))
        cost = self.transaction_cost * turnover

        # Relative return reward, penalized for turnover and risk.
        excess = port_ret - bench_ret
        concentration = float(np.sum(new_weights ** 2))

        reward = (
            excess
            - cost
            - self.turnover_lambda * turnover
            - self.risk_lambda * concentration
        )

        self.weights = new_weights
        self.t += 1

        terminated = self.t >= self.T - 1
        truncated = False

        info = {
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "turnover": turnover,
            "cost": cost,
            "weights": new_weights,
        }

        obs = self._get_obs() if not terminated else np.zeros(self.observation_space.shape, dtype=np.float32)
        return obs, float(reward), terminated, truncated, info


def prepare_sac_data(panel):
    # Use only the strongest V9 names. This keeps SAC small and learnable.
    last_date = panel["date"].max()
    latest = panel[panel["date"] == last_date].copy()

    score_col = "score_regime_adjusted"
    selected = latest.sort_values(score_col, ascending=False).head(TOP_N)["ticker"].tolist()

    pd.DataFrame({"ticker": selected}).to_csv(os.path.join(OUT_DIR, "v10_selected_assets.csv"), index=False)
    print("Selected assets for SAC:", selected)

    monthly_ret = get_monthly_returns_for_assets(selected)

    feature_cols = [
        "ret_1m", "ret_3m", "ret_6m", "ret_12m",
        "vol_3m", "vol_6m", "drawdown_6m",
        "volume_z", "beta_12m",
        "score_momentum", "score_behavioral", "score_bayesian",
        "score_sentiment", "score_regime_adjusted",
        "bull_regime", "bear_or_stress_regime",
    ]

    dates = sorted(panel["date"].unique())
    common_dates = []
    X_list = []
    R_list = []
    B_list = []

    for date in dates:
        if date not in monthly_ret.index:
            continue

        idx = monthly_ret.index.get_loc(date) + 1
        if idx >= len(monthly_ret.index):
            continue

        next_date = monthly_ret.index[idx]
        if next_date not in monthly_ret.index:
            continue

        g = panel[(panel["date"] == date) & (panel["ticker"].isin(selected))].copy()
        if len(g) != len(selected):
            continue

        g = g.set_index("ticker").loc[selected]

        X = g[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        R = monthly_ret.loc[next_date, selected].replace([np.inf, -np.inf], np.nan).fillna(0.0).values
        B = monthly_ret.loc[next_date, BENCHMARK]

        common_dates.append(next_date)
        X_list.append(X)
        R_list.append(R)
        B_list.append(B)

    X = np.array(X_list, dtype=np.float32)
    R = np.array(R_list, dtype=np.float32)
    B = np.array(B_list, dtype=np.float32)
    dates = pd.to_datetime(common_dates)

    # Normalize features using train data only.
    split = int(len(dates) * 0.70)
    flat_train = X[:split].reshape(-1, X.shape[-1])
    mean = flat_train.mean(axis=0)
    std = flat_train.std(axis=0) + 1e-6
    X = (X - mean) / std
    X = np.clip(X, -5, 5)

    return X, R, B, dates, selected, split


def run_equal_weight(R, B, dates, start_idx):
    test_R = R[start_idx:]
    test_B = B[start_idx:]
    test_dates = dates[start_idx:]

    port = test_R.mean(axis=1)
    out = pd.DataFrame({
        "date": test_dates,
        "strategy_return": port,
        "spy_return": test_B,
    }).set_index("date")

    out["strategy_equity"] = (1 + out["strategy_return"]).cumprod()
    out["spy_equity"] = (1 + out["spy_return"]).cumprod()
    return out


def evaluate_sac(model, env, dates, start_idx):
    obs, _ = env.reset()
    rows = []

    while True:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        current_index = env.t - 1
        if current_index >= start_idx:
            rows.append({
                "date": dates[current_index],
                "strategy_return": info["portfolio_return"],
                "spy_return": info["benchmark_return"],
                "turnover": info["turnover"],
                "reward": reward,
            })

        if done or truncated:
            break

    out = pd.DataFrame(rows).set_index("date")
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
    panel = build_or_load_panel()
    X, R, B, dates, selected, split = prepare_sac_data(panel)

    print("SAC data:")
    print("X:", X.shape)
    print("R:", R.shape)
    print("Split date:", dates[split])

    train_env = V10SACPortfolioEnv(
        feature_tensor=X[:split],
        return_matrix=R[:split],
        benchmark_returns=B[:split],
        transaction_cost=0.001,
        risk_lambda=0.02,
        turnover_lambda=0.001,
        max_weight=0.20,
    )

    full_env = V10SACPortfolioEnv(
        feature_tensor=X,
        return_matrix=R,
        benchmark_returns=B,
        transaction_cost=0.001,
        risk_lambda=0.02,
        turnover_lambda=0.001,
        max_weight=0.20,
    )

    n_actions = train_env.action_space.shape[-1]
    action_noise = NormalActionNoise(mean=np.zeros(n_actions), sigma=0.05 * np.ones(n_actions))

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=3e-4,
        buffer_size=50_000,
        batch_size=64,
        tau=0.02,
        gamma=0.95,
        train_freq=1,
        gradient_steps=1,
        action_noise=action_noise,
        verbose=1,
        seed=42,
    )

    print("Training V10 SAC...")
    model.learn(total_timesteps=40_000)

    model.save(os.path.join(OUT_DIR, "v10_sac_model"))

    sac_bt = evaluate_sac(model, full_env, dates, split)
    sac_bt.to_csv(os.path.join(OUT_DIR, "v10_sac_backtest.csv"))

    ew_bt = run_equal_weight(R, B, dates, split)
    ew_bt.to_csv(os.path.join(OUT_DIR, "v10_equal_weight_selected_backtest.csv"))

    sac_stats = performance(sac_bt)
    sac_stats["name"] = "v10_sac_v9_features"

    ew_stats = performance(ew_bt)
    ew_stats["name"] = "equal_weight_v9_selected"

    summary = pd.DataFrame([sac_stats, ew_stats])
    summary = summary[["name"] + [c for c in summary.columns if c != "name"]]
    summary.to_csv(os.path.join(OUT_DIR, "v10_sac_summary.csv"), index=False)

    print("\n========== V10 FINAL SUMMARY ==========")
    print(summary)
    print("\nSaved V10 outputs in:", OUT_DIR)


if __name__ == "__main__":
    main()