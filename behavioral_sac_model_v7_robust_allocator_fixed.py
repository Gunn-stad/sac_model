"""
behavioral_sac_model_v7_robust_allocator.py

V7 Robust Allocator

Purpose:
    A stronger pre-paper-trading experiment that tests:
    - 2-3 random seeds
    - fixed entropy coefficient to avoid entropy collapse
    - best-validation checkpoint selection
    - stronger regime modeling
    - asset-rotation rewards
    - safe-growth / diversified ETF universe

Run in Colab:
    %cd /content/sac_model
    !python behavioral_sac_model_v7_robust_allocator.py

Outputs:
    /content/drive/MyDrive/lokaverkefni_bs/results/v7_robust_allocator/
        v7_seed_sweep_summary.csv
        v7_seed_sweep_summary.json
        seed_*/metrics.json
        seed_*/best_model.zip
"""

import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = "/content/drive/MyDrive/lokaverkefni_bs"
CACHE_DIR = f"{BASE_DIR}/data_cache"
BEHAVIORAL_DIR = f"{BASE_DIR}/behavioral_data"
RESULTS_DIR = f"{BASE_DIR}/results/v7_robust_allocator"
MODELS_DIR = f"{BASE_DIR}/models"

for d in [CACHE_DIR, BEHAVIORAL_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

print("RUNNING FILE:", __file__)
print("BASE_DIR:", BASE_DIR)
print("CACHE_DIR:", CACHE_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("BEHAVIORAL_DIR:", BEHAVIORAL_DIR)

UNIVERSE_NAME = "v7_robust_allocator_safe_growth"

TICKERS = [
    "SPY", "QQQ", "VTI", "DIA", "IWM",
    "XLK", "SMH", "SOXX", "BOTZ", "ARKK",
    "XLP", "XLU", "XLV",
    "SCHD", "VIG", "USMV", "SPLV",
    "SHY", "BIL", "SGOV", "IEF", "TLT", "TIP",
    "GLD",
]

GROWTH_ASSETS = {"QQQ", "XLK", "SMH", "SOXX", "BOTZ", "ARKK", "SPY", "VTI", "IWM"}
DEFENSIVE_ASSETS = {"XLP", "XLU", "XLV", "SHY", "BIL", "SGOV", "IEF", "TLT", "TIP", "GLD", "USMV", "SPLV", "VIG", "SCHD"}

INTERVAL = "1h"
PERIOD = "730d"
TRAIN_FRAC = 0.70
VAL_FRAC = 0.15
SEEDS = [0, 1, 2]
TOTAL_TIMESTEPS = 80_000
EVAL_FREQ = 10_000
BOTTOM_K = 5
LEARNING_RATE = 1e-4

MIN_EXPOSURE = 0.75
MAX_WEIGHT = 0.25
TRANSACTION_COST = 0.0005
SMOOTHING = 0.35
REBALANCE_EVERY = 3

LAMBDA_BENCH = 1.25
LAMBDA_TURNOVER = 0.002
LAMBDA_RISK = 0.02
LAMBDA_CONC = 0.01
CASH_PENALTY = 0.001
LAMBDA_ROTATION = 0.015
LAMBDA_STRESS_DD = 0.010
ENT_COEF = 0.005

print("Using universe:", UNIVERSE_NAME)
print("Number of assets:", len(TICKERS))
print("Seeds:", SEEDS)
print("Fixed ent_coef:", ENT_COEF)


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)


def safe_sharpe(returns: np.ndarray, periods_per_year: float = 252 * 6.5) -> float:
    returns = np.asarray(returns, dtype=np.float64)
    returns = returns[np.isfinite(returns)]
    if len(returns) < 2 or np.std(returns) < 1e-12:
        return 0.0
    return float(np.sqrt(periods_per_year) * np.mean(returns) / np.std(returns))


def max_drawdown(equity: np.ndarray) -> float:
    equity = np.asarray(equity, dtype=np.float64)
    peak = np.maximum.accumulate(equity)
    dd = equity / np.maximum(peak, 1e-12) - 1.0
    return float(abs(np.min(dd)))


def softmax(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=np.float64)
    x = x - np.max(x)
    e = np.exp(x)
    return e / max(e.sum(), 1e-12)


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-x))


def zscore_series(s: pd.Series, window: int = 240) -> pd.Series:
    mu = s.rolling(window, min_periods=max(10, window // 10)).mean()
    sd = s.rolling(window, min_periods=max(10, window // 10)).std()
    return ((s - mu) / (sd + 1e-8)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def download_or_load(ticker: str) -> pd.DataFrame:
    import yfinance as yf
    path = Path(CACHE_DIR) / f"{ticker}_{INTERVAL}_{PERIOD}_auto_adjust.parquet"
    if path.exists():
        print(f"[cache] {ticker} -> {path}")
        df = pd.read_parquet(path)
        df = normalize_datetime_index(df)
        return df
    print(f"[download] {ticker}")
    df = yf.download(ticker, period=PERIOD, interval=INTERVAL, auto_adjust=True, progress=False, threads=False)
    if df is None or len(df) == 0:
        raise ValueError(f"No data downloaded for {ticker}")
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]
    df = df.reset_index()
    dt_col = "Datetime" if "Datetime" in df.columns else "Date"
    df = df.rename(columns={dt_col: "datetime"})
    df["datetime"] = pd.to_datetime(df["datetime"], utc=True).dt.tz_convert(None)
    df = df.set_index("datetime").sort_index()
    df = normalize_datetime_index(df)
    keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
    df = df[keep].dropna()
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def build_close_panel(tickers):
    closes = []
    for t in tickers:
        try:
            df = download_or_load(t)
            df = normalize_datetime_index(df)
            closes.append(df["Close"].rename(t))
        except Exception as e:
            print(f"[warning] skipping {t}: {e}")
    if not closes:
        raise RuntimeError("No valid ticker data loaded.")
    close = pd.concat(closes, axis=1).sort_index()
    close = close.ffill().dropna(how="all")
    close = close.dropna(axis=1, how="all").ffill().dropna()
    print("Valid assets:", list(close.columns))
    return close


def stochastic_features(returns: pd.DataFrame):
    feats = []
    windows = [6, 24, 72]
    for w in windows:
        mu = returns.rolling(w, min_periods=max(2, w // 4)).mean().fillna(0.0)
        sig = returns.rolling(w, min_periods=max(2, w // 4)).std().replace(0, np.nan).bfill().fillna(1e-6)
        z = ((returns - mu) / (sig + 1e-8)).clip(-5, 5).fillna(0.0)
        rv = (returns ** 2).rolling(w, min_periods=max(2, w // 4)).sum().fillna(0.0)
        jump = (z.abs() > 2.5).astype(float)
        feats.extend([mu.values, sig.values, z.values, rv.values, jump.values])
    return np.concatenate(feats, axis=1).astype(np.float32)


def build_regime_features(close: pd.DataFrame, returns: pd.DataFrame) -> pd.DataFrame:
    cols = {}
    market_price = close.mean(axis=1)
    market_ret = returns.mean(axis=1).fillna(0.0)
    market_vol_24 = market_ret.rolling(24, min_periods=6).std().fillna(0.0)
    market_vol_72 = market_ret.rolling(72, min_periods=12).std().fillna(0.0)
    market_trend_24 = market_ret.rolling(24, min_periods=6).sum().fillna(0.0)
    market_trend_72 = market_ret.rolling(72, min_periods=12).sum().fillna(0.0)
    market_dd = (market_price / market_price.cummax() - 1.0).fillna(0.0)
    vol_z = zscore_series(market_vol_24, 240)
    stress_index = (0.40 * vol_z.clip(lower=0) + 0.35 * (-market_trend_24).clip(lower=0) * 10 + 0.25 * (-market_dd).clip(lower=0)).fillna(0.0)
    cols.update({
        "market_ret": market_ret,
        "market_vol_24": market_vol_24,
        "market_vol_72": market_vol_72,
        "market_trend_24": market_trend_24,
        "market_trend_72": market_trend_72,
        "market_drawdown": market_dd,
        "vol_z": vol_z,
        "stress_index": stress_index,
        "high_vol_regime": (vol_z > 0.75).astype(float),
        "drawdown_regime": (market_dd < -0.05).astype(float),
        "risk_off_regime": ((vol_z > 0.75) | (market_trend_24 < 0)).astype(float),
    })
    def rel_strength(a, b, name):
        if a in close.columns and b in close.columns:
            ratio = (close[a] / close[b]).replace([np.inf, -np.inf], np.nan).ffill().fillna(1.0)
            cols[name] = ratio.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0)
            cols[name + "_z"] = zscore_series(cols[name], 240)
    rel_strength("QQQ", "XLP", "qqq_vs_xlp")
    rel_strength("XLK", "XLU", "xlk_vs_xlu")
    rel_strength("SPY", "TLT", "spy_vs_tlt")
    rel_strength("GLD", "SPY", "gld_vs_spy")
    return pd.DataFrame(cols).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def build_features(close):
    returns = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    asset_features = []
    for w in [1, 6, 24, 72]:
        asset_features.append(close.pct_change(w).replace([np.inf, -np.inf], np.nan).fillna(0.0).values)
    for w in [24, 72]:
        ma = close.rolling(w, min_periods=max(3, w // 4)).mean()
        asset_features.append((close / ma - 1.0).replace([np.inf, -np.inf], np.nan).fillna(0.0).values)
    asset_features.append(stochastic_features(returns))
    X_asset = np.concatenate(asset_features, axis=1).astype(np.float32)
    regime_df = build_regime_features(close, returns)
    X = np.concatenate([X_asset, regime_df.values.astype(np.float32)], axis=1).astype(np.float32)
    return X, returns.values.astype(np.float32), list(close.columns), regime_df


def load_behavioral_features(index):
    cols = []
    stock_path = Path(BEHAVIORAL_DIR) / "stock_behavioral_signals.csv"
    global_path = Path(BEHAVIORAL_DIR) / "global_event_signals.csv"
    if stock_path.exists():
        print("[behavioral] loading", stock_path)
        try:
            s = pd.read_csv(stock_path)
            s["datetime"] = pd.to_datetime(s["datetime"]).dt.tz_localize(None).dt.floor("h")
            numeric_cols = [c for c in s.columns if c not in ["datetime", "ticker"]]
            if numeric_cols:
                agg = s.groupby("datetime")[numeric_cols].mean()
                cols.append(agg.reindex(index.floor("h")).ffill().fillna(0.0).values.astype(np.float32))
        except Exception as e:
            print("[behavioral warning]", e)
    if global_path.exists():
        print("[behavioral] loading", global_path)
        try:
            g = pd.read_csv(global_path)
            g["datetime"] = pd.to_datetime(g["datetime"]).dt.tz_localize(None).dt.floor("h")
            g = g.set_index("datetime").sort_index()
            cols.append(g.reindex(index.floor("h")).ffill().fillna(0.0).values.astype(np.float32))
        except Exception as e:
            print("[global behavioral warning]", e)
    if not cols:
        print("[behavioral] no files found; using zeros")
        return np.zeros((len(index), 4), dtype=np.float32)
    return np.concatenate(cols, axis=1).astype(np.float32)


def evaluate_weights(returns, weights_series, name, cash_series=None):
    n_steps, n_assets = returns.shape
    equity = [1.0]
    rets = []
    turnovers = []
    prev_w = np.zeros(n_assets)
    if cash_series is None:
        cash_series = np.zeros(n_steps)
    for t in range(n_steps - 1):
        w = weights_series[t]
        turnover = np.sum(np.abs(w - prev_w))
        cost = TRANSACTION_COST * turnover
        r = float(np.dot(w, returns[t + 1]))
        net_r = (1.0 + r) * (1.0 - cost) - 1.0
        equity.append(equity[-1] * (1.0 + net_r))
        rets.append(net_r)
        turnovers.append(turnover)
        prev_w = w.copy()
    equity = np.array(equity)
    rets = np.array(rets)
    return {
        "name": name,
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": max_drawdown(equity),
        "sharpe": safe_sharpe(rets),
        "avg_turnover": float(np.mean(turnovers)) if turnovers else 0.0,
        "avg_cash": float(np.mean(cash_series)) if cash_series is not None else 0.0,
        "final_equity": float(equity[-1]),
    }


def equal_weight_baseline(returns):
    n_steps, n_assets = returns.shape
    w = np.ones((n_steps, n_assets), dtype=np.float32) / n_assets
    return evaluate_weights(returns, w, "Baseline_EQ")


def bottom_k_baseline(returns):
    n_steps, n_assets = returns.shape
    rolling = pd.DataFrame(returns).rolling(24, min_periods=3).sum().fillna(0.0).values
    w = np.zeros((n_steps, n_assets), dtype=np.float32)
    for t in range(n_steps):
        idx = np.argsort(rolling[t])[:min(BOTTOM_K, n_assets)]
        w[t, idx] = 1.0 / len(idx)
    return evaluate_weights(returns, w, f"Bottom{BOTTOM_K}_ret_24h")


def make_env_class(growth_idx, defensive_idx):
    import gymnasium as gym
    from gymnasium import spaces
    class PortfolioEnv(gym.Env):
        metadata = {"render_modes": []}
        def __init__(self, X, H, R, regime_df, tickers, split_name="train"):
            super().__init__()
            self.X = X.astype(np.float32)
            self.H = H.astype(np.float32)
            self.R = R.astype(np.float32)
            self.regime_df = regime_df.reset_index(drop=True)
            self.tickers = tickers
            self.n_assets = R.shape[1]
            self.split_name = split_name
            self.obs_dim = self.X.shape[1] + self.H.shape[1] + self.n_assets + 1 + self.n_assets
            self.action_dim = 1 + self.n_assets
            self.observation_space = spaces.Box(-10, 10, shape=(self.obs_dim,), dtype=np.float32)
            self.action_space = spaces.Box(-5, 5, shape=(self.action_dim,), dtype=np.float32)
        def _baseline_weights(self, t):
            lookback = min(24, t)
            if lookback <= 3:
                idx = np.arange(min(BOTTOM_K, self.n_assets))
            else:
                score = self.R[t - lookback:t].sum(axis=0)
                idx = np.argsort(score)[:min(BOTTOM_K, self.n_assets)]
            b = np.zeros(self.n_assets, dtype=np.float32)
            b[idx] = 1.0 / len(idx)
            return b
        def _obs(self):
            b = self._baseline_weights(self.t)
            obs = np.concatenate([self.X[self.t], self.H[self.t], self.w, np.array([self.cash], dtype=np.float32), b])
            return np.nan_to_num(obs, nan=0.0, posinf=10.0, neginf=-10.0).astype(np.float32)
        def reset(self, seed=None, options=None):
            super().reset(seed=seed)
            self.t = 24
            self.w = np.zeros(self.n_assets, dtype=np.float32)
            self.cash = 1.0
            self.value = 1.0
            self.records = []
            return self._obs(), {}
        def _regime_values(self):
            row = self.regime_df.iloc[min(self.t, len(self.regime_df) - 1)]
            stress = float(row.get("stress_index", 0.0))
            risk_off = float(row.get("risk_off_regime", 0.0))
            drawdown_regime = float(row.get("drawdown_regime", 0.0))
            high_vol = float(row.get("high_vol_regime", 0.0))
            stress_norm = float(np.clip(sigmoid(2.0 * stress) - 0.5, 0, 1) * 2)
            return float(np.clip(0.45 * stress_norm + 0.25 * risk_off + 0.15 * drawdown_regime + 0.15 * high_vol, 0, 1))
        def step(self, action):
            action = np.asarray(action, dtype=np.float32)
            exposure = MIN_EXPOSURE + (1.0 - MIN_EXPOSURE) * sigmoid(action[0])
            logits = action[1:]
            b = self._baseline_weights(self.t)
            tilt = softmax(logits)
            proposed = 0.50 * b + 0.50 * tilt
            proposed = np.clip(proposed, 0, MAX_WEIGHT)
            proposed = proposed / max(proposed.sum(), 1e-12)
            proposed = exposure * proposed
            proposed_cash = max(0.0, 1.0 - proposed.sum())
            if self.t % REBALANCE_EVERY == 0:
                new_w = (1 - SMOOTHING) * self.w + SMOOTHING * proposed
                new_cash = (1 - SMOOTHING) * self.cash + SMOOTHING * proposed_cash
            else:
                new_w = self.w.copy()
                new_cash = self.cash
            turnover = float(np.sum(np.abs(new_w - self.w)) + abs(new_cash - self.cash))
            cost = TRANSACTION_COST * turnover
            asset_ret = self.R[self.t + 1]
            port_ret = float(np.dot(new_w, asset_ret))
            bench_ret = float(np.mean(asset_ret))
            net_growth = max((1 + port_ret) * (1 - cost), 1e-8)
            raw_reward = math.log(net_growth)
            conc = float(np.sum(new_w ** 2))
            risk_proxy = float(np.var(asset_ret) * conc)
            risk_off_score = self._regime_values()
            defensive_weight = float(new_w[defensive_idx].sum()) if len(defensive_idx) else 0.0
            growth_weight = float(new_w[growth_idx].sum()) if len(growth_idx) else 0.0
            rotation_score = risk_off_score * defensive_weight + (1.0 - risk_off_score) * growth_weight
            stress_dd_penalty = risk_off_score * growth_weight
            reward = (raw_reward + LAMBDA_BENCH * (port_ret - bench_ret) + LAMBDA_ROTATION * rotation_score - LAMBDA_STRESS_DD * stress_dd_penalty - LAMBDA_TURNOVER * turnover - LAMBDA_RISK * risk_proxy - LAMBDA_CONC * conc - CASH_PENALTY * new_cash)
            self.value *= net_growth
            self.w = new_w.astype(np.float32)
            self.cash = float(new_cash)
            self.records.append({"t": int(self.t), "portfolio_value": float(self.value), "portfolio_return": port_ret, "benchmark_return": bench_ret, "turnover": turnover, "cash": self.cash, "risk_off_score": risk_off_score, "defensive_weight": defensive_weight, "growth_weight": growth_weight, "rotation_score": rotation_score, "reward": float(reward)})
            self.t += 1
            terminated = self.t >= len(self.R) - 2
            return self._obs(), float(reward), terminated, False, self.records[-1]
    return PortfolioEnv


def evaluate_model(model, EnvClass, X, H, R, regime_df, tickers, name):
    env = EnvClass(X, H, R, regime_df, tickers, split_name="eval")
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    records = env.records
    equity = np.array([r["portfolio_value"] for r in records], dtype=np.float64)
    port_returns = np.array([r["portfolio_return"] for r in records], dtype=np.float64)
    turnover = np.array([r["turnover"] for r in records], dtype=np.float64)
    cash = np.array([r["cash"] for r in records], dtype=np.float64)
    metrics = {
        "name": name,
        "total_return": float(equity[-1] - 1.0),
        "max_drawdown": max_drawdown(equity),
        "sharpe": safe_sharpe(port_returns),
        "avg_turnover": float(turnover.mean()),
        "avg_cash": float(cash.mean()),
        "avg_defensive_weight": float(np.mean([r["defensive_weight"] for r in records])),
        "avg_growth_weight": float(np.mean([r["growth_weight"] for r in records])),
        "avg_risk_off_score": float(np.mean([r["risk_off_score"] for r in records])),
        "final_equity": float(equity[-1]),
    }
    return metrics, records


def run_seed(seed, data_pack):
    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback
    set_seed(seed)
    EnvClass = data_pack["EnvClass"]
    tickers = data_pack["tickers"]
    seed_dir = Path(RESULTS_DIR) / f"seed_{seed}"
    seed_dir.mkdir(parents=True, exist_ok=True)
    def make_train_env():
        return Monitor(EnvClass(data_pack["X_train"], data_pack["H_train"], data_pack["R_train"], data_pack["Reg_train"], tickers, split_name="train"))
    def make_val_env():
        return Monitor(EnvClass(data_pack["X_val"], data_pack["H_val"], data_pack["R_val"], data_pack["Reg_val"], tickers, split_name="val"))
    train_env = DummyVecEnv([make_train_env])
    val_env = DummyVecEnv([make_val_env])
    eval_callback = EvalCallback(val_env, best_model_save_path=str(seed_dir), log_path=str(seed_dir), eval_freq=EVAL_FREQ, deterministic=True, render=False)
    model = SAC("MlpPolicy", train_env, learning_rate=LEARNING_RATE, buffer_size=200_000, batch_size=256, tau=0.005, gamma=0.99, train_freq=1, gradient_steps=1, ent_coef=ENT_COEF, verbose=1, seed=seed, policy_kwargs=dict(net_arch=[256, 256]))
    print(f"\n=== TRAINING seed={seed}, timesteps={TOTAL_TIMESTEPS}, ent_coef={ENT_COEF} ===")
    model.learn(total_timesteps=TOTAL_TIMESTEPS, callback=eval_callback)
    final_path = seed_dir / "final_model.zip"
    model.save(final_path)
    best_path = seed_dir / "best_model.zip"
    eval_model = SAC.load(best_path) if best_path.exists() else model
    val_metrics, val_records = evaluate_model(eval_model, EnvClass, data_pack["X_val"], data_pack["H_val"], data_pack["R_val"], data_pack["Reg_val"], tickers, f"VAL_v7_seed{seed}")
    test_metrics, test_records = evaluate_model(eval_model, EnvClass, data_pack["X_test"], data_pack["H_test"], data_pack["R_test"], data_pack["Reg_test"], tickers, f"TEST_v7_seed{seed}")
    result = {"seed": seed, "ent_coef": ENT_COEF, "total_timesteps": TOTAL_TIMESTEPS, "val": val_metrics, "test": test_metrics, "best_model_path": str(best_path), "final_model_path": str(final_path)}
    (seed_dir / "metrics.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    (seed_dir / "val_records.json").write_text(json.dumps(val_records, indent=2), encoding="utf-8")
    (seed_dir / "test_records.json").write_text(json.dumps(test_records, indent=2), encoding="utf-8")
    print("VAL:", val_metrics)
    print("TEST:", test_metrics)
    return result


def main():
    close = build_close_panel(TICKERS)
    X, R, tickers, regime_df = build_features(close)
    H = load_behavioral_features(close.index)
    n = len(close)
    n_train = int(n * TRAIN_FRAC)
    n_val = int(n * VAL_FRAC)
    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n)
    print("Rows (train/val/test):", n_train, n_val, n - n_train - n_val)
    growth_idx = np.array([i for i, t in enumerate(tickers) if t in GROWTH_ASSETS], dtype=int)
    defensive_idx = np.array([i for i, t in enumerate(tickers) if t in DEFENSIVE_ASSETS], dtype=int)
    print("Growth assets:", [tickers[i] for i in growth_idx])
    print("Defensive assets:", [tickers[i] for i in defensive_idx])
    EnvClass = make_env_class(growth_idx, defensive_idx)
    data_pack = {
        "X_train": X[train_slice], "H_train": H[train_slice], "R_train": R[train_slice], "Reg_train": regime_df.iloc[train_slice].reset_index(drop=True),
        "X_val": X[val_slice], "H_val": H[val_slice], "R_val": R[val_slice], "Reg_val": regime_df.iloc[val_slice].reset_index(drop=True),
        "X_test": X[test_slice], "H_test": H[test_slice], "R_test": R[test_slice], "Reg_test": regime_df.iloc[test_slice].reset_index(drop=True),
        "tickers": tickers, "EnvClass": EnvClass,
    }
    baseline_eq = equal_weight_baseline(data_pack["R_test"])
    bottom_k = bottom_k_baseline(data_pack["R_test"])
    print("\n=== Baseline references on test ===")
    print(baseline_eq)
    print(bottom_k)
    metadata = {"universe": UNIVERSE_NAME, "tickers": tickers, "growth_assets": [tickers[i] for i in growth_idx], "defensive_assets": [tickers[i] for i in defensive_idx], "n_rows": int(n), "n_assets": len(tickers), "X_dim": int(X.shape[1]), "H_dim": int(H.shape[1]), "seeds": SEEDS, "total_timesteps": TOTAL_TIMESTEPS, "ent_coef": ENT_COEF, "min_exposure": MIN_EXPOSURE, "lambda_rotation": LAMBDA_ROTATION}
    Path(RESULTS_DIR, "metadata_v7.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    all_results = []
    for seed in SEEDS:
        try:
            res = run_seed(seed, data_pack)
            all_results.append(res)
        except Exception as e:
            print(f"[ERROR] seed {seed} failed:", e)
            all_results.append({"seed": seed, "error": str(e)})
        Path(RESULTS_DIR, "v7_seed_sweep_summary.json").write_text(json.dumps(all_results, indent=2), encoding="utf-8")
    rows = []
    rows.append({"kind": "baseline", **baseline_eq})
    rows.append({"kind": "baseline", **bottom_k})
    for res in all_results:
        if "test" in res:
            row = {"kind": "v7_test", "seed": res["seed"], "ent_coef": res["ent_coef"], "total_timesteps": res["total_timesteps"], **res["test"], "val_return": res["val"]["total_return"], "val_sharpe": res["val"]["sharpe"], "best_model_path": res["best_model_path"]}
            rows.append(row)
    df = pd.DataFrame(rows)
    csv_path = Path(RESULTS_DIR) / "v7_seed_sweep_summary.csv"
    df.to_csv(csv_path, index=False)
    print("\n=== V7 SUMMARY ===")
    print(df)
    print("\nSaved:")
    print(csv_path)
    print(Path(RESULTS_DIR) / "v7_seed_sweep_summary.json")


if __name__ == "__main__":
    main()
