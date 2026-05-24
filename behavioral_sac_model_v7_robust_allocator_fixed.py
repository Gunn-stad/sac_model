"""
behavioral_sac_model_v7_robust_allocator_fixed_v2.py

Fixed V7 robust allocator:
- fixes timezone mismatch
- fixes normalize_datetime_index missing function
- safe-growth diversified allocator
- fixed entropy coefficient
- multi-seed evaluation
"""

import os
import json
import math
import random
from pathlib import Path

import numpy as np
import pandas as pd

# -----------------------------
# Paths
# -----------------------------
BASE_DIR = "/content/drive/MyDrive/lokaverkefni_bs"
CACHE_DIR = f"{BASE_DIR}/data_cache"
BEHAVIORAL_DIR = f"{BASE_DIR}/behavioral_data"
RESULTS_DIR = f"{BASE_DIR}/results/v7_robust_allocator"
MODELS_DIR = f"{BASE_DIR}/models"

for d in [CACHE_DIR, BEHAVIORAL_DIR, RESULTS_DIR, MODELS_DIR]:
    os.makedirs(d, exist_ok=True)

# -----------------------------
# IMPORTANT FIX
# -----------------------------
def normalize_datetime_index(df):
    """
    Convert all datetime indexes to tz-naive UTC-normalized indexes.
    Fixes:
        Cannot join tz-naive with tz-aware DatetimeIndex
    """

    df = df.copy()

    if not isinstance(df.index, pd.DatetimeIndex):
        df.index = pd.to_datetime(
            df.index,
            utc=True,
            errors="coerce",
        )
    else:
        df.index = pd.to_datetime(
            df.index,
            utc=True,
            errors="coerce",
        )

    df = df[~df.index.isna()]

    df.index = df.index.tz_convert(None)

    df = df.sort_index()

    df = df[~df.index.duplicated(keep="last")]

    return df


print("RUNNING FILE:", __file__)

# -----------------------------
# Universe
# -----------------------------
TICKERS = [
    "SPY", "QQQ", "VTI", "DIA", "IWM",
    "XLK", "SMH", "SOXX", "BOTZ", "ARKK",
    "XLP", "XLU", "XLV",
    "SCHD", "VIG", "USMV", "SPLV",
    "SHY", "BIL", "SGOV", "IEF", "TLT", "TIP",
    "GLD",
]

SEEDS = [0, 1, 2]

INTERVAL = "1h"
PERIOD = "730d"

TOTAL_TIMESTEPS = 80000
ENT_COEF = 0.005

TRAIN_FRAC = 0.70
VAL_FRAC = 0.15

# -----------------------------
# Utils
# -----------------------------
def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)

def sharpe_ratio(x):
    x = np.asarray(x)
    if len(x) < 2 or np.std(x) < 1e-12:
        return 0.0
    return float(np.sqrt(252 * 6.5) * np.mean(x) / np.std(x))

def max_drawdown(equity):
    equity = np.asarray(equity)
    peak = np.maximum.accumulate(equity)
    dd = equity / peak - 1.0
    return float(abs(np.min(dd)))

# -----------------------------
# Data
# -----------------------------
def download_or_load(ticker):
    import yfinance as yf

    path = Path(CACHE_DIR) / f"{ticker}_{INTERVAL}_{PERIOD}_auto_adjust.parquet"

    if path.exists():
        print(f"[cache] {ticker} -> {path}")
        df = pd.read_parquet(path)
        df = normalize_datetime_index(df)
        return df

    print(f"[download] {ticker}")

    df = yf.download(
        ticker,
        period=PERIOD,
        interval=INTERVAL,
        auto_adjust=True,
        progress=False,
        threads=False,
    )

    if isinstance(df.columns, pd.MultiIndex):
        df.columns = [c[0] for c in df.columns]

    df = df.reset_index()

    dt_col = "Datetime" if "Datetime" in df.columns else "Date"

    df = df.rename(columns={dt_col: "datetime"})

    df["datetime"] = pd.to_datetime(
        df["datetime"],
        utc=True,
    ).dt.tz_convert(None)

    df = df.set_index("datetime")

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

            closes.append(
                df["Close"].rename(t)
            )

        except Exception as e:
            print(f"[warning] skipping {t}: {e}")

    if not closes:
        raise RuntimeError("No valid ticker data loaded.")

    close = pd.concat(closes, axis=1)

    close = normalize_datetime_index(close)

    close = close.ffill().dropna()

    return close

# -----------------------------
# Features
# -----------------------------
def build_features(close):
    returns = close.pct_change().fillna(0.0)

    feats = []

    for w in [1, 6, 24, 72]:
        r = close.pct_change(w).fillna(0.0)
        feats.append(r.values)

    X = np.concatenate(feats, axis=1).astype(np.float32)

    return X, returns.values.astype(np.float32)

# -----------------------------
# Environment
# -----------------------------
def make_env():
    import gymnasium as gym
    from gymnasium import spaces

    class PortfolioEnv(gym.Env):

        def __init__(self, X, R):
            super().__init__()

            self.X = X
            self.R = R

            self.n_assets = R.shape[1]

            self.observation_space = spaces.Box(
                low=-10,
                high=10,
                shape=(X.shape[1] + self.n_assets,),
                dtype=np.float32,
            )

            self.action_space = spaces.Box(
                low=-5,
                high=5,
                shape=(self.n_assets,),
                dtype=np.float32,
            )

        def reset(self, seed=None, options=None):
            self.t = 24
            self.w = np.ones(self.n_assets) / self.n_assets
            self.value = 1.0
            self.equity = [1.0]

            return self._obs(), {}

        def _obs(self):
            return np.concatenate([
                self.X[self.t],
                self.w,
            ]).astype(np.float32)

        def step(self, action):

            logits = np.asarray(action)

            e = np.exp(logits - np.max(logits))
            w = e / np.sum(e)

            ret = float(np.dot(w, self.R[self.t + 1]))

            self.value *= (1 + ret)

            self.equity.append(self.value)

            reward = math.log(max(1e-8, 1 + ret))

            self.w = w

            self.t += 1

            done = self.t >= len(self.R) - 2

            return self._obs(), reward, done, False, {
                "portfolio_value": self.value
            }

    return PortfolioEnv

# -----------------------------
# Evaluation
# -----------------------------
def evaluate_model(model, EnvClass, X, R):

    env = EnvClass(X, R)

    obs, _ = env.reset()

    done = False

    returns = []

    while not done:

        action, _ = model.predict(
            obs,
            deterministic=True,
        )

        obs, reward, terminated, truncated, info = env.step(action)

        done = terminated or truncated

        returns.append(reward)

    equity = np.array(env.equity)

    return {
        "total_return": float(equity[-1] - 1),
        "max_drawdown": max_drawdown(equity),
        "sharpe": sharpe_ratio(returns),
        "final_equity": float(equity[-1]),
    }

# -----------------------------
# Main
# -----------------------------
def main():

    close = build_close_panel(TICKERS)

    X, R = build_features(close)

    n = len(close)

    n_train = int(n * TRAIN_FRAC)

    n_val = int(n * VAL_FRAC)

    train_slice = slice(0, n_train)
    val_slice = slice(n_train, n_train + n_val)
    test_slice = slice(n_train + n_val, n)

    X_train = X[train_slice]
    R_train = R[train_slice]

    X_val = X[val_slice]
    R_val = R[val_slice]

    X_test = X[test_slice]
    R_test = R[test_slice]

    from stable_baselines3 import SAC
    from stable_baselines3.common.monitor import Monitor
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    EnvClass = make_env()

    results = []

    for seed in SEEDS:

        print(f"\n=== SEED {seed} ===")

        set_seed(seed)

        def make_train():
            return Monitor(
                EnvClass(X_train, R_train)
            )

        def make_val():
            return Monitor(
                EnvClass(X_val, R_val)
            )

        train_env = DummyVecEnv([make_train])

        val_env = DummyVecEnv([make_val])

        seed_dir = Path(RESULTS_DIR) / f"seed_{seed}"

        seed_dir.mkdir(
            parents=True,
            exist_ok=True,
        )

        callback = EvalCallback(
            val_env,
            best_model_save_path=str(seed_dir),
            log_path=str(seed_dir),
            eval_freq=10000,
            deterministic=True,
            render=False,
        )

        model = SAC(
            "MlpPolicy",
            train_env,
            learning_rate=1e-4,
            batch_size=256,
            buffer_size=200000,
            gamma=0.99,
            tau=0.005,
            ent_coef=ENT_COEF,
            verbose=1,
            seed=seed,
            policy_kwargs=dict(
                net_arch=[256, 256]
            ),
        )

        model.learn(
            total_timesteps=TOTAL_TIMESTEPS,
            callback=callback,
        )

        best_model = seed_dir / "best_model.zip"

        if best_model.exists():
            model = SAC.load(best_model)

        metrics = evaluate_model(
            model,
            EnvClass,
            X_test,
            R_test,
        )

        metrics["seed"] = seed

        results.append(metrics)

        print(metrics)

        with open(seed_dir / "metrics.json", "w") as f:
            json.dump(metrics, f, indent=2)

    df = pd.DataFrame(results)

    out_csv = Path(RESULTS_DIR) / "v7_seed_results.csv"

    df.to_csv(out_csv, index=False)

    print("\n=== FINAL RESULTS ===")
    print(df)

    print("\nSaved:")
    print(out_csv)

if __name__ == "__main__":
    main()
