"""
Analyze replay logs by market regime before paper/live trading.

This does NOT train a model. It reads a replay JSON file produced by your model
and creates regime/stress statistics.

Expected replay fields vary by script, so this parser is defensive.

Recommended input:
    /content/drive/MyDrive/lokaverkefni_bs/results/top20_trade_replay/top20_episode_log_*.json
"""

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

BASE_DIR = Path("/content/drive/MyDrive/lokaverkefni_bs")
RESULTS_DIR = BASE_DIR / "results" / "pre_live_tests" / "regime_stress"
RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Change this if needed.
REPLAY_FILE = BASE_DIR / "results" / "top20_trade_replay" / "top20_episode_log_v5_1.json"

def load_replay(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"Replay file not found: {path}")

    data = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(data, dict):
        for key in ["steps", "episode", "log", "records", "replay"]:
            if key in data and isinstance(data[key], list):
                data = data[key]
                break

    if not isinstance(data, list):
        raise ValueError("Replay JSON must be a list or contain a list under steps/episode/log/records/replay.")

    df = pd.DataFrame(data)
    return df

def find_col(df: pd.DataFrame, candidates):
    for c in candidates:
        if c in df.columns:
            return c
    return None

def max_drawdown(equity: pd.Series) -> float:
    running_max = equity.cummax()
    dd = equity / running_max - 1.0
    return float(abs(dd.min()))

def sharpe_from_returns(r: pd.Series, periods_per_year=252*6.5) -> float:
    r = pd.Series(r).replace([np.inf, -np.inf], np.nan).dropna()
    if len(r) < 2 or r.std() == 0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / r.std())

def summarize_slice(name: str, d: pd.DataFrame, equity_col: str, ret_col: str | None):
    if len(d) == 0:
        return {"regime": name, "n": 0}

    equity = d[equity_col].astype(float)
    if ret_col:
        rets = d[ret_col].astype(float)
    else:
        rets = equity.pct_change().fillna(0.0)

    return {
        "regime": name,
        "n": int(len(d)),
        "start_equity": float(equity.iloc[0]),
        "end_equity": float(equity.iloc[-1]),
        "total_return": float(equity.iloc[-1] / equity.iloc[0] - 1.0) if equity.iloc[0] != 0 else np.nan,
        "max_drawdown": max_drawdown(equity),
        "sharpe_proxy": sharpe_from_returns(rets),
        "avg_return": float(rets.mean()),
        "volatility": float(rets.std()),
    }

def main():
    df = load_replay(REPLAY_FILE)

    equity_col = find_col(df, ["portfolio_value", "equity", "portfolio_equity", "value", "V"])
    ret_col = find_col(df, ["portfolio_return", "port_return", "return", "reward_raw"])
    market_ret_col = find_col(df, ["market_return", "benchmark_return", "equal_weight_return", "bench_return"])
    vol_col = find_col(df, ["volatility_regime", "market_volatility", "realized_vol", "vol"])

    if equity_col is None:
        raise ValueError(f"Could not find equity column. Columns found: {list(df.columns)}")

    # If no market return exists, use portfolio return proxy for regime classification.
    if market_ret_col is None:
        if ret_col is None:
            df["_ret_proxy"] = df[equity_col].astype(float).pct_change().fillna(0.0)
            market_ret_col = "_ret_proxy"
        else:
            market_ret_col = ret_col

    df["_market_ret"] = df[market_ret_col].astype(float).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    df["_rolling_vol"] = df["_market_ret"].rolling(24, min_periods=5).std().fillna(0.0)
    df["_rolling_ret"] = df["_market_ret"].rolling(24, min_periods=5).sum().fillna(0.0)

    vol_q75 = df["_rolling_vol"].quantile(0.75)
    vol_q25 = df["_rolling_vol"].quantile(0.25)

    regimes = {
        "all": df,
        "high_volatility": df[df["_rolling_vol"] >= vol_q75],
        "low_volatility": df[df["_rolling_vol"] <= vol_q25],
        "market_stress_negative_24h": df[df["_rolling_ret"] < 0],
        "market_positive_24h": df[df["_rolling_ret"] > 0],
        "large_negative_shock": df[df["_market_ret"] <= df["_market_ret"].quantile(0.05)],
        "large_positive_shock": df[df["_market_ret"] >= df["_market_ret"].quantile(0.95)],
    }

    rows = [summarize_slice(name, d, equity_col, ret_col) for name, d in regimes.items()]
    out = pd.DataFrame(rows)
    out_path = RESULTS_DIR / "regime_stress_summary.csv"
    out.to_csv(out_path, index=False)

    print("Saved regime/stress summary:")
    print(out_path)
    print(out)

if __name__ == "__main__":
    main()
