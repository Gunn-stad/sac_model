"""
behavioral_sac_model_v5_1_options_info.py

Behavioral / event-aware SAC portfolio model.

This file expects these optional files from behavioral_data_pipeline.py:
  BASE_DIR/behavioral_data/stock_behavioral_signals.csv
  BASE_DIR/behavioral_data/global_event_signals.csv

It falls back to price/volume proxies and zeros if files do not exist.

Usage in Colab:
  from google.colab import drive
  drive.mount('/content/drive')
  !pip install numpy pandas yfinance pyarrow matplotlib torch stable-baselines3 gymnasium shimmy
  !python behavioral_data_pipeline.py
  !python behavioral_sac_model_v5_1_options_info.py
"""

from __future__ import annotations

import os
import json
import math
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Tuple, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")
print("RUNNING FILE:", __file__)

try:
    import gymnasium as gym
    from gymnasium import spaces
except Exception as e:
    raise ImportError("Install gymnasium: pip install gymnasium") from e


# ======================
# Config / paths
# ======================
IS_COLAB = os.path.exists("/content/drive")
BASE_DIR = os.environ.get("SAC_BASE_DIR", "/content/drive/MyDrive/lokaverkefni_bs" if IS_COLAB else ".")
CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results", "top20_trade_replay")
MODELS_DIR = os.path.join(BASE_DIR, "models")
BEHAVIORAL_DIR = os.path.join(BASE_DIR, "behavioral_data")

for p in [CACHE_DIR, RESULTS_DIR, MODELS_DIR, BEHAVIORAL_DIR]:
    os.makedirs(p, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("CACHE_DIR:", CACHE_DIR)
print("RESULTS_DIR:", RESULTS_DIR)
print("BEHAVIORAL_DIR:", BEHAVIORAL_DIR)

TOP20_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",
    "META", "TSLA", "JPM", "XOM", "UNH",
    "LLY", "AVGO", "COST", "WMT", "HD",
    "PG", "JNJ", "BAC", "ABBV", "CRM",
]


@dataclass
class ModelConfig:
    tickers: List[str]
    period: str = "730d"
    interval: str = "1h"
    universe_name: str = "top20"
    bottom_k: int = 5
    total_timesteps: int = 120_000
    seed: int = 0
    train_frac: float = 0.70
    val_frac: float = 0.15
    transaction_cost: float = 0.001
    rebalance_every: int = 3
    smooth_beta: float = 0.25
    w_max: float = 0.35
    lambda_risk: float = 0.006    # v5: lower risk penalty; still controls drawdown
    lambda_conc: float = 0.00025 # v5: lower concentration penalty
    lambda_exp: float = 0.000    # keep direct exposure penalty off
    lambda_cash: float = -0.0020 # v5.1: same cash penalty as v5
    lambda_bench: float = 1.25   # v5.1: same benchmark reward as v5
    min_exposure: float = 0.75   # v5.1: force at least 75% stock exposure
    max_exposure: float = 1.00   # v5: allow full stock exposure
    periods_per_year: float = 252 * 6.5
    learning_rate: float = 1e-4
    batch_size: int = 256
    gamma: float = 0.99
    tau: float = 0.005


def set_global_seed(seed: int = 0) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    try:
        import torch
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        torch.set_num_threads(1)
    except Exception:
        pass


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    z = np.asarray(x, dtype=np.float64) - np.max(x)
    ez = np.exp(z)
    s = np.sum(ez)
    if s <= 1e-12:
        return np.ones_like(z, dtype=np.float64) / len(z)
    return ez / s


def safe_zscore(s: pd.Series, window: int = 24) -> pd.Series:
    mean = s.rolling(window, min_periods=max(3, window // 4)).mean()
    std = s.rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
    return ((s - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def project_to_caps(weights: np.ndarray, w_max: float, max_iter: int = 100) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).copy()
    n = w.size
    if n == 0:
        return w.astype(np.float32)
    if w_max * n < 1.0 - 1e-12:
        raise ValueError(f"Infeasible caps: n={n}, w_max={w_max}")
    w = np.maximum(w, 0.0)
    s = w.sum()
    w = np.ones(n) / n if s <= 1e-12 else w / s
    free = np.ones(n, dtype=bool)
    out = np.zeros(n, dtype=np.float64)
    remaining = 1.0
    for _ in range(max_iter):
        if not np.any(free):
            break
        wf = w[free]
        if wf.sum() <= 1e-12:
            out[free] = remaining / np.sum(free)
            break
        alloc = remaining * wf / wf.sum()
        hit_cap = alloc > w_max
        free_idx = np.where(free)[0]
        if not np.any(hit_cap):
            out[free_idx] = alloc
            break
        capped_idx = free_idx[hit_cap]
        out[capped_idx] = w_max
        remaining = 1.0 - out.sum()
        free[capped_idx] = False
        if remaining <= 1e-12:
            break
    if remaining > 1e-10 and np.any(free):
        out[free] += remaining / np.sum(free)
    out = np.clip(out, 0.0, w_max)
    out = out / max(out.sum(), 1e-12)
    return out.astype(np.float32)


def turnover(w_new: np.ndarray, w_old: np.ndarray) -> float:
    return float(np.sum(np.abs(np.asarray(w_new) - np.asarray(w_old))))


def max_drawdown(equity_curve: np.ndarray) -> float:
    v = np.asarray(equity_curve, dtype=np.float64)
    peak = np.maximum.accumulate(v)
    dd = (peak - v) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def sharpe_ratio(returns: np.ndarray, periods_per_year: float) -> float:
    r = np.asarray(returns, dtype=np.float64)
    if r.size < 2:
        return float("nan")
    mu = np.mean(r)
    sd = np.std(r, ddof=1)
    if sd <= 1e-12:
        return float("nan")
    return float((mu / sd) * np.sqrt(periods_per_year))


# ======================
# Data
# ======================
def download_hourly_ohlcv(tickers: List[str], period: str, interval: str, cache_dir: str) -> Dict[str, pd.DataFrame]:
    import yfinance as yf
    os.makedirs(cache_dir, exist_ok=True)
    frames: Dict[str, pd.DataFrame] = {}
    for t in tickers:
        cache_path = os.path.join(cache_dir, f"{t}_{interval}_{period}_auto_adjust.parquet")
        if os.path.exists(cache_path):
            print(f"[cache] {t} -> {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print(f"[download] {t}")
            df = yf.download(t, period=period, interval=interval, auto_adjust=True, progress=False, threads=True)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.title)
            keep = [c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]
            df = df[keep].dropna()
            df.to_parquet(cache_path)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).floor("h")
        frames[t] = df[~df.index.duplicated(keep="last")]
    return frames


def align_close_volume(frames: Dict[str, pd.DataFrame], tickers: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
    idx = None
    for t in tickers:
        idx = frames[t].index if idx is None else idx.intersection(frames[t].index)
    idx = idx.sort_values()
    close = pd.DataFrame({t: frames[t].reindex(idx)["Close"].astype(float) for t in tickers}, index=idx)
    volume = pd.DataFrame({t: frames[t].reindex(idx).get("Volume", pd.Series(0, index=idx)).astype(float) for t in tickers}, index=idx)
    return close.dropna(), volume.reindex(close.dropna().index).fillna(0.0)


def make_market_features(close: pd.DataFrame, volume: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rets = close.pct_change().replace([np.inf, -np.inf], np.nan).fillna(0.0)
    feats = []
    market_ret = rets.mean(axis=1)
    for t in close.columns:
        p = close[t]
        r1 = rets[t]
        f = pd.DataFrame(index=close.index)
        f[f"{t}_ret_1h"] = r1
        for h in [4, 24, 48, 72]:
            f[f"{t}_ret_{h}h"] = p.pct_change(h).fillna(0.0)
        for h in [8, 24, 72]:
            ma = p.rolling(h, min_periods=max(3, h // 3)).mean()
            f[f"{t}_dist_ma_{h}"] = (p / ma - 1).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        f[f"{t}_vol_24h"] = r1.rolling(24, min_periods=6).std(ddof=0).fillna(0.0)
        f[f"{t}_ret1_z24"] = safe_zscore(r1, 24)
        f[f"{t}_volume_z24"] = safe_zscore(volume[t], 24)
        cov = r1.rolling(24, min_periods=8).cov(market_ret)
        var = market_ret.rolling(24, min_periods=8).var(ddof=0)
        f[f"{t}_beta_24"] = (cov / var.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(1.0)
        market_cum24 = (1 + market_ret).rolling(24, min_periods=6).apply(np.prod, raw=True).fillna(1.0) - 1
        f[f"{t}_relret_24"] = f[f"{t}_ret_24h"] - market_cum24
        feats.append(f)
    X = pd.concat(feats, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    return X, rets.astype(np.float32)




def make_options_info_features(close: pd.DataFrame, returns: pd.DataFrame, volume: pd.DataFrame, tickers: List[str]) -> pd.DataFrame:
    """
    Options-market expectation features used as STATE INFORMATION ONLY.
    Historical option chains are not always available for free, so this function creates useful proxies:
    implied-vol proxy, IV rank, downside skew proxy, put/call-pressure proxy.
    If you later create options_info_signals.csv, this can be extended to load real IV and put/call values.
    """
    blocks = []
    market_ret = returns.mean(axis=1)
    for t in tickers:
        r = returns[t].astype(float)
        vol = r.rolling(24, min_periods=8).std(ddof=0).fillna(0.0) * np.sqrt(252 * 6.5)
        vol_min = vol.rolling(252, min_periods=50).min()
        vol_max = vol.rolling(252, min_periods=50).max()
        iv_rank = ((vol - vol_min) / (vol_max - vol_min).replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        downside = r.clip(upper=0).abs().rolling(24, min_periods=8).mean().fillna(0.0)
        upside = r.clip(lower=0).rolling(24, min_periods=8).mean().fillna(0.0)
        skew_proxy = (downside - upside).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        volume_z = safe_zscore(volume[t], 24).astype(float)
        put_call_proxy = (np.maximum(-safe_zscore(r, 24), 0.0) * np.maximum(volume_z, 0.0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        f = pd.DataFrame(index=close.index)
        f[f"{t}_iv_proxy_annualized"] = vol
        f[f"{t}_iv_rank_proxy"] = iv_rank
        f[f"{t}_downside_skew_proxy"] = skew_proxy
        f[f"{t}_put_call_pressure_proxy"] = put_call_proxy
        f[f"{t}_options_fear_proxy"] = iv_rank * put_call_proxy
        blocks.append(f)
    out = pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)

    real_path = os.path.join(BEHAVIORAL_DIR, "options_info_signals.csv")
    if os.path.exists(real_path):
        print("[options] loading real options info", real_path)
        opt = pd.read_csv(real_path)
        opt["datetime"] = pd.to_datetime(opt["datetime"], errors="coerce").dt.floor("h")
        opt["ticker"] = opt["ticker"].astype(str).str.upper()
        wanted = ["implied_volatility", "put_call_ratio", "iv_rank", "skew", "options_volume_z", "open_interest_z"]
        wide_parts = []
        for col in wanted:
            if col not in opt.columns:
                opt[col] = 0.0
            opt[col] = pd.to_numeric(opt[col], errors="coerce").fillna(0.0)
            piv = opt.pivot_table(index="datetime", columns="ticker", values=col, aggfunc="mean")
            piv = piv.reindex(close.index).ffill().fillna(0.0)
            piv = piv.reindex(columns=tickers, fill_value=0.0)
            piv.columns = [f"{t}_real_{col}" for t in tickers]
            wide_parts.append(piv)
        real = pd.concat(wide_parts, axis=1).astype(np.float32)
        out = pd.concat([out, real], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    else:
        print("[options] No options_info_signals.csv found; using options proxies only.")
    return out


def load_external_behavioral_features(index: pd.DatetimeIndex, tickers: List[str]) -> Tuple[pd.DataFrame, Dict[str, Any]]:
    stock_path = os.path.join(BEHAVIORAL_DIR, "stock_behavioral_signals.csv")
    global_path = os.path.join(BEHAVIORAL_DIR, "global_event_signals.csv")
    meta: Dict[str, Any] = {"stock_path": stock_path, "global_path": global_path}

    blocks = []
    if os.path.exists(stock_path):
        print("[behavioral] loading", stock_path)
        s = pd.read_csv(stock_path)
        s["datetime"] = pd.to_datetime(s["datetime"], errors="coerce").dt.floor("h")
        s["ticker"] = s["ticker"].astype(str).str.upper()
        wanted = [
            "sentiment_score", "news_count_z", "news_negative_share", "google_trend_z",
            "attention_shock", "crowd_disagreement", "event_shock",
            "proxy_attention_z", "proxy_overreaction", "proxy_panic",
        ]
        for col in wanted:
            if col not in s.columns:
                s[col] = 0.0
            s[col] = pd.to_numeric(s[col], errors="coerce").fillna(0.0)
        wide_parts = []
        for col in wanted:
            piv = s.pivot_table(index="datetime", columns="ticker", values=col, aggfunc="mean")
            piv = piv.reindex(index).ffill().fillna(0.0)
            piv = piv.reindex(columns=tickers, fill_value=0.0)
            piv.columns = [f"{t}_{col}" for t in tickers]
            wide_parts.append(piv)
        blocks.append(pd.concat(wide_parts, axis=1))
        meta["stock_behavioral_loaded"] = True
        meta["stock_behavioral_columns"] = wanted
    else:
        print("[behavioral] No stock_behavioral_signals.csv found; using zeros.")
        meta["stock_behavioral_loaded"] = False
        zero_cols = {f"{t}_sentiment_score": 0.0 for t in tickers}
        blocks.append(pd.DataFrame(zero_cols, index=index))

    if os.path.exists(global_path):
        print("[behavioral] loading", global_path)
        g = pd.read_csv(global_path)
        g["datetime"] = pd.to_datetime(g["datetime"], errors="coerce").dt.floor("h")
        g = g.dropna(subset=["datetime"]).drop_duplicates("datetime", keep="last").set_index("datetime")
        wanted_g = [
            "vix_z", "vix_spike", "market_attention_z", "political_uncertainty",
            "rate_policy_signal", "inflation_event_signal", "macro_event_signal",
            "tariff_risk_signal", "influential_person_event",
        ]
        for col in wanted_g:
            if col not in g.columns:
                g[col] = 0.0
            g[col] = pd.to_numeric(g[col], errors="coerce").fillna(0.0)
        G = g[wanted_g].reindex(index).ffill().fillna(0.0)
        blocks.append(G)
        meta["global_behavioral_loaded"] = True
        meta["global_behavioral_columns"] = wanted_g
    else:
        print("[behavioral] No global_event_signals.csv found; using zeros.")
        meta["global_behavioral_loaded"] = False
        blocks.append(pd.DataFrame({"political_uncertainty": 0.0, "rate_policy_signal": 0.0}, index=index))

    H = pd.concat(blocks, axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    meta["n_behavioral_features"] = int(H.shape[1])
    meta["behavioral_feature_names"] = list(H.columns)
    return H, meta


def save_feature_diagnostics(X: pd.DataFrame, H: pd.DataFrame, meta: Dict[str, Any]) -> None:
    diag = pd.DataFrame({
        "feature": H.columns,
        "mean": H.mean().values,
        "std": H.std(ddof=0).values,
        "min": H.min().values,
        "max": H.max().values,
        "nonzero_fraction": (H.abs() > 1e-12).mean().values,
    })
    diag.to_csv(os.path.join(RESULTS_DIR, "behavioral_feature_diagnostics_v5_1_options_info.csv"), index=False)
    with open(os.path.join(RESULTS_DIR, "feature_metadata_v5_1_options_info.json"), "w", encoding="utf-8") as f:
        json.dump(meta, f, indent=2)
    print("[saved] diagnostics ->", os.path.join(RESULTS_DIR, "behavioral_feature_diagnostics_v5_1_options_info.csv"))


# ======================
# Baselines and env
# ======================
def contrarian_weights_from_features(close: pd.DataFrame, tickers: List[str], bottom_k: int, w_max: float) -> pd.DataFrame:
    ret24 = close.pct_change(24).replace([np.inf, -np.inf], np.nan).fillna(0.0)
    rows = []
    for _, row in ret24.iterrows():
        selected = row.nsmallest(bottom_k).index.tolist()
        w = np.zeros(len(tickers), dtype=np.float32)
        for t in selected:
            w[tickers.index(t)] = 1.0 / bottom_k
        w = project_to_caps(w, w_max)
        rows.append(w)
    return pd.DataFrame(rows, index=close.index, columns=tickers, dtype=np.float32)


class BehavioralContrarianPortfolioEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, features: pd.DataFrame, returns: pd.DataFrame, base_weights: pd.DataFrame, cfg: ModelConfig):
        super().__init__()
        self.features = features.reset_index(drop=True).astype(np.float32)
        self.returns = returns.reset_index(drop=True).astype(np.float32)
        self.base_weights = base_weights.reset_index(drop=True).astype(np.float32)
        self.timestamps = features.index if isinstance(features.index, pd.DatetimeIndex) else None
        self.cfg = cfg
        self.n_assets = len(cfg.tickers)
        self.feature_dim = self.features.shape[1]
        self.obs_dim = self.feature_dim + self.n_assets + 1 + self.n_assets
        self.action_dim = self.n_assets + 1
        self.action_space = spaces.Box(low=-5.0, high=5.0, shape=(self.action_dim,), dtype=np.float32)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(self.obs_dim,), dtype=np.float32)
        self.reset()

    def _get_obs(self) -> np.ndarray:
        feat = self.features.iloc[self.t].values.astype(np.float32)
        base = self.base_weights.iloc[self.t].values.astype(np.float32)
        return np.concatenate([feat, self.weights.astype(np.float32), np.array([self.cash], dtype=np.float32), base]).astype(np.float32)

    def reset(self, *, seed: Optional[int] = None, options: Optional[dict] = None):
        super().reset(seed=seed)
        self.t = 0
        self.equity = 1.0
        self.weights = np.zeros(self.n_assets, dtype=np.float32)
        self.cash = 1.0
        self.equity_curve = [self.equity]
        self.step_returns = []
        self.logs = []
        return self._get_obs(), {}

    def step(self, action: np.ndarray):
        action = np.asarray(action, dtype=np.float32)
        old_weights = self.weights.copy()
        old_cash = float(self.cash)
        # v5.1 exposure transform:
        # Old version: exposure = sigmoid(action[0]), which allowed the agent to sit near 0% stocks.
        # New version: keep exposure inside [min_exposure, max_exposure].
        exposure_raw = float(sigmoid(action[0]))
        exposure = float(self.cfg.min_exposure + (self.cfg.max_exposure - self.cfg.min_exposure) * exposure_raw)
        raw_tilt = softmax(action[1:])
        base = self.base_weights.iloc[self.t].values.astype(np.float32)
        mask = base > 1e-8
        if mask.any():
            masked = np.zeros(self.n_assets, dtype=np.float64)
            denom = float(raw_tilt[mask].sum())
            masked[mask] = raw_tilt[mask] / max(denom, 1e-12)
        else:
            masked = raw_tilt
        mixed = 0.5 * base + 0.5 * masked
        mixed = project_to_caps(mixed, self.cfg.w_max)
        prop_w = exposure * mixed
        prop_cash = max(0.0, 1.0 - float(np.sum(prop_w)))

        if (self.t + 1) % self.cfg.rebalance_every == 0:
            target_w = (1 - self.cfg.smooth_beta) * old_weights + self.cfg.smooth_beta * prop_w
            if target_w.sum() > 1.0:
                target_w = project_to_caps(target_w, self.cfg.w_max)
            target_cash = max(0.0, 1.0 - float(target_w.sum()))
        else:
            target_w = old_weights
            target_cash = old_cash

        to = turnover(target_w, old_weights) + abs(target_cash - old_cash)
        cost = self.cfg.transaction_cost * to
        next_r = self.returns.iloc[min(self.t + 1, len(self.returns) - 1)].values.astype(np.float64)
        port_ret = float(np.dot(target_w, next_r))
        bench_ret = float(np.mean(next_r))
        growth = max((1.0 + port_ret) * (1.0 - cost), 1e-12)
        new_equity = self.equity * growth
        raw_reward = math.log(growth)
        conc = float(np.sum(target_w ** 2))
        risk_proxy = float(np.var(next_r) * conc)
        exp = float(np.sum(target_w))
        reward = (
            raw_reward
            - self.cfg.lambda_risk * risk_proxy
            - self.cfg.lambda_conc * conc
            - self.cfg.lambda_exp * exp
            + self.cfg.lambda_cash * target_cash
            + self.cfg.lambda_bench * (port_ret - bench_ret)
        )

        self.weights = target_w.astype(np.float32)
        self.cash = float(target_cash)
        self.equity = float(new_equity)
        self.equity_curve.append(self.equity)
        self.step_returns.append(growth - 1.0)
        self.logs.append({
            "t": int(self.t),
            "equity": self.equity,
            "reward": float(reward),
            "raw_reward": float(raw_reward),
            "portfolio_return": port_ret,
            "benchmark_return": bench_ret,
            "turnover": float(to),
            "cost": float(cost),
            "cash": float(self.cash),
            "exposure_action": exposure,
            "exposure_raw": exposure_raw,
            "weights": self.weights.tolist(),
            "base_weights": base.tolist(),
        })

        self.t += 1
        terminated = self.t >= len(self.features) - 2
        truncated = False
        obs = self._get_obs() if not terminated else np.zeros(self.obs_dim, dtype=np.float32)
        info = self.logs[-1]
        return obs, float(reward), terminated, truncated, info


def evaluate_strategy(name: str, returns: pd.DataFrame, weights_df: Optional[pd.DataFrame], cfg: ModelConfig) -> Dict[str, float]:
    equity = 1.0
    curve = [equity]
    rets = []
    tos = []
    cash_vals = []
    old_w = np.zeros(len(cfg.tickers), dtype=np.float32)
    old_cash = 1.0
    for i in range(len(returns) - 1):
        if weights_df is None:
            w = np.ones(len(cfg.tickers), dtype=np.float32) / len(cfg.tickers)
        else:
            w = weights_df.iloc[i].values.astype(np.float32)
        cash = max(0.0, 1.0 - float(w.sum()))
        to = turnover(w, old_w) + abs(cash - old_cash)
        cost = cfg.transaction_cost * to if weights_df is not None else 0.0
        r = float(np.dot(w, returns.iloc[i + 1].values))
        growth = max((1 + r) * (1 - cost), 1e-12)
        equity *= growth
        curve.append(equity)
        rets.append(growth - 1.0)
        tos.append(to)
        cash_vals.append(cash)
        old_w, old_cash = w, cash
    return {
        "name": name,
        "total_return": float(equity - 1.0),
        "max_drawdown": max_drawdown(np.asarray(curve)),
        "sharpe": sharpe_ratio(np.asarray(rets), cfg.periods_per_year),
        "avg_turnover": float(np.mean(tos)) if tos else 0.0,
        "avg_cash": float(np.mean(cash_vals)) if cash_vals else 0.0,
        "final_equity": float(equity),
    }


def evaluate_sac_model(model, env: BehavioralContrarianPortfolioEnv, cfg: ModelConfig) -> Dict[str, float]:
    obs, _ = env.reset()
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated
    rets = np.asarray(env.step_returns)
    return {
        "name": "Hybrid_Contrarian_SAC_v5_1_options_info",
        "total_return": float(env.equity - 1.0),
        "max_drawdown": max_drawdown(np.asarray(env.equity_curve)),
        "sharpe": sharpe_ratio(rets, cfg.periods_per_year),
        "avg_turnover": float(np.mean([x["turnover"] for x in env.logs])) if env.logs else 0.0,
        "avg_cash": float(np.mean([x["cash"] for x in env.logs])) if env.logs else 0.0,
        "final_equity": float(env.equity),
    }


def split_time(X: pd.DataFrame, rets: pd.DataFrame, base: pd.DataFrame, cfg: ModelConfig):
    n = len(X)
    n_train = int(n * cfg.train_frac)
    n_val = int(n * cfg.val_frac)
    train = slice(0, n_train)
    val = slice(n_train, n_train + n_val)
    test = slice(n_train + n_val, n)
    return (
        X.iloc[train], rets.iloc[train], base.iloc[train],
        X.iloc[val], rets.iloc[val], base.iloc[val],
        X.iloc[test], rets.iloc[test], base.iloc[test],
    )


def main() -> None:
    cfg = ModelConfig(tickers=TOP20_TICKERS)
    set_global_seed(cfg.seed)
    print("Using universe:", cfg.universe_name)
    print("Number of stocks:", len(cfg.tickers))
    print("Contrarian basket size (bottom_k):", cfg.bottom_k)

    frames = download_hourly_ohlcv(cfg.tickers, cfg.period, cfg.interval, CACHE_DIR)
    close, volume = align_close_volume(frames, cfg.tickers)
    X_market, returns = make_market_features(close, volume)
    O = make_options_info_features(close.reindex(X_market.index), returns.reindex(X_market.index), volume.reindex(X_market.index), cfg.tickers)
    X_market = pd.concat([X_market, O], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    H, meta = load_external_behavioral_features(X_market.index, cfg.tickers)
    X = pd.concat([X_market, H], axis=1).replace([np.inf, -np.inf], np.nan).fillna(0.0).astype(np.float32)
    base_w = contrarian_weights_from_features(close.reindex(X.index), cfg.tickers, cfg.bottom_k, cfg.w_max)
    base_w = base_w.reindex(X.index).ffill().fillna(0.0).astype(np.float32)
    returns = returns.reindex(X.index).fillna(0.0).astype(np.float32)

    meta.update({
        "n_market_features": int(X_market.shape[1]),
        "n_total_features": int(X.shape[1]),
        "tickers": cfg.tickers,
        "config": cfg.__dict__,
    })
    save_feature_diagnostics(X_market, H, meta)

    X_train, r_train, b_train, X_val, r_val, b_val, X_test, r_test, b_test = split_time(X, returns, base_w, cfg)
    print("Rows (train/val/test):", len(X_train), len(X_val), len(X_test))

    from stable_baselines3 import SAC
    from stable_baselines3.common.callbacks import EvalCallback
    from stable_baselines3.common.monitor import Monitor

    train_env = Monitor(BehavioralContrarianPortfolioEnv(X_train, r_train, b_train, cfg))
    val_env = Monitor(BehavioralContrarianPortfolioEnv(X_val, r_val, b_val, cfg))
    test_env = BehavioralContrarianPortfolioEnv(X_test, r_test, b_test, cfg)

    eval_callback = EvalCallback(
        val_env,
        best_model_save_path=RESULTS_DIR,
        log_path=RESULTS_DIR,
        eval_freq=10_000,
        deterministic=True,
        render=False,
        n_eval_episodes=1,
    )

    model = SAC(
        "MlpPolicy",
        train_env,
        learning_rate=cfg.learning_rate,
        batch_size=cfg.batch_size,
        gamma=cfg.gamma,
        tau=cfg.tau,
        seed=cfg.seed,
        verbose=1,
        policy_kwargs=dict(net_arch=[256, 256]),
    )
    model.learn(total_timesteps=cfg.total_timesteps, callback=eval_callback)

    final_model_path = os.path.join(MODELS_DIR, "behavioral_sac_top20_v5_1_options_info_final.zip")
    model.save(final_model_path)
    print("[saved] final model ->", final_model_path)

    best_path = os.path.join(RESULTS_DIR, "best_model.zip")
    if os.path.exists(best_path):
        model_eval = SAC.load(best_path)
        print("[eval] using best validation model")
    else:
        model_eval = model
        print("[eval] best validation model missing; using final model")

    eq = evaluate_strategy("Baseline_EQ", r_test, None, cfg)
    bottom = evaluate_strategy(f"Bottom{cfg.bottom_k}_ret_24h", r_test, b_test, cfg)
    sac = evaluate_sac_model(model_eval, test_env, cfg)
    results = [eq, bottom, sac]

    print("\n=== Final evaluation on test ===")
    for row in results:
        print(row)
    summary_path = os.path.join(RESULTS_DIR, "summary_results_v5_1_options_info.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2)
    print("[saved] summary ->", summary_path)

    print("\n=== Exporting one human-readable replay ===")
    replay_path = os.path.join(RESULTS_DIR, "top20_episode_log_v5_1_options_info.json")
    with open(replay_path, "w", encoding="utf-8") as f:
        json.dump(test_env.logs, f, indent=2)
    print("[saved] trade replay log ->", replay_path)
    print("Done.")
    print("Replay file:", replay_path)
    print("Best model:", best_path)
    print("Final saved model:", final_model_path)


if __name__ == "__main__":
    main()
