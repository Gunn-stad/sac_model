import os
import json
import math
import random
import warnings
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple, Optional

import numpy as np
import pandas as pd
import yfinance as yf

warnings.filterwarnings("ignore")
print("RUNNING FILE:", __file__)


# ======================
# Storage / Colab setup
# ======================
# Edit this file in VS Code, push to GitHub, then run it in Google Colab.
# When Google Drive is mounted in Colab, all large files are stored there
# instead of on your laptop or inside the temporary Colab runtime.
IS_COLAB = os.path.exists("/content/drive")

if IS_COLAB:
    BASE_DIR = "/content/drive/MyDrive/lokaverkefni_bs"
else:
    BASE_DIR = os.path.abspath(".")

CACHE_DIR = os.path.join(BASE_DIR, "data_cache")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
MODELS_DIR = os.path.join(BASE_DIR, "models")
LOGS_DIR = os.path.join(BASE_DIR, "logs")

for _path in [CACHE_DIR, RESULTS_DIR, MODELS_DIR, LOGS_DIR]:
    os.makedirs(_path, exist_ok=True)

print("BASE_DIR:", BASE_DIR)
print("CACHE_DIR:", CACHE_DIR)
print("RESULTS_DIR:", RESULTS_DIR)


# ======================
# Behavioral feature switch
# ======================
# Keep this True so the observation space is ready for behavioral finance,
# crowd wisdom, and influential-person/event signals. At first these features
# are safe placeholders or simple proxies from prices/volume. Later you can
# replace them with real Reddit/X/news/Google Trends/Fed/political-event data.
USE_BEHAVIORAL_FEATURES = True
# If True, the code tries to load real behavioral/event data from CSV files.
# If the files are not found, the model still runs using market-derived proxies
# and zero-filled placeholder columns.
USE_EXTERNAL_BEHAVIORAL_DATA = True

# Put optional CSV files here in Google Drive:
#   /content/drive/MyDrive/lokaverkefni_bs/behavioral_data/
# or locally:
#   ./behavioral_data/
BEHAVIORAL_DATA_DIR = os.path.join(BASE_DIR, "behavioral_data")
os.makedirs(BEHAVIORAL_DATA_DIR, exist_ok=True)

# External stock-level CSV expected long format:
#   timestamp,ticker,news_sentiment,social_sentiment,social_mentions_z,google_trends_z,
#   influencer_event_shock,political_event_shock,crowd_disagreement
#
# External global CSV expected format:
#   timestamp,fear_greed_index,political_uncertainty,rate_policy_signal,tariff_risk_signal,
#   fed_speech_event,president_event,ceo_influencer_event



# ======================
# Universe
# ======================
TOP20_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",
    "META", "TSLA", "JPM", "XOM", "UNH",
    "LLY", "AVGO", "COST", "WMT", "HD",
    "PG", "JNJ", "BAC", "ABBV", "CRM",
]


# ======================
# Utils
# ======================
def set_global_seed(seed: int = 0) -> None:
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


def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-x))


def softmax(x: np.ndarray) -> np.ndarray:
    z = x - np.max(x)
    ez = np.exp(z)
    s = np.sum(ez)
    if s <= 0:
        return np.ones_like(x) / len(x)
    return ez / s


def project_to_caps(weights: np.ndarray, w_max: float, max_iter: int = 100) -> np.ndarray:
    w = np.asarray(weights, dtype=np.float64).copy()
    w[w < 0.0] = 0.0
    s = w.sum()
    if s <= 0:
        return w.astype(np.float32)

    w /= s
    for _ in range(max_iter):
        over = w > w_max
        if not np.any(over):
            break
        excess = np.sum(w[over] - w_max)
        w[over] = w_max
        under = ~over
        under_sum = np.sum(w[under])
        if under_sum <= 1e-12:
            break
        w[under] += excess * (w[under] / under_sum)
        w[w < 0.0] = 0.0
        s = w.sum()
        if s > 0:
            w /= s
    return w.astype(np.float32)


def max_drawdown(equity: np.ndarray) -> float:
    peak = np.maximum.accumulate(equity)
    dd = (peak - equity) / np.maximum(peak, 1e-12)
    return float(np.max(dd))


def sharpe_ratio(rets: np.ndarray, periods_per_year: float) -> float:
    if len(rets) < 2:
        return 0.0
    mu = np.mean(rets)
    sd = np.std(rets)
    if sd <= 1e-12:
        return 0.0
    return float((mu / sd) * math.sqrt(periods_per_year))


def summarize(
    name: str,
    equity: np.ndarray,
    rets: np.ndarray,
    tos: np.ndarray,
    cashes: np.ndarray,
    periods_per_year: float,
) -> Dict[str, Any]:
    return {
        "name": name,
        "total_return": float(equity[-1] / equity[0] - 1.0),
        "max_drawdown": float(max_drawdown(equity)),
        "sharpe": float(sharpe_ratio(rets, periods_per_year)),
        "avg_turnover": float(np.mean(tos)) if len(tos) else 0.0,
        "avg_cash": float(np.mean(cashes)) if len(cashes) else 0.0,
        "final_equity": float(equity[-1]),
    }


def split_train_val_test(df: pd.DataFrame, train_frac=0.70, val_frac=0.15):
    n = len(df)
    i_train = int(n * train_frac)
    i_val = int(n * (train_frac + val_frac))
    return df.iloc[:i_train], df.iloc[i_train:i_val], df.iloc[i_val:]


def choose_bottom_k(n: int) -> int:
    if n <= 10:
        return 3
    if n <= 20:
        return 5
    return 6


# ======================
# Data
# ======================
def download_close_volume(
    tickers: List[str],
    period: str = "730d",
    interval: str = "1h",
    cache_dir: str = CACHE_DIR,
) -> Tuple[pd.DataFrame, pd.DataFrame]:
    close_frames = []
    vol_frames = []
    os.makedirs(cache_dir, exist_ok=True)

    for tkr in tickers:
        cache_path = os.path.join(cache_dir, f"{tkr}_{interval}_{period}_auto_adjust.parquet")

        if os.path.exists(cache_path):
            print(f"[cache] {tkr} -> {cache_path}")
            df = pd.read_parquet(cache_path)
        else:
            print(f"[download] {tkr} ({period}, {interval})")
            df = yf.download(
                tkr,
                period=period,
                interval=interval,
                auto_adjust=True,
                progress=False,
                threads=False,
            )
            if df is None or df.empty:
                raise ValueError(f"No data for {tkr}")

            if isinstance(df.columns, pd.MultiIndex):
                cols = [c[0] for c in df.columns]
                df.columns = cols

            df.to_parquet(cache_path)

        if isinstance(df.columns, pd.MultiIndex):
            cols = [c[0] for c in df.columns]
            df.columns = cols

        close_frames.append(df["Close"].rename(tkr))
        vol_frames.append(df["Volume"].rename(tkr))

    closes = pd.concat(close_frames, axis=1).sort_index()
    vols = pd.concat(vol_frames, axis=1).sort_index()

    closes = closes.dropna()
    vols = vols.reindex(closes.index).ffill().fillna(0.0)

    return closes, vols


def _rolling_zscore(x: pd.DataFrame, window: int = 24) -> pd.DataFrame:
    return (x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-12)



def _read_csv_with_timestamp(path: str) -> Optional[pd.DataFrame]:
    """Read a CSV that has a timestamp/date/datetime column and return a time-indexed DataFrame."""
    if not os.path.exists(path):
        return None

    df = pd.read_csv(path)
    if df.empty:
        return None

    candidates = ["timestamp", "datetime", "date", "time", "Date", "Datetime"]
    ts_col = next((c for c in candidates if c in df.columns), None)
    if ts_col is None:
        raise ValueError(f"{path} must contain one timestamp column, e.g. 'timestamp'.")

    df[ts_col] = pd.to_datetime(df[ts_col], errors="coerce", utc=True)
    df = df.dropna(subset=[ts_col]).sort_values(ts_col).set_index(ts_col)
    # Match yfinance timezone style as safely as possible.
    try:
        df.index = df.index.tz_convert(None)
    except Exception:
        try:
            df.index = df.index.tz_localize(None)
        except Exception:
            pass
    return df


def _safe_zscore_series(x: pd.Series, window: int = 24) -> pd.Series:
    return ((x - x.rolling(window).mean()) / (x.rolling(window).std() + 1e-12)).replace([np.inf, -np.inf], np.nan)


def _align_external_time_features(df: pd.DataFrame, target_index: pd.Index, shift_one_step: bool = True) -> pd.DataFrame:
    """
    Align external data to the price timestamps.

    We forward-fill past known values, then optionally shift by one step. The shift is
    important because it reduces look-ahead leakage: the agent at time t should not
    see a news/event value that would only be known after t.
    """
    out = df.sort_index().reindex(target_index, method="ffill")
    if shift_one_step:
        out = out.shift(1)
    return out.fillna(0.0)


def load_external_stock_behavioral_features(
    target_index: pd.Index,
    tickers: List[str],
    data_dir: str = BEHAVIORAL_DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Optional real stock-level behavioral signals.

    Expected file:
        stock_behavioral_signals.csv

    Expected long format:
        timestamp,ticker,news_sentiment,social_sentiment,social_mentions_z,
        google_trends_z,influencer_event_shock,political_event_shock,crowd_disagreement

    The output columns use the same MultiIndex layout as the rest of the feature matrix:
        (ticker, feature_name)
    """
    path = os.path.join(data_dir, "stock_behavioral_signals.csv")
    df = _read_csv_with_timestamp(path)
    if df is None:
        print("[behavioral] No stock_behavioral_signals.csv found; using proxies/placeholders.")
        return None
    if "ticker" not in df.columns:
        raise ValueError("stock_behavioral_signals.csv must contain a 'ticker' column.")

    signal_cols = [
        "news_sentiment",
        "social_sentiment",
        "social_mentions_z",
        "google_trends_z",
        "influencer_event_shock",
        "political_event_shock",
        "crowd_disagreement",
    ]
    present = [c for c in signal_cols if c in df.columns]
    if not present:
        raise ValueError(f"stock_behavioral_signals.csv must contain at least one of: {signal_cols}")

    blocks = []
    for col in present:
        wide = df.pivot_table(index=df.index, columns="ticker", values=col, aggfunc="last")
        wide = wide.reindex(columns=tickers)
        wide = _align_external_time_features(wide, target_index, shift_one_step=True)
        wide.columns = pd.MultiIndex.from_product([wide.columns, [col]])
        blocks.append(wide)

    out = pd.concat(blocks, axis=1).sort_index(axis=1).astype(np.float32)
    print(f"[behavioral] Loaded stock-level behavioral signals from {path}: {out.shape[1]} columns")
    return out


def load_external_global_behavioral_features(
    target_index: pd.Index,
    data_dir: str = BEHAVIORAL_DATA_DIR,
) -> Optional[pd.DataFrame]:
    """
    Optional real global/event signals.

    Expected file:
        global_event_signals.csv

    Expected columns:
        timestamp,fear_greed_index,political_uncertainty,rate_policy_signal,
        tariff_risk_signal,fed_speech_event,president_event,ceo_influencer_event
    """
    path = os.path.join(data_dir, "global_event_signals.csv")
    df = _read_csv_with_timestamp(path)
    if df is None:
        print("[behavioral] No global_event_signals.csv found; using proxies/placeholders.")
        return None

    signal_cols = [
        "fear_greed_index",
        "political_uncertainty",
        "rate_policy_signal",
        "tariff_risk_signal",
        "fed_speech_event",
        "president_event",
        "ceo_influencer_event",
    ]
    present = [c for c in signal_cols if c in df.columns]
    if not present:
        raise ValueError(f"global_event_signals.csv must contain at least one of: {signal_cols}")

    out = _align_external_time_features(df[present], target_index, shift_one_step=True)
    out.columns = pd.MultiIndex.from_product([["global"], out.columns])
    print(f"[behavioral] Loaded global event signals from {path}: {out.shape[1]} columns")
    return out.astype(np.float32)


def save_behavioral_feature_diagnostics(
    feats: pd.DataFrame,
    rets1: pd.DataFrame,
    tickers: List[str],
    out_path: str,
    top_n: int = 40,
) -> None:
    """
    Save simple diagnostics showing which behavioral features have the strongest
    absolute correlation with next-step returns. This is not proof of causality, but
    it is a useful sanity check for whether the added features contain signal.
    """
    rows = []
    next_rets = rets1.shift(-1).reindex(feats.index)

    for col in feats.columns:
        ticker, feature = col
        if ticker == "global":
            y = next_rets.mean(axis=1)
        elif ticker in tickers:
            y = next_rets[ticker]
        else:
            continue

        x = feats[col]
        valid = x.notna() & y.notna() & np.isfinite(x) & np.isfinite(y)
        if valid.sum() < 50:
            continue
        corr = float(np.corrcoef(x[valid].to_numpy(), y[valid].to_numpy())[0, 1])
        if np.isfinite(corr):
            rows.append({"ticker": ticker, "feature": feature, "corr_with_next_return": corr, "abs_corr": abs(corr)})

    diag = pd.DataFrame(rows).sort_values("abs_corr", ascending=False).head(top_n)
    diag.to_csv(out_path, index=False)
    print(f"[saved] behavioral feature diagnostics -> {out_path}")


def build_behavioral_features(closes: pd.DataFrame, vols: pd.DataFrame) -> pd.DataFrame:
    """
    Behavioral-finance / wisdom-of-the-crowd / event-aware feature block.

    This version has two layers:
      1. Real market-behavior proxies from price and volume.
      2. Safe placeholder columns for future external data, such as news sentiment,
         Reddit/X/Stocktwits mentions, Google Trends, Fed/political events,
         CEO/influencer events, and sector-specific policy shocks.

    Important: the SAC agent should receive these as numerical state features.
    Do not hard-code rules like "if politician says X, buy Y". Let the agent
    learn from historical relationships during backtesting.
    """
    rets1 = closes.pct_change()
    market_ret = rets1.mean(axis=1)

    feature_blocks = []

    def add_stock_block(name: str, block: pd.DataFrame):
        b = block.copy().reindex(index=closes.index, columns=closes.columns)
        b.columns = pd.MultiIndex.from_product([b.columns, [name]])
        feature_blocks.append(b)

    # --- Behavioral proxies per stock ---
    # Attention / crowd-interest proxy: abnormal volume.
    mention_proxy = _rolling_zscore(vols, 24).clip(-10, 10)
    add_stock_block("crowd_attention_proxy", mention_proxy)

    # Panic / euphoria proxy: large negative/positive short-term moves.
    panic_proxy = (-_rolling_zscore(rets1, 24)).clip(-10, 10)
    euphoria_proxy = (_rolling_zscore(rets1, 24)).clip(-10, 10)
    add_stock_block("panic_proxy", panic_proxy)
    add_stock_block("euphoria_proxy", euphoria_proxy)

    # Overreaction and possible reversal proxy: strong 24h move relative to recent volatility.
    ret_24h = closes / closes.shift(24) - 1.0
    vol_24h = rets1.rolling(24).std()
    overreaction_proxy = (ret_24h / (vol_24h * math.sqrt(24) + 1e-12)).clip(-10, 10)
    add_stock_block("overreaction_proxy", overreaction_proxy)

    # Herding proxy: stock return aligned with market return, scaled by attention.
    aligned = np.sign(rets1).mul(np.sign(market_ret), axis=0)
    herding_proxy = (aligned * mention_proxy).clip(-10, 10)
    add_stock_block("herding_proxy", herding_proxy)

    # --- Placeholder per-stock columns for later real external data ---
    zeros_stock = pd.DataFrame(0.0, index=closes.index, columns=closes.columns)
    for name in [
        "news_sentiment",
        "social_sentiment",
        "social_mentions_z",
        "google_trends_z",
        "influencer_event_shock",
        "political_event_shock",
        "crowd_disagreement",
    ]:
        add_stock_block(name, zeros_stock)

    stock_behavior = pd.concat(feature_blocks, axis=1).sort_index(axis=1)

    # --- Global market/event placeholders and proxies ---
    global_features = pd.DataFrame(index=closes.index)
    global_features["global", "market_fear_proxy"] = (-_rolling_zscore(market_ret.to_frame("mkt"), 24)["mkt"]).clip(-10, 10)
    global_features["global", "market_euphoria_proxy"] = (_rolling_zscore(market_ret.to_frame("mkt"), 24)["mkt"]).clip(-10, 10)
    global_features["global", "market_attention_proxy"] = _rolling_zscore(vols.mean(axis=1).to_frame("vol"), 24)["vol"].clip(-10, 10)

    # These are placeholders to be filled later from real event datasets.
    global_features["global", "fear_greed_index"] = 0.0
    global_features["global", "political_uncertainty"] = 0.0
    global_features["global", "rate_policy_signal"] = 0.0
    global_features["global", "tariff_risk_signal"] = 0.0
    global_features["global", "fed_speech_event"] = 0.0
    global_features["global", "president_event"] = 0.0
    global_features["global", "ceo_influencer_event"] = 0.0
    global_features.columns = pd.MultiIndex.from_tuples(global_features.columns)

    out = pd.concat([stock_behavior, global_features], axis=1).replace([np.inf, -np.inf], np.nan)

    if USE_EXTERNAL_BEHAVIORAL_DATA:
        ext_stock = load_external_stock_behavioral_features(closes.index, list(closes.columns), BEHAVIORAL_DATA_DIR)
        ext_global = load_external_global_behavioral_features(closes.index, BEHAVIORAL_DATA_DIR)

        # External values override same-named placeholders/proxies when present.
        if ext_stock is not None:
            for col in ext_stock.columns:
                out[col] = ext_stock[col]
        if ext_global is not None:
            for col in ext_global.columns:
                out[col] = ext_global[col]

    return out.astype(np.float32)


def build_features(closes: pd.DataFrame, vols: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame]:
    rets1 = closes.pct_change()

    feature_blocks = []

    def add_block(name: str, block: pd.DataFrame):
        b = block.copy()
        b.columns = pd.MultiIndex.from_product([b.columns, [name]])
        feature_blocks.append(b)

    add_block("ret_1h", rets1)
    add_block("ret_4h", closes / closes.shift(4) - 1.0)
    add_block("ret_24h", closes / closes.shift(24) - 1.0)
    add_block("ret_48h", closes / closes.shift(48) - 1.0)
    add_block("ret_72h", closes / closes.shift(72) - 1.0)

    ma_24 = closes.rolling(24).mean()
    dist_ma_24 = closes / ma_24 - 1.0
    add_block("dist_ma_24h", dist_ma_24)

    vol_24 = rets1.rolling(24).std()
    add_block("vol_24h", vol_24)

    vol_z = (vols - vols.rolling(24).mean()) / (vols.rolling(24).std() + 1e-12)
    add_block("vol_z_24h", vol_z)

    feats = pd.concat(feature_blocks, axis=1).sort_index()

    if USE_BEHAVIORAL_FEATURES:
        behavioral_feats = build_behavioral_features(closes, vols)
        feats = pd.concat([feats, behavioral_feats], axis=1).sort_index(axis=1)

    full = pd.concat([feats, rets1], axis=1).dropna()
    feats = feats.loc[full.index]
    rets1 = rets1.loc[full.index]

    return feats, rets1


def extract_feature_block(feats: pd.DataFrame, feature_name: str) -> pd.DataFrame:
    return feats.xs(feature_name, axis=1, level=1)


def compute_bottomk_base_weights(
    feats: pd.DataFrame,
    tickers: List[str],
    score_feature: str = "ret_24h",
    bottom_k: int = 5,
    w_max: float = 0.20,
) -> pd.DataFrame:
    score_block = extract_feature_block(feats, score_feature).reindex(columns=tickers)
    rows = []

    for _, row in score_block.iterrows():
        scores = row.to_numpy(dtype=np.float64)
        valid = np.isfinite(scores)
        w = np.zeros(len(tickers), dtype=np.float32)
        if np.any(valid):
            idx = np.where(valid)[0]
            ranked = idx[np.argsort(scores[idx])]
            chosen = ranked[: min(bottom_k, len(ranked))]
            if len(chosen) > 0:
                w[chosen] = 1.0 / len(chosen)
                w = project_to_caps(w, w_max)
        rows.append(w)

    return pd.DataFrame(rows, index=score_block.index, columns=tickers)


# ======================
# Environment
# ======================
import gymnasium as gym
from gymnasium import spaces


@dataclass
class EnvConfig:
    kappa: float = 0.0015
    w_max: float = 0.20
    episode_len: int = 24 * 10
    seed: int = 0
    rebalance_every: int = 6
    action_smooth: float = 0.10
    risk_lambda: float = 0.10
    hold_cash_bonus: float = 0.0
    concentration_lambda: float = 0.0005
    benchmark_lambda: float = 0.05
    exposure_penalty_lambda: float = 0.0
    start_mode: str = "equal_weight"


class HybridContrarianEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(
        self,
        feats: pd.DataFrame,
        rets1: pd.DataFrame,
        base_weights: pd.DataFrame,
        tickers: List[str],
        cfg: EnvConfig,
    ):
        super().__init__()
        self.feats = feats
        self.rets1 = rets1
        self.base_weights = base_weights.reindex(index=feats.index, columns=tickers)
        self.tickers = tickers
        self.n = len(tickers)
        self.cfg = cfg

        self.feature_cols = list(self.feats.columns)
        self.feat_dim = len(self.feature_cols)

        self.action_space = spaces.Box(low=-10.0, high=10.0, shape=(self.n + 1,), dtype=np.float32)
        obs_dim = self.feat_dim + self.n + 1 + self.n
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)

        self.rng = np.random.default_rng(cfg.seed)

        self.t0 = 0
        self.t = 0
        self.w = np.zeros(self.n, dtype=np.float32)
        self.cash = 1.0
        self.V = 1.0

    def _obs(self) -> np.ndarray:
        feat_vec = self.feats.iloc[self.t0 + self.t].to_numpy(dtype=np.float32)
        base_vec = self.base_weights.iloc[self.t0 + self.t].to_numpy(dtype=np.float32)
        return np.concatenate([
            feat_vec,
            self.w.astype(np.float32),
            np.array([self.cash], dtype=np.float32),
            base_vec.astype(np.float32),
        ]).astype(np.float32)

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        max_start = len(self.feats) - self.cfg.episode_len - 2
        if max_start <= 0:
            self.t0 = 0
        else:
            self.t0 = int(self.rng.integers(0, max_start))

        self.t = 0
        self.V = 1.0

        if self.cfg.start_mode == "equal_weight":
            self.w = np.ones(self.n, dtype=np.float32) / self.n
            self.w = project_to_caps(self.w, self.cfg.w_max)
            self.cash = float(max(0.0, 1.0 - np.sum(self.w)))
        else:
            self.w = np.zeros(self.n, dtype=np.float32)
            self.cash = 1.0

        return self._obs(), {}

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        exposure_logit = action[0]
        tilt_logits = action[1:]

        exposure = float(sigmoid(np.array([exposure_logit]))[0])
        tilt = softmax(tilt_logits)

        base_w = self.base_weights.iloc[self.t0 + self.t].to_numpy(dtype=np.float64)

        mask = (base_w > 0).astype(np.float64)
        if mask.sum() > 0:
            masked = tilt * mask
            masked_sum = masked.sum()
            if masked_sum > 0:
                tilt_masked = masked / masked_sum
            else:
                tilt_masked = base_w.copy()
        else:
            tilt_masked = tilt.copy()

        w_mix = 0.5 * base_w + 0.5 * tilt_masked
        w_mix = project_to_caps(w_mix, self.cfg.w_max).astype(np.float64)

        w_prop = exposure * w_mix
        cash_prop = float(max(0.0, 1.0 - np.sum(w_prop)))

        if ((self.t + 1) % self.cfg.rebalance_every) == 0:
            beta = self.cfg.action_smooth
            w_target = (1.0 - beta) * self.w + beta * w_prop
            w_target = project_to_caps(w_target, self.cfg.w_max)
            cash_target = float(max(0.0, 1.0 - np.sum(w_target)))
        else:
            w_target = self.w.copy()
            cash_target = self.cash

        target_full = np.append(w_target, cash_target)
        current_full = np.append(self.w, self.cash)
        to = float(np.sum(np.abs(target_full - current_full)))
        cost = self.cfg.kappa * to

        r_next = self.rets1.iloc[self.t0 + self.t + 1].reindex(self.tickers).to_numpy(dtype=np.float64)
        port_ret = float(np.dot(w_target, r_next))
        bench_ret = float(np.mean(r_next))

        raw = math.log(max(1e-12, (1.0 + port_ret) * (1.0 - cost)))
        risk_proxy = float(np.var(r_next) * np.sum(w_target ** 2))
        concentration = float(np.sum(w_target ** 2))
        exposure_pen = float(np.sum(w_target))

        reward = (
            raw
            - self.cfg.risk_lambda * risk_proxy
            - self.cfg.concentration_lambda * concentration
            - self.cfg.exposure_penalty_lambda * exposure_pen
            + self.cfg.hold_cash_bonus * cash_target
            + self.cfg.benchmark_lambda * (port_ret - bench_ret)
        )

        self.V = self.V * (1.0 + port_ret) * (1.0 - cost)
        self.w = w_target.astype(np.float32)
        self.cash = cash_target

        self.t += 1
        terminated = self.t >= self.cfg.episode_len
        truncated = False

        info = {
            "V": float(self.V),
            "turnover": float(to),
            "cost": float(cost),
            "port_ret": float(port_ret),
            "bench_ret": float(bench_ret),
            "cash": float(self.cash),
        }

        return self._obs(), float(reward), terminated, truncated, info


# ======================
# Backtests
# ======================
def backtest_equal_weight(rets: pd.DataFrame, w_max: float):
    n = rets.shape[1]
    w = np.ones(n, dtype=np.float32) / n
    w = project_to_caps(w, w_max)
    cash = float(max(0.0, 1.0 - np.sum(w)))

    equity = [1.0]
    ret_list, to_list, cash_list, weights_hist = [], [], [], []

    for t in range(len(rets) - 1):
        r_next = rets.iloc[t + 1].to_numpy(dtype=np.float64)
        port_ret = float(np.dot(w, r_next))
        V_next = equity[-1] * (1.0 + port_ret)

        equity.append(V_next)
        ret_list.append(port_ret)
        to_list.append(0.0)
        cash_list.append(cash)
        weights_hist.append(w.copy())

    return np.array(equity), np.array(ret_list), np.array(to_list), np.array(cash_list), np.vstack(weights_hist)


def backtest_bottomk_ret24h(
    feats: pd.DataFrame,
    rets: pd.DataFrame,
    env_cfg: EnvConfig,
    tickers: List[str],
    bottom_k: int,
):
    base_w = compute_bottomk_base_weights(feats, tickers, "ret_24h", bottom_k, env_cfg.w_max)

    equity = [1.0]
    ret_list, to_list, cash_list, weights_hist = [], [], [], []

    w = np.zeros(len(tickers), dtype=np.float32)
    cash = 1.0

    for t in range(len(rets) - 1):
        w_target = base_w.iloc[t].to_numpy(dtype=np.float64)
        cash_target = float(max(0.0, 1.0 - np.sum(w_target)))

        to = float(np.sum(np.abs(np.append(w_target, cash_target) - np.append(w, cash))))
        cost = env_cfg.kappa * to

        r_next = rets.iloc[t + 1].reindex(tickers).to_numpy(dtype=np.float64)
        port_ret = float(np.dot(w_target, r_next))

        V_next = equity[-1] * (1.0 + port_ret) * (1.0 - cost)
        equity.append(V_next)
        ret_list.append(port_ret - cost)
        to_list.append(to)
        cash_list.append(cash_target)
        weights_hist.append(w_target.copy())

        w = w_target.astype(np.float32)
        cash = cash_target

    return np.array(equity), np.array(ret_list), np.array(to_list), np.array(cash_list), np.vstack(weights_hist)


def backtest_hybrid_model(
    feats: pd.DataFrame,
    rets: pd.DataFrame,
    base_weights: pd.DataFrame,
    model,
    env_cfg: EnvConfig,
    tickers: List[str],
):
    env = HybridContrarianEnv(feats, rets, base_weights, tickers, env_cfg)
    obs, _ = env.reset()

    equity = [1.0]
    ret_list, to_list, cash_list, weights_hist = [], [], [], []

    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, _, info = env.step(action)

        equity.append(info["V"])
        ret_list.append(info["port_ret"] - info["cost"])
        to_list.append(info["turnover"])
        cash_list.append(info["cash"])
        weights_hist.append(env.w.copy())

    return np.array(equity), np.array(ret_list), np.array(to_list), np.array(cash_list), np.vstack(weights_hist)


# ======================
# Trade replay helpers
# ======================
def classify_trades(
    tickers: List[str],
    w_before: np.ndarray,
    w_after: np.ndarray,
    threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    out = []
    for i, tkr in enumerate(tickers):
        before = float(w_before[i])
        after = float(w_after[i])
        delta = after - before

        if delta > threshold:
            action = "BUY"
        elif delta < -threshold:
            action = "SELL"
        else:
            action = "HOLD"

        out.append(
            {
                "ticker": tkr,
                "weight_before": before,
                "weight_after": after,
                "delta_weight": delta,
                "action": action,
            }
        )
    return out


def top_trade_lists(trades: List[Dict[str, Any]], top_k: int = 5) -> Dict[str, Any]:
    buys = [x for x in trades if x["action"] == "BUY"]
    sells = [x for x in trades if x["action"] == "SELL"]
    holds = [x for x in trades if x["action"] == "HOLD"]

    buys = sorted(buys, key=lambda x: x["delta_weight"], reverse=True)[:top_k]
    sells = sorted(sells, key=lambda x: x["delta_weight"])[:top_k]

    return {
        "buys": buys,
        "sells": sells,
        "holds_count": len(holds),
    }


def run_trade_replay_episode(
    model,
    env,
    prices: pd.DataFrame,
    tickers: List[str],
    out_json_path: str,
    action_threshold: float = 0.01,
) -> List[Dict[str, Any]]:
    obs, _ = env.reset()
    done = False
    truncated = False

    episode_log: List[Dict[str, Any]] = []

    while not (done or truncated):
        t_abs = env.t0 + env.t
        ts_now = prices.index[t_abs]
        price_row = prices.iloc[t_abs].to_dict()

        w_before = env.w.copy()
        cash_before = float(env.cash)
        V_before = float(env.V)

        action, _ = model.predict(obs, deterministic=True)
        obs, reward, done, truncated, info = env.step(action)

        w_after = env.w.copy()
        cash_after = float(env.cash)
        V_after = float(env.V)

        trades = classify_trades(
            tickers=tickers,
            w_before=w_before,
            w_after=w_after,
            threshold=action_threshold,
        )
        ranked = top_trade_lists(trades, top_k=5)

        record = {
            "timestamp": str(ts_now),
            "portfolio_value_before": V_before,
            "portfolio_value_after": V_after,
            "cash_before": cash_before,
            "cash_after": cash_after,
            "reward": float(reward),
            "turnover": float(info.get("turnover", 0.0)),
            "cost": float(info.get("cost", 0.0)),
            "portfolio_return": float(info.get("port_ret", 0.0)),
            "benchmark_return": float(info.get("bench_ret", 0.0)),
            "prices": {k: float(v) for k, v in price_row.items()},
            "weights_before": {tickers[i]: float(w_before[i]) for i in range(len(tickers))},
            "weights_after": {tickers[i]: float(w_after[i]) for i in range(len(tickers))},
            "top_buys": ranked["buys"],
            "top_sells": ranked["sells"],
            "holds_count": ranked["holds_count"],
            "all_trades": trades,
        }
        episode_log.append(record)

    with open(out_json_path, "w", encoding="utf-8") as f:
        json.dump(episode_log, f, indent=2)

    print(f"[saved] trade replay log -> {out_json_path}")
    return episode_log


# ======================
# Main
# ======================
def main():
    set_global_seed(0)

    universe_name = "top20"
    tickers = TOP20_TICKERS.copy()
    print("Using universe:", universe_name)
    print("Number of stocks:", len(tickers))

    bottom_k = choose_bottom_k(len(tickers))
    print("Contrarian basket size (bottom_k):", bottom_k)

    out_dir = os.path.join(RESULTS_DIR, "top20_trade_replay")
    os.makedirs(out_dir, exist_ok=True)

    closes, vols = download_close_volume(tickers, period="730d", interval="1h", cache_dir=CACHE_DIR)
    feats, rets1 = build_features(closes, vols)

    # Save feature metadata/diagnostics so we can inspect whether the behavioral block
    # is active and whether any feature has predictive correlation with next returns.
    metadata_path = os.path.join(out_dir, "feature_metadata.json")
    with open(metadata_path, "w", encoding="utf-8") as f:
        json.dump({
            "use_behavioral_features": USE_BEHAVIORAL_FEATURES,
            "use_external_behavioral_data": USE_EXTERNAL_BEHAVIORAL_DATA,
            "n_feature_columns": int(feats.shape[1]),
            "feature_columns": [str(c) for c in feats.columns],
            "behavioral_data_dir": BEHAVIORAL_DATA_DIR,
        }, f, indent=2)
    print("[saved] feature metadata ->", metadata_path)

    diag_path = os.path.join(out_dir, "behavioral_feature_diagnostics.csv")
    save_behavioral_feature_diagnostics(feats, rets1, tickers, diag_path, top_n=50)

    prices_all = closes.loc[feats.index].copy()

    feats_train, feats_val, feats_test = split_train_val_test(feats, 0.70, 0.15)
    rets_train, rets_val, rets_test = split_train_val_test(rets1, 0.70, 0.15)
    prices_train, prices_val, prices_test = split_train_val_test(prices_all, 0.70, 0.15)

    print("Rows (train/val/test):", len(feats_train), len(feats_val), len(feats_test))

    env_cfg = EnvConfig(
        kappa=0.0015,
        w_max=0.20,
        episode_len=min(24 * 10, len(feats_train) - 2),
        seed=0,
        rebalance_every=6,
        action_smooth=0.10,
        risk_lambda=0.10,
        hold_cash_bonus=0.0,
        concentration_lambda=0.0005,
        benchmark_lambda=0.05,
        exposure_penalty_lambda=0.0,
        start_mode="equal_weight",
    )

    base_w_train = compute_bottomk_base_weights(feats_train, tickers, "ret_24h", bottom_k, env_cfg.w_max)
    base_w_val = compute_bottomk_base_weights(feats_val, tickers, "ret_24h", bottom_k, env_cfg.w_max)
    base_w_test = compute_bottomk_base_weights(feats_test, tickers, "ret_24h", bottom_k, env_cfg.w_max)

    from stable_baselines3 import SAC
    from stable_baselines3.common.vec_env import DummyVecEnv
    from stable_baselines3.common.callbacks import EvalCallback

    def make_train_env():
        return HybridContrarianEnv(feats_train, rets_train, base_w_train, tickers, env_cfg)

    def make_val_env():
        return HybridContrarianEnv(feats_val, rets_val, base_w_val, tickers, env_cfg)

    train_env = DummyVecEnv([make_train_env])
    eval_env = DummyVecEnv([make_val_env])

    sac_params = dict(
        learning_rate=1e-4,
        buffer_size=300000,
        batch_size=256,
        tau=0.005,
        gamma=0.99,
        train_freq=4,
        gradient_steps=1,
        learning_starts=10000,
        ent_coef="auto",
        policy_kwargs=dict(net_arch=[256, 256]),
        seed=0,
        verbose=1,
    )

    eval_freq = 5000
    total_timesteps = 150000

    eval_callback = EvalCallback(
        eval_env,
        best_model_save_path=out_dir,
        log_path=out_dir,
        eval_freq=eval_freq,
        n_eval_episodes=1,
        deterministic=True,
        render=False,
        verbose=0,
    )

    print("\n=== Training SAC ===")
    model = SAC("MlpPolicy", train_env, **sac_params)
    model.learn(total_timesteps=total_timesteps, callback=eval_callback)

    best_path = os.path.join(out_dir, "best_model.zip")
    best_model = SAC.load(best_path, env=train_env) if os.path.exists(best_path) else model

    final_model_path = os.path.join(MODELS_DIR, "behavioral_sac_top20_final.zip")
    best_model.save(final_model_path)
    print("[saved] final model ->", final_model_path)

    periods_per_year = 252 * 6.5

    print("\n=== Final evaluation on test ===")
    base_eq_t, base_ret_t, base_to_t, base_cash_t, _ = backtest_equal_weight(rets_test, env_cfg.w_max)
    base_test = summarize("Baseline_EQ", base_eq_t, base_ret_t, base_to_t, base_cash_t, periods_per_year)

    contr_eq_t, contr_ret_t, contr_to_t, contr_cash_t, _ = backtest_bottomk_ret24h(
        feats_test, rets_test, env_cfg, tickers, bottom_k
    )
    contr_test = summarize(f"Bottom{bottom_k}_ret_24h", contr_eq_t, contr_ret_t, contr_to_t, contr_cash_t, periods_per_year)

    hyb_eq_t, hyb_ret_t, hyb_to_t, hyb_cash_t, _ = backtest_hybrid_model(
        feats_test, rets_test, base_w_test, best_model, env_cfg, tickers
    )
    hyb_test = summarize("Hybrid_Contrarian_SAC", hyb_eq_t, hyb_ret_t, hyb_to_t, hyb_cash_t, periods_per_year)

    print(base_test)
    print(contr_test)
    print(hyb_test)

    summary_path = os.path.join(out_dir, "summary_results.json")
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump({
            "baseline_equal_weight": base_test,
            "bottomk_contrarian": contr_test,
            "hybrid_behavioral_sac": hyb_test,
            "behavioral_features_enabled": USE_BEHAVIORAL_FEATURES,
            "tickers": tickers,
            "bottom_k": bottom_k,
        }, f, indent=2)
    print("[saved] summary ->", summary_path)

    print("\n=== Exporting one human-readable replay ===")
    replay_env = HybridContrarianEnv(
        feats=feats_test,
        rets1=rets_test,
        base_weights=base_w_test,
        tickers=tickers,
        cfg=EnvConfig(
            kappa=env_cfg.kappa,
            w_max=env_cfg.w_max,
            episode_len=min(120, len(feats_test) - 2),
            seed=123,
            rebalance_every=env_cfg.rebalance_every,
            action_smooth=env_cfg.action_smooth,
            risk_lambda=env_cfg.risk_lambda,
            hold_cash_bonus=env_cfg.hold_cash_bonus,
            concentration_lambda=env_cfg.concentration_lambda,
            benchmark_lambda=env_cfg.benchmark_lambda,
            exposure_penalty_lambda=env_cfg.exposure_penalty_lambda,
            start_mode=env_cfg.start_mode,
        ),
    )

    replay_json = os.path.join(out_dir, "top20_episode_log.json")
    run_trade_replay_episode(
        model=best_model,
        env=replay_env,
        prices=prices_test.reindex(feats_test.index)[tickers].copy(),
        tickers=tickers,
        out_json_path=replay_json,
        action_threshold=0.01,
    )

    print("\nDone.")
    print("Replay file:", replay_json)
    print("Best model:", best_path if os.path.exists(best_path) else "[current in-memory model]")
    print("Final saved model:", final_model_path)


if __name__ == "__main__":
    main()