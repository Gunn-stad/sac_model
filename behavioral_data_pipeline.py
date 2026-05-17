"""
behavioral_data_pipeline.py

Builds real behavioral/news/crowd/macro signal files for the Behavioral SAC model.

Outputs, by default:
  /content/drive/MyDrive/lokaverkefni_bs/behavioral_data/stock_behavioral_signals.csv
  /content/drive/MyDrive/lokaverkefni_bs/behavioral_data/global_event_signals.csv

Usage in Colab:
  !pip install pandas numpy yfinance requests textblob pytrends vaderSentiment
  !python behavioral_data_pipeline.py

Optional API keys:
  NEWSAPI_KEY          -> https://newsapi.org/
  ALPHAVANTAGE_API_KEY -> https://www.alphavantage.co/

If no API keys are provided, this still creates usable proxy files from price/volume,
VIX, and optional Google Trends when pytrends works.
"""

from __future__ import annotations

import os
import time
import math
import json
import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    import requests
except Exception:  # pragma: no cover
    requests = None

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except Exception:  # pragma: no cover
    SentimentIntensityAnalyzer = None

try:
    from textblob import TextBlob
except Exception:  # pragma: no cover
    TextBlob = None

try:
    from pytrends.request import TrendReq
except Exception:  # pragma: no cover
    TrendReq = None


TOP20_TICKERS = [
    "AAPL", "MSFT", "AMZN", "GOOGL", "NVDA",
    "META", "TSLA", "JPM", "XOM", "UNH",
    "LLY", "AVGO", "COST", "WMT", "HD",
    "PG", "JNJ", "BAC", "ABBV", "CRM",
]

GLOBAL_KEYWORDS = [
    "Federal Reserve", "interest rates", "inflation", "CPI", "tariffs",
    "recession", "election", "war", "oil prices", "market crash",
]


@dataclass
class PipelineConfig:
    tickers: List[str]
    period: str = "730d"
    interval: str = "1h"
    base_dir: str = "/content/drive/MyDrive/lokaverkefni_bs"
    out_subdir: str = "behavioral_data"
    cache_subdir: str = "data_cache"
    use_newsapi: bool = True
    use_alpha_vantage: bool = True
    use_google_trends: bool = True
    max_news_per_ticker: int = 80
    sleep_seconds: float = 1.0

    @property
    def out_dir(self) -> str:
        return os.path.join(self.base_dir, self.out_subdir)

    @property
    def cache_dir(self) -> str:
        return os.path.join(self.base_dir, self.cache_subdir)


def ensure_dirs(cfg: PipelineConfig) -> None:
    os.makedirs(cfg.out_dir, exist_ok=True)
    os.makedirs(cfg.cache_dir, exist_ok=True)


def safe_zscore(s: pd.Series, window: int = 24) -> pd.Series:
    mean = s.rolling(window, min_periods=max(3, window // 4)).mean()
    std = s.rolling(window, min_periods=max(3, window // 4)).std(ddof=0)
    return ((s - mean) / std.replace(0, np.nan)).replace([np.inf, -np.inf], np.nan).fillna(0.0)


def sentiment_score(text: str) -> float:
    text = str(text or "")[:5000]
    if not text.strip():
        return 0.0
    if SentimentIntensityAnalyzer is not None:
        analyzer = sentiment_score._analyzer  # type: ignore[attr-defined]
        return float(analyzer.polarity_scores(text)["compound"])
    if TextBlob is not None:
        return float(TextBlob(text).sentiment.polarity)
    return 0.0


if SentimentIntensityAnalyzer is not None:
    sentiment_score._analyzer = SentimentIntensityAnalyzer()  # type: ignore[attr-defined]


def download_price_data(cfg: PipelineConfig) -> Dict[str, pd.DataFrame]:
    if yf is None:
        raise ImportError("yfinance is required. Install with: pip install yfinance")

    frames: Dict[str, pd.DataFrame] = {}
    for ticker in cfg.tickers:
        cache_path = os.path.join(cfg.cache_dir, f"{ticker}_{cfg.interval}_{cfg.period}_auto_adjust.parquet")
        if os.path.exists(cache_path):
            df = pd.read_parquet(cache_path)
            print(f"[cache] {ticker} -> {cache_path}")
        else:
            print(f"[download] {ticker} from yfinance")
            df = yf.download(
                ticker,
                period=cfg.period,
                interval=cfg.interval,
                auto_adjust=True,
                progress=False,
                threads=True,
            )
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            df = df.rename(columns=str.title)
            df = df[[c for c in ["Open", "High", "Low", "Close", "Volume"] if c in df.columns]].dropna()
            df.to_parquet(cache_path)
        df.index = pd.to_datetime(df.index, utc=True).tz_convert(None).floor("h")
        frames[ticker] = df[~df.index.duplicated(keep="last")]
    return frames


def common_index(frames: Dict[str, pd.DataFrame]) -> pd.DatetimeIndex:
    idx: Optional[pd.DatetimeIndex] = None
    for df in frames.values():
        idx = df.index if idx is None else idx.intersection(df.index)
    if idx is None or len(idx) == 0:
        raise ValueError("No overlapping timestamps found across tickers.")
    return idx.sort_values()


def build_price_volume_proxies(frames: Dict[str, pd.DataFrame], tickers: List[str]) -> pd.DataFrame:
    idx = common_index(frames)
    rows = []
    for ticker in tickers:
        df = frames[ticker].reindex(idx).copy()
        close = df["Close"].astype(float)
        vol = df.get("Volume", pd.Series(0.0, index=idx)).astype(float)
        ret1 = close.pct_change().fillna(0.0)
        ret24 = close.pct_change(24).fillna(0.0)
        vol_z = safe_zscore(vol, 24)
        abs_ret_z = safe_zscore(ret1.abs(), 24)
        attention_proxy = (0.5 * vol_z + 0.5 * abs_ret_z).clip(-5, 5)
        overreaction_proxy = (-ret24 * attention_proxy.clip(lower=0)).replace([np.inf, -np.inf], np.nan).fillna(0.0)
        panic_proxy = ((ret1 < 0).astype(float) * attention_proxy.clip(lower=0)).fillna(0.0)

        tmp = pd.DataFrame({
            "datetime": idx,
            "ticker": ticker,
            "proxy_attention_z": attention_proxy.values,
            "proxy_overreaction": overreaction_proxy.values,
            "proxy_panic": panic_proxy.values,
            "proxy_volume_z": vol_z.values,
            "proxy_abs_return_z": abs_ret_z.values,
        })
        rows.append(tmp)
    return pd.concat(rows, ignore_index=True)


def fetch_newsapi_rows(ticker: str, api_key: str, max_items: int = 80) -> List[dict]:
    if requests is None or not api_key:
        return []
    url = "https://newsapi.org/v2/everything"
    params = {
        "q": f"{ticker} stock OR {ticker} earnings OR {ticker} shares",
        "language": "en",
        "sortBy": "publishedAt",
        "pageSize": min(max_items, 100),
        "apiKey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=20)
        if r.status_code != 200:
            print(f"[newsapi] {ticker}: status {r.status_code} {r.text[:120]}")
            return []
        data = r.json()
    except Exception as e:
        print(f"[newsapi] {ticker}: {e}")
        return []

    rows = []
    for art in data.get("articles", []):
        dt = art.get("publishedAt")
        title = art.get("title") or ""
        desc = art.get("description") or ""
        content = art.get("content") or ""
        text = " ".join([title, desc, content])
        rows.append({
            "datetime": dt,
            "ticker": ticker,
            "headline": title,
            "source": (art.get("source") or {}).get("name", "newsapi"),
            "sentiment_score": sentiment_score(text),
            "news_count": 1.0,
        })
    return rows


def fetch_alpha_vantage_news_rows(ticker: str, api_key: str, max_items: int = 80) -> List[dict]:
    if requests is None or not api_key:
        return []
    url = "https://www.alphavantage.co/query"
    params = {
        "function": "NEWS_SENTIMENT",
        "tickers": ticker,
        "limit": min(max_items, 1000),
        "apikey": api_key,
    }
    try:
        r = requests.get(url, params=params, timeout=25)
        if r.status_code != 200:
            print(f"[alpha_vantage] {ticker}: status {r.status_code} {r.text[:120]}")
            return []
        data = r.json()
    except Exception as e:
        print(f"[alpha_vantage] {ticker}: {e}")
        return []

    rows = []
    for art in data.get("feed", []):
        dt_raw = art.get("time_published", "")
        try:
            dt = pd.to_datetime(dt_raw, format="%Y%m%dT%H%M%S", utc=True)
        except Exception:
            dt = pd.to_datetime(dt_raw, errors="coerce", utc=True)
        title = art.get("title") or ""
        score = art.get("overall_sentiment_score", None)
        if score is None:
            score = sentiment_score(title + " " + str(art.get("summary") or ""))
        rows.append({
            "datetime": dt,
            "ticker": ticker,
            "headline": title,
            "source": "alpha_vantage",
            "sentiment_score": float(score),
            "news_count": 1.0,
        })
    return rows


def aggregate_news_rows(rows: List[dict]) -> pd.DataFrame:
    if not rows:
        return pd.DataFrame(columns=["datetime", "ticker", "sentiment_score", "news_count", "news_sentiment_abs", "news_negative_share"])
    df = pd.DataFrame(rows)
    df["datetime"] = pd.to_datetime(df["datetime"], errors="coerce", utc=True).dt.tz_convert(None).dt.floor("h")
    df = df.dropna(subset=["datetime", "ticker"])
    df["sentiment_score"] = pd.to_numeric(df["sentiment_score"], errors="coerce").fillna(0.0)
    df["news_count"] = pd.to_numeric(df.get("news_count", 1.0), errors="coerce").fillna(1.0)
    df["news_sentiment_abs"] = df["sentiment_score"].abs()
    df["negative_flag"] = (df["sentiment_score"] < -0.05).astype(float)
    agg = df.groupby(["datetime", "ticker"], as_index=False).agg(
        sentiment_score=("sentiment_score", "mean"),
        news_count=("news_count", "sum"),
        news_sentiment_abs=("news_sentiment_abs", "mean"),
        news_negative_share=("negative_flag", "mean"),
    )
    return agg


def build_news_features(cfg: PipelineConfig) -> pd.DataFrame:
    rows: List[dict] = []
    newsapi_key = os.environ.get("NEWSAPI_KEY", "")
    av_key = os.environ.get("ALPHAVANTAGE_API_KEY", "")

    if cfg.use_newsapi and newsapi_key:
        print("[news] using NewsAPI")
        for ticker in cfg.tickers:
            rows.extend(fetch_newsapi_rows(ticker, newsapi_key, cfg.max_news_per_ticker))
            time.sleep(cfg.sleep_seconds)
    else:
        print("[news] NEWSAPI_KEY not found; skipping NewsAPI")

    if cfg.use_alpha_vantage and av_key:
        print("[news] using Alpha Vantage NEWS_SENTIMENT")
        for ticker in cfg.tickers:
            rows.extend(fetch_alpha_vantage_news_rows(ticker, av_key, cfg.max_news_per_ticker))
            time.sleep(max(cfg.sleep_seconds, 12.0))  # free tier rate limits are strict
    else:
        print("[news] ALPHAVANTAGE_API_KEY not found; skipping Alpha Vantage")

    return aggregate_news_rows(rows)


def build_google_trends_features(cfg: PipelineConfig, index: pd.DatetimeIndex) -> pd.DataFrame:
    if not cfg.use_google_trends or TrendReq is None:
        print("[trends] pytrends unavailable or disabled; skipping")
        return pd.DataFrame(columns=["datetime", "ticker", "google_trend", "google_trend_z"])
    try:
        pytrends = TrendReq(hl="en-US", tz=0)
    except Exception as e:
        print(f"[trends] init failed: {e}")
        return pd.DataFrame(columns=["datetime", "ticker", "google_trend", "google_trend_z"])

    start = index.min().date().isoformat()
    end = index.max().date().isoformat()
    timeframe = f"{start} {end}"
    rows = []

    for ticker in cfg.tickers:
        try:
            # Use ticker plus company stock context. Hourly trends over 730d are not always available,
            # so daily values are forward-filled to the model's hourly index.
            pytrends.build_payload([f"{ticker} stock"], timeframe=timeframe)
            data = pytrends.interest_over_time()
            if data.empty:
                continue
            col = [c for c in data.columns if c != "isPartial"][0]
            daily = data[col].astype(float)
            daily.index = pd.to_datetime(daily.index).tz_localize(None)
            hourly = daily.reindex(pd.date_range(index.min().floor("D"), index.max().ceil("D"), freq="D")).ffill()
            hourly = hourly.reindex(index.floor("D"), method="ffill")
            trend_z = safe_zscore(pd.Series(hourly.values, index=index), 24 * 7)
            rows.append(pd.DataFrame({
                "datetime": index,
                "ticker": ticker,
                "google_trend": hourly.values,
                "google_trend_z": trend_z.values,
            }))
            print(f"[trends] {ticker} ok")
            time.sleep(cfg.sleep_seconds)
        except Exception as e:
            print(f"[trends] {ticker} failed: {e}")
    if not rows:
        return pd.DataFrame(columns=["datetime", "ticker", "google_trend", "google_trend_z"])
    return pd.concat(rows, ignore_index=True)


def build_stock_behavioral_signals(cfg: PipelineConfig, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = common_index(frames)
    proxies = build_price_volume_proxies(frames, cfg.tickers)
    news = build_news_features(cfg)
    trends = build_google_trends_features(cfg, idx)

    base = proxies.copy()
    for extra in [news, trends]:
        if not extra.empty:
            extra["datetime"] = pd.to_datetime(extra["datetime"]).dt.floor("h")
            base = base.merge(extra, on=["datetime", "ticker"], how="left")

    for col in ["sentiment_score", "news_count", "news_sentiment_abs", "news_negative_share", "google_trend", "google_trend_z"]:
        if col not in base.columns:
            base[col] = 0.0
        base[col] = pd.to_numeric(base[col], errors="coerce").fillna(0.0)

    # Derived crowd/news variables
    base["news_count_z"] = base.groupby("ticker")["news_count"].transform(lambda s: safe_zscore(s, 24 * 5))
    base["attention_shock"] = (
        0.35 * base["proxy_attention_z"]
        + 0.35 * base["news_count_z"]
        + 0.30 * base["google_trend_z"]
    ).replace([np.inf, -np.inf], np.nan).fillna(0.0).clip(-5, 5)
    base["crowd_disagreement"] = (
        base["news_sentiment_abs"].fillna(0.0) + base["proxy_abs_return_z"].abs().clip(0, 5) / 5.0
    ).clip(0, 5)
    base["event_shock"] = (
        base["attention_shock"].clip(lower=0) * base["news_sentiment_abs"].fillna(0.0)
    ).clip(0, 5)

    cols = [
        "datetime", "ticker",
        "sentiment_score", "news_count", "news_count_z", "news_negative_share",
        "google_trend", "google_trend_z",
        "attention_shock", "crowd_disagreement", "event_shock",
        "proxy_attention_z", "proxy_overreaction", "proxy_panic", "proxy_volume_z", "proxy_abs_return_z",
    ]
    return base[cols].sort_values(["datetime", "ticker"]).reset_index(drop=True)


def fetch_vix(index: pd.DatetimeIndex, cfg: PipelineConfig) -> pd.Series:
    if yf is None:
        return pd.Series(0.0, index=index)
    cache_path = os.path.join(cfg.cache_dir, f"VIX_{cfg.interval}_{cfg.period}.parquet")
    try:
        if os.path.exists(cache_path):
            vix = pd.read_parquet(cache_path)
        else:
            vix = yf.download("^VIX", period=cfg.period, interval=cfg.interval, progress=False, auto_adjust=True)
            if isinstance(vix.columns, pd.MultiIndex):
                vix.columns = vix.columns.get_level_values(0)
            vix = vix.rename(columns=str.title)
            vix.to_parquet(cache_path)
        vix.index = pd.to_datetime(vix.index, utc=True).tz_convert(None).floor("h")
        s = vix["Close"].astype(float).reindex(index).ffill().bfill()
        return s
    except Exception as e:
        print(f"[global] VIX failed: {e}")
        return pd.Series(0.0, index=index)


def build_global_event_signals(cfg: PipelineConfig, frames: Dict[str, pd.DataFrame]) -> pd.DataFrame:
    idx = common_index(frames)
    vix = fetch_vix(idx, cfg)
    vix_z = safe_zscore(vix, 24 * 5)
    vix_spike = (vix_z > 1.5).astype(float)

    # Simple calendar-style approximations. Replace these later with exact FRED/Econoday/FOMC data.
    dates = pd.Series(idx.date, index=idx)
    hour = pd.Series(idx.hour, index=idx)
    weekday = pd.Series(idx.weekday, index=idx)
    day = pd.Series(idx.day, index=idx)

    # CPI often around first half of month; FOMC roughly every 6 weeks. These are weak proxies.
    cpi_proxy = (((day >= 10) & (day <= 14) & (hour >= 8) & (hour <= 16))).astype(float)
    fed_window_proxy = (((day >= 14) & (day <= 22) & (weekday <= 2) & (hour >= 12) & (hour <= 20))).astype(float)
    jobs_proxy = (((day <= 7) & (weekday == 4) & (hour >= 8) & (hour <= 16))).astype(float)

    market_close = pd.concat([frames[t]["Close"].reindex(idx).astype(float) for t in cfg.tickers], axis=1)
    market_ret = market_close.pct_change().mean(axis=1).fillna(0.0)
    market_attention = safe_zscore(market_ret.abs(), 24)

    df = pd.DataFrame({
        "datetime": idx,
        "vix": vix.values,
        "vix_z": vix_z.values,
        "vix_spike": vix_spike.values,
        "market_attention_z": market_attention.values,
        "cpi_event_proxy": cpi_proxy.values,
        "fed_event_proxy": fed_window_proxy.values,
        "jobs_event_proxy": jobs_proxy.values,
    })
    df["political_uncertainty"] = (0.55 * df["vix_spike"] + 0.45 * df["market_attention_z"].clip(lower=0) / 5).clip(0, 1)
    df["rate_policy_signal"] = (df["fed_event_proxy"] + 0.25 * df["vix_spike"]).clip(0, 1)
    df["inflation_event_signal"] = (df["cpi_event_proxy"] + 0.25 * df["vix_spike"]).clip(0, 1)
    df["macro_event_signal"] = (df["fed_event_proxy"] + df["cpi_event_proxy"] + df["jobs_event_proxy"]).clip(0, 1)
    df["tariff_risk_signal"] = 0.0
    df["influential_person_event"] = 0.0
    return df.sort_values("datetime").reset_index(drop=True)


def save_outputs(cfg: PipelineConfig, stock_df: pd.DataFrame, global_df: pd.DataFrame) -> Tuple[str, str]:
    ensure_dirs(cfg)
    stock_path = os.path.join(cfg.out_dir, "stock_behavioral_signals.csv")
    global_path = os.path.join(cfg.out_dir, "global_event_signals.csv")
    stock_df.to_csv(stock_path, index=False)
    global_df.to_csv(global_path, index=False)
    metadata = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "tickers": cfg.tickers,
        "rows_stock": int(len(stock_df)),
        "rows_global": int(len(global_df)),
        "stock_columns": list(stock_df.columns),
        "global_columns": list(global_df.columns),
        "notes": "NewsAPI/AlphaVantage are used only when API keys exist. Otherwise proxy features are generated.",
    }
    with open(os.path.join(cfg.out_dir, "behavioral_data_metadata.json"), "w", encoding="utf-8") as f:
        json.dump(metadata, f, indent=2)
    return stock_path, global_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--base-dir", default=os.environ.get("SAC_BASE_DIR", "/content/drive/MyDrive/lokaverkefni_bs"))
    parser.add_argument("--tickers", default=",".join(TOP20_TICKERS))
    parser.add_argument("--period", default="730d")
    parser.add_argument("--interval", default="1h")
    parser.add_argument("--no-newsapi", action="store_true")
    parser.add_argument("--no-alpha-vantage", action="store_true")
    parser.add_argument("--no-google-trends", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cfg = PipelineConfig(
        tickers=[t.strip().upper() for t in args.tickers.split(",") if t.strip()],
        period=args.period,
        interval=args.interval,
        base_dir=args.base_dir,
        use_newsapi=not args.no_newsapi,
        use_alpha_vantage=not args.no_alpha_vantage,
        use_google_trends=not args.no_google_trends,
    )
    ensure_dirs(cfg)
    print("RUNNING behavioral_data_pipeline.py")
    print("BASE_DIR:", cfg.base_dir)
    print("OUT_DIR:", cfg.out_dir)
    print("TICKERS:", cfg.tickers)

    frames = download_price_data(cfg)
    stock_df = build_stock_behavioral_signals(cfg, frames)
    global_df = build_global_event_signals(cfg, frames)
    stock_path, global_path = save_outputs(cfg, stock_df, global_df)

    print("\nDone.")
    print("Saved stock signals ->", stock_path)
    print("Saved global signals ->", global_path)
    print("Stock signal rows:", len(stock_df))
    print("Global signal rows:", len(global_df))


if __name__ == "__main__":
    main()
