"""
Stock Picker — Hybrid AI + Quant (with graceful fallbacks)
-----------------------------------------------------------
This script extends a traditional quantitative stock picker with AI-driven
signals from news/sentiment and lightweight topic momentum. It preserves
robust data fallbacks and optional dependencies so it runs in constrained
environments.

Key features
------------
- **Price providers (auto-fallback):**
    1) yfinance (preferred)
    2) Stooq via pandas-datareader (no API key)
    3) Alpha Vantage (needs env `ALPHAVANTAGE_API_KEY`)

- **AI signals (optional, graceful degrade):**
    - Google News RSS headlines (via `feedparser`) per ticker
    - FinBERT headline sentiment (via `transformers`), else neutral if missing
    - Topic momentum via TF-IDF on headlines (via `sklearn`), else neutral if missing

- **Scoring fusion:**
    - Combine Quant (CAGR, Sharpe) + AI (Sentiment, TopicMomentum)
    - Optional simple weight search to maximize recent in-sample Sharpe

- **Portfolio build:**
    - Rank by Final_Score → pick top N → equal-weight portfolio
    - Basic backtest over the lookback window for sanity (optional)

How to run
----------
- Minimal: `pip install pandas numpy matplotlib` and optionally
    `pip install yfinance feedparser transformers pandas-datareader scikit-learn`.
- If `transformers` not installed, sentiment = 0 (neutral). If `feedparser`
    or `sklearn` are missing, AI features gracefully degrade to 0.

Note: This file is self-contained; adjust `CONFIG` at the bottom.
"""
from __future__ import annotations
import datetime
import os
import math
import time
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional

import numpy as np
import pandas as pd

# ------------------------------
# Optional imports (graceful)
# ------------------------------
try:
    import yfinance as yf  # type: ignore
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr  # type: ignore
except Exception:
    pdr = None

try:
    import requests  # Alpha Vantage fallback
except Exception:
    requests = None

try:
    import feedparser  # Google News RSS
except Exception:
    feedparser = None

try:
    from transformers import pipeline  # FinBERT
except Exception:
    pipeline = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer  # Topic momentum
except Exception:
    TfidfVectorizer = None

warnings.filterwarnings("ignore")

# ------------------------------
# Utilities
# ------------------------------

def _min_max(s: pd.Series) -> pd.Series:
    """
    Normalize a pandas Series to the [0, 1] range.
    If all values are the same, returns all zeros.
    """
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def _zscore(s: pd.Series) -> pd.Series:
    """
    Standardize a pandas Series to zero mean and unit variance.
    If all values are the same or NaN, returns all zeros.
    """
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def _annualize_return(prices: pd.Series) -> float:
    """
    Compute the annualized return (CAGR) from a price series.
    """
    if len(prices) < 2:
        return 0.0
    total_ret = prices.iloc[-1] / prices.iloc[0]
    days = len(prices)
    return total_ret ** (252 / days) - 1


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """
    Compute the annualized Sharpe ratio for a return series.
    """
    if returns.std(ddof=0) == 0 or returns.empty:
        return 0.0
    return (returns.mean() - rf/252) / returns.std(ddof=0) * math.sqrt(252)


def _max_drawdown(prices: pd.Series) -> float:
    """
    Compute the maximum drawdown (largest peak-to-trough drop) for a price series.
    """
    if prices.empty:
        return 0.0
    cummax = prices.cummax()
    dd = (prices / cummax - 1.0).min()
    return float(dd)

# ------------------------------
# Price loaders with fallbacks
# ------------------------------

def load_prices(tickers: List[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """
    Load adjusted close prices for a list of tickers using multiple providers with fallbacks.
    Tries yfinance, then Stooq (via pandas-datareader), then Alpha Vantage.
    """
    # 1) Try yfinance (preferred)
    if yf is not None:
        try:
            print("Attempting to fetch data using yfinance...")
            data = {}
            for ticker in tickers:
                print(f"Fetching data for {ticker}...")
                ticker_data = yf.Ticker(ticker)
                df = ticker_data.history(period=period, interval=interval, auto_adjust=True)
                if not df.empty:
                    data[ticker] = df["Close"].rename(ticker)
            if data:
                return pd.concat(data.values(), axis=1)
        except Exception as e:
            print(f"yfinance error: {str(e)}")

    # 2) Try Stooq via pandas-datareader
    if pdr is not None:
        try:
            # Stooq expects no dot suffix for US tickers; many work as-is
            frames = []
            for t in tickers:
                try:
                    df = pdr.DataReader(t, "stooq")
                    df = df.sort_index()["Close"].rename(t)
                    frames.append(df)
                except Exception:
                    continue
            if frames:
                return pd.concat(frames, axis=1)
        except Exception:
            pass

    # 3) Try Alpha Vantage (Daily Adjusted)
    apikey = os.getenv("ALPHAVANTAGE_API_KEY")
    if apikey and requests is not None:
        data = {}
        for t in tickers:
            try:
                url = (
                    "https://www.alphavantage.co/query?function=TIME_SERIES_DAILY_ADJUSTED"
                    f"&symbol={t}&outputsize=compact&apikey={apikey}"
                )
                r = requests.get(url, timeout=15)
                j = r.json()
                ts = j.get("Time Series (Daily)", {})
                s = pd.Series({pd.to_datetime(k): float(v["5. adjusted close"]) for k, v in ts.items()})
                s = s.sort_index().rename(t)
                if not s.empty:
                    data[t] = s
            except Exception:
                continue
        if data:
            return pd.concat(data.values(), axis=1)

    # If all providers fail, raise an error
    raise RuntimeError("No price provider available. Install yfinance or pandas-datareader, or set ALPHAVANTAGE_API_KEY.")

# ------------------------------
# AI: Headlines, Sentiment (FinBERT), Topic Momentum
# ------------------------------

def fetch_headlines(ticker: str, max_items: int = 15, lookback_days: int = 14) -> List[str]:
    """
    Fetch recent news headlines for a ticker using Google News RSS.
    Returns a list of headline strings (max_items, within lookback_days).
    """
    if feedparser is None:
        return []
    q = f"{ticker} stock when:{lookback_days}d"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    try:
        print(f"Fetching from URL: {url}")
        feed = feedparser.parse(url)
        titles = []
        for e in feed.entries[:max_items]:
            title = getattr(e, "title", "").strip()
            if title:
                titles.append(title)
        return titles
    except Exception as e:
        print(f"Error fetching headlines: {str(e)}")
        return []


class SentimentModel:
    def __init__(self):
        """
        Initialize the sentiment model using FinBERT (if available).
        Sets self.available and self._pipe accordingly.
        """
        self.available = pipeline is not None
        self._pipe = None
        if self.available:
            try:
                # ProsusAI/finbert is the common finance sentiment model
                self._pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
            except Exception:
                self.available = False
                self._pipe = None

    def score(self, texts: List[str]) -> float:
        """
        Return mean sentiment in [-1, 1] for a list of texts.
        Uses FinBERT if available, else returns 0 (neutral).
        """
        if not texts:
            return 0.0
        if not self.available or self._pipe is None:
            return 0.0
        try:
            preds = self._pipe(texts)
            # FinBERT labels: positive/neutral/negative
            # Build a robust mapping that handles different label formats
            mapping = {"positive": 1.0, "pos": 1.0, "negative": -1.0, "neg": -1.0, "neutral": 0.0, "neutrality": 0.0}
            vals = []
            for p in preds:
                lab = p.get("label", "neutral")
                if isinstance(lab, str):
                    lab_key = lab.lower()
                else:
                    lab_key = str(lab).lower()
                vals.append(mapping.get(lab_key, 0.0))
            return float(np.mean(vals)) if vals else 0.0
        except Exception:
            return 0.0

    def explain(self, texts: List[str], top_k_examples: int = 2) -> Dict[str, object]:
        """
        Return per-headline predictions and summary counts and examples for explainability.
        Returns a dict with keys:
         - mean: float (same as score)
         - counts: dict with keys 'positive','neutral','negative'
         - examples: dict with keys 'positive','neutral','negative' each a list of example headlines
         - raw: list of (headline, label, score)
        """
        out = {"mean": 0.0, "counts": {"positive": 0, "neutral": 0, "negative": 0}, "examples": {"positive": [], "neutral": [], "negative": []}, "raw": []}
        if not texts:
            return out
        if not self.available or self._pipe is None:
            return out
        try:
            preds = self._pipe(texts)
            mapping = {"positive": 1.0, "pos": 1.0, "negative": -1.0, "neg": -1.0, "neutral": 0.0, "neutrality": 0.0}
            vals = []
            for text, p in zip(texts, preds):
                lab = p.get("label", "neutral")
                if isinstance(lab, str):
                    lab_key = lab.lower()
                else:
                    lab_key = str(lab).lower()
                score = mapping.get(lab_key, 0.0)
                vals.append(score)
                # normalize label to positive/neutral/negative for counts
                if score > 0:
                    cat = "positive"
                elif score < 0:
                    cat = "negative"
                else:
                    cat = "neutral"
                out["counts"][cat] += 1
                if len(out["examples"][cat]) < top_k_examples:
                    out["examples"][cat].append(text)
                out["raw"].append({"text": text, "label": lab, "mapped": score})
            out["mean"] = float(np.mean(vals)) if vals else 0.0
            return out
        except Exception:
            return out


def topic_momentum(texts: List[str]) -> float:
    """
    Compute a topic momentum score using TF-IDF for bullish finance terms in headlines.
    Returns value in [0,1] after sigmoid squashing; 0 if unavailable.
    """
    if not texts:
        return 0.0
    if TfidfVectorizer is None:
        return 0.0
    bullish_terms = [
        "beat", "beats", "outperform", "strong guidance", "raise guidance", "upgrade",
        "resilient", "tailwind", "record", "surge", "accelerate", "profitability",
        "margin expansion", "contract win", "strategic partnership", "ai", "chip demand",
    ]
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2048)
        X = vec.fit_transform(texts)
        vocab = vec.get_feature_names_out()
        # Find indices of bullish terms in the vocabulary
        idx = [np.where(vocab == term)[0][0] for term in bullish_terms if term in vocab]
        if not idx:
            return 0.0
        # Average tf-idf for bullish terms across all headlines
        weights = X.toarray()[:, idx].mean()
        # squash to [0,1] using a sigmoid
        return float(1 / (1 + math.exp(-8 * (weights - 0.05))))
    except Exception:
        return 0.0

# ------------------------------
# Signal computation
# ------------------------------

def compute_quant_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """
    Compute quantitative signals (CAGR, Sharpe, Volatility, Max Drawdown) for each ticker.
    Returns a DataFrame indexed by ticker.
    """
    rets = prices.pct_change().dropna()
    metrics = []
    for t in prices.columns:
        s = prices[t].dropna()
        r = rets[t].dropna()
        metrics.append({
            "Ticker": t,
            "CAGR": _annualize_return(s),
            "Sharpe": _sharpe(r),
            "Volatility": r.std(ddof=0) * math.sqrt(252),
            "MaxDD": _max_drawdown(s),
        })
    return pd.DataFrame(metrics).set_index("Ticker")


def compute_daily_ai_signals(tickers: List[str], max_news: int = 12) -> pd.DataFrame:
    """
    For each ticker, fetch and analyze news for today and yesterday separately.
    Returns a DataFrame with one row per ticker per day (date, ticker, sentiment, explainability, etc).
    """
    print("Initializing sentiment model...")
    sent_model = SentimentModel()
    rows = []
    today = datetime.date.today()
    for day_offset in [0, 1]:
        day = today - datetime.timedelta(days=day_offset)
        day_str = day.strftime("%Y-%m-%d")
        for t in tickers:
            print(f"\nProcessing AI signals for {t} on {day_str}...")
            # Fetch headlines for this ticker and this day only
            headlines = fetch_headlines_for_date(t, day, max_items=max_news)
            print(f"Found {len(headlines)} headlines for {day_str}")
            sent = sent_model.score(headlines)
            explain = sent_model.explain(headlines, top_k_examples=2) if headlines and sent_model.available else {"mean": sent, "counts": {"positive": 0, "neutral": 0, "negative": 0}, "examples": {"positive": [], "neutral": [], "negative": []}}
            pos_count = explain.get("counts", {}).get("positive", 0)
            neu_count = explain.get("counts", {}).get("neutral", 0)
            neg_count = explain.get("counts", {}).get("negative", 0)
            pos_examples = " | ".join(explain.get("examples", {}).get("positive", []))
            neg_examples = " | ".join(explain.get("examples", {}).get("negative", []))
            neu_examples = " | ".join(explain.get("examples", {}).get("neutral", []))
            topic = topic_momentum(headlines)
            rows.append({
                "Date": day_str,
                "Ticker": t,
                "Sentiment": sent,
                "TopicMomentum": topic,
                "Headlines": len(headlines),
                "Sent_Pos": pos_count,
                "Sent_Neutral": neu_count,
                "Sent_Neg": neg_count,
                "Sent_Examples_Pos": pos_examples,
                "Sent_Examples_Neg": neg_examples,
                "Sent_Examples_Neutral": neu_examples,
            })
            time.sleep(0.1)
    df = pd.DataFrame(rows)
    return df


@dataclass
class Weights:
    """Weights used to fuse quant and AI signals into a final score."""
    CAGR: float = 0.45
    Sharpe: float = 0.35
    Sentiment: float = 0.12
    TopicMomentum: float = 0.08

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.CAGR, self.Sharpe, self.Sentiment, self.TopicMomentum)


def compute_daily_scores(quant: pd.DataFrame, ai_daily: pd.DataFrame, weights: Optional[Weights] = None) -> pd.DataFrame:
    """
    Fuse quantitative signals (per-ticker) with daily AI signals (per-ticker-per-date)
    to compute a Final_Score per ticker per date. Returns a ranked DataFrame.
    """
    if weights is None:
        weights = Weights()

    # Ensure quant has the required columns
    quant = quant.copy()
    if "CAGR" not in quant.columns or "Sharpe" not in quant.columns:
        raise ValueError("quant must contain 'CAGR' and 'Sharpe' columns")

    # Normalize the static quant metrics across tickers
    quant["CAGR_n"] = _min_max(quant["CAGR"])
    quant["Sharpe_n"] = _min_max(quant["Sharpe"])

    # Merge quant normalized metrics into daily AI rows
    merged = ai_daily.merge(quant[["CAGR_n", "Sharpe_n"]], left_on="Ticker", right_index=True, how="left")
    merged["CAGR_n"] = merged["CAGR_n"].fillna(0.0)
    merged["Sharpe_n"] = merged["Sharpe_n"].fillna(0.0)

    # For Sentiment and TopicMomentum, normalize per-date across tickers
    merged["Sentiment_n"] = merged.groupby("Date")["Sentiment"].transform(lambda s: _min_max(s))
    merged["TopicMomentum_n"] = merged.groupby("Date")["TopicMomentum"].transform(lambda s: _min_max(s))

    # Fill NaNs conservatively
    merged["Sentiment_n"] = merged["Sentiment_n"].fillna(0.0)
    merged["TopicMomentum_n"] = merged["TopicMomentum_n"].fillna(0.0)

    w = weights
    merged["Final_Score"] = (
        w.CAGR * merged["CAGR_n"] +
        w.Sharpe * merged["Sharpe_n"] +
        w.Sentiment * merged["Sentiment_n"] +
        w.TopicMomentum * merged["TopicMomentum_n"]
    )

    # Rank within each date
    merged = merged.sort_values(["Date", "Final_Score"], ascending=[True, False])
    return merged
# Helper to fetch headlines for a specific date (today or yesterday)
def fetch_headlines_for_date(ticker: str, day: datetime.date, max_items: int = 12) -> List[str]:
    """
    Fetch headlines for a ticker for a specific day (using Google News RSS, filter by date in title/published).
    """
    if feedparser is None:
        return []
    # Google News RSS does not support exact date filtering, so we fetch recent and filter manually
    q = f"{ticker} stock"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    try:
        feed = feedparser.parse(url)
        titles = []
        for e in feed.entries:
            # Try to parse the published date
            pubdate = None
            if hasattr(e, "published_parsed") and e.published_parsed:
                pubdate = datetime.date(*e.published_parsed[:3])
            elif hasattr(e, "updated_parsed") and e.updated_parsed:
                pubdate = datetime.date(*e.updated_parsed[:3])
            if pubdate == day:
                title = getattr(e, "title", "").strip()
                if title:
                    titles.append(title)
            if len(titles) >= max_items:
                break
        return titles
    except Exception as ex:
        print(f"Error fetching headlines for {ticker} on {day}: {ex}")
        return []

# ------------------------------
# Score fusion & simple weight search
# ------------------------------
@dataclass
class Weights:
    CAGR: float = 0.45
    Sharpe: float = 0.35
    Sentiment: float = 0.12
    TopicMomentum: float = 0.08

    def as_tuple(self) -> Tuple[float, float, float, float]:
        return (self.CAGR, self.Sharpe, self.Sentiment, self.TopicMomentum)


def fuse_scores(quant: pd.DataFrame, ai: pd.DataFrame, weights: Weights) -> pd.DataFrame:
    df = quant.join(ai, how="left").fillna({"Sentiment": 0.0, "TopicMomentum": 0.0, "Headlines": 0})
    # Normalize pro-return metrics upward, risk metrics downward
    df["CAGR_n"] = _min_max(df["CAGR"])  # up
    df["Sharpe_n"] = _min_max(df["Sharpe"])  # up
    df["Sentiment_n"] = _min_max(df["Sentiment"])  # up
    df["TopicMomentum_n"] = _min_max(df["TopicMomentum"])  # up

    w = weights
    df["Final_Score"] = (
        w.CAGR * df["CAGR_n"] +
        w.Sharpe * df["Sharpe_n"] +
        w.Sentiment * df["Sentiment_n"] +
        w.TopicMomentum * df["TopicMomentum_n"]
    )
    return df.sort_values("Final_Score", ascending=False)


def simple_weight_search(prices: pd.DataFrame, base_ai: pd.DataFrame, grid: Optional[List[Weights]] = None, top_k: int = 5) -> Weights:
    """Brute-force small grid to maximize in-sample portfolio Sharpe.
    Uses equal-weight top_k picks for quick calibration.
    """
    if grid is None:
        grid = []
        for c in [0.3, 0.4, 0.5]:
            for s in [0.2, 0.3, 0.4]:
                for se in [0.05, 0.1, 0.2]:
                    tm = max(0.0, 1.0 - (c + s + se))
                    grid.append(Weights(CAGR=c, Sharpe=s, Sentiment=se, TopicMomentum=tm))

    rets = prices.pct_change().dropna()
    best_w = None
    best_score = -1e9

    for w in grid:
        quant = compute_quant_signals(prices)
        ranked = fuse_scores(quant, base_ai, w)
        picks = ranked.head(top_k).index.tolist()
        if not picks:
            continue
        port = rets[picks].mean(axis=1)
        s = _sharpe(port)
        if s > best_score:
            best_score = s
            best_w = w

    return best_w or Weights()

# ------------------------------
# Portfolio construction & (optional) quick backtest
# ------------------------------

def build_portfolio(ranked: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    picks = ranked.head(top_k).copy()
    if picks.empty:
        return picks
    picks["Weight"] = 1.0 / len(picks)
    cols = ["CAGR", "Sharpe", "Sentiment", "TopicMomentum", "Final_Score", "Weight", "Headlines"]
    existing = [c for c in cols if c in picks.columns]
    return picks[existing]


def quick_backtest(prices: pd.DataFrame, tickers: List[str]) -> Dict[str, float]:
    rets = prices.pct_change().dropna()
    if not set(tickers).issubset(rets.columns):
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    port = rets[tickers].mean(axis=1)
    eq_curve = (1 + port).cumprod()
    stats = {
        "CAGR": _annualize_return(eq_curve),
        "Sharpe": _sharpe(port),
        "MaxDD": _max_drawdown(eq_curve),
    }
    return stats

# ------------------------------
# Runner
# ------------------------------
@dataclass
class Config:
    tickers: List[str]
    period: str = "9mo"  # use enough history for Sharpe/CAGR
    interval: str = "1d"
    top_k: int = 6
    optimize_weights: bool = True
    max_news: int = 12
    lookback_days: int = 14


def run(cfg: Config) -> Dict[str, object]:
    """
    For daily mode: only fetch and analyze news for today and yesterday, per ticker.
    """
    print("Loading price data...")
    prices = load_prices(cfg.tickers, period=cfg.period, interval=cfg.interval).dropna(how="all")
    prices = prices.dropna(axis=1)

    # Compute signals (quantitative only, for reference)
    print("Computing quantitative signals...")
    quant = compute_quant_signals(prices)

    # Compute daily AI signals (today and yesterday)
    print("Computing daily AI signals (today and yesterday)...")
    ai_daily = compute_daily_ai_signals(quant.index.tolist(), max_news=cfg.max_news)

    # Save daily AI signals to CSV
    ai_daily.to_csv("daily_news_sentiment.csv", index=False)
    print("Saved: daily_news_sentiment.csv")

    # Compute fused scores (quant + AI) for each date
    print("Computing daily fused scores...")
    daily_ranked = compute_daily_scores(quant, ai_daily)
    daily_ranked.to_csv("daily_ranked.csv", index=False)
    print("Saved: daily_ranked.csv")

    return {
        "prices": prices,
        "quant": quant,
        "ai_daily": ai_daily,
        "daily_ranked": daily_ranked,
    }


if __name__ == "__main__":
    # ------------------
    # CONFIGURE HERE
    # ------------------
    cfg = Config(
        tickers=[
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
            "META", "AVGO", "TSLA", "BRK-B", "JPM",
        ],
        period="12mo",
        interval="1d",
        top_k=5,
        optimize_weights=True,
        max_news=12,
        lookback_days=14,
    )

    out = run(cfg)
    print("\n=== Daily News Sentiment (today and yesterday) ===")
    print(out["ai_daily"])  # DataFrame
