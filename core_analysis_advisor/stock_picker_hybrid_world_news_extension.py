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
    - Optional simple weight search to maximize recent in-sample Sharpe<f

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

def fetch_world_news(max_items: int = 15, lookback_days: int = 14) -> List[Tuple[str, List[str]]]:
    """
    Fetch recent world news that might affect stocks from Google News RSS.
    Returns a list of tuples (category, headlines).
    """
    if feedparser is None:
        return []
    
    # Categories that might affect stocks
    categories = [
        ("economy", "global economy OR economic crisis OR inflation OR recession"),
        ("geopolitics", "geopolitical tensions OR trade war OR sanctions"),
        ("technology", "technology sector OR semiconductor OR AI OR cybersecurity"),
        ("energy", "oil prices OR energy crisis OR renewable energy"),
        ("finance", "federal reserve OR interest rates OR banking sector")
    ]
    
    all_news = []
    for category, query in categories:
        try:
            url = f"https://news.google.com/rss/search?q={requests.utils.quote(query)}+when:{lookback_days}d&hl=en-US&gl=US&ceid=US:en"
            print(f"Fetching world news for {category} from URL: {url}")
            feed = feedparser.parse(url)
            headlines = []
            for e in feed.entries[:max_items]:
                title = getattr(e, "title", "").strip()
                if title:
                    headlines.append(title)
            if headlines:
                all_news.append((category, headlines))
        except Exception as e:
            print(f"Error fetching {category} news: {str(e)}")
            continue
    
    return all_news


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


def compute_world_news_impact(world_news: List[Tuple[str, List[str]]], sent_model: SentimentModel) -> Dict[str, float]:
    """
    Compute sentiment impact scores for different categories of world news.
    Returns a dictionary of category -> sentiment score.
    """
    impacts = {}
    for category, headlines in world_news:
        sentiment = sent_model.score(headlines)
        impacts[category] = sentiment
        print(f"World news {category} sentiment: {sentiment:.3f}")
    return impacts

def compute_ai_signals(tickers: List[str], max_news: int = 120, lookback_days: int = 14) -> pd.DataFrame:
    """
    Compute AI-driven signals for each ticker:
      - Fetches recent news headlines (both stock-specific and world news)
      - Computes sentiment using FinBERT
      - Computes topic momentum using TF-IDF
    Returns a DataFrame indexed by ticker with explainability columns.
    """
    print("Initializing sentiment model...")
    sent_model = SentimentModel()
    
    print("\nFetching and analyzing world news...")
    world_news = fetch_world_news(max_items=max_news, lookback_days=lookback_days)
    world_impacts = compute_world_news_impact(world_news, sent_model)
    
    rows = []
    for t in tickers:
        print(f"\nProcessing AI signals for {t}...")
        print("Fetching stock-specific headlines...")
        headlines = fetch_headlines(t, max_items=max_news, lookback_days=lookback_days)
        print(f"Found {len(headlines)} headlines")
        
        print("Computing stock-specific sentiment...")
        stock_sentiment = sent_model.score(headlines)
        print(f"Stock-specific sentiment: {stock_sentiment:.3f}")
        
        # Combine stock-specific sentiment with relevant world news impact
        world_sentiment = 0.0
        relevant_categories = 0
        
        # Basic sector-based relevance (can be extended with more sophisticated mapping)
        if any(tech in t for tech in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AVGO"]):  # Tech stocks
            if "technology" in world_impacts:
                world_sentiment += world_impacts["technology"] * 1.0  # Full weight for tech news on tech stocks
                relevant_categories += 1
        
        if any(bank in t for bank in ["JPM", "GS", "MS", "BAC", "WFC"]):  # Financial stocks
            if "finance" in world_impacts:
                world_sentiment += world_impacts["finance"] * 1.0  # Full weight for finance news on bank stocks
                relevant_categories += 1
                
        if any(energy in t for energy in ["XOM", "CVX", "COP", "EOG"]):  # Energy stocks
            if "energy" in world_impacts:
                world_sentiment += world_impacts["energy"] * 1.0  # Full weight for energy news on energy stocks
                relevant_categories += 1
        
        # All stocks are affected by economy and geopolitics
        for cat in ["economy", "geopolitics"]:
            if cat in world_impacts:
                world_sentiment += world_impacts[cat] * 0.5  # Half weight for general categories
                relevant_categories += 0.5
        
        # Normalize world sentiment if we have any relevant categories
        if relevant_categories > 0:
            world_sentiment /= relevant_categories
        
        # Combine sentiments (60% stock-specific, 40% world)
        combined_sentiment = 0.6 * stock_sentiment + 0.4 * world_sentiment
        
        print(f"World news impact: {world_sentiment:.3f}")
        print(f"Combined sentiment: {combined_sentiment:.3f}")
        
        # Get detailed sentiment explanation for stock-specific news
        explain = sent_model.explain(headlines, top_k_examples=2) if headlines and sent_model.available else {"mean": stock_sentiment, "counts": {"positive": 0, "neutral": 0, "negative": 0}, "examples": {"positive": [], "neutral": [], "negative": []}}
        pos_count = explain.get("counts", {}).get("positive", 0)
        neu_count = explain.get("counts", {}).get("neutral", 0)
        neg_count = explain.get("counts", {}).get("negative", 0)
        pos_examples = " | ".join(explain.get("examples", {}).get("positive", []))
        neg_examples = " | ".join(explain.get("examples", {}).get("negative", []))
        neu_examples = " | ".join(explain.get("examples", {}).get("neutral", []))
        
        print("Computing topic momentum...")
        topic = topic_momentum(headlines)
        print(f"Topic momentum score: {topic:.3f}")
        
        rows.append({
            "Ticker": t,
            "StockSemanticScore": stock_sentiment,
            "WorldNewsScore": world_sentiment,
            "CombinedSentiment": combined_sentiment,
            "TopicMomentum": topic,
            "Headlines": len(headlines),
            "Sent_Pos": pos_count,
            "Sent_Neutral": neu_count,
            "Sent_Neg": neg_count,
            "Sent_Examples_Pos": pos_examples,
            "Sent_Examples_Neg": neg_examples,
            "Sent_Examples_Neutral": neu_examples,
        })
        # Sleep to avoid hammering the RSS endpoint
        time.sleep(0.1)
    df = pd.DataFrame(rows).set_index("Ticker")
    return df

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
    df = quant.join(ai, how="left").fillna({"CombinedSentiment": 0.0, "TopicMomentum": 0.0, "Headlines": 0})
    # Normalize pro-return metrics upward, risk metrics downward
    df["CAGR_n"] = _min_max(df["CAGR"])  # up
    df["Sharpe_n"] = _min_max(df["Sharpe"])  # up
    df["CombinedSentiment_n"] = _min_max(df["CombinedSentiment"])  # up
    df["TopicMomentum_n"] = _min_max(df["TopicMomentum"])  # up

    w = weights
    df["Final_Score"] = (
        w.CAGR * df["CAGR_n"] +
        w.Sharpe * df["Sharpe_n"] +
        w.Sentiment * df["CombinedSentiment_n"] +
        w.TopicMomentum * df["TopicMomentum_n"]
    )
    # Place StockSemanticScore and WorldNewsScore next to each other in output
    cols = list(df.columns)
    for col in ["StockSemanticScore", "WorldNewsScore"]:
        if col in cols:
            cols.remove(col)
    # Insert after Final_Score
    idx = cols.index("Final_Score") + 1 if "Final_Score" in cols else len(cols)
    cols = cols[:idx] + ["StockSemanticScore", "WorldNewsScore"] + cols[idx:]
    return df[cols].sort_values("Final_Score", ascending=False)


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
    print("Loading price data...")
    prices = load_prices(cfg.tickers, period=cfg.period, interval=cfg.interval).dropna(how="all")
    # Only drop columns with more than 20% missing data (keep tickers with sufficient history)
    threshold = int(0.8 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    # Forward-fill remaining gaps (common for holidays/weekends)
    prices = prices.ffill()

    # Compute signals
    print("Computing quantitative signals...")
    quant = compute_quant_signals(prices)
    print("Computing AI signals (this may take a few minutes)...")
    ai = compute_ai_signals(quant.index.tolist(), max_news=cfg.max_news, lookback_days=cfg.lookback_days)

    # Optional weight search
    weights = Weights()
    if cfg.optimize_weights:
        weights = simple_weight_search(prices, ai, top_k=cfg.top_k)

    ranked = fuse_scores(quant, ai, weights)
    portfolio = build_portfolio(ranked, top_k=cfg.top_k)

    # Quick backtest on the same window (sanity check only)
    bt = quick_backtest(prices, portfolio.index.tolist())

    return {
        "prices": prices,
        "quant": quant,
        "ai": ai,
        "weights": weights.__dict__,
        "ranked": ranked,
        "portfolio": portfolio,
        "backtest": bt,
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
    print("\n=== Learned Weights ===")
    print(json.dumps(out["weights"], indent=2))

    print("\n=== Top Picks (Equal-Weight) ===")
    print(out["portfolio"])  # DataFrame

    print("\n=== Backtest (in-sample, sanity only) ===")
    print(out["backtest"])  # dict

    # Save artifacts
    try:
        out["ranked"].to_csv("ranked_signals.csv")
        out["portfolio"].to_csv("portfolio_selection.csv")
        print("\nSaved: ranked_signals.csv, portfolio_selection.csv")

        # Save daily news sentiment data
        sentiment_data = pd.DataFrame({
            'Ticker': out['ai'].index,
            'StockSemanticScore': out['ai']['StockSemanticScore'],
            'WorldNewsScore': out['ai']['WorldNewsScore'],
            'CombinedSentiment': out['ai']['CombinedSentiment'],
            'Date': pd.Timestamp.now().date()
        })
        sentiment_data.to_csv('world_daily_news_sentiment.csv', mode='a', header=not os.path.exists('world_daily_news_sentiment.csv'), index=False)
        print("Saved: world_daily_news_sentiment.csv")

        # Save scores to separate CSV files (no Excel dependency)
        out["ranked"].to_csv("ranked_signals_full.csv")
        out["portfolio"].to_csv("portfolio_full.csv")
        sentiment_data.to_csv("sentiment_scores_full.csv", index=False)
        print("Saved: ranked_signals_full.csv, portfolio_full.csv, sentiment_scores_full.csv")
    except Exception as e:
        print(f"Error saving files: {e}")
