"""
Stock Picker — Social Media Sentiment Extension (Reddit + StockTwits + Google News)
-----------------------------------------------------------------------------------
This script extends the hybrid stock picker with multi-source sentiment analysis:
- Google News RSS (via feedparser)
- Reddit (via PRAW - Python Reddit API Wrapper)
- StockTwits (via public API)

Combines sentiment from all three sources with weighted averaging and FinBERT analysis.

Key features
------------
- **Multi-source sentiment aggregation:**
    - Google News: Professional financial news headlines
    - Reddit: Community discussions from r/wallstreetbets, r/stocks, r/investing
    - StockTwits: Real-time trader sentiment and discussions
    
- **Weighted sentiment fusion:**
    - Configurable weights per source (default: 40% News, 30% Reddit, 30% StockTwits)
    - Volume-weighted sentiment (more mentions = higher confidence)
    
- **Engagement metrics:**
    - Reddit: upvotes, comments, post engagement
    - StockTwits: likes, sentiment distribution
    - Google News: headline count, recency

Setup
-----
1. Install dependencies:
   pip install praw requests feedparser transformers scikit-learn yfinance pandas numpy

2. Configure Reddit API credentials (get from https://www.reddit.com/prefs/apps):
   Set environment variables:
   - REDDIT_CLIENT_ID
   - REDDIT_CLIENT_SECRET
   - REDDIT_USER_AGENT (e.g., "stock_picker_bot/1.0")

3. StockTwits uses public API (no auth required, but rate-limited)

Note: All data sources gracefully degrade if unavailable or rate-limited.
"""
from __future__ import annotations
import os
import math
import time
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime, timedelta
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
    import requests
except Exception:
    requests = None

try:
    import feedparser
except Exception:
    feedparser = None

try:
    from transformers import pipeline  # FinBERT
except Exception:
    pipeline = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

try:
    import praw  # Reddit API
except Exception:
    praw = None

warnings.filterwarnings("ignore")

# ------------------------------
# Utilities (same as base script)
# ------------------------------

def _min_max(s: pd.Series) -> pd.Series:
    """Normalize a pandas Series to the [0, 1] range."""
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def _zscore(s: pd.Series) -> pd.Series:
    """Standardize a pandas Series to zero mean and unit variance."""
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def _annualize_return(prices: pd.Series) -> float:
    """Compute the annualized return (CAGR) from a price series."""
    if len(prices) < 2:
        return 0.0
    total_ret = prices.iloc[-1] / prices.iloc[0]
    days = len(prices)
    return total_ret ** (252 / days) - 1


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Compute the annualized Sharpe ratio for a return series."""
    if returns.std(ddof=0) == 0 or returns.empty:
        return 0.0
    return (returns.mean() - rf/252) / returns.std(ddof=0) * math.sqrt(252)


def _max_drawdown(prices: pd.Series) -> float:
    """Compute the maximum drawdown for a price series."""
    if prices.empty:
        return 0.0
    cummax = prices.cummax()
    dd = (prices / cummax - 1.0).min()
    return float(dd)

# ------------------------------
# Price loaders (same as base)
# ------------------------------

def load_prices(tickers: List[str], period: str = "1y", interval: str = "1d") -> pd.DataFrame:
    """Load adjusted close prices using yfinance with fallbacks."""
    if yf is None:
        raise RuntimeError("yfinance is required. Install: pip install yfinance")
    
    print("Fetching price data using yfinance...")
    data = {}
    for ticker in tickers:
        print(f"Fetching data for {ticker}...")
        try:
            ticker_data = yf.Ticker(ticker)
            df = ticker_data.history(period=period, interval=interval, auto_adjust=True)
            if not df.empty:
                data[ticker] = df["Close"].rename(ticker)
        except Exception as e:
            print(f"Error fetching {ticker}: {str(e)}")
            continue
    
    if data:
        return pd.concat(data.values(), axis=1)
    raise RuntimeError("Failed to load price data for any ticker")

# ------------------------------
# Multi-Source Data Fetchers
# ------------------------------

def fetch_google_news(ticker: str, max_items: int = 15, lookback_days: int = 14) -> List[Dict[str, object]]:
    """
    Fetch Google News headlines for a ticker.
    Returns list of dicts with keys: text, source, timestamp
    """
    if feedparser is None or requests is None:
        return []
    
    q = f"{ticker} stock when:{lookback_days}d"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(url)
        results = []
        for e in feed.entries[:max_items]:
            title = getattr(e, "title", "").strip()
            if title:
                results.append({
                    "text": title,
                    "source": "google_news",
                    "timestamp": getattr(e, "published_parsed", None),
                    "engagement": 1.0  # Neutral weight for news
                })
        return results
    except Exception as e:
        print(f"Error fetching Google News for {ticker}: {str(e)}")
        return []


def fetch_reddit_posts(ticker: str, max_items: int = 50, lookback_days: int = 14) -> List[Dict[str, object]]:
    """
    Fetch Reddit posts mentioning a ticker from finance subreddits.
    Returns list of dicts with keys: text, source, timestamp, engagement
    """
    if praw is None:
        print("Reddit API (praw) not available. Install: pip install praw")
        return []
    
    client_id = os.getenv("REDDIT_CLIENT_ID")
    client_secret = os.getenv("REDDIT_CLIENT_SECRET")
    user_agent = os.getenv("REDDIT_USER_AGENT", "stock_picker_bot/1.0")
    
    if not client_id or not client_secret:
        print("Reddit API credentials not configured. Set REDDIT_CLIENT_ID and REDDIT_CLIENT_SECRET")
        return []
    
    try:
        reddit = praw.Reddit(
            client_id=client_id,
            client_secret=client_secret,
            user_agent=user_agent
        )
        
        subreddits = ["wallstreetbets", "stocks", "investing", "StockMarket"]
        results = []
        cutoff = datetime.now() - timedelta(days=lookback_days)
        
        for subreddit_name in subreddits:
            try:
                subreddit = reddit.subreddit(subreddit_name)
                # Search for ticker symbol
                for submission in subreddit.search(f"${ticker} OR {ticker}", time_filter="week", limit=max_items // len(subreddits)):
                    post_time = datetime.fromtimestamp(submission.created_utc)
                    
                    if post_time < cutoff:
                        continue
                    
                    # Combine title and selftext
                    text = submission.title
                    if submission.selftext:
                        text += " " + submission.selftext[:500]  # Limit body length
                    
                    # Calculate engagement score (upvotes + comments)
                    engagement = math.log1p(submission.score + submission.num_comments)
                    
                    results.append({
                        "text": text,
                        "source": f"reddit_{subreddit_name}",
                        "timestamp": post_time,
                        "engagement": engagement,
                        "upvotes": submission.score,
                        "comments": submission.num_comments
                    })
                    
                    if len(results) >= max_items:
                        break
            except Exception as e:
                print(f"Error fetching from r/{subreddit_name}: {str(e)}")
                continue
            
            if len(results) >= max_items:
                break
            
            time.sleep(0.5)  # Rate limiting
        
        return results
    
    except Exception as e:
        print(f"Error fetching Reddit posts for {ticker}: {str(e)}")
        return []


def fetch_stocktwits(ticker: str, max_items: int = 30) -> List[Dict[str, object]]:
    """
    Fetch StockTwits messages for a ticker using public API.
    Returns list of dicts with keys: text, source, timestamp, engagement, native_sentiment
    """
    if requests is None:
        return []
    
    url = f"https://api.stocktwits.com/api/2/streams/symbol/{ticker}.json"
    
    # Add headers to mimic browser request
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/91.0.4472.124 Safari/537.36',
        'Accept': 'application/json',
        'Referer': f'https://stocktwits.com/symbol/{ticker}'
    }
    
    try:
        response = requests.get(url, headers=headers, timeout=10)
        if response.status_code == 403:
            print(f"StockTwits blocked request for {ticker} (403). API may require authentication or have stricter rate limits.")
            return []
        elif response.status_code == 429:
            print(f"StockTwits rate limit exceeded for {ticker}. Wait before retrying.")
            return []
        elif response.status_code != 200:
            print(f"StockTwits API error for {ticker}: {response.status_code}")
            return []
        
        data = response.json()
        messages = data.get("messages", [])
        
        results = []
        for msg in messages[:max_items]:
            text = msg.get("body", "").strip()
            if not text:
                continue
            
            # Extract native sentiment if available
            entities = msg.get("entities", {})
            sentiment_data = entities.get("sentiment", {})
            native_sentiment = sentiment_data.get("basic")  # "Bullish" or "Bearish"
            
            # Map native sentiment to score
            sentiment_score = 0.0
            if native_sentiment:
                if native_sentiment.lower() == "bullish":
                    sentiment_score = 1.0
                elif native_sentiment.lower() == "bearish":
                    sentiment_score = -1.0
            
            # Engagement from likes
            likes = msg.get("likes", {}).get("total", 0)
            engagement = math.log1p(likes)
            
            # Parse timestamp
            created_at = msg.get("created_at")
            timestamp = None
            if created_at:
                try:
                    timestamp = datetime.strptime(created_at, "%Y-%m-%dT%H:%M:%SZ")
                except Exception:
                    pass
            
            results.append({
                "text": text,
                "source": "stocktwits",
                "timestamp": timestamp,
                "engagement": engagement,
                "likes": likes,
                "native_sentiment": sentiment_score
            })
        
        return results
    
    except Exception as e:
        print(f"Error fetching StockTwits for {ticker}: {str(e)}")
        return []


def fetch_world_news(max_items: int = 15, lookback_days: int = 14) -> List[Tuple[str, List[str]]]:
    """Fetch world news that might affect stocks (same as base script)."""
    if feedparser is None or requests is None:
        return []
    
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
            feed = feedparser.parse(url)
            headlines = []
            for e in feed.entries[:max_items]:
                title = getattr(e, "title", "").strip()
                if title:
                    headlines.append(title)
            if headlines:
                all_news.append((category, headlines))
        except Exception:
            continue
        
        time.sleep(0.1)
    
    return all_news

# ------------------------------
# Sentiment Model (FinBERT)
# ------------------------------

class SentimentModel:
    def __init__(self):
        """Initialize FinBERT sentiment model."""
        self.available = pipeline is not None
        self._pipe = None
        if self.available:
            try:
                self._pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert")
                print("FinBERT model loaded successfully")
            except Exception as e:
                print(f"Failed to load FinBERT: {str(e)}")
                self.available = False
                self._pipe = None

    def score(self, texts: List[str]) -> float:
        """Return mean sentiment in [-1, 1] for a list of texts."""
        if not texts:
            return 0.0
        if not self.available or self._pipe is None:
            return 0.0
        try:
            preds = self._pipe(texts)
            mapping = {
                "positive": 1.0, "pos": 1.0,
                "negative": -1.0, "neg": -1.0,
                "neutral": 0.0, "neutrality": 0.0
            }
            vals = []
            for p in preds:
                lab = p.get("label", "neutral")
                lab_key = lab.lower() if isinstance(lab, str) else str(lab).lower()
                vals.append(mapping.get(lab_key, 0.0))
            return float(np.mean(vals)) if vals else 0.0
        except Exception as e:
            print(f"Sentiment scoring error: {str(e)}")
            return 0.0

    def score_with_weights(self, items: List[Dict[str, object]]) -> Tuple[float, Dict[str, int]]:
        """
        Score sentiment with engagement weighting.
        Returns (weighted_sentiment, source_counts)
        """
        if not items:
            return 0.0, {}
        
        texts = [item["text"] for item in items]
        weights = [item.get("engagement", 1.0) for item in items]
        
        if not self.available or self._pipe is None:
            # Use native sentiment if available (StockTwits)
            scores = [item.get("native_sentiment", 0.0) for item in items]
        else:
            try:
                preds = self._pipe(texts)
                mapping = {
                    "positive": 1.0, "pos": 1.0,
                    "negative": -1.0, "neg": -1.0,
                    "neutral": 0.0, "neutrality": 0.0
                }
                scores = []
                for i, p in enumerate(preds):
                    lab = p.get("label", "neutral")
                    lab_key = lab.lower() if isinstance(lab, str) else str(lab).lower()
                    score = mapping.get(lab_key, 0.0)
                    
                    # If StockTwits and has native sentiment, blend it
                    if items[i].get("source") == "stocktwits" and items[i].get("native_sentiment") != 0.0:
                        native = items[i]["native_sentiment"]
                        score = 0.5 * score + 0.5 * native  # Blend FinBERT + native
                    
                    scores.append(score)
            except Exception as e:
                print(f"Sentiment scoring error: {str(e)}")
                scores = [item.get("native_sentiment", 0.0) for item in items]
        
        # Weighted average
        total_weight = sum(weights)
        if total_weight == 0:
            weighted_sentiment = 0.0
        else:
            weighted_sentiment = sum(s * w for s, w in zip(scores, weights)) / total_weight
        
        # Count sources
        source_counts = {}
        for item in items:
            source = item.get("source", "unknown")
            source_counts[source] = source_counts.get(source, 0) + 1
        
        return float(weighted_sentiment), source_counts


def topic_momentum(texts: List[str]) -> float:
    """Compute topic momentum using TF-IDF for bullish terms."""
    if not texts or TfidfVectorizer is None:
        return 0.0
    
    bullish_terms = [
        "beat", "beats", "outperform", "strong guidance", "raise guidance", "upgrade",
        "resilient", "tailwind", "record", "surge", "accelerate", "profitability",
        "margin expansion", "contract win", "strategic partnership", "ai", "chip demand",
        "moon", "rocket", "bullish", "buy", "long", "calls"  # Social media terms
    ]
    
    try:
        vec = TfidfVectorizer(stop_words="english", max_features=2048)
        X = vec.fit_transform(texts)
        vocab = vec.get_feature_names_out()
        idx = [np.where(vocab == term)[0][0] for term in bullish_terms if term in vocab]
        if not idx:
            return 0.0
        weights = X.toarray()[:, idx].mean()
        return float(1 / (1 + math.exp(-8 * (weights - 0.05))))
    except Exception:
        return 0.0

# ------------------------------
# Signal Computation
# ------------------------------

def compute_quant_signals(prices: pd.DataFrame) -> pd.DataFrame:
    """Compute quantitative signals (CAGR, Sharpe, Volatility, Max Drawdown)."""
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
    """Compute sentiment impact for world news categories."""
    impacts = {}
    for category, headlines in world_news:
        sentiment = sent_model.score(headlines)
        impacts[category] = sentiment
        print(f"World news {category} sentiment: {sentiment:.3f}")
    return impacts


def compute_social_ai_signals(
    tickers: List[str],
    max_news: int = 15,
    max_reddit: int = 50,
    max_stocktwits: int = 30,
    lookback_days: int = 14,
    weights: Optional[Dict[str, float]] = None
) -> pd.DataFrame:
    """
    Compute AI signals using multi-source sentiment (Google News + Reddit + StockTwits).
    
    Args:
        tickers: List of stock tickers
        max_news: Max Google News headlines per ticker
        max_reddit: Max Reddit posts per ticker
        max_stocktwits: Max StockTwits messages per ticker
        lookback_days: Days to look back for news/posts
        weights: Source weights dict {"news": w1, "reddit": w2, "stocktwits": w3}
                 Default: {"news": 0.4, "reddit": 0.3, "stocktwits": 0.3}
    
    Returns:
        DataFrame with sentiment scores, topic momentum, and source breakdowns
    """
    if weights is None:
        weights = {"news": 0.4, "reddit": 0.3, "stocktwits": 0.3}
    
    print("Initializing sentiment model...")
    sent_model = SentimentModel()
    
    print("\nFetching and analyzing world news...")
    world_news = fetch_world_news(max_items=max_news, lookback_days=lookback_days)
    world_impacts = compute_world_news_impact(world_news, sent_model)
    
    rows = []
    for t in tickers:
        print(f"\n{'='*60}")
        print(f"Processing {t}...")
        print(f"{'='*60}")
        
        # Fetch from all sources
        print("Fetching Google News...")
        news_items = fetch_google_news(t, max_items=max_news, lookback_days=lookback_days)
        print(f"  Found {len(news_items)} news headlines")
        
        print("Fetching Reddit posts...")
        reddit_items = fetch_reddit_posts(t, max_items=max_reddit, lookback_days=lookback_days)
        print(f"  Found {len(reddit_items)} Reddit posts")
        
        print("Fetching StockTwits messages...")
        stocktwits_items = fetch_stocktwits(t, max_items=max_stocktwits)
        print(f"  Found {len(stocktwits_items)} StockTwits messages")
        
        # Compute sentiment per source
        news_sentiment = 0.0
        reddit_sentiment = 0.0
        stocktwits_sentiment = 0.0
        
        news_sources = {}
        reddit_sources = {}
        stocktwits_sources = {}
        
        if news_items:
            news_sentiment, news_sources = sent_model.score_with_weights(news_items)
            print(f"  News sentiment: {news_sentiment:.3f}")
        
        if reddit_items:
            reddit_sentiment, reddit_sources = sent_model.score_with_weights(reddit_items)
            print(f"  Reddit sentiment: {reddit_sentiment:.3f}")
            # Calculate average engagement
            avg_engagement = np.mean([item.get("engagement", 0) for item in reddit_items])
            print(f"  Reddit avg engagement: {avg_engagement:.2f}")
        
        if stocktwits_items:
            stocktwits_sentiment, stocktwits_sources = sent_model.score_with_weights(stocktwits_items)
            print(f"  StockTwits sentiment: {stocktwits_sentiment:.3f}")
        
        # Weighted combination of sources
        social_sentiment = (
            weights["news"] * news_sentiment +
            weights["reddit"] * reddit_sentiment +
            weights["stocktwits"] * stocktwits_sentiment
        )
        print(f"  Combined social sentiment: {social_sentiment:.3f}")
        
        # World news impact (same logic as base script)
        world_sentiment = 0.0
        relevant_categories = 0
        
        # Sector-based relevance
        if any(tech in t for tech in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AVGO"]):
            if "technology" in world_impacts:
                world_sentiment += world_impacts["technology"] * 1.0
                relevant_categories += 1
        
        if any(bank in t for bank in ["JPM", "GS", "MS", "BAC", "WFC"]):
            if "finance" in world_impacts:
                world_sentiment += world_impacts["finance"] * 1.0
                relevant_categories += 1
        
        if any(energy in t for energy in ["XOM", "CVX", "COP", "EOG"]):
            if "energy" in world_impacts:
                world_sentiment += world_impacts["energy"] * 1.0
                relevant_categories += 1
        
        for cat in ["economy", "geopolitics"]:
            if cat in world_impacts:
                world_sentiment += world_impacts[cat] * 0.5
                relevant_categories += 0.5
        
        if relevant_categories > 0:
            world_sentiment /= relevant_categories
        
        # Final combined sentiment (60% social, 40% world)
        combined_sentiment = 0.6 * social_sentiment + 0.4 * world_sentiment
        print(f"  World news impact: {world_sentiment:.3f}")
        print(f"  Final combined sentiment: {combined_sentiment:.3f}")
        
        # Topic momentum from all text
        all_texts = [item["text"] for item in news_items + reddit_items + stocktwits_items]
        topic = topic_momentum(all_texts)
        print(f"  Topic momentum: {topic:.3f}")
        
        rows.append({
            "Ticker": t,
            "NewsSentiment": news_sentiment,
            "RedditSentiment": reddit_sentiment,
            "StockTwitsSentiment": stocktwits_sentiment,
            "SocialSentiment": social_sentiment,
            "WorldNewsScore": world_sentiment,
            "CombinedSentiment": combined_sentiment,
            "TopicMomentum": topic,
            "NewsCount": len(news_items),
            "RedditCount": len(reddit_items),
            "StockTwitsCount": len(stocktwits_items),
            "TotalMentions": len(all_texts)
        })
        
        time.sleep(0.5)  # Rate limiting
    
    return pd.DataFrame(rows).set_index("Ticker")

# ------------------------------
# Score Fusion
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
    """Fuse quantitative and AI signals into final scores."""
    df = quant.join(ai, how="left").fillna({
        "CombinedSentiment": 0.0,
        "TopicMomentum": 0.0,
        "TotalMentions": 0
    })
    
    # Normalize all metrics
    df["CAGR_n"] = _min_max(df["CAGR"])
    df["Sharpe_n"] = _min_max(df["Sharpe"])
    df["CombinedSentiment_n"] = _min_max(df["CombinedSentiment"])
    df["TopicMomentum_n"] = _min_max(df["TopicMomentum"])
    
    w = weights
    df["Final_Score"] = (
        w.CAGR * df["CAGR_n"] +
        w.Sharpe * df["Sharpe_n"] +
        w.Sentiment * df["CombinedSentiment_n"] +
        w.TopicMomentum * df["TopicMomentum_n"]
    )
    
    return df.sort_values("Final_Score", ascending=False)


def build_portfolio(ranked: pd.DataFrame, top_k: int = 5) -> pd.DataFrame:
    """Select top K stocks for equal-weight portfolio."""
    picks = ranked.head(top_k).copy()
    if picks.empty:
        return picks
    picks["Weight"] = 1.0 / len(picks)
    
    cols = ["CAGR", "Sharpe", "CombinedSentiment", "TopicMomentum", 
            "Final_Score", "Weight", "TotalMentions", "NewsCount", 
            "RedditCount", "StockTwitsCount"]
    existing = [c for c in cols if c in picks.columns]
    return picks[existing]


def quick_backtest(prices: pd.DataFrame, tickers: List[str]) -> Dict[str, float]:
    """Simple backtest of equal-weight portfolio."""
    rets = prices.pct_change().dropna()
    if not set(tickers).issubset(rets.columns):
        return {"CAGR": 0.0, "Sharpe": 0.0, "MaxDD": 0.0}
    port = rets[tickers].mean(axis=1)
    eq_curve = (1 + port).cumprod()
    return {
        "CAGR": _annualize_return(eq_curve),
        "Sharpe": _sharpe(port),
        "MaxDD": _max_drawdown(eq_curve),
    }

# ------------------------------
# Runner
# ------------------------------

@dataclass
class Config:
    tickers: List[str]
    period: str = "9mo"
    interval: str = "1d"
    top_k: int = 5
    max_news: int = 15
    max_reddit: int = 50
    max_stocktwits: int = 30
    lookback_days: int = 14
    source_weights: Dict[str, float] = None

    def __post_init__(self):
        if self.source_weights is None:
            self.source_weights = {"news": 0.4, "reddit": 0.3, "stocktwits": 0.3}


def run(cfg: Config) -> Dict[str, object]:
    """Execute full pipeline: load prices, compute signals, rank, build portfolio."""
    print("="*60)
    print("STOCK PICKER - SOCIAL SENTIMENT EDITION")
    print("="*60)
    
    print("\nLoading price data...")
    prices = load_prices(cfg.tickers, period=cfg.period, interval=cfg.interval).dropna(how="all")
    
    # Only drop columns with >20% missing data
    threshold = int(0.8 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill()  # Forward-fill remaining gaps
    
    print(f"Loaded data for {len(prices.columns)} tickers: {list(prices.columns)}")
    
    # Compute signals
    print("\nComputing quantitative signals...")
    quant = compute_quant_signals(prices)
    
    print("\nComputing social sentiment AI signals...")
    ai = compute_social_ai_signals(
        quant.index.tolist(),
        max_news=cfg.max_news,
        max_reddit=cfg.max_reddit,
        max_stocktwits=cfg.max_stocktwits,
        lookback_days=cfg.lookback_days,
        weights=cfg.source_weights
    )
    
    # Fuse scores
    print("\nFusing scores...")
    weights = Weights()
    ranked = fuse_scores(quant, ai, weights)
    portfolio = build_portfolio(ranked, top_k=cfg.top_k)
    
    # Backtest
    print("\nRunning backtest...")
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
            "META", "AVGO", "TSLA", "GME", "AMC",
        ],
        period="12mo",
        interval="1d",
        top_k=5,
        max_news=15,
        max_reddit=50,
        max_stocktwits=0,  # Disabled: StockTwits blocks automated requests with Cloudflare
        lookback_days=14,
        source_weights={"news": 0.5, "reddit": 0.5, "stocktwits": 0.0}  # 50/50 News + Reddit
    )

    out = run(cfg)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\n=== Portfolio Weights ===")
    print(json.dumps(out["weights"], indent=2))

    print("\n=== Top Picks (Equal-Weight) ===")
    print(out["portfolio"])

    print("\n=== Backtest (in-sample) ===")
    print(json.dumps(out["backtest"], indent=2))
    
    print("\n=== Full Rankings (Top 10) ===")
    display_cols = ["CAGR", "Sharpe", "CombinedSentiment", "TopicMomentum", 
                    "Final_Score", "TotalMentions", "NewsCount", "RedditCount", "StockTwitsCount"]
    available_cols = [c for c in display_cols if c in out["ranked"].columns]
    print(out["ranked"][available_cols].head(10))

    # Save artifacts
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out["ranked"].to_csv(f"ranked_signals_social_{timestamp}.csv")
        out["portfolio"].to_csv(f"portfolio_selection_social_{timestamp}.csv")
        out["ai"].to_csv(f"ai_signals_social_{timestamp}.csv")
        
        print(f"\nSaved artifacts with timestamp: {timestamp}")
        print(f"  - ranked_signals_social_{timestamp}.csv")
        print(f"  - portfolio_selection_social_{timestamp}.csv")
        print(f"  - ai_signals_social_{timestamp}.csv")
    except Exception as e:
        print(f"\nError saving files: {e}")
