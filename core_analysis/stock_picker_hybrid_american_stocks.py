"""
US Stock Market Analysis with News Sentiment

This module adapts the core stock picker for US markets (NYSE, NASDAQ) with:
- US-specific news sources and sectors
- English sentiment analysis optimized for financial news
- US market-specific technical indicators and patterns
"""

from __future__ import annotations
import os, math, time, warnings
from typing import List, Tuple, Dict, Optional
from urllib.parse import quote_plus
import numpy as np
import pandas as pd
from stock_picker_hybrid_egyptian_news import PredictConfig, add_directional_assistant


# Reuse core building blocks but customize for US markets
from stock_picker_hybrid_egyptian_news import (
    Config, Weights, fuse_scores, build_portfolio, quick_backtest,
    compute_quant_signals, load_prices, _annualize_return, _sharpe, _max_drawdown,
    SentimentModel
)

# Optional dependencies (graceful fallbacks)
try:
    import feedparser
except Exception:
    feedparser = None

try:
    from transformers import pipeline
except Exception:
    pipeline = None

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

warnings.filterwarnings("ignore")

# ------------------------------
# US Market News Fetchers
# ------------------------------

def fetch_headlines_us(ticker: str, max_items: int = 15, lookback_days: int = 14) -> List[str]:
    """
    Google News RSS for US stocks.
    Example query: "AAPL stock price NYSE NASDAQ when:14d"
    """
    if feedparser is None:
        return []
    
    try:
        from urllib.parse import quote_plus
    except ImportError:
        return []
    
    # Add exchange info to improve relevance
    exchanges = "NYSE NASDAQ"
    q = f"{ticker} stock price {exchanges} when:{lookback_days}d"
    encoded_q = quote_plus(q)
    url = f"https://news.google.com/rss/search?q={encoded_q}&hl=en&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(url)
        titles = []
        for e in feed.entries[:max_items]:
            t = getattr(e, "title", "").strip()
            if t:
                titles.append(t)
        return titles
    except Exception:
        return []


def fetch_world_news_us(max_items: int = 15, lookback_days: int = 14) -> List[Tuple[str, List[str]]]:
    """
    US market-focused world/macro categories.
    """
    if feedparser is None:
        return []
    
    categories = [
        ("economy", "US economy GDP inflation interest rates Federal Reserve"),
        ("geopolitics", "US trade relations sanctions geopolitical tensions"),
        ("energy", "oil prices natural gas energy sector commodities"),
        ("finance", "Wall Street banking sector Federal Reserve monetary policy"),
        ("technology", "tech stocks FAANG artificial intelligence cybersecurity"),
        ("consumer", "retail sales consumer spending e-commerce"),
        ("healthcare", "healthcare biotech pharmaceutical FDA approval"),
        ("industrial", "manufacturing industrial production supply chain")
    ]
    
    out = []
    for cat, query in categories:
        try:
            encoded_query = quote_plus(f"{query} when:{lookback_days}d")
            url = f"https://news.google.com/rss/search?q={encoded_query}&hl=en&gl=US&ceid=US:en"
            feed = feedparser.parse(url)
            heads = []
            for e in feed.entries[:max_items]:
                t = getattr(e, "title", "").strip()
                if t:
                    heads.append(t)
            if heads:
                out.append((cat, heads))
        except Exception:
            continue
        time.sleep(0.05)  # Rate limiting
    return out


def save_analysis_to_csv(quant: pd.DataFrame, ai: pd.DataFrame, ranked: pd.DataFrame, suffix: str = "us") -> List[str]:
    """Save all analysis results to separate CSV files."""
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    filenames = []
    
    # Save rankings
    rankings_file = f"rankings_{suffix}_{timestamp}.csv"
    ranked.to_csv(rankings_file)
    filenames.append(rankings_file)
    
    # Save quantitative signals
    quant_file = f"quant_signals_{suffix}_{timestamp}.csv"
    quant.to_csv(quant_file)
    filenames.append(quant_file)
    
    # Save AI signals
    ai_file = f"ai_signals_{suffix}_{timestamp}.csv"
    ai.to_csv(ai_file)
    filenames.append(ai_file)
    
    return filenames


def compute_ai_signals_us(
    tickers: List[str],
    max_news: int = 120,
    lookback_days: int = 14
) -> pd.DataFrame:
    """US market-focused AI signals pipeline."""
    print("Initializing sentiment model...")
    sent_model = SentimentModel()

    print("\nFetching and analyzing US market news...")
    world_news = fetch_world_news_us(max_items=max_news, lookback_days=lookback_days)
    world_impacts = {}
    for category, headlines in world_news:
        sentiment = sent_model.score(headlines)
        world_impacts[category] = sentiment
        print(f"[world] {category} sentiment: {sentiment:.3f}")

    rows = []
    for t in tickers:
        print(f"\n[AI] Processing {t} ...")
        headlines = fetch_headlines_us(t, max_items=max_news, lookback_days=lookback_days)
        
        explain = sent_model.explain(headlines, top_k_examples=2)
        stock_sentiment = explain.get("mean", 0.0)
        pos_count = explain.get("counts", {}).get("positive", 0)
        neu_count = explain.get("counts", {}).get("neutral", 0) 
        neg_count = explain.get("counts", {}).get("negative", 0)
        pos_examples = " | ".join(explain.get("examples", {}).get("positive", []))
        neg_examples = " | ".join(explain.get("examples", {}).get("negative", []))
        neu_examples = " | ".join(explain.get("examples", {}).get("neutral", []))

        # US Market Sector Classification
        # Tech sector (FAANG + major tech)
        is_tech = any(key in t.upper() for key in ["AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA", "AMD", "INTC"])
        # Financial sector
        is_fin = any(key in t.upper() for key in ["JPM", "BAC", "WFC", "GS", "MS", "C", "BLK"])
        # Energy sector
        is_energy = any(key in t.upper() for key in ["XOM", "CVX", "COP", "SLB", "EOG", "PXD"])
        # Healthcare
        is_health = any(key in t.upper() for key in ["JNJ", "PFE", "MRK", "ABBV", "LLY", "UNH"])
        # Consumer
        is_consumer = any(key in t.upper() for key in ["WMT", "TGT", "COST", "HD", "MCD", "NKE", "SBUX"])

        # Topic momentum using TF-IDF
        topic_score = 0.0
        if TfidfVectorizer is not None and headlines:
            try:
                vectorizer = TfidfVectorizer(max_features=100)
                X = vectorizer.fit_transform([" ".join(headlines)])
                top_weights = sorted(
                    zip(vectorizer.get_feature_names_out(), X.toarray()[0]),
                    key=lambda x: abs(x[1]),
                    reverse=True
                )
                topic_score = sum(abs(w) for _, w in top_weights[:10])
            except Exception:
                pass

        # Sector-specific world news impact
        world_sentiment = 0.0
        relevant = 0.0
        
        # Tech sector weights tech and consumer news more heavily
        if is_tech:
            if "technology" in world_impacts:
                world_sentiment += world_impacts["technology"] * 1.5
                relevant += 1.5
            if "consumer" in world_impacts:
                world_sentiment += world_impacts["consumer"] * 0.5
                relevant += 0.5
                
        # Financial sector weights finance and economy news more heavily
        if is_fin:
            if "finance" in world_impacts:
                world_sentiment += world_impacts["finance"] * 1.5
                relevant += 1.5
            if "economy" in world_impacts:
                world_sentiment += world_impacts["economy"] * 1.0
                relevant += 1.0
                
        # Energy sector weights energy and geopolitical news more heavily
        if is_energy:
            if "energy" in world_impacts:
                world_sentiment += world_impacts["energy"] * 1.5
                relevant += 1.5
            if "geopolitics" in world_impacts:
                world_sentiment += world_impacts["geopolitics"] * 1.0
                relevant += 1.0
                
        # Healthcare sector weights healthcare and consumer news
        if is_health:
            if "healthcare" in world_impacts:
                world_sentiment += world_impacts["healthcare"] * 1.5
                relevant += 1.5
            if "consumer" in world_impacts:
                world_sentiment += world_impacts["consumer"] * 0.5
                relevant += 0.5
                
        # Consumer sector weights consumer and economy news
        if is_consumer:
            if "consumer" in world_impacts:
                world_sentiment += world_impacts["consumer"] * 1.5
                relevant += 1.5
            if "economy" in world_impacts:
                world_sentiment += world_impacts["economy"] * 1.0
                relevant += 1.0

        # All stocks are affected by economy and geopolitics to some degree
        if "economy" in world_impacts and not is_fin and not is_consumer:
            world_sentiment += world_impacts["economy"] * 0.5
            relevant += 0.5
        if "geopolitics" in world_impacts and not is_energy:
            world_sentiment += world_impacts["geopolitics"] * 0.3
            relevant += 0.3

        if relevant > 0:
            world_sentiment /= relevant

        # Combined sentiment with stock-specific vs world news weighting
        combined_sentiment = 0.7 * stock_sentiment + 0.3 * world_sentiment  # US stocks typically more company-specific

        print(f"  headlines found: {len(headlines)}")
        print(f"  stock sentiment: {stock_sentiment:.3f}")
        print(f"  world impact: {world_sentiment:.3f}")
        print(f"  combined: {combined_sentiment:.3f}")

        rows.append({
            "Ticker": t,
            "Sentiment": combined_sentiment,
            "StockSentiment": stock_sentiment,
            "WorldNewsScore": world_sentiment,
            "TopicMomentum": topic_score,
            "Headlines": len(headlines),
            "Sent_Pos": pos_count,
            "Sent_Neutral": neu_count,
            "Sent_Neg": neg_count,
            "Sent_Examples_Pos": pos_examples,
            "Sent_Examples_Neg": neg_examples,
            "Sent_Examples_Neutral": neu_examples,
        })
        # Prevent rate limiting
        time.sleep(0.1)

    df = pd.DataFrame(rows).set_index("Ticker")
    for cat in world_impacts:
        df[f"World_{cat}"] = world_impacts[cat]
    
    return df


if __name__ == "__main__":
    # Example usage with major US stocks
    cfg = Config(
        # Example tickers from different sectors
        tickers=[
            # Tech
            "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
            # Finance
            "JPM", "BAC", "GS",
            # Healthcare
            "JNJ", "PFE", "UNH",
            # Energy
            "XOM", "CVX", "COP",
            # Consumer
            "WMT", "HD", "MCD"
        ],
        period="1y",
        interval="1d",
        max_news=50,
        lookback_days=14
    )
    
    # Load price data
    print("Loading US market price data...")
    prices = load_prices(cfg.tickers, period=cfg.period, interval=cfg.interval)
    
    # Compute quantitative signals
    print("\nComputing quantitative signals...")
    quant = compute_quant_signals(prices)
    
    # Compute AI signals
    print("\nComputing AI signals...")
    ai = compute_ai_signals_us(
        quant.index.tolist(),
        max_news=cfg.max_news,
        lookback_days=cfg.lookback_days
    )

    # Map to core schema if needed
    if "Sentiment" in ai.columns and "CombinedSentiment" not in ai.columns:
        ai["CombinedSentiment"] = ai["Sentiment"]
    if "StockSentiment" in ai.columns and "StockSentimentScore" not in ai.columns:
        ai = ai.rename(columns={"StockSentiment": "StockSentimentScore"})

    
    # Fuse signals and rank stocks
    ranked = fuse_scores(quant, ai, Weights())
    print("\nRanked US stocks:")
    print(ranked)
    
    # Save results to CSV files
    csv_files = save_analysis_to_csv(quant, ai, ranked, suffix="us")
    print("\nResults saved to:")
    for f in csv_files:
        print(f"- {f}")
    
    # Display summary
    print("\nSummary of saved data:")
    print(f"- Rankings: {len(ranked)} tickers")
    print(f"- Quantitative signals: {len(quant.columns)} metrics")
    print(f"- AI signals: {len(ai.columns)} metrics")

    # --- Directional Warning Assistant (US) ---
    dcfg = PredictConfig(
        horizon_days=5,
        prob_buy=0.60,
        prob_sell=0.40,
        watch_band=0.05,
        vol_window=20,
        risk_mult_stop=1.25,
        risk_mult_tp=2.0,
        require_headlines=2,
    )
    alert = add_directional_assistant(prices, ai, dcfg)
    print("\n=== Directional Warnings (US, horizon: {}d) ===".format(dcfg.horizon_days))
    print(alert["warnings"])

    # Save
    alert["warnings"].to_csv("directional_warnings_us.csv")
