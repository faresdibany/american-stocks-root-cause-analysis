"""
Stock Picker — Advanced Quantitative Analysis Extension
--------------------------------------------------------
This script extends the hybrid stock picker with comprehensive technical and 
statistical analysis including:

- Fundamental Metrics: P/E ratio, P/B ratio, EPS, Market Cap
- Technical Indicators: MA Crossovers, MACD, RSI, Bollinger Bands
- Statistical Models: ARIMA forecasts, Kalman filtering
- Volatility Models: GARCH for risk-adjusted returns
- Mean Reversion: Z-Score analysis, fair value estimation
- Momentum Indicators: Price momentum, volume trends, ADX

Features
--------
- **Fundamental Analysis:**
    - P/E, P/B, EPS, Market Cap, Revenue Growth from yfinance
    - Dividend yield, Profit margins
    
- **Technical Indicators:**
    - SMA/EMA crossovers (50/200 day)
    - MACD (12, 26, 9) with signal line crossover
    - RSI (14) with divergence detection
    - Bollinger Bands (20, 2) for volatility
    - ADX for trend strength

- **Statistical Models:**
    - ARIMA(p,d,q) for price forecasting
    - Kalman Filter for noise reduction
    - GARCH(1,1) for volatility forecasting

- **Mean Reversion Analysis:**
    - Z-Score relative to moving average
    - Fair value estimation (regression to mean)
    - Bollinger Band position

Dependencies
-----------
pip install yfinance pandas numpy scipy statsmodels arch scikit-learn ta-lib
Note: ta-lib requires binary installation on Windows
Alternative: pip install pandas-ta (pure Python alternative)

Usage
-----
python stock_picker_advanced_quantitative.py
"""
from __future__ import annotations
import os
import math
import time
import json
import warnings
from dataclasses import dataclass
from typing import Dict, List, Tuple, Optional
from datetime import datetime
import numpy as np
import pandas as pd

# Core dependencies
try:
    import yfinance as yf
except Exception:
    yf = None

# Statistical models
try:
    from statsmodels.tsa.arima.model import ARIMA
    from statsmodels.tsa.stattools import adfuller
except Exception:
    ARIMA = None
    adfuller = None

try:
    from arch import arch_model
except Exception:
    arch_model = None

try:
    from scipy import signal
    from scipy.signal import butter, filtfilt
except Exception:
    signal = None

# Technical analysis - try multiple libraries
try:
    import talib as ta
    TALIB_AVAILABLE = True
except Exception:
    TALIB_AVAILABLE = False

try:
    import pandas_ta as pta
    PANDAS_TA_AVAILABLE = True
except Exception:
    PANDAS_TA_AVAILABLE = False

# AI/NLP dependencies
try:
    import feedparser
except Exception:
    feedparser = None

try:
    from transformers import pipeline
    import torch
    # Enable GPU if available
    DEVICE = 0 if torch.cuda.is_available() else -1
    if DEVICE == 0:
        print(f"🚀 GPU acceleration enabled: {torch.cuda.get_device_name(0)}")
    else:
        print("💻 Running on CPU (GPU not available)")
except Exception:
    pipeline = None
    DEVICE = -1
    print("⚠️  Transformers not available, AI sentiment disabled")

try:
    from sklearn.feature_extraction.text import TfidfVectorizer
except Exception:
    TfidfVectorizer = None

try:
    import requests
except Exception:
    requests = None

warnings.filterwarnings("ignore")

# ==============================
# Utility Functions
# ==============================

def _min_max(s: pd.Series) -> pd.Series:
    """Normalize to [0, 1] range."""
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def _zscore(s: pd.Series) -> pd.Series:
    """Standardize to zero mean, unit variance."""
    if s.std(ddof=0) == 0 or s.isna().all():
        return pd.Series(0.0, index=s.index)
    return (s - s.mean()) / s.std(ddof=0)


def _annualize_return(prices: pd.Series) -> float:
    """Compute CAGR."""
    if len(prices) < 2:
        return 0.0
    total_ret = prices.iloc[-1] / prices.iloc[0]
    days = len(prices)
    return total_ret ** (252 / days) - 1


def _sharpe(returns: pd.Series, rf: float = 0.0) -> float:
    """Compute annualized Sharpe ratio."""
    if returns.std(ddof=0) == 0 or returns.empty:
        return 0.0
    return (returns.mean() - rf/252) / returns.std(ddof=0) * math.sqrt(252)


def _max_drawdown(prices: pd.Series) -> float:
    """Compute maximum drawdown."""
    if prices.empty:
        return 0.0
    cummax = prices.cummax()
    dd = (prices / cummax - 1.0).min()
    return float(dd)

# ==============================
# Data Loading
# ==============================

def load_prices(tickers: List[str], period: str = "2y", interval: str = "1d") -> pd.DataFrame:
    """Load adjusted close prices for technical analysis (needs longer history)."""
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
    raise RuntimeError("Failed to load price data")


def load_full_data(tickers: List[str], period: str = "2y") -> Dict[str, pd.DataFrame]:
    """Load full OHLCV data for each ticker."""
    if yf is None:
        raise RuntimeError("yfinance is required")
    
    full_data = {}
    for ticker in tickers:
        try:
            ticker_obj = yf.Ticker(ticker)
            df = ticker_obj.history(period=period, auto_adjust=True)
            if not df.empty:
                full_data[ticker] = df
        except Exception as e:
            print(f"Error loading full data for {ticker}: {str(e)}")
    
    return full_data

# ==============================
# Fundamental Metrics
# ==============================

def get_fundamental_metrics(tickers: List[str]) -> pd.DataFrame:
    """
    Fetch fundamental metrics: P/E, P/B, Market Cap, EPS, etc.
    Returns DataFrame indexed by ticker.
    """
    if yf is None:
        return pd.DataFrame()
    
    print("\nFetching fundamental metrics...")
    metrics = []
    
    for ticker in tickers:
        try:
            stock = yf.Ticker(ticker)
            info = stock.info
            
            metrics.append({
                "Ticker": ticker,
                "PE_Ratio": info.get("trailingPE", np.nan),
                "Forward_PE": info.get("forwardPE", np.nan),
                "PB_Ratio": info.get("priceToBook", np.nan),
                "PS_Ratio": info.get("priceToSalesTrailing12Months", np.nan),
                "EPS": info.get("trailingEps", np.nan),
                "Market_Cap": info.get("marketCap", np.nan),
                "Dividend_Yield": info.get("dividendYield", 0.0),
                "Profit_Margin": info.get("profitMargins", np.nan),
                "ROE": info.get("returnOnEquity", np.nan),
                "Beta": info.get("beta", np.nan),
            })
            time.sleep(0.1)  # Rate limiting
        except Exception as e:
            print(f"Error fetching fundamentals for {ticker}: {str(e)}")
            metrics.append({"Ticker": ticker})
    
    df = pd.DataFrame(metrics).set_index("Ticker")
    return df

# ==============================
# Technical Indicators
# ==============================

def calculate_ma_crossover(prices: pd.Series, short: int = 50, long: int = 200) -> Dict[str, float]:
    """
    Calculate moving average crossover signals.
    Returns dict with SMA values and crossover status.
    """
    if len(prices) < long:
        return {"SMA_Short": np.nan, "SMA_Long": np.nan, "MA_Signal": 0}
    
    sma_short = prices.rolling(window=short).mean().iloc[-1]
    sma_long = prices.rolling(window=long).mean().iloc[-1]
    
    # Signal: 1 (bullish crossover), -1 (bearish), 0 (no clear signal)
    if sma_short > sma_long * 1.02:  # 2% buffer
        signal = 1
    elif sma_short < sma_long * 0.98:
        signal = -1
    else:
        signal = 0
    
    return {
        "SMA_Short": sma_short,
        "SMA_Long": sma_long,
        "MA_Signal": signal,
        "MA_Distance": (sma_short / sma_long - 1) * 100  # % distance
    }


def calculate_macd(prices: pd.Series, fast: int = 12, slow: int = 26, signal_period: int = 9) -> Dict[str, float]:
    """
    Calculate MACD indicator.
    Returns MACD line, signal line, histogram, and crossover signal.
    """
    if len(prices) < slow + signal_period:
        return {"MACD": np.nan, "MACD_Signal": np.nan, "MACD_Hist": np.nan, "MACD_Crossover": 0}
    
    ema_fast = prices.ewm(span=fast, adjust=False).mean()
    ema_slow = prices.ewm(span=slow, adjust=False).mean()
    macd_line = ema_fast - ema_slow
    signal_line = macd_line.ewm(span=signal_period, adjust=False).mean()
    histogram = macd_line - signal_line
    
    # Crossover detection (last 2 periods)
    if len(histogram) >= 2:
        if histogram.iloc[-2] < 0 and histogram.iloc[-1] > 0:
            crossover = 1  # Bullish
        elif histogram.iloc[-2] > 0 and histogram.iloc[-1] < 0:
            crossover = -1  # Bearish
        else:
            crossover = 0
    else:
        crossover = 0
    
    return {
        "MACD": macd_line.iloc[-1],
        "MACD_Signal": signal_line.iloc[-1],
        "MACD_Hist": histogram.iloc[-1],
        "MACD_Crossover": crossover
    }


def calculate_rsi(prices: pd.Series, period: int = 14) -> Dict[str, float]:
    """
    Calculate RSI and detect divergences.
    Returns RSI value and divergence signal.
    """
    if len(prices) < period + 10:
        return {"RSI": np.nan, "RSI_Signal": 0, "RSI_Divergence": 0}
    
    delta = prices.diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
    
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    current_rsi = rsi.iloc[-1]
    
    # Signal based on overbought/oversold
    if current_rsi > 70:
        signal = -1  # Overbought
    elif current_rsi < 30:
        signal = 1  # Oversold
    else:
        signal = 0
    
    # Simple divergence detection (price vs RSI trend last 20 periods)
    if len(prices) >= 20:
        price_trend = np.polyfit(range(20), prices.iloc[-20:].values, 1)[0]
        rsi_trend = np.polyfit(range(20), rsi.iloc[-20:].values, 1)[0]
        
        # Bullish divergence: price down, RSI up
        # Bearish divergence: price up, RSI down
        if price_trend < 0 and rsi_trend > 0:
            divergence = 1  # Bullish
        elif price_trend > 0 and rsi_trend < 0:
            divergence = -1  # Bearish
        else:
            divergence = 0
    else:
        divergence = 0
    
    return {
        "RSI": current_rsi,
        "RSI_Signal": signal,
        "RSI_Divergence": divergence
    }


def calculate_bollinger_bands(prices: pd.Series, period: int = 20, std_dev: float = 2.0) -> Dict[str, float]:
    """
    Calculate Bollinger Bands and position relative to bands.
    """
    if len(prices) < period:
        return {"BB_Upper": np.nan, "BB_Middle": np.nan, "BB_Lower": np.nan, "BB_Position": np.nan}
    
    sma = prices.rolling(window=period).mean()
    std = prices.rolling(window=period).std()
    
    upper_band = sma + (std * std_dev)
    lower_band = sma - (std * std_dev)
    
    current_price = prices.iloc[-1]
    bb_middle = sma.iloc[-1]
    bb_upper = upper_band.iloc[-1]
    bb_lower = lower_band.iloc[-1]
    
    # Position: 0 = lower band, 0.5 = middle, 1 = upper band
    bb_position = (current_price - bb_lower) / (bb_upper - bb_lower) if bb_upper != bb_lower else 0.5
    
    return {
        "BB_Upper": bb_upper,
        "BB_Middle": bb_middle,
        "BB_Lower": bb_lower,
        "BB_Position": bb_position,
        "BB_Width": (bb_upper - bb_lower) / bb_middle  # Normalized width
    }


def calculate_adx(high: pd.Series, low: pd.Series, close: pd.Series, period: int = 14) -> float:
    """
    Calculate Average Directional Index (ADX) for trend strength.
    Returns ADX value (0-100, higher = stronger trend).
    """
    if len(close) < period * 2:
        return np.nan
    
    # True Range
    high_low = high - low
    high_close = np.abs(high - close.shift())
    low_close = np.abs(low - close.shift())
    tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    
    # Directional Movement
    up_move = high - high.shift()
    down_move = low.shift() - low
    
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0)
    
    # Smoothed indicators
    atr = tr.rolling(window=period).mean()
    plus_di = 100 * pd.Series(plus_dm).rolling(window=period).mean() / atr
    minus_di = 100 * pd.Series(minus_dm).rolling(window=period).mean() / atr
    
    # ADX
    dx = 100 * np.abs(plus_di - minus_di) / (plus_di + minus_di)
    adx = dx.rolling(window=period).mean()
    
    return adx.iloc[-1] if not adx.empty else np.nan

# ==============================
# Statistical Models
# ==============================

def forecast_arima(prices: pd.Series, order: Tuple[int, int, int] = (2, 1, 2), forecast_periods: int = 5) -> Dict[str, float]:
    """
    Fit ARIMA model and forecast future prices.
    Returns forecast value and direction.
    """
    if ARIMA is None or len(prices) < 50:
        return {"ARIMA_Forecast": np.nan, "ARIMA_Direction": 0, "ARIMA_Return": np.nan}
    
    try:
        # Fit model
        model = ARIMA(prices.values, order=order)
        fitted = model.fit()
        
        # Forecast
        forecast = fitted.forecast(steps=forecast_periods)
        forecast_price = forecast[-1]
        current_price = prices.iloc[-1]
        
        # Expected return
        arima_return = (forecast_price / current_price - 1) * 100
        direction = 1 if forecast_price > current_price else -1
        
        return {
            "ARIMA_Forecast": forecast_price,
            "ARIMA_Direction": direction,
            "ARIMA_Return": arima_return
        }
    except Exception as e:
        print(f"ARIMA error: {str(e)}")
        return {"ARIMA_Forecast": np.nan, "ARIMA_Direction": 0, "ARIMA_Return": np.nan}


def apply_kalman_filter(prices: pd.Series) -> pd.Series:
    """
    Apply Kalman filter for noise reduction and trend extraction.
    Returns smoothed price series.
    """
    if len(prices) < 10:
        return prices
    
    try:
        # Simple Kalman filter implementation
        n = len(prices)
        filtered = np.zeros(n)
        
        # Initial estimates
        x_est = prices.iloc[0]
        p_est = 1.0
        
        # Process and measurement noise
        q = 0.01  # Process noise
        r = 0.1   # Measurement noise
        
        for i in range(n):
            # Prediction
            x_pred = x_est
            p_pred = p_est + q
            
            # Update
            k = p_pred / (p_pred + r)  # Kalman gain
            x_est = x_pred + k * (prices.iloc[i] - x_pred)
            p_est = (1 - k) * p_pred
            
            filtered[i] = x_est
        
        return pd.Series(filtered, index=prices.index)
    except Exception:
        return prices


def forecast_garch(returns: pd.Series, horizon: int = 5) -> Dict[str, float]:
    """
    Fit GARCH(1,1) model to forecast volatility.
    Returns forecasted volatility and risk-adjusted score.
    """
    if arch_model is None or len(returns) < 100:
        return {"GARCH_Vol": np.nan, "GARCH_RiskScore": 0}
    
    try:
        # Remove NaNs
        clean_returns = returns.dropna() * 100  # Scale for numerical stability
        
        if len(clean_returns) < 100:
            return {"GARCH_Vol": np.nan, "GARCH_RiskScore": 0}
        
        # Fit GARCH(1,1)
        model = arch_model(clean_returns, vol='Garch', p=1, q=1)
        fitted = model.fit(disp='off')
        
        # Forecast volatility
        forecast = fitted.forecast(horizon=horizon)
        forecast_vol = np.sqrt(forecast.variance.values[-1, :].mean())
        
        # Risk score: higher volatility = lower score
        # Normalize: typical daily vol is 1-5%
        risk_score = max(0, 1 - (forecast_vol / 5.0))
        
        return {
            "GARCH_Vol": forecast_vol,
            "GARCH_RiskScore": risk_score
        }
    except Exception as e:
        print(f"GARCH error: {str(e)}")
        return {"GARCH_Vol": np.nan, "GARCH_RiskScore": 0}

# ==============================
# AI/Sentiment Analysis (from world_news_extension)
# ==============================

def fetch_headlines(ticker: str, max_items: int = 15, lookback_days: int = 14) -> List[str]:
    """
    Fetch recent news headlines for a ticker using Google News RSS.
    Returns a list of headline strings.
    """
    if feedparser is None or requests is None:
        return []
    
    q = f"{ticker} stock when:{lookback_days}d"
    url = f"https://news.google.com/rss/search?q={requests.utils.quote(q)}&hl=en-US&gl=US&ceid=US:en"
    
    try:
        feed = feedparser.parse(url)
        titles = []
        for e in feed.entries[:max_items]:
            title = getattr(e, "title", "").strip()
            if title:
                titles.append(title)
        return titles
    except Exception as e:
        print(f"Error fetching headlines for {ticker}: {str(e)}")
        return []


def fetch_world_news(max_items: int = 15, lookback_days: int = 14) -> List[Tuple[str, List[str]]]:
    """
    Fetch recent world news that might affect stocks from Google News RSS and Bloomberg.
    Returns a list of tuples (category, headlines).
    """
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
    
    # Fetch from Google News
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
        except Exception as e:
            print(f"Error fetching {category} news from Google: {str(e)}")
            continue
        
        time.sleep(0.1)
    
    # Fetch from Bloomberg Business
    try:
        print("Fetching Bloomberg Business news...")
        bloomberg_url = "https://feeds.bloomberg.com/markets/news.rss"
        feed = feedparser.parse(bloomberg_url)
        headlines = []
        for e in feed.entries[:max_items]:
            title = getattr(e, "title", "").strip()
            if title:
                headlines.append(title)
        if headlines:
            all_news.append(("bloomberg_business", headlines))
            print(f"  ✓ Retrieved {len(headlines)} Bloomberg headlines")
    except Exception as e:
        print(f"Error fetching Bloomberg Business news: {str(e)}")
    
    return all_news


class SentimentModel:
    """FinBERT-based sentiment analysis model."""
    
    def __init__(self):
        """Initialize the sentiment model using FinBERT (if available)."""
        self.available = pipeline is not None
        self._pipe = None
        if self.available:
            try:
                # Use GPU device if available (DEVICE is set at module level)
                self._pipe = pipeline("sentiment-analysis", model="ProsusAI/finbert", device=DEVICE)
                device_name = "GPU" if DEVICE == 0 else "CPU"
                print(f"✅ FinBERT sentiment model loaded successfully on {device_name}")
            except Exception as e:
                print(f"❌ Failed to load FinBERT: {str(e)}")
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


def topic_momentum(texts: List[str]) -> float:
    """
    Compute topic momentum score using TF-IDF for bullish finance terms.
    Returns value in [0,1] after sigmoid squashing; 0 if unavailable.
    """
    if not texts or TfidfVectorizer is None:
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
        idx = [np.where(vocab == term)[0][0] for term in bullish_terms if term in vocab]
        if not idx:
            return 0.0
        weights = X.toarray()[:, idx].mean()
        return float(1 / (1 + math.exp(-8 * (weights - 0.05))))
    except Exception:
        return 0.0


def compute_world_news_impact(
    world_news: List[Tuple[str, List[str]]],
    sent_model: SentimentModel
) -> Dict[str, float]:
    """
    Compute sentiment impact scores for different categories of world news.
    Returns a dictionary of category -> sentiment score.
    """
    impacts = {}
    for category, headlines in world_news:
        sentiment = sent_model.score(headlines)
        impacts[category] = sentiment
        print(f"  World news {category} sentiment: {sentiment:.3f}")
    return impacts


def compute_ai_signals(
    tickers: List[str],
    max_news: int = 50,
    lookback_days: int = 14
) -> pd.DataFrame:
    """
    Compute AI-driven signals for each ticker:
      - Fetches recent news headlines (both stock-specific and world news)
      - Computes sentiment using FinBERT
      - Computes topic momentum using TF-IDF
    Returns a DataFrame indexed by ticker.
    """
    print("\nInitializing sentiment model...")
    sent_model = SentimentModel()
    
    print("Fetching and analyzing world news...")
    world_news = fetch_world_news(max_items=max_news, lookback_days=lookback_days)
    world_impacts = compute_world_news_impact(world_news, sent_model)
    
    rows = []
    for t in tickers:
        print(f"\nProcessing AI signals for {t}...")
        
        # Fetch stock-specific headlines
        headlines = fetch_headlines(t, max_items=max_news, lookback_days=lookback_days)
        print(f"  Found {len(headlines)} headlines")
        
        # Stock-specific sentiment
        stock_sentiment = sent_model.score(headlines)
        print(f"  Stock sentiment: {stock_sentiment:.3f}")
        
        # World news impact (sector-based relevance)
        world_sentiment = 0.0
        relevant_categories = 0
        
        # Tech stocks
        if any(tech in t for tech in ["NVDA", "AAPL", "MSFT", "GOOGL", "META", "AVGO", "AMD", "INTC"]):
            if "technology" in world_impacts:
                world_sentiment += world_impacts["technology"] * 1.0
                relevant_categories += 1
        
        # Financial stocks
        if any(bank in t for bank in ["JPM", "GS", "MS", "BAC", "WFC", "V", "MA"]):
            if "finance" in world_impacts:
                world_sentiment += world_impacts["finance"] * 1.0
                relevant_categories += 1
        
        # Energy stocks
        if any(energy in t for energy in ["XOM", "CVX", "COP", "EOG"]):
            if "energy" in world_impacts:
                world_sentiment += world_impacts["energy"] * 1.0
                relevant_categories += 1
        
        # All stocks affected by economy and geopolitics
        for cat in ["economy", "geopolitics"]:
            if cat in world_impacts:
                world_sentiment += world_impacts[cat] * 0.5
                relevant_categories += 0.5
        
        if relevant_categories > 0:
            world_sentiment /= relevant_categories
        
        # Combined sentiment (60% stock-specific, 40% world)
        combined_sentiment = 0.6 * stock_sentiment + 0.4 * world_sentiment
        print(f"  World impact: {world_sentiment:.3f}")
        print(f"  Combined sentiment: {combined_sentiment:.3f}")
        
        # Topic momentum
        topic = topic_momentum(headlines)
        print(f"  Topic momentum: {topic:.3f}")
        
        rows.append({
            "Ticker": t,
            "StockSentiment": stock_sentiment,
            "WorldSentiment": world_sentiment,
            "CombinedSentiment": combined_sentiment,
            "TopicMomentum": topic,
            "HeadlineCount": len(headlines)
        })
        
        time.sleep(0.2)  # Rate limiting
    
    return pd.DataFrame(rows).set_index("Ticker")


# ==============================
# Mean Reversion Analysis
# ==============================

def calculate_zscore_analysis(prices: pd.Series, window: int = 60) -> Dict[str, float]:
    """
    Calculate Z-Score for mean reversion analysis.
    Returns Z-Score and mean reversion signal.
    """
    if len(prices) < window:
        return {"ZScore": np.nan, "ZScore_Signal": 0, "Distance_From_Mean": np.nan}
    
    # Rolling mean and std
    rolling_mean = prices.rolling(window=window).mean()
    rolling_std = prices.rolling(window=window).std()
    
    current_price = prices.iloc[-1]
    mean_price = rolling_mean.iloc[-1]
    std_price = rolling_std.iloc[-1]
    
    if std_price == 0:
        return {"ZScore": 0, "ZScore_Signal": 0, "Distance_From_Mean": 0}
    
    zscore = (current_price - mean_price) / std_price
    
    # Signal based on Z-Score thresholds
    if zscore > 2:
        signal = -1  # Overvalued, expect reversion down
    elif zscore < -2:
        signal = 1  # Undervalued, expect reversion up
    else:
        signal = 0
    
    distance_pct = (current_price / mean_price - 1) * 100
    
    return {
        "ZScore": zscore,
        "ZScore_Signal": signal,
        "Distance_From_Mean": distance_pct,
        "Mean_Price": mean_price
    }


def estimate_fair_value(prices: pd.Series, window: int = 120) -> Dict[str, float]:
    """
    Estimate fair value using multiple methods.
    Returns fair value estimate and upside potential.
    """
    if len(prices) < window:
        return {"Fair_Value": np.nan, "Upside_Potential": np.nan, "Fair_Value_Signal": 0}
    
    current_price = prices.iloc[-1]
    
    # Method 1: Simple moving average
    sma = prices.rolling(window=window).mean().iloc[-1]
    
    # Method 2: Exponential moving average (gives more weight to recent)
    ema = prices.ewm(span=window, adjust=False).mean().iloc[-1]
    
    # Method 3: Linear regression trend
    recent_prices = prices.iloc[-window:]
    x = np.arange(len(recent_prices))
    coeffs = np.polyfit(x, recent_prices.values, 1)
    trend_value = coeffs[0] * (len(recent_prices) - 1) + coeffs[1]
    
    # Weighted average of methods
    fair_value = 0.3 * sma + 0.3 * ema + 0.4 * trend_value
    
    # Upside potential
    upside = (fair_value / current_price - 1) * 100
    
    # Signal
    if upside > 10:
        signal = 1  # Undervalued
    elif upside < -10:
        signal = -1  # Overvalued
    else:
        signal = 0
    
    return {
        "Fair_Value": fair_value,
        "Upside_Potential": upside,
        "Fair_Value_Signal": signal
    }

# ==============================
# Comprehensive Analysis
# ==============================

def compute_advanced_quant_signals(
    prices: pd.DataFrame,
    full_data: Dict[str, pd.DataFrame],
    include_ai: bool = True,
    max_news: int = 50,
    lookback_days: int = 14
) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
    """
    Compute all advanced quantitative signals for each ticker.
    Returns tuple of (quant_signals, ai_signals).
    """
    print("\nComputing advanced quantitative signals...")
    
    rows = []
    for ticker in prices.columns:
        print(f"Analyzing {ticker}...")
        
        price_series = prices[ticker].dropna()
        if len(price_series) < 100:
            print(f"  Insufficient data for {ticker}")
            continue
        
        returns = price_series.pct_change().dropna()
        
        # Get OHLCV data
        ohlcv = full_data.get(ticker)
        
        # Basic metrics
        metrics = {
            "Ticker": ticker,
            "Current_Price": price_series.iloc[-1],
            "CAGR": _annualize_return(price_series),
            "Sharpe": _sharpe(returns),
            "Volatility": returns.std() * np.sqrt(252),
            "MaxDD": _max_drawdown(price_series)
        }
        
        # Moving Average Crossover
        ma_data = calculate_ma_crossover(price_series)
        metrics.update(ma_data)
        
        # MACD
        macd_data = calculate_macd(price_series)
        metrics.update(macd_data)
        
        # RSI
        rsi_data = calculate_rsi(price_series)
        metrics.update(rsi_data)
        
        # Bollinger Bands
        bb_data = calculate_bollinger_bands(price_series)
        metrics.update(bb_data)
        
        # ADX (requires OHLC)
        if ohlcv is not None and len(ohlcv) > 30:
            adx = calculate_adx(ohlcv['High'], ohlcv['Low'], ohlcv['Close'])
            metrics["ADX"] = adx
        else:
            metrics["ADX"] = np.nan
        
        # ARIMA Forecast
        arima_data = forecast_arima(price_series)
        metrics.update(arima_data)
        
        # Kalman Filter (trend extraction)
        kalman_prices = apply_kalman_filter(price_series)
        kalman_trend = (kalman_prices.iloc[-1] / kalman_prices.iloc[-20] - 1) * 100 if len(kalman_prices) >= 20 else 0
        metrics["Kalman_Trend"] = kalman_trend
        
        # GARCH Volatility Forecast
        garch_data = forecast_garch(returns)
        metrics.update(garch_data)
        
        # Z-Score Analysis
        zscore_data = calculate_zscore_analysis(price_series)
        metrics.update(zscore_data)
        
        # Fair Value Estimation
        fair_value_data = estimate_fair_value(price_series)
        metrics.update(fair_value_data)
        
        rows.append(metrics)
    
    df = pd.DataFrame(rows).set_index("Ticker")
    
    # Compute AI signals if requested
    ai_signals = None
    if include_ai:
        ai_signals = compute_ai_signals(
            df.index.tolist(),
            max_news=max_news,
            lookback_days=lookback_days
        )
    
    return df, ai_signals

# ==============================
# Score Fusion
# ==============================

@dataclass
class AdvancedWeights:
    """Weights for combining signals."""
    CAGR: float = 0.12
    Sharpe: float = 0.12
    MA_Signal: float = 0.08
    MACD_Signal: float = 0.08
    RSI_Signal: float = 0.06
    ARIMA_Return: float = 0.10
    GARCH_Risk: float = 0.08
    ZScore_Signal: float = 0.08
    Fair_Value_Signal: float = 0.08
    Sentiment: float = 0.12
    TopicMomentum: float = 0.08


def fuse_advanced_scores(
    quant: pd.DataFrame,
    fundamentals: pd.DataFrame,
    ai_signals: Optional[pd.DataFrame],
    weights: AdvancedWeights
) -> pd.DataFrame:
    """
    Fuse all quantitative, fundamental, and AI signals into final score.
    """
    print("\nFusing advanced scores...")
    
    # Merge dataframes
    df = quant.join(fundamentals, how="left")
    
    # Add AI signals if available
    if ai_signals is not None:
        df = df.join(ai_signals, how="left")
        df["CombinedSentiment"] = df["CombinedSentiment"].fillna(0)
        df["TopicMomentum"] = df["TopicMomentum"].fillna(0)
    else:
        df["CombinedSentiment"] = 0.0
        df["TopicMomentum"] = 0.0
    
    # Normalize metrics for scoring
    df["CAGR_n"] = _min_max(df["CAGR"])
    df["Sharpe_n"] = _min_max(df["Sharpe"])
    df["ARIMA_Return_n"] = _min_max(df["ARIMA_Return"].fillna(0))
    df["GARCH_RiskScore_n"] = df["GARCH_RiskScore"].fillna(0.5)
    df["CombinedSentiment_n"] = _min_max(df["CombinedSentiment"])
    df["TopicMomentum_n"] = _min_max(df["TopicMomentum"])
    
    # Technical signals (already -1, 0, 1)
    df["MA_Signal_n"] = (df["MA_Signal"] + 1) / 2  # Scale to [0, 1]
    df["MACD_Crossover_n"] = (df["MACD_Crossover"] + 1) / 2
    df["RSI_Signal_n"] = (df["RSI_Signal"] + 1) / 2
    df["ZScore_Signal_n"] = (df["ZScore_Signal"] + 1) / 2
    df["Fair_Value_Signal_n"] = (df["Fair_Value_Signal"] + 1) / 2
    
    # Composite technical score
    df["Technical_Score"] = (
        df["MA_Signal_n"] * 0.25 +
        df["MACD_Crossover_n"] * 0.25 +
        df["RSI_Signal_n"] * 0.25 +
        (df["RSI_Divergence"] + 1) / 2 * 0.25
    )
    
    # Final score
    w = weights
    df["Final_Score"] = (
        w.CAGR * df["CAGR_n"] +
        w.Sharpe * df["Sharpe_n"] +
        w.MA_Signal * df["MA_Signal_n"] +
        w.MACD_Signal * df["MACD_Crossover_n"] +
        w.RSI_Signal * df["RSI_Signal_n"] +
        w.ARIMA_Return * df["ARIMA_Return_n"] +
        w.GARCH_Risk * df["GARCH_RiskScore_n"] +
        w.ZScore_Signal * df["ZScore_Signal_n"] +
        w.Fair_Value_Signal * df["Fair_Value_Signal_n"] +
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
    
    cols = ["Current_Price", "CAGR", "Sharpe", "Final_Score", "Weight",
            "CombinedSentiment", "TopicMomentum", "MA_Signal", "MACD_Crossover", 
            "RSI", "ARIMA_Return", "Upside_Potential", "ZScore", "PE_Ratio"]
    existing = [c for c in cols if c in picks.columns]
    return picks[existing]

# ==============================
# Runner
# ==============================

@dataclass
class Config:
    tickers: List[str]
    period: str = "2y"  # Longer period for technical analysis
    interval: str = "1d"
    top_k: int = 5
    include_ai: bool = True  # Include AI sentiment analysis
    max_news: int = 50
    lookback_days: int = 14


def run(cfg: Config) -> Dict[str, object]:
    """Execute full advanced quantitative pipeline."""
    print("="*60)
    print("STOCK PICKER - ADVANCED QUANTITATIVE ANALYSIS")
    print("="*60)
    
    # Load data
    print("\nLoading price data...")
    prices = load_prices(cfg.tickers, period=cfg.period, interval=cfg.interval)
    
    # Clean data
    threshold = int(0.8 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold)
    prices = prices.ffill()
    
    print(f"Loaded data for {len(prices.columns)} tickers")
    
    # Load full OHLCV data
    print("\nLoading full OHLCV data...")
    full_data = load_full_data(prices.columns.tolist(), period=cfg.period)
    
    # Get fundamentals
    fundamentals = get_fundamental_metrics(prices.columns.tolist())
    
    # Compute advanced signals (including AI if enabled)
    quant, ai_signals = compute_advanced_quant_signals(
        prices,
        full_data,
        include_ai=cfg.include_ai,
        max_news=cfg.max_news,
        lookback_days=cfg.lookback_days
    )
    
    # Fuse scores
    weights = AdvancedWeights()
    ranked = fuse_advanced_scores(quant, fundamentals, ai_signals, weights)
    portfolio = build_portfolio(ranked, top_k=cfg.top_k)
    
    return {
        "prices": prices,
        "fundamentals": fundamentals,
        "quant": quant,
        "ai_signals": ai_signals,
        "weights": weights.__dict__,
        "ranked": ranked,
        "portfolio": portfolio,
    }


if __name__ == "__main__":
    # ------------------
    # CONFIGURE HERE
    # ------------------
    # The Magnificent Seven - The 7 mega-cap tech stocks driving market performance
    cfg = Config(
        tickers=[
            "AAPL",   # Apple - iPhone, Mac, Services
            "MSFT",   # Microsoft - Cloud, Windows, Office
            "GOOGL",  # Alphabet/Google - Search, Ads, Cloud
            "AMZN",   # Amazon - E-commerce, AWS Cloud
            "NVDA",   # NVIDIA - AI chips, GPUs
            "META",   # Meta/Facebook - Social media, AI
            "TSLA",   # Tesla - Electric vehicles, AI/Autonomy
        ],
        period="2y",
        interval="1d",
        top_k=7  # Return all 7 stocks
    )

    out = run(cfg)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\n=== Top Portfolio Picks ===")
    print(out["portfolio"])
    
    print("\n=== Full Rankings (Top 10) ===")
    display_cols = [
        "Current_Price", "CAGR", "Sharpe", "Final_Score",
        "CombinedSentiment", "TopicMomentum",
        "MA_Signal", "MACD_Crossover", "RSI", "RSI_Signal",
        "ARIMA_Return", "Upside_Potential", "ZScore", "PE_Ratio"
    ]
    available = [c for c in display_cols if c in out["ranked"].columns]
    print(out["ranked"][available].head(10))
    
    # Save artifacts
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out["ranked"].to_csv(f"ranked_signals_advanced_{timestamp}.csv")
        out["portfolio"].to_csv(f"portfolio_advanced_{timestamp}.csv")
        out["fundamentals"].to_csv(f"fundamentals_{timestamp}.csv")
        
        if out["ai_signals"] is not None:
            out["ai_signals"].to_csv(f"ai_signals_advanced_{timestamp}.csv")
            print(f"\n✓ Saved artifacts with timestamp: {timestamp}")
            print(f"  - ranked_signals_advanced_{timestamp}.csv")
            print(f"  - portfolio_advanced_{timestamp}.csv")
            print(f"  - fundamentals_{timestamp}.csv")
            print(f"  - ai_signals_advanced_{timestamp}.csv")
        else:
            print(f"\n✓ Saved artifacts with timestamp: {timestamp}")
            print(f"  - ranked_signals_advanced_{timestamp}.csv")
            print(f"  - portfolio_advanced_{timestamp}.csv")
            print(f"  - fundamentals_{timestamp}.csv")
    except Exception as e:
        print(f"\nError saving files: {e}")
