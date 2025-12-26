"""
Stock Picker — Advanced Quantitative Analysis with NLG Explanations
-------------------------------------------------------------------
This script extends the advanced quantitative analysis with Natural Language Generation (NLG)
to provide human-readable explanations for stock rankings and scores.

Features
--------
- All advanced quantitative metrics from stock_picker_advanced_quantitative.py
- AI sentiment analysis with FinBERT
- **Natural Language Explanations:**
    - Why each stock was ranked in its position
    - Strengths and weaknesses breakdown
    - Technical indicator interpretations
    - Sentiment analysis summaries
    - Risk factors and opportunities
    - Actionable insights

Output
------
Creates additional file: explanations_YYYYMMDD_HHMMSS.txt
Contains detailed narrative explanations for each ticker's score and ranking.

Dependencies
-----------
Same as stock_picker_advanced_quantitative.py
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

# Import all dependencies from advanced quantitative script
try:
    import yfinance as yf
except Exception:
    yf = None

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

try:
    import requests
except Exception:
    requests = None

warnings.filterwarnings("ignore")

# Import all utility functions and classes from advanced quantitative script
# (For brevity, I'll reference the key functions - in practice, copy all from the original file)

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
# NLG Explanation Generator
# ==============================

class StockExplainer:
    """Generates natural language explanations for stock rankings."""
    
    def __init__(self):
        """Initialize the explainer."""
        pass
    
    def explain_sentiment(self, row: pd.Series) -> str:
        """Generate explanation for sentiment scores."""
        stock_sent = row.get("StockSentiment", 0)
        world_sent = row.get("WorldSentiment", 0)
        combined = row.get("CombinedSentiment", 0)
        topic = row.get("TopicMomentum", 0)
        headline_count = row.get("HeadlineCount", 0)
        
        parts = []
        
        # Overall sentiment
        if combined > 0.15:
            parts.append(f"📈 **Strong positive sentiment** (score: {combined:.2f}) based on {headline_count} recent headlines.")
        elif combined > 0.05:
            parts.append(f"📊 **Moderately positive sentiment** (score: {combined:.2f}) from {headline_count} news articles.")
        elif combined > -0.05:
            parts.append(f"➡️ **Neutral sentiment** (score: {combined:.2f}) across {headline_count} headlines.")
        elif combined > -0.15:
            parts.append(f"📉 **Moderately negative sentiment** (score: {combined:.2f}) from {headline_count} sources.")
        else:
            parts.append(f"⚠️ **Significantly negative sentiment** (score: {combined:.2f}) based on {headline_count} headlines.")
        
        # Stock-specific vs world news
        if abs(stock_sent) > abs(world_sent) + 0.1:
            parts.append(f"Stock-specific news drives the narrative (stock: {stock_sent:.2f} vs world: {world_sent:.2f}).")
        elif abs(world_sent) > abs(stock_sent) + 0.1:
            parts.append(f"Macro factors dominate the sentiment (world: {world_sent:.2f} vs stock: {stock_sent:.2f}).")
        
        # Topic momentum
        if topic > 0.6:
            parts.append(f"🚀 **High media buzz** with bullish keywords appearing frequently (momentum: {topic:.2f}).")
        elif topic > 0.5:
            parts.append(f"Moderate attention in financial media (momentum: {topic:.2f}).")
        else:
            parts.append(f"Limited recent media coverage or neutral language (momentum: {topic:.2f}).")
        
        return " ".join(parts)
    
    def explain_technical_indicators(self, row: pd.Series) -> str:
        """Generate explanation for technical indicators."""
        parts = []
        
        # Moving averages
        ma_signal = row.get("MA_Signal", 0)
        ma_distance = row.get("MA_Distance", 0)
        
        if ma_signal == 1:
            parts.append(f"📊 **Bullish MA crossover**: 50-day SMA is {abs(ma_distance):.1f}% above 200-day SMA (golden cross pattern).")
        elif ma_signal == -1:
            parts.append(f"📉 **Bearish MA crossover**: 50-day SMA is {abs(ma_distance):.1f}% below 200-day SMA (death cross pattern).")
        else:
            parts.append(f"➡️ **Neutral trend**: Moving averages are converging (distance: {ma_distance:.1f}%).")
        
        # MACD
        macd_crossover = row.get("MACD_Crossover", 0)
        macd_hist = row.get("MACD_Hist", 0)
        
        if macd_crossover == 1:
            parts.append(f"✅ **MACD bullish crossover** detected with positive histogram ({macd_hist:.3f}).")
        elif macd_crossover == -1:
            parts.append(f"❌ **MACD bearish crossover** with negative histogram ({macd_hist:.3f}).")
        elif macd_hist > 0:
            parts.append(f"MACD shows positive momentum (histogram: {macd_hist:.3f}).")
        else:
            parts.append(f"MACD indicates weak momentum (histogram: {macd_hist:.3f}).")
        
        # RSI
        rsi = row.get("RSI", 50)
        rsi_signal = row.get("RSI_Signal", 0)
        rsi_divergence = row.get("RSI_Divergence", 0)
        
        if rsi_signal == 1:
            parts.append(f"💪 **Oversold condition** (RSI: {rsi:.1f}) suggests potential bounce.")
        elif rsi_signal == -1:
            parts.append(f"⚠️ **Overbought territory** (RSI: {rsi:.1f}) signals caution.")
        else:
            parts.append(f"RSI at {rsi:.1f} indicates balanced momentum.")
        
        if rsi_divergence == 1:
            parts.append(f"📈 **Bullish divergence**: Price declining but RSI rising (reversal signal).")
        elif rsi_divergence == -1:
            parts.append(f"📉 **Bearish divergence**: Price rising but RSI falling (warning sign).")
        
        # Bollinger Bands
        bb_position = row.get("BB_Position", 0.5)
        bb_width = row.get("BB_Width", 0)
        
        if bb_position > 0.8:
            parts.append(f"Price near upper Bollinger Band ({bb_position*100:.0f}% position) - potentially extended.")
        elif bb_position < 0.2:
            parts.append(f"Price near lower Bollinger Band ({bb_position*100:.0f}% position) - oversold signal.")
        
        if bb_width > 0.1:
            parts.append(f"High volatility regime (BB width: {bb_width:.2f}).")
        elif bb_width < 0.05:
            parts.append(f"Low volatility (BB width: {bb_width:.2f}) - potential breakout brewing.")
        
        return " ".join(parts)
    
    def explain_statistical_models(self, row: pd.Series) -> str:
        """Generate explanation for statistical forecasts."""
        parts = []
        
        # ARIMA
        arima_return = row.get("ARIMA_Return", 0)
        arima_direction = row.get("ARIMA_Direction", 0)
        
        if not pd.isna(arima_return):
            if arima_return > 5:
                parts.append(f"📈 **ARIMA forecast** projects {arima_return:.1f}% upside over next 5 days.")
            elif arima_return > 0:
                parts.append(f"Modest ARIMA forecast of +{arima_return:.1f}% gain expected.")
            elif arima_return > -5:
                parts.append(f"ARIMA suggests mild downside of {arima_return:.1f}%.")
            else:
                parts.append(f"⚠️ **ARIMA warning**: {arima_return:.1f}% decline forecasted.")
        
        # GARCH volatility
        garch_vol = row.get("GARCH_Vol", 0)
        garch_risk = row.get("GARCH_RiskScore", 0)
        
        if not pd.isna(garch_vol):
            if garch_vol > 4:
                parts.append(f"⚠️ **High volatility** regime: GARCH forecasts {garch_vol:.1f}% daily vol (risk score: {garch_risk:.2f}).")
            elif garch_vol > 2:
                parts.append(f"Moderate volatility expected: {garch_vol:.1f}% daily (risk score: {garch_risk:.2f}).")
            else:
                parts.append(f"✅ **Low volatility** environment: {garch_vol:.1f}% daily vol provides stable outlook.")
        
        # Kalman trend
        kalman_trend = row.get("Kalman_Trend", 0)
        
        if kalman_trend > 5:
            parts.append(f"Kalman filter identifies strong uptrend ({kalman_trend:.1f}% over 20 days).")
        elif kalman_trend < -5:
            parts.append(f"Kalman filter detects downtrend ({kalman_trend:.1f}% over 20 days).")
        
        return " ".join(parts)
    
    def explain_mean_reversion(self, row: pd.Series) -> str:
        """Generate explanation for mean reversion analysis."""
        parts = []
        
        # Z-Score
        zscore = row.get("ZScore", 0)
        distance_from_mean = row.get("Distance_From_Mean", 0)
        
        if zscore > 2:
            parts.append(f"⚠️ **Significantly overvalued** (Z-Score: {zscore:.2f}): Price is {abs(distance_from_mean):.1f}% above 60-day mean, suggesting mean reversion downward.")
        elif zscore > 1:
            parts.append(f"Moderately extended (Z-Score: {zscore:.2f}): {abs(distance_from_mean):.1f}% above average.")
        elif zscore < -2:
            parts.append(f"💰 **Significantly undervalued** (Z-Score: {zscore:.2f}): Price {abs(distance_from_mean):.1f}% below mean suggests bounce opportunity.")
        elif zscore < -1:
            parts.append(f"Trading below average (Z-Score: {zscore:.2f}): {abs(distance_from_mean):.1f}% discount to mean.")
        else:
            parts.append(f"Price near statistical mean (Z-Score: {zscore:.2f}).")
        
        # Fair value
        upside = row.get("Upside_Potential", 0)
        fair_value = row.get("Fair_Value", 0)
        current_price = row.get("Current_Price", 0)
        
        if not pd.isna(upside):
            if upside > 15:
                parts.append(f"🎯 **Substantial upside**: Fair value at ${fair_value:.2f} implies {upside:.1f}% gain from current ${current_price:.2f}.")
            elif upside > 5:
                parts.append(f"Moderate upside: {upside:.1f}% to fair value (${fair_value:.2f}).")
            elif upside < -15:
                parts.append(f"⚠️ **Overvalued**: {abs(upside):.1f}% above fair value estimate of ${fair_value:.2f}.")
            elif upside < -5:
                parts.append(f"Slightly rich: {abs(upside):.1f}% premium to fair value.")
            else:
                parts.append(f"Trading near fair value (${fair_value:.2f}, {upside:+.1f}%).")
        
        return " ".join(parts)
    
    def explain_fundamentals(self, row: pd.Series) -> str:
        """Generate explanation for fundamental metrics."""
        parts = []
        
        pe_ratio = row.get("PE_Ratio", np.nan)
        forward_pe = row.get("Forward_PE", np.nan)
        pb_ratio = row.get("PB_Ratio", np.nan)
        roe = row.get("ROE", np.nan)
        profit_margin = row.get("Profit_Margin", np.nan)
        dividend_yield = row.get("Dividend_Yield", 0)
        
        # P/E analysis
        if not pd.isna(pe_ratio):
            if pe_ratio < 15:
                parts.append(f"💎 **Value play**: Low P/E of {pe_ratio:.1f}x suggests undervaluation.")
            elif pe_ratio < 25:
                parts.append(f"Reasonable valuation: P/E at {pe_ratio:.1f}x.")
            elif pe_ratio < 50:
                parts.append(f"Growth premium: P/E of {pe_ratio:.1f}x reflects high expectations.")
            else:
                parts.append(f"⚠️ **Expensive**: P/E ratio at {pe_ratio:.1f}x requires strong growth to justify.")
        
        # Profitability
        if not pd.isna(roe):
            if roe > 0.20:
                parts.append(f"✅ **Excellent profitability**: ROE of {roe*100:.1f}% demonstrates strong returns.")
            elif roe > 0.10:
                parts.append(f"Solid ROE of {roe*100:.1f}%.")
            elif roe > 0:
                parts.append(f"Modest ROE of {roe*100:.1f}%.")
            else:
                parts.append(f"⚠️ Negative ROE ({roe*100:.1f}%) indicates profitability concerns.")
        
        if not pd.isna(profit_margin):
            if profit_margin > 0.20:
                parts.append(f"Wide profit margins ({profit_margin*100:.1f}%) provide cushion.")
            elif profit_margin < 0.05:
                parts.append(f"Thin margins ({profit_margin*100:.1f}%) limit flexibility.")
        
        # Dividend
        if dividend_yield > 0.03:
            parts.append(f"💰 Attractive dividend yield of {dividend_yield*100:.1f}% provides income.")
        
        return " ".join(parts)
    
    def explain_performance(self, row: pd.Series) -> str:
        """Generate explanation for historical performance."""
        cagr = row.get("CAGR", 0)
        sharpe = row.get("Sharpe", 0)
        volatility = row.get("Volatility", 0)
        maxdd = row.get("MaxDD", 0)
        
        parts = []
        
        # Returns
        if cagr > 0.5:
            parts.append(f"🚀 **Outstanding returns**: {cagr*100:.1f}% annualized gain over analysis period.")
        elif cagr > 0.2:
            parts.append(f"Strong historical performance with {cagr*100:.1f}% CAGR.")
        elif cagr > 0:
            parts.append(f"Modest gains of {cagr*100:.1f}% annually.")
        else:
            parts.append(f"⚠️ Negative returns: {cagr*100:.1f}% CAGR.")
        
        # Risk-adjusted returns
        if sharpe > 1.5:
            parts.append(f"✅ **Excellent risk-adjusted returns** (Sharpe: {sharpe:.2f}) - strong reward per unit of risk.")
        elif sharpe > 0.8:
            parts.append(f"Good Sharpe ratio of {sharpe:.2f}.")
        elif sharpe > 0:
            parts.append(f"Moderate risk/reward (Sharpe: {sharpe:.2f}).")
        else:
            parts.append(f"⚠️ Poor risk-adjusted performance (Sharpe: {sharpe:.2f}).")
        
        # Volatility
        if volatility > 0.5:
            parts.append(f"⚠️ **High volatility** ({volatility*100:.1f}% annualized) - expect large swings.")
        elif volatility > 0.3:
            parts.append(f"Moderate volatility: {volatility*100:.1f}% annually.")
        else:
            parts.append(f"Low volatility ({volatility*100:.1f}%) provides stability.")
        
        # Drawdown
        if maxdd < -0.4:
            parts.append(f"⚠️ **Severe drawdowns**: Max loss of {abs(maxdd)*100:.1f}% shows high risk.")
        elif maxdd < -0.2:
            parts.append(f"Moderate drawdown risk (max: {abs(maxdd)*100:.1f}%).")
        else:
            parts.append(f"Resilient to losses (max drawdown: {abs(maxdd)*100:.1f}%).")
        
        return " ".join(parts)
    
    def generate_overall_summary(self, row: pd.Series, rank: int, total: int) -> str:
        """Generate overall summary and recommendation."""
        ticker = row.name
        final_score = row.get("Final_Score", 0)
        
        # Determine tier
        percentile = rank / total
        if percentile <= 0.2:
            tier = "🏆 **TOP TIER**"
            recommendation = "STRONG BUY"
        elif percentile <= 0.4:
            tier = "⭐ **STRONG PICK**"
            recommendation = "BUY"
        elif percentile <= 0.6:
            tier = "✅ **SOLID CHOICE**"
            recommendation = "HOLD/BUY"
        elif percentile <= 0.8:
            tier = "⚠️ **PROCEED WITH CAUTION**"
            recommendation = "HOLD"
        else:
            tier = "❌ **AVOID**"
            recommendation = "SELL/AVOID"
        
        summary = f"""
{'='*80}
{ticker} - RANK #{rank} of {total} | SCORE: {final_score:.4f}
{tier} | {recommendation}
{'='*80}
"""
        return summary
    
    def generate_key_strengths(self, row: pd.Series) -> str:
        """Identify and list key strengths."""
        strengths = []
        
        if row.get("CAGR", 0) > 0.3:
            strengths.append(f"✓ Strong historical returns ({row['CAGR']*100:.1f}% CAGR)")
        if row.get("Sharpe", 0) > 1.2:
            strengths.append(f"✓ Excellent risk-adjusted performance (Sharpe: {row['Sharpe']:.2f})")
        if row.get("CombinedSentiment", 0) > 0.1:
            strengths.append(f"✓ Positive market sentiment ({row['CombinedSentiment']:.2f})")
        if row.get("TopicMomentum", 0) > 0.55:
            strengths.append(f"✓ High media buzz and positive narrative")
        if row.get("MA_Signal", 0) == 1:
            strengths.append(f"✓ Bullish moving average crossover")
        if row.get("RSI_Signal", 0) == 1:
            strengths.append(f"✓ Oversold RSI suggests bounce opportunity")
        if row.get("ARIMA_Return", 0) > 3:
            strengths.append(f"✓ Positive price forecast (+{row['ARIMA_Return']:.1f}%)")
        if row.get("ZScore", 0) < -1.5:
            strengths.append(f"✓ Undervalued vs historical mean")
        if row.get("Upside_Potential", 0) > 10:
            strengths.append(f"✓ Significant upside to fair value (+{row['Upside_Potential']:.1f}%)")
        if row.get("PE_Ratio", 999) < 20:
            strengths.append(f"✓ Attractive valuation (P/E: {row['PE_Ratio']:.1f}x)")
        if row.get("ROE", 0) > 0.18:
            strengths.append(f"✓ High profitability (ROE: {row['ROE']*100:.1f}%)")
        
        if not strengths:
            strengths.append("Limited clear strengths identified")
        
        return "\n".join(strengths)
    
    def generate_key_risks(self, row: pd.Series) -> str:
        """Identify and list key risks/weaknesses."""
        risks = []
        
        if row.get("CAGR", 0) < 0:
            risks.append(f"⚠ Negative historical returns ({row['CAGR']*100:.1f}% CAGR)")
        if row.get("Sharpe", 0) < 0.5:
            risks.append(f"⚠ Poor risk-adjusted performance (Sharpe: {row['Sharpe']:.2f})")
        if row.get("Volatility", 0) > 0.5:
            risks.append(f"⚠ High volatility ({row['Volatility']*100:.1f}% annualized)")
        if row.get("MaxDD", 0) < -0.35:
            risks.append(f"⚠ Severe drawdown history ({abs(row['MaxDD'])*100:.1f}% max loss)")
        if row.get("CombinedSentiment", 0) < -0.15:
            risks.append(f"⚠ Negative market sentiment ({row['CombinedSentiment']:.2f})")
        if row.get("MA_Signal", 0) == -1:
            risks.append(f"⚠ Bearish moving average death cross")
        if row.get("RSI_Signal", 0) == -1:
            risks.append(f"⚠ Overbought RSI suggests pullback risk")
        if row.get("ARIMA_Return", 0) < -3:
            risks.append(f"⚠ Negative price forecast ({row['ARIMA_Return']:.1f}%)")
        if row.get("ZScore", 0) > 2:
            risks.append(f"⚠ Overvalued vs historical mean (Z-Score: {row['ZScore']:.2f})")
        if row.get("Upside_Potential", 0) < -10:
            risks.append(f"⚠ Trading above fair value ({abs(row['Upside_Potential']):.1f}% premium)")
        if row.get("PE_Ratio", 0) > 60:
            risks.append(f"⚠ Expensive valuation (P/E: {row['PE_Ratio']:.1f}x)")
        if row.get("GARCH_Vol", 0) > 4:
            risks.append(f"⚠ High forecasted volatility ({row['GARCH_Vol']:.1f}%)")
        
        if not risks:
            risks.append("No major red flags identified")
        
        return "\n".join(risks)
    
    def generate_full_explanation(self, row: pd.Series, rank: int, total: int) -> str:
        """Generate complete NLG explanation for a stock."""
        sections = []
        
        # Header
        sections.append(self.generate_overall_summary(row, rank, total))
        
        # Key strengths
        sections.append("\n📊 KEY STRENGTHS:")
        sections.append(self.generate_key_strengths(row))
        
        # Key risks
        sections.append("\n⚠️  KEY RISKS:")
        sections.append(self.generate_key_risks(row))
        
        # Performance analysis
        sections.append("\n📈 HISTORICAL PERFORMANCE:")
        sections.append(self.explain_performance(row))
        
        # Sentiment analysis
        if not pd.isna(row.get("CombinedSentiment")):
            sections.append("\n🗞️  SENTIMENT ANALYSIS:")
            sections.append(self.explain_sentiment(row))
        
        # Technical indicators
        sections.append("\n📊 TECHNICAL INDICATORS:")
        sections.append(self.explain_technical_indicators(row))
        
        # Statistical models
        sections.append("\n🔬 STATISTICAL FORECASTS:")
        sections.append(self.explain_statistical_models(row))
        
        # Mean reversion
        sections.append("\n↩️  MEAN REVERSION ANALYSIS:")
        sections.append(self.explain_mean_reversion(row))
        
        # Fundamentals
        if not pd.isna(row.get("PE_Ratio")):
            sections.append("\n💼 FUNDAMENTAL METRICS:")
            sections.append(self.explain_fundamentals(row))
        
        sections.append("\n" + "="*80 + "\n")
        
        return "\n".join(sections)

# ==============================
# Import computation functions from advanced quantitative
# (These would be copied from stock_picker_advanced_quantitative.py)
# For brevity, using import statement
# ==============================

# Import the run function and all necessary components
import sys
sys.path.append(os.path.dirname(__file__))

try:
    from stock_picker_advanced_quantitative import (
        load_prices, load_full_data, get_fundamental_metrics,
        compute_advanced_quant_signals, fuse_advanced_scores,
        build_portfolio, AdvancedWeights, Config as BaseConfig
    )
    ADVANCED_QUANT_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import from stock_picker_advanced_quantitative.py: {e}")
    print("Please ensure stock_picker_advanced_quantitative.py is in the same directory.")
    ADVANCED_QUANT_AVAILABLE = False

# ==============================
# Extended Config with NLG options
# ==============================

@dataclass
class Config:
    tickers: List[str]
    period: str = "2y"
    interval: str = "1d"
    top_k: int = 5
    include_ai: bool = True
    max_news: int = 50
    lookback_days: int = 14
    generate_explanations: bool = True
    explanations_top_n: int = 10  # Generate detailed explanations for top N stocks


def run_with_explanations(cfg: Config) -> Dict[str, object]:
    """Execute full pipeline with NLG explanations."""
    if not ADVANCED_QUANT_AVAILABLE:
        raise RuntimeError("stock_picker_advanced_quantitative.py must be available")
    
    print("="*60)
    print("STOCK PICKER - ADVANCED QUANTITATIVE WITH NLG")
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
    
    # Compute advanced signals
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
    
    # Generate NLG explanations
    explanations_text = ""
    if cfg.generate_explanations:
        print("\n" + "="*60)
        print("GENERATING NATURAL LANGUAGE EXPLANATIONS")
        print("="*60)
        
        explainer = StockExplainer()
        explanations = []
        
        # Generate for top N stocks
        top_n = min(cfg.explanations_top_n, len(ranked))
        for i, (ticker, row) in enumerate(ranked.head(top_n).iterrows(), 1):
            print(f"Generating explanation for #{i}: {ticker}...")
            explanation = explainer.generate_full_explanation(row, i, len(ranked))
            explanations.append(explanation)
        
        explanations_text = "\n".join(explanations)
    
    return {
        "prices": prices,
        "fundamentals": fundamentals,
        "quant": quant,
        "ai_signals": ai_signals,
        "weights": weights.__dict__,
        "ranked": ranked,
        "portfolio": portfolio,
        "explanations": explanations_text
    }


if __name__ == "__main__":
    # ------------------
    # CONFIGURE HERE
    # ------------------
    cfg = Config(
        tickers=[
            "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
            "META", "AVGO", "TSLA", "JPM", "V",
        ],
        period="2y",
        interval="1d",
        top_k=5,
        include_ai=True,
        max_news=50,
        lookback_days=14,
        generate_explanations=True,
        explanations_top_n=10
    )

    out = run_with_explanations(cfg)
    
    print("\n" + "="*60)
    print("RESULTS")
    print("="*60)
    
    print("\n=== Top Portfolio Picks ===")
    print(out["portfolio"])
    
    print("\n=== Full Rankings (Top 10) ===")
    display_cols = [
        "Current_Price", "CAGR", "Sharpe", "Final_Score",
        "CombinedSentiment", "TopicMomentum",
        "MA_Signal", "MACD_Crossover", "RSI",
        "ARIMA_Return", "Upside_Potential", "ZScore", "PE_Ratio"
    ]
    available = [c for c in display_cols if c in out["ranked"].columns]
    print(out["ranked"][available].head(10))
    
    # Save artifacts
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create output directory
        output_dir = f"nlg_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)
        print(f"\nCreating output directory: {output_dir}/")
        
        out["ranked"].to_csv(os.path.join(output_dir, f"ranked_signals_nlg_{timestamp}.csv"))
        out["portfolio"].to_csv(os.path.join(output_dir, f"portfolio_nlg_{timestamp}.csv"))
        out["fundamentals"].to_csv(os.path.join(output_dir, f"fundamentals_nlg_{timestamp}.csv"))
        
        if out["ai_signals"] is not None:
            out["ai_signals"].to_csv(os.path.join(output_dir, f"ai_signals_nlg_{timestamp}.csv"))
        
        # Save NLG explanations to text file
        if out["explanations"]:
            explanation_file = os.path.join(output_dir, f"explanations_{timestamp}.txt")
            with open(explanation_file, 'w', encoding='utf-8') as f:
                f.write("STOCK ANALYSIS - NATURAL LANGUAGE EXPLANATIONS\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("="*80 + "\n\n")
                f.write(out["explanations"])
            
            print(f"\n✓ Saved all artifacts to folder: {output_dir}/")
            print(f"  - ranked_signals_nlg_{timestamp}.csv")
            print(f"  - portfolio_nlg_{timestamp}.csv")
            print(f"  - fundamentals_nlg_{timestamp}.csv")
            if out["ai_signals"] is not None:
                print(f"  - ai_signals_nlg_{timestamp}.csv")
            print(f"  - explanations_{timestamp}.txt ⭐")
            print(f"\n📖 Open {output_dir}/explanations_{timestamp}.txt for detailed analysis of each stock!")
        
    except Exception as e:
        print(f"\nError saving files: {e}")
