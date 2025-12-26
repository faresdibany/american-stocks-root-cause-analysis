"""
Stock Picker — Advanced Quantitative Analysis with Driver-Aware NLG Explanations
-------------------------------------------------------------------------------
This script extends stock_picker_advanced_quantitative.py with Natural Language Generation (NLG)
and *driver discovery* to produce driver-aware explanations and ranking.

Key upgrades vs original stock_picker_nlg_explanations.py
---------------------------------------------------------
1) Driver Discovery (worldwide, best-effort):
   - Infers which drivers dominate (beta/sector vs idiosyncratic vs rates vs commodity)
   - Produces: mode, R², market-like share, top drivers
2) Driver-Conditional Explanations:
   - Always prints DRIVER PROFILE first
   - Suppresses or de-emphasizes signals that are not appropriate for the mode
3) Replacement Ranking Logic:
   - Adds ModeAwareScore that re-weights signals based on driver mode
   - Optionally re-sorts ranked table and portfolio selection accordingly
4) Safety fixes:
   - Clamps dividend yield for NLG to avoid absurd outputs if upstream data is wrong.

Output
------
Creates additional file: explanations_YYYYMMDD_HHMMSS.txt
Contains driver-aware narrative explanations for each ticker.

Notes
-----
- This is not investment advice.
- Worldwide proxies are best-effort; missing factors are automatically dropped in driver_discovery.
"""

from __future__ import annotations

import os
import math
import sys
import warnings
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Any

import numpy as np
import pandas as pd

warnings.filterwarnings("ignore")

# Local imports (same folder)
sys.path.append(os.path.dirname(__file__))

# ------------------------------
# Advanced quant pipeline import
# ------------------------------
try:
    from stock_picker_advanced_quantitative import (
        load_prices, load_full_data, get_fundamental_metrics,
        compute_advanced_quant_signals, fuse_advanced_scores,
        build_portfolio, AdvancedWeights
    )
    ADVANCED_QUANT_AVAILABLE = True
except Exception as e:
    print(f"Warning: Could not import from stock_picker_advanced_quantitative.py: {e}")
    ADVANCED_QUANT_AVAILABLE = False

# ------------------------------
# Driver discovery import
# ------------------------------
try:
    from driver_discovery import discover_driver_profile
    DRIVER_DISCOVERY_AVAILABLE = True
except Exception as e:
    print(f"Warning: driver_discovery.py not available: {e}")
    DRIVER_DISCOVERY_AVAILABLE = False


# ==============================
# Utilities
# ==============================

def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.nunique(dropna=True) <= 1:
        return pd.Series(0.0, index=s.index)
    return (s - s.min()) / (s.max() - s.min())


def _safe_float(x, default=0.0) -> float:
    try:
        if x is None or (isinstance(x, float) and np.isnan(x)):
            return float(default)
        return float(x)
    except Exception:
        return float(default)


def _clamp_dividend_yield(dy: float) -> float:
    """
    Clamp dividend yield for NLG readability.
    Upstream sources sometimes return broken values (e.g., 182%).
    """
    dy = _safe_float(dy, 0.0)
    if dy < 0:
        return 0.0
    # If > 20% it's likely wrong for most large caps; clamp for narrative text.
    return min(dy, 0.20)


# ==============================
# Mode-aware ranking logic
# ==============================

def _mode_weights(mode: str) -> Dict[str, float]:
    """
    Weights for ModeAwareScore.
    Columns are expected to exist in ranked table; missing columns become zero.
    """
    if mode == "beta_or_sector_driven":
        return {"CAGR": 0.45, "Sharpe": 0.40, "Sent": 0.10, "Topic": 0.05}
    if mode == "narrative_or_idiosyncratic":
        return {"CAGR": 0.20, "Sharpe": 0.25, "Sent": 0.35, "Topic": 0.20}
    if mode == "rates_sensitive":
        return {"CAGR": 0.30, "Sharpe": 0.50, "Sent": 0.15, "Topic": 0.05}
    if mode == "commodity_oil_sensitive":
        return {"CAGR": 0.35, "Sharpe": 0.45, "Sent": 0.15, "Topic": 0.05}
    return {"CAGR": 0.35, "Sharpe": 0.35, "Sent": 0.20, "Topic": 0.10}


def apply_mode_aware_ranking(ranked: pd.DataFrame, driver_profiles: Dict[str, Dict[str, Any]]) -> pd.DataFrame:
    """
    Adds:
      - driver_mode, driver_r2, driver_market_like_share
      - ModeAwareScore
    and returns ranked sorted by ModeAwareScore desc.

    This is intentionally simple and robust.
    """
    if ranked is None or ranked.empty:
        return ranked

    df = ranked.copy()

    df["CAGR_n"] = _minmax(df["CAGR"]) if "CAGR" in df.columns else 0.0
    df["Sharpe_n"] = _minmax(df["Sharpe"]) if "Sharpe" in df.columns else 0.0
    df["Sent_n"] = _minmax(df["CombinedSentiment"]) if "CombinedSentiment" in df.columns else 0.0
    df["Topic_n"] = _minmax(df["TopicMomentum"]) if "TopicMomentum" in df.columns else 0.0

    # Attach driver info
    modes, r2s, mls = {}, {}, {}
    for t in df.index:
        prof = (driver_profiles or {}).get(t, {}) or {}
        modes[t] = prof.get("mode", "mixed")
        r2s[t] = prof.get("r2", np.nan)
        mls[t] = prof.get("market_like_share", np.nan)

    df["driver_mode"] = pd.Series(modes)
    df["driver_r2"] = pd.Series(r2s)
    df["driver_market_like_share"] = pd.Series(mls)

    # Jump risk proxy from profile: if idio share is high, treat as higher gap-risk
    idio_share = {}
    for t in df.index:
        prof = (driver_profiles or {}).get(t, {}) or {}
        idio_share[t] = _safe_float((prof.get("idiosyncratic") or {}).get("share"), np.nan)

    df["driver_idio_share"] = pd.Series(idio_share)

    scores = []
    for t in df.index:
        mode = str(df.loc[t, "driver_mode"])
        w = _mode_weights(mode)

        base = (
            w["CAGR"] * _safe_float(df.loc[t, "CAGR_n"]) +
            w["Sharpe"] * _safe_float(df.loc[t, "Sharpe_n"]) +
            w["Sent"] * _safe_float(df.loc[t, "Sent_n"]) +
            w["Topic"] * _safe_float(df.loc[t, "Topic_n"])
        )

        # Penalties / adjustments
        if mode == "narrative_or_idiosyncratic":
            # penalize extreme idiosyncratic share
            base -= 0.20 * max(0.0, _safe_float(df.loc[t, "driver_idio_share"]) - 0.55)
        elif mode == "rates_sensitive":
            # small stability bump if Sharpe is good
            base += 0.05 * _safe_float(df.loc[t, "Sharpe_n"])
        elif mode == "beta_or_sector_driven":
            # reward high R² (drivers explain moves; more hedgeable)
            base += 0.05 * _safe_float(df.loc[t, "driver_r2"])
        scores.append(base)

    df["ModeAwareScore"] = pd.Series(scores, index=df.index)
    return df.sort_values("ModeAwareScore", ascending=False)


# ==============================
# NLG Explanation Generator (Driver-aware)
# ==============================

class StockExplainer:
    """
    Driver-aware explainer:
    - prints DRIVER PROFILE first
    - conditionally suppresses sections depending on driver mode
    """

    def __init__(self, driver_profiles: Optional[Dict[str, Dict[str, Any]]] = None):
        self.driver_profiles = driver_profiles or {}

    def _driver_profile_section(self, ticker: str) -> str:
        prof = (self.driver_profiles or {}).get(ticker, {}) or {}
        if not prof:
            return "DRIVER PROFILE:\n- (driver discovery unavailable)\n"

        mode = prof.get("mode", "mixed")
        r2 = prof.get("r2", None)
        ml = prof.get("market_like_share", None)
        idio = (prof.get("idiosyncratic") or {}).get("share", None)
        top = prof.get("driver_rank", [])[:5]

        lines = ["DRIVER PROFILE:"]
        lines.append(f"- Mode: {mode}")
        if r2 is not None and not (isinstance(r2, float) and np.isnan(r2)):
            lines.append(f"- Explained by factors (R²): {float(r2):.2f}")
        if ml is not None and not (isinstance(ml, float) and np.isnan(ml)):
            lines.append(f"- Market-like share (market/sector/region/global): {float(ml):.2f}")
        if idio is not None and not (isinstance(idio, float) and np.isnan(idio)):
            lines.append(f"- Idiosyncratic share: {float(idio):.2f}")

        if isinstance(top, list) and top:
            pretty = []
            for d in top:
                try:
                    pretty.append(f"{d.get('driver')} ({d.get('share',0):.2f})")
                except Exception:
                    pass
            if pretty:
                lines.append("- Top discovered drivers: " + ", ".join(pretty))

        checklist = prof.get("analysis_checklist") or []
        if checklist:
            lines.append("INTERPRETATION (how to analyze):")
            for it in checklist[:4]:
                lines.append(f"- {it}")

        return "\n".join(lines) + "\n"

    def _mode(self, ticker: str) -> str:
        prof = (self.driver_profiles or {}).get(ticker, {}) or {}
        return str(prof.get("mode", "mixed"))

    # --- Existing explanation helpers (kept, but will be conditionally used) ---

    def explain_sentiment(self, row: pd.Series) -> str:
        stock_sent = row.get("StockSentiment", 0)
        world_sent = row.get("WorldSentiment", 0)
        combined = row.get("CombinedSentiment", 0)
        topic = row.get("TopicMomentum", 0)
        headline_count = row.get("HeadlineCount", 0)

        parts = []
        if combined > 0.15:
            parts.append(f"📈 **Strong positive sentiment** (score: {combined:.2f}) based on {headline_count} headlines.")
        elif combined > 0.05:
            parts.append(f"📊 **Moderately positive sentiment** (score: {combined:.2f}) from {headline_count} headlines.")
        elif combined > -0.05:
            parts.append(f"➡️ **Neutral sentiment** (score: {combined:.2f}) across {headline_count} headlines.")
        elif combined > -0.15:
            parts.append(f"📉 **Moderately negative sentiment** (score: {combined:.2f}) from {headline_count} headlines.")
        else:
            parts.append(f"⚠️ **Significantly negative sentiment** (score: {combined:.2f}) based on {headline_count} headlines.")

        if abs(stock_sent) > abs(world_sent) + 0.1:
            parts.append(f"Stock-specific news dominates sentiment (stock: {stock_sent:.2f} vs world: {world_sent:.2f}).")
        elif abs(world_sent) > abs(stock_sent) + 0.1:
            parts.append(f"Macro news dominates sentiment (world: {world_sent:.2f} vs stock: {stock_sent:.2f}).")

        if topic > 0.6:
            parts.append(f"🚀 **High media buzz** (momentum: {topic:.2f}).")
        elif topic > 0.5:
            parts.append(f"Moderate media attention (momentum: {topic:.2f}).")
        else:
            parts.append(f"Limited media buzz (momentum: {topic:.2f}).")

        return " ".join(parts)

    def explain_technical_indicators(self, row: pd.Series) -> str:
        parts = []
        ma_signal = row.get("MA_Signal", 0)
        ma_distance = row.get("MA_Distance", 0)

        if ma_signal == 1:
            parts.append(f"📊 **Bullish MA crossover**: 50-day SMA is {abs(ma_distance):.1f}% above 200-day SMA.")
        elif ma_signal == -1:
            parts.append(f"📉 **Bearish MA crossover**: 50-day SMA is {abs(ma_distance):.1f}% below 200-day SMA.")
        else:
            parts.append(f"➡️ **Neutral MA trend** (distance: {ma_distance:.1f}%).")

        macd_hist = row.get("MACD_Hist", 0)
        if macd_hist > 0:
            parts.append(f"MACD momentum positive (hist: {macd_hist:.3f}).")
        else:
            parts.append(f"MACD momentum weak (hist: {macd_hist:.3f}).")

        rsi = row.get("RSI", 50)
        rsi_signal = row.get("RSI_Signal", 0)
        if rsi_signal == 1:
            parts.append(f"💪 RSI oversold (RSI: {rsi:.1f}) suggests bounce risk/reward improving.")
        elif rsi_signal == -1:
            parts.append(f"⚠️ RSI overbought (RSI: {rsi:.1f}) suggests pullback risk.")
        else:
            parts.append(f"RSI {rsi:.1f} is neutral.")

        bb_position = row.get("BB_Position", 0.5)
        bb_width = row.get("BB_Width", 0)
        if bb_position > 0.8:
            parts.append(f"Near upper Bollinger band ({bb_position*100:.0f}%) → potentially extended.")
        elif bb_position < 0.2:
            parts.append(f"Near lower Bollinger band ({bb_position*100:.0f}%) → potentially oversold.")
        if bb_width > 0.1:
            parts.append(f"Volatility elevated (BB width: {bb_width:.2f}).")

        return " ".join(parts)

    def explain_statistical_models(self, row: pd.Series) -> str:
        parts = []
        arima_return = row.get("ARIMA_Return", np.nan)
        if not pd.isna(arima_return):
            if arima_return > 3:
                parts.append(f"ARIMA projects mild upside: +{arima_return:.1f}% (5d).")
            elif arima_return < -3:
                parts.append(f"ARIMA warns mild downside: {arima_return:.1f}% (5d).")
            else:
                parts.append(f"ARIMA near-flat: {arima_return:.1f}% (5d).")
        garch_vol = row.get("GARCH_Vol", np.nan)
        if not pd.isna(garch_vol):
            parts.append(f"GARCH daily vol est.: {garch_vol:.1f}%.")
        return " ".join(parts)

    def explain_mean_reversion(self, row: pd.Series) -> str:
        zscore = row.get("ZScore", 0)
        dist = row.get("Distance_From_Mean", 0)
        upside = row.get("Upside_Potential", np.nan)
        fair_value = row.get("Fair_Value", np.nan)
        px = row.get("Current_Price", np.nan)

        parts = []
        if zscore > 2:
            parts.append(f"Overextended vs mean (Z: {zscore:.2f}, {abs(dist):.1f}% above mean).")
        elif zscore < -2:
            parts.append(f"Undervalued vs mean (Z: {zscore:.2f}, {abs(dist):.1f}% below mean).")
        else:
            parts.append(f"Near mean (Z: {zscore:.2f}).")

        if not pd.isna(upside) and not pd.isna(fair_value) and not pd.isna(px):
            parts.append(f"Fair value est.: ${fair_value:.2f} ({upside:+.1f}%).")

        return " ".join(parts)

    def explain_fundamentals(self, row: pd.Series) -> str:
        parts = []
        pe = row.get("PE_Ratio", np.nan)
        roe = row.get("ROE", np.nan)
        pm = row.get("Profit_Margin", np.nan)
        dy = _clamp_dividend_yield(row.get("Dividend_Yield", 0))

        if not pd.isna(pe):
            if pe < 15:
                parts.append(f"💎 Low P/E ({pe:.1f}x) suggests value tilt.")
            elif pe < 35:
                parts.append(f"Moderate valuation (P/E {pe:.1f}x).")
            else:
                parts.append(f"Growth premium (P/E {pe:.1f}x).")

        if not pd.isna(roe):
            parts.append(f"ROE: {roe*100:.1f}%.")

        if not pd.isna(pm):
            parts.append(f"Margins: {pm*100:.1f}%.")

        if dy > 0.03:
            parts.append(f"Dividend: ~{dy*100:.1f}% (clamped for display).")

        return " ".join(parts)

    def explain_performance(self, row: pd.Series) -> str:
        cagr = row.get("CAGR", 0)
        sharpe = row.get("Sharpe", 0)
        vol = row.get("Volatility", 0)
        mdd = row.get("MaxDD", 0)

        parts = []
        parts.append(f"CAGR: {cagr*100:.1f}%, Sharpe: {sharpe:.2f}, Vol: {vol*100:.1f}%, MaxDD: {abs(mdd)*100:.1f}%.")

        return " ".join(parts)

    # --- Recommendation / positioning guidance (driver-aware) ---

    def positioning_guidance(self, ticker: str, rank: int, total: int) -> str:
        mode = self._mode(ticker)
        percentile = rank / max(total, 1)

        # Confidence tier still shown, but not BUY/SELL
        if percentile <= 0.2:
            tier = "🏆 TOP TIER"
        elif percentile <= 0.4:
            tier = "⭐ STRONG PICK"
        elif percentile <= 0.6:
            tier = "✅ SOLID"
        elif percentile <= 0.8:
            tier = "⚠️ CAUTION"
        else:
            tier = "❌ LOW RANK"

        # Mode-specific guidance
        if mode == "narrative_or_idiosyncratic":
            guide = "Positioning: catalyst/narrative stock → size for gap risk; use technicals for timing; prioritize news/catalyst monitoring over mean-reversion."
        elif mode == "rates_sensitive":
            guide = "Positioning: macro-conditional (rates/curve) → evaluate under rate regime; hedge/size around macro event days; avoid over-relying on RSI/ARIMA."
        elif mode == "commodity_oil_sensitive":
            guide = "Positioning: commodity regime exposure → track oil/USD and geopolitics; treat earnings as cycle confirmation."
        elif mode == "beta_or_sector_driven":
            guide = "Positioning: beta/sector exposure (hedgeable) → compare vs market+sector; use hedges; single headlines less explanatory on most days."
        else:
            guide = "Positioning: mixed drivers → combine macro/sector context with idiosyncratic news; avoid single-factor decisions."

        return f"{tier} | {guide}"

    def generate_key_strengths(self, row: pd.Series, ticker: str) -> str:
        mode = self._mode(ticker)
        strengths = []

        if row.get("CAGR", 0) > 0.3:
            strengths.append(f"✓ Strong returns ({row['CAGR']*100:.1f}% CAGR)")
        if row.get("Sharpe", 0) > 1.2:
            strengths.append(f"✓ Strong risk-adjusted profile (Sharpe: {row['Sharpe']:.2f})")
        if row.get("CombinedSentiment", 0) > 0.1 and mode != "beta_or_sector_driven":
            strengths.append(f"✓ Positive narrative/sentiment ({row['CombinedSentiment']:.2f})")
        if row.get("MA_Signal", 0) == 1 and mode in {"beta_or_sector_driven", "mixed"}:
            strengths.append("✓ Trend support (bullish MA structure)")

        if not strengths:
            strengths.append("No standout strengths from the selected signals.")

        return "\n".join(strengths)

    def generate_key_risks(self, row: pd.Series, ticker: str) -> str:
        mode = self._mode(ticker)
        risks = []

        if row.get("Volatility", 0) > 0.5:
            risks.append(f"⚠ High volatility ({row['Volatility']*100:.1f}% ann.)")
        if row.get("MaxDD", 0) < -0.35:
            risks.append(f"⚠ Severe drawdowns ({abs(row['MaxDD'])*100:.1f}%)")

        if mode == "narrative_or_idiosyncratic":
            risks.append("⚠ High idiosyncratic/jump risk: large moves can occur on news/catalysts.")
        if mode == "rates_sensitive":
            risks.append("⚠ Rate sensitivity: macro shocks can dominate near-term returns.")
        if mode == "commodity_oil_sensitive":
            risks.append("⚠ Commodity sensitivity: oil regime shifts can dominate.")

        if row.get("PE_Ratio", 0) > 60:
            risks.append(f"⚠ Valuation stretch (P/E: {row['PE_Ratio']:.1f}x)")

        if not risks:
            risks.append("No major red flags from the selected risk checks.")

        return "\n".join(risks)

    def generate_full_explanation(self, row: pd.Series, rank: int, total: int) -> str:
        ticker = str(row.name)
        lines = []

        lines.append("=" * 80)
        score = row.get("ModeAwareScore", row.get("Final_Score", 0.0))
        lines.append(f"{ticker} - RANK #{rank} of {total} | SCORE: {score:.4f}")
        lines.append(self.positioning_guidance(ticker, rank, total))
        lines.append("=" * 80)
        lines.append("")

        # DRIVER PROFILE always first
        lines.append(self._driver_profile_section(ticker))

        # Strengths / risks (mode-aware)
        lines.append("📊 KEY STRENGTHS:")
        lines.append(self.generate_key_strengths(row, ticker))
        lines.append("")
        lines.append("⚠️  KEY RISKS:")
        lines.append(self.generate_key_risks(row, ticker))
        lines.append("")

        # Performance
        lines.append("📈 HISTORICAL PERFORMANCE:")
        lines.append(self.explain_performance(row))
        lines.append("")

        # Sentiment (always ok, but its weight differs by mode)
        if not pd.isna(row.get("CombinedSentiment", np.nan)):
            lines.append("🗞️  SENTIMENT ANALYSIS:")
            lines.append(self.explain_sentiment(row))
            lines.append("")

        mode = self._mode(ticker)

        # Technical indicators:
        # - useful for beta/sector and mixed
        # - allowed but de-emphasized for narrative (as timing)
        # - de-emphasized for rates/commodity
        show_tech = mode in {"beta_or_sector_driven", "mixed", "narrative_or_idiosyncratic"}
        if show_tech:
            lines.append("📊 TECHNICAL INDICATORS:")
            tech = self.explain_technical_indicators(row)
            if mode == "narrative_or_idiosyncratic":
                tech = "Timing-only (secondary): " + tech
            if mode in {"rates_sensitive", "commodity_oil_sensitive"}:
                tech = "Secondary (macro dominates): " + tech
            lines.append(tech)
            lines.append("")

        # Statistical forecasts:
        # - generally not useful for narrative/rates/commodity; keep short or skip
        if mode in {"beta_or_sector_driven", "mixed"}:
            lines.append("🔬 STATISTICAL FORECASTS (low weight):")
            lines.append(self.explain_statistical_models(row))
            lines.append("")

        # Mean reversion:
        # - skip for narrative and beta names; can be shown for mixed as context
        if mode in {"mixed"}:
            lines.append("↩️  MEAN REVERSION (context only):")
            lines.append(self.explain_mean_reversion(row))
            lines.append("")

        # Fundamentals (always allowed, but interpret lightly for narrative names)
        if not pd.isna(row.get("PE_Ratio", np.nan)):
            lines.append("💼 FUNDAMENTALS:")
            fundamentals = self.explain_fundamentals(row)
            if mode == "narrative_or_idiosyncratic":
                fundamentals = "Useful for long-horizon context; near-term is narrative-driven. " + fundamentals
            lines.append(fundamentals)
            lines.append("")

        lines.append("=" * 80)
        lines.append("")
        return "\n".join(lines)


# ==============================
# Config
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
    explanations_top_n: int = 10
    use_mode_aware_ranking: bool = True   # NEW: replace ranking logic
    driver_min_obs: int = 90              # NEW: discovery min observations


def run_with_explanations(cfg: Config) -> Dict[str, object]:
    if not ADVANCED_QUANT_AVAILABLE:
        raise RuntimeError("stock_picker_advanced_quantitative.py must be available")

    print("=" * 60)
    print("STOCK PICKER - ADVANCED QUANT + DRIVER-AWARE NLG")
    print("=" * 60)

    print("\nLoading price data...")
    prices = load_prices(cfg.tickers, period=cfg.period, interval=cfg.interval)

    threshold = int(0.8 * len(prices))
    prices = prices.dropna(axis=1, thresh=threshold).ffill()
    tickers = prices.columns.tolist()

    print(f"Loaded data for {len(tickers)} tickers")

    print("\nLoading full OHLCV data...")
    full_data = load_full_data(tickers, period=cfg.period)

    fundamentals = get_fundamental_metrics(tickers)

    quant, ai_signals = compute_advanced_quant_signals(
        prices,
        full_data,
        include_ai=cfg.include_ai,
        max_news=cfg.max_news,
        lookback_days=cfg.lookback_days
    )

    weights = AdvancedWeights()
    ranked = fuse_advanced_scores(quant, fundamentals, ai_signals, weights)

    # ------------------------------
    # Driver discovery (worldwide)
    # ------------------------------
    driver_profiles: Dict[str, Dict[str, Any]] = {}
    if DRIVER_DISCOVERY_AVAILABLE:
        print("\nRunning driver discovery (best-effort)...")
        # build a date window from prices index
        start = prices.index.min().strftime("%Y-%m-%d")
        end = (prices.index.max() + pd.Timedelta(days=1)).strftime("%Y-%m-%d")

        # discover for all tickers used in ranking
        for t in ranked.index:
            try:
                prof = discover_driver_profile(
                    ticker=t,
                    start=start,
                    end=end,
                    headlines_by_date=None,
                    min_obs=cfg.driver_min_obs,
                )
                if prof.get("available"):
                    driver_profiles[t] = prof
            except Exception:
                continue

    # ------------------------------
    # Apply mode-aware ranking
    # ------------------------------
    if cfg.use_mode_aware_ranking and driver_profiles:
        ranked = apply_mode_aware_ranking(ranked, driver_profiles)

    portfolio = build_portfolio(ranked, top_k=cfg.top_k)

    # ------------------------------
    # NLG explanations
    # ------------------------------
    explanations_text = ""
    if cfg.generate_explanations:
        print("\n" + "=" * 60)
        print("GENERATING DRIVER-AWARE EXPLANATIONS")
        print("=" * 60)

        explainer = StockExplainer(driver_profiles=driver_profiles)

        explanations = []
        top_n = min(cfg.explanations_top_n, len(ranked))
        for i, (ticker, row) in enumerate(ranked.head(top_n).iterrows(), 1):
            print(f"Generating explanation for #{i}: {ticker}...")
            explanations.append(explainer.generate_full_explanation(row, i, len(ranked)))

        explanations_text = "\n".join(explanations)

    return {
        "prices": prices,
        "fundamentals": fundamentals,
        "quant": quant,
        "ai_signals": ai_signals,
        "weights": weights.__dict__,
        "ranked": ranked,
        "portfolio": portfolio,
        "explanations": explanations_text,
        "driver_profiles": driver_profiles,
    }


if __name__ == "__main__":
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
        explanations_top_n=10,
        use_mode_aware_ranking=True,
        driver_min_obs=90,
    )

    out = run_with_explanations(cfg)

    print("\n" + "=" * 60)
    print("RESULTS")
    print("=" * 60)

    print("\n=== Top Portfolio Picks ===")
    print(out["portfolio"])

    print("\n=== Full Rankings (Top 10) ===")
    display_cols = [
        "Current_Price", "CAGR", "Sharpe", "Final_Score", "ModeAwareScore",
        "CombinedSentiment", "TopicMomentum",
        "driver_mode", "driver_r2", "driver_market_like_share",
        "MA_Signal", "RSI", "ARIMA_Return",
        "Upside_Potential", "ZScore", "PE_Ratio"
    ]
    available = [c for c in display_cols if c in out["ranked"].columns]
    print(out["ranked"][available].head(10))

    # Save artifacts
    try:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_dir = f"nlg_analysis_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

        out["ranked"].to_csv(os.path.join(output_dir, f"ranked_signals_nlg_{timestamp}.csv"))
        out["portfolio"].to_csv(os.path.join(output_dir, f"portfolio_nlg_{timestamp}.csv"))
        out["fundamentals"].to_csv(os.path.join(output_dir, f"fundamentals_nlg_{timestamp}.csv"))

        if out["ai_signals"] is not None:
            out["ai_signals"].to_csv(os.path.join(output_dir, f"ai_signals_nlg_{timestamp}.csv"))

        if out.get("explanations"):
            explanation_file = os.path.join(output_dir, f"explanations_{timestamp}.txt")
            with open(explanation_file, "w", encoding="utf-8") as f:
                f.write("STOCK ANALYSIS - DRIVER-AWARE NLG EXPLANATIONS\n")
                f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                f.write("=" * 80 + "\n\n")
                f.write(out["explanations"])

        print(f"\n✓ Saved artifacts to folder: {output_dir}/")
        if out.get("explanations"):
            print(f"  - explanations_{timestamp}.txt ⭐")

    except Exception as e:
        print(f"\nError saving files: {e}")
