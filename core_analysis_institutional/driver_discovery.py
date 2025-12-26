"""driver_discovery.py

Generalized, outcome-driven driver discovery for *any* stock ticker (worldwide, best-effort).

Goal
----
Given a stock ticker and a date range, infer which drivers matter most by:
1) inferring metadata (country, sector) via yfinance (best-effort)
2) selecting proxy benchmarks: market ETF + sector ETF (best-effort)
3) building a broad candidate factor dictionary (market/sector + macro/style)
4) fitting a regularized linear factor model (ridge) to estimate exposures
5) ranking drivers by average absolute contribution
6) deriving an "analysis mode" (beta/sector-driven vs narrative/idio vs rates vs commodity)

This module is designed to be called from rca_pipeline before running stock_driver_analysis,
so you can pass the chosen market/sector proxies and the full factor dict.

Interfaces (stable)
-------------------
- infer_metadata(ticker) -> dict
- build_candidate_factors(meta) -> dict[str,str]
- discover_driver_profile(ticker, start, end, *, headlines_by_date=None, factor_tickers=None) -> dict

Notes
-----
- This is *not* investment advice.
- Worldwide coverage is best-effort; if some ETFs are unavailable in your data provider,
  the fitter will automatically drop missing series and continue.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple

import math

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception as e:  # pragma: no cover
    yf = None  # type: ignore


# ----------------------------
# ETF proxy maps (best-effort)
# ----------------------------

# US Select Sector SPDRs (commonly available)
SECTOR_TO_ETF_US: Dict[str, str] = {
    "Technology": "XLK",
    "Financial Services": "XLF",
    "Financial": "XLF",
    "Healthcare": "XLV",
    "Health Care": "XLV",
    "Consumer Cyclical": "XLY",
    "Consumer Defensive": "XLP",
    "Consumer Discretionary": "XLY",
    "Consumer Staples": "XLP",
    "Energy": "XLE",
    "Industrials": "XLI",
    "Basic Materials": "XLB",
    "Materials": "XLB",
    "Utilities": "XLU",
    "Real Estate": "XLRE",
    "Communication Services": "XLC",
}

# Common country ETFs (US-listed iShares/others). Not exhaustive.
COUNTRY_TO_ETF: Dict[str, str] = {
    "United States": "SPY",
    "USA": "SPY",
    "US": "SPY",
    "Germany": "EWG",
    "United Kingdom": "EWU",
    "UK": "EWU",
    "France": "EWQ",
    "Italy": "EWI",
    "Spain": "EWP",
    "Switzerland": "EWL",
    "Netherlands": "EWN",  # iShares Netherlands (thin)
    "Sweden": "EWD",
    "Norway": "ENOR",      # Global X Norway
    "Denmark": "EDEN",     # iShares Denmark (thin)
    "Austria": "EWO",      # iShares Austria (thin)
    "Belgium": "EWK",      # iShares Belgium (thin)
    "Ireland": "EIRL",
    "Japan": "EWJ",
    "China": "MCHI",
    "Hong Kong": "EWH",
    "Taiwan": "EWT",
    "South Korea": "EWY",
    "India": "INDA",
    "Australia": "EWA",
    "Canada": "EWC",
    "Brazil": "EWZ",
    "Mexico": "EWW",
    "South Africa": "EZA",
    "Singapore": "EWS",
    "Malaysia": "EWM",
    "Indonesia": "EIDO",
    "Thailand": "THD",
    "Vietnam": "VNM",
}

# Region proxies
REGION_DEFAULTS: Dict[str, str] = {
    "developed_ex_us": "EFA",
    "emerging": "EEM",
    "global": "VT",
    "europe": "VGK",
    "asia_pacific": "VPL",
}

# Broad style + macro proxies (US-listed; widely available)
DEFAULT_MACRO_STYLE_FACTORS: Dict[str, str] = {
    # styles
    "growth": "QQQ",
    "value": "IWD",
    "small": "IWM",
    # macro
    "rates_intermediate": "IEF",
    "rates_long": "TLT",
    "usd": "UUP",
    "oil": "USO",
    "gold": "GLD",
    "vix": "VXX",
}


# ----------------------------
# Utilities
# ----------------------------

def _safe_float(x) -> Optional[float]:
    try:
        if x is None:
            return None
        return float(x)
    except Exception:
        return None


def infer_metadata(ticker: str) -> Dict[str, Any]:
    """Infer metadata for the given ticker via yfinance, best-effort."""
    meta: Dict[str, Any] = {
        "ticker": ticker,
        "country": None,
        "currency": None,
        "exchange": None,
        "sector": None,
        "industry": None,
        "quoteType": None,
        "marketCap": None,
    }

    if yf is None:
        return meta

    try:
        tk = yf.Ticker(ticker)
        info = getattr(tk, "info", None) or {}
        # yfinance info fields vary by ticker/exchange
        meta.update({
            "country": info.get("country") or info.get("region"),
            "currency": info.get("currency"),
            "exchange": info.get("exchange") or info.get("fullExchangeName"),
            "sector": info.get("sector"),
            "industry": info.get("industry"),
            "quoteType": info.get("quoteType"),
            "marketCap": info.get("marketCap"),
        })
    except Exception:
        pass

    return meta


def _infer_region(meta: Dict[str, Any]) -> str:
    """Rough region inference from country."""
    c = (meta.get("country") or "")
    c = str(c)
    if not c:
        return "global"
    europe = {
        "Germany","United Kingdom","France","Italy","Spain","Switzerland","Netherlands",
        "Sweden","Norway","Denmark","Austria","Belgium","Ireland"
    }
    apac = {"Japan","Australia","New Zealand","Singapore","Hong Kong","South Korea","Taiwan"}
    emerging = {"China","India","Brazil","Mexico","South Africa","Indonesia","Malaysia","Thailand","Vietnam"}

    if c in europe:
        return "europe"
    if c in apac:
        return "asia_pacific"
    if c in emerging:
        return "emerging"
    if c in {"United States","USA","US"}:
        return "us"
    # default
    return "global"


def pick_market_proxy(meta: Dict[str, Any]) -> str:
    """Pick a market proxy ETF for the stock's primary market, best-effort."""
    country = meta.get("country")
    if isinstance(country, str) and country.strip():
        if country in COUNTRY_TO_ETF:
            return COUNTRY_TO_ETF[country]

    # fallback by region
    region = _infer_region(meta)
    if region == "us":
        return "SPY"
    if region == "europe":
        return REGION_DEFAULTS["europe"]
    if region == "asia_pacific":
        return REGION_DEFAULTS["asia_pacific"]
    if region == "emerging":
        return REGION_DEFAULTS["emerging"]
    return REGION_DEFAULTS["global"]


def pick_sector_proxy(meta: Dict[str, Any]) -> Optional[str]:
    """Pick a sector proxy ETF. Defaults to US sector ETFs as global liquid proxies."""
    sector = meta.get("sector")
    if isinstance(sector, str) and sector.strip():
        return SECTOR_TO_ETF_US.get(sector) or SECTOR_TO_ETF_US.get(sector.title())
    return None


def build_candidate_factors(meta: Dict[str, Any]) -> Dict[str, str]:
    """Build a full factor dict (market + sector + macro/style) for this stock."""
    factors: Dict[str, str] = {}

    market = pick_market_proxy(meta)
    factors["market"] = market

    sector = pick_sector_proxy(meta)
    if sector:
        factors["sector"] = sector

    # Add region proxy if market proxy is a country ETF that can be noisy; helps separate country vs global beta.
    region = _infer_region(meta)
    if region in REGION_DEFAULTS and region not in {"us"}:
        factors["region"] = REGION_DEFAULTS[region]

    # Add global baseline always (helps decompose global vs local)
    factors.setdefault("global", REGION_DEFAULTS["global"])

    # Add macro/style factors
    factors.update(DEFAULT_MACRO_STYLE_FACTORS)

    return factors


# ----------------------------
# Ridge regression + scoring
# ----------------------------

def _ridge_fit(X: np.ndarray, y: np.ndarray, alpha: float = 5.0) -> Tuple[float, np.ndarray]:
    """Ridge regression with intercept (alpha is L2 strength)."""
    # Add intercept column
    X1 = np.column_stack([np.ones(len(X)), X])
    # Solve (X'X + aI) b = X'y
    XtX = X1.T @ X1
    I = np.eye(XtX.shape[0])
    I[0, 0] = 0.0  # don't penalize intercept
    A = XtX + alpha * I
    Xty = X1.T @ y
    b = np.linalg.solve(A, Xty)
    intercept = float(b[0])
    betas = b[1:]
    return intercept, betas


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0:
        return 0.0
    return max(0.0, 1.0 - ssr / sst)


def _fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not available")

    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)
    # yfinance returns multiindex columns if multiple tickers
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = data[["Close"]].rename(columns={"Close": tickers[0]})
    # Ensure columns named by ticker
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]
    return px


def _compute_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna(how="all")


def discover_driver_profile(
    ticker: str,
    start: str,
    end: str,
    *,
    headlines_by_date: Optional[Dict[str, List[str]]] = None,
    factor_tickers: Optional[Dict[str, str]] = None,
    ridge_alpha: float = 5.0,
    min_obs: int = 90,
) -> Dict[str, Any]:
    """Discover dominant drivers and return a profile dict for pipeline use."""
    meta = infer_metadata(ticker)
    factors = dict(factor_tickers) if factor_tickers else build_candidate_factors(meta)

    # Ensure market exists
    factors.setdefault("market", pick_market_proxy(meta))
    # Sector may be None; ok.

    # Fetch prices (best-effort; drop missing)
    tickers_to_fetch = [ticker] + list(dict.fromkeys(factors.values()))
    tickers_to_fetch = [t for t in tickers_to_fetch if isinstance(t, str) and t.strip()]
    tickers_to_fetch = list(dict.fromkeys(tickers_to_fetch))

    out: Dict[str, Any] = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "metadata": meta,
        "factor_tickers": factors,
        "headlines_days": len(headlines_by_date or {}),
        "available": False,
    }

    try:
        px = _fetch_prices(tickers_to_fetch, start=start, end=end)
    except Exception as e:
        out["error"] = f"price_fetch_failed: {e}"
        return out

    if ticker not in px.columns:
        out["error"] = "ticker_prices_missing"
        return out

    rets = _compute_returns(px).dropna(how="all")
    # Build factor matrix using those factors that actually exist
    factor_cols: List[str] = []
    factor_names: List[str] = []
    for name, tkr in factors.items():
        if tkr in rets.columns:
            factor_cols.append(tkr)
            factor_names.append(name)

    # Drop factors that are identical tickers (avoid duplicates)
    # We keep the first occurrence.
    seen = set()
    keep_idx = []
    for i, (nm, col) in enumerate(zip(factor_names, factor_cols)):
        if col in seen:
            continue
        seen.add(col)
        keep_idx.append(i)
    factor_cols = [factor_cols[i] for i in keep_idx]
    factor_names = [factor_names[i] for i in keep_idx]

    df = rets[[ticker] + factor_cols].dropna()
    if len(df) < min_obs or len(factor_cols) < 1:
        out["error"] = f"insufficient_data: n={len(df)}, factors={len(factor_cols)}"
        return out

    y = df[ticker].values.astype(float)
    X = df[factor_cols].values.astype(float)

    # Fit ridge
    try:
        intercept, betas = _ridge_fit(X, y, alpha=ridge_alpha)
        yhat = intercept + X @ betas
        r2 = _r2(y, yhat)
        resid = y - yhat
    except Exception as e:
        out["error"] = f"ridge_fit_failed: {e}"
        return out

    # Importance by average absolute contribution
    contrib = X * betas.reshape(1, -1)
    mean_abs_contrib = np.mean(np.abs(contrib), axis=0)
    idio_importance = float(np.mean(np.abs(resid)))

    # Normalize to shares
    total_imp = float(np.sum(mean_abs_contrib) + idio_importance)
    if total_imp <= 0:
        shares = np.zeros_like(mean_abs_contrib)
        idio_share = 0.0
    else:
        shares = mean_abs_contrib / total_imp
        idio_share = idio_importance / total_imp

    driver_rank = sorted(
        [
            {"driver": nm, "ticker": col, "beta": float(b), "importance": float(imp), "share": float(sh)}
            for nm, col, b, imp, sh in zip(factor_names, factor_cols, betas, mean_abs_contrib, shares)
        ],
        key=lambda d: d["share"],
        reverse=True
    )

    # Market-like share (market + sector + region if present)
    share_map = {d["driver"]: d["share"] for d in driver_rank}
    market_like = float(share_map.get("market", 0.0) + share_map.get("sector", 0.0) + share_map.get("region", 0.0) + share_map.get("global", 0.0) * 0.25)

    # Mode inference (outcome-driven)
    mode = "mixed"
    if idio_share >= 0.55:
        mode = "narrative_or_idiosyncratic"
    elif share_map.get("rates_long", 0.0) + share_map.get("rates_intermediate", 0.0) >= 0.25:
        mode = "rates_sensitive"
    elif share_map.get("oil", 0.0) >= 0.18:
        mode = "commodity_oil_sensitive"
    elif market_like >= 0.65:
        mode = "beta_or_sector_driven"

    # Recommendations on what to focus on (analysis checklist)
    checklist: List[str] = []
    if mode == "narrative_or_idiosyncratic":
        checklist += [
            "Prioritize catalysts and narrative: earnings/guidance, product cycle, regulatory/geopolitical headlines.",
            "Use jump-risk sizing: expect gap moves; consider event calendar / options hedges.",
            "Use technicals mainly for timing; avoid relying on mean reversion.",
        ]
    elif mode == "rates_sensitive":
        checklist += [
            "Track yields/curve and macro event days (CPI, central bank).",
            "Consider hedging with duration proxy (TLT/IEF) or reducing size in high-rate volatility regimes.",
        ]
    elif mode == "commodity_oil_sensitive":
        checklist += [
            "Track commodity regime (oil) + USD; treat earnings as cycle confirmation.",
            "Stress-test against commodity shocks and geopolitics.",
        ]
    elif mode == "beta_or_sector_driven":
        checklist += [
            "Treat as beta/sector exposure: compare vs market+sector; use hedges (index/sector ETF) for risk control.",
            "Avoid over-attribution to single headlines when the move is explained by factors.",
        ]
    else:
        checklist += [
            "Combine factor context (market/sector/macro) with idiosyncratic news; avoid single-factor conclusions.",
        ]

    # Choose final market/sector proxies (for downstream driver report)
    market_proxy = factors.get("market")
    sector_proxy = factors.get("sector")

    out.update({
        "available": True,
        "n_obs": int(len(df)),
        "ridge_alpha": float(ridge_alpha),
        "r2": float(r2),
        "intercept": float(intercept),
        "betas": {nm: float(b) for nm, b in zip(factor_names, betas)},
        "driver_rank": driver_rank,
        "idiosyncratic": {"importance": idio_importance, "share": float(idio_share)},
        "market_like_share": float(market_like),
        "mode": mode,
        "analysis_checklist": checklist,
        "selected": {
            "market_proxy": market_proxy,
            "sector_proxy": sector_proxy,
        },
        "notes": [
            "Driver importance is mean(|beta_k * factor_return_k|) relative to idiosyncratic residual.",
            "Missing factors are dropped automatically if data is unavailable.",
        ],
    })

    return out
