"""driver_discovery_v2.py

Outcome-driven driver discovery (v2): more robust + less "everything is idiosyncratic".

Fixes vs v1
-----------
- Standardizes returns before ridge fit (prevents excessive shrinkage + incomparable factor scales).
- Uses a pruned, de-correlated default factor set (reduces multicollinearity).
- Computes driver importance on standardized contributions (prevents high-vol factors like VIX dominating).
- Adds a sanity fallback: if fit quality is poor for large/liquid stocks, mark as "beta_or_sector_driven_uncertain"
  and warn that discovery is unreliable rather than confidently claiming narrative dominance.

Interfaces
----------
- infer_metadata(ticker) -> dict
- build_candidate_factors(meta) -> dict[str,str]
- discover_driver_profile(ticker, start, end, *, headlines_by_date=None, factor_tickers=None, ...) -> dict

Notes
-----
- Best-effort worldwide: missing ETFs are dropped automatically.
- Not investment advice.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

try:
    import yfinance as yf  # type: ignore
except Exception:  # pragma: no cover
    yf = None  # type: ignore


# ----------------------------
# Proxy maps (best-effort)
# ----------------------------

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
}

REGION_DEFAULTS: Dict[str, str] = {
    "developed_ex_us": "EFA",
    "emerging": "EEM",
    "global": "VT",
    "europe": "VGK",
    "asia_pacific": "VPL",
}

# Pruned default factor set (de-correlated-ish, liquid)
DEFAULT_FACTORS_PRUNED: Dict[str, str] = {
    # style
    "growth": "QQQ",
    # macro
    "rates_long": "TLT",
    "usd": "UUP",
    "oil": "USO",
    "gold": "GLD",
    "vix": "VXX",
}


# ----------------------------
# Helpers
# ----------------------------

def infer_metadata(ticker: str) -> Dict[str, Any]:
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
    c = str(meta.get("country") or "")
    if not c:
        return "global"
    europe = {"Germany","United Kingdom","France","Italy","Spain","Switzerland","Netherlands","Sweden","Norway","Denmark","Austria","Belgium","Ireland"}
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
    return "global"


def pick_market_proxy(meta: Dict[str, Any]) -> str:
    country = meta.get("country")
    if isinstance(country, str) and country.strip():
        if country in COUNTRY_TO_ETF:
            return COUNTRY_TO_ETF[country]
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
    sector = meta.get("sector")
    if isinstance(sector, str) and sector.strip():
        return SECTOR_TO_ETF_US.get(sector) or SECTOR_TO_ETF_US.get(sector.title())
    return None


def build_candidate_factors(meta: Dict[str, Any]) -> Dict[str, str]:
    """
    Returns a full factor dict, but pruned to reduce collinearity:
    - Always: market
    - If available: sector
    - If non-US: add a regional ETF (helps separate local vs global beta)
    - Always: pruned macro/style factors (growth, rates_long, usd, oil, gold, vix)
    """
    factors: Dict[str, str] = {}
    market = pick_market_proxy(meta)
    factors["market"] = market

    sector = pick_sector_proxy(meta)
    if sector:
        factors["sector"] = sector

    region = _infer_region(meta)
    if region in {"europe","asia_pacific","emerging"}:
        factors["region"] = REGION_DEFAULTS[region]

    # Add global only if market is a country/region proxy (helps decompose global vs local)
    if market not in {"VT"}:
        factors["global"] = REGION_DEFAULTS["global"]

    factors.update(DEFAULT_FACTORS_PRUNED)
    return factors


def _fetch_prices(tickers: List[str], start: str, end: str) -> pd.DataFrame:
    if yf is None:
        raise RuntimeError("yfinance not available")
    data = yf.download(tickers=tickers, start=start, end=end, auto_adjust=True, progress=False)
    if isinstance(data.columns, pd.MultiIndex):
        px = data["Close"].copy()
    else:
        px = data[["Close"]].rename(columns={"Close": tickers[0]})
    if isinstance(px.columns, pd.MultiIndex):
        px.columns = [c[0] for c in px.columns]
    return px


def _log_returns(px: pd.DataFrame) -> pd.DataFrame:
    return np.log(px).diff().dropna(how="all")


def _zscore(mat: np.ndarray) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    mu = np.nanmean(mat, axis=0)
    sd = np.nanstd(mat, axis=0)
    sd = np.where(sd < 1e-12, 1.0, sd)
    z = (mat - mu) / sd
    return z, mu, sd


def _ridge_fit_std(Xz: np.ndarray, yz: np.ndarray, alpha: float) -> np.ndarray:
    """
    Ridge on standardized data, no intercept (since standardized).
    """
    XtX = Xz.T @ Xz
    A = XtX + alpha * np.eye(XtX.shape[0])
    Xty = Xz.T @ yz
    b = np.linalg.solve(A, Xty)
    return b


def _r2(y: np.ndarray, yhat: np.ndarray) -> float:
    ssr = float(np.sum((y - yhat) ** 2))
    sst = float(np.sum((y - np.mean(y)) ** 2))
    if sst <= 0:
        return 0.0
    return max(0.0, 1.0 - ssr / sst)


def discover_driver_profile(
    ticker: str,
    start: str,
    end: str,
    *,
    headlines_by_date: Optional[Dict[str, List[str]]] = None,
    factor_tickers: Optional[Dict[str, str]] = None,
    ridge_alpha: float = 10.0,
    min_obs: int = 120,
) -> Dict[str, Any]:
    """
    Returns a driver profile dict.
    - Uses standardized ridge to estimate exposures.
    - Importance computed from standardized contributions.
    """
    meta = infer_metadata(ticker)
    factors = dict(factor_tickers) if factor_tickers else build_candidate_factors(meta)
    factors.setdefault("market", pick_market_proxy(meta))

    tickers_to_fetch = [ticker] + list(dict.fromkeys([v for v in factors.values() if v]))
    tickers_to_fetch = list(dict.fromkeys([t for t in tickers_to_fetch if isinstance(t, str) and t.strip()]))

    out: Dict[str, Any] = {
        "ticker": ticker,
        "start": start,
        "end": end,
        "metadata": meta,
        "factor_tickers": factors,
        "headlines_days": len(headlines_by_date or {}),
        "available": False,
        "warnings": [],
    }

    try:
        px = _fetch_prices(tickers_to_fetch, start=start, end=end)
    except Exception as e:
        out["error"] = f"price_fetch_failed: {e}"
        return out

    if ticker not in px.columns:
        out["error"] = "ticker_prices_missing"
        return out

    rets = _log_returns(px).dropna(how="all")

    # keep only factors with data; de-dup identical tickers
    factor_cols: List[str] = []
    factor_names: List[str] = []
    seen = set()
    for name, tkr in factors.items():
        if tkr in seen:
            continue
        if tkr in rets.columns:
            factor_cols.append(tkr)
            factor_names.append(name)
            seen.add(tkr)

    df = rets[[ticker] + factor_cols].dropna()
    if len(df) < min_obs or len(factor_cols) < 2:
        out["error"] = f"insufficient_data: n={len(df)}, factors={len(factor_cols)}"
        return out

    y = df[ticker].values.astype(float)
    X = df[factor_cols].values.astype(float)

    # Standardize
    Xz, Xmu, Xsd = _zscore(X)
    yz, ymu, ysd = _zscore(y.reshape(-1, 1))
    yz = yz.reshape(-1)

    # Fit ridge on standardized (no intercept needed)
    try:
        betas_std = _ridge_fit_std(Xz, yz, alpha=ridge_alpha)
        yhat_z = Xz @ betas_std
        r2 = _r2(yz, yhat_z)
        resid_z = yz - yhat_z
    except Exception as e:
        out["error"] = f"ridge_fit_failed: {e}"
        return out

    # Convert standardized betas to original scale (approx): beta = (ysd / Xsd) * beta_std
    beta_orig = (ysd.reshape(-1)[0] / Xsd) * betas_std

    # Importance via standardized contributions
    contrib_z = Xz * betas_std.reshape(1, -1)
    mean_abs_contrib = np.mean(np.abs(contrib_z), axis=0)
    idio_importance = float(np.mean(np.abs(resid_z)))

    total_imp = float(np.sum(mean_abs_contrib) + idio_importance)
    if total_imp <= 1e-12:
        shares = np.zeros_like(mean_abs_contrib)
        idio_share = 0.0
    else:
        shares = mean_abs_contrib / total_imp
        idio_share = idio_importance / total_imp

    driver_rank = sorted(
        [
            {
                "driver": nm,
                "ticker": col,
                "beta": float(beta_orig[i]),
                "beta_std": float(betas_std[i]),
                "share": float(shares[i]),
            }
            for i, (nm, col) in enumerate(zip(factor_names, factor_cols))
        ],
        key=lambda d: d["share"],
        reverse=True,
    )

    share_map = {d["driver"]: d["share"] for d in driver_rank}
    market_like = float(
        share_map.get("market", 0.0) +
        share_map.get("sector", 0.0) +
        share_map.get("region", 0.0) +
        share_map.get("global", 0.0)
    )

    # Mode inference (more conservative)
    mode = "mixed"
    if idio_share >= 0.65:
        mode = "narrative_or_idiosyncratic"
    elif (share_map.get("rates_long", 0.0)) >= 0.25:
        mode = "rates_sensitive"
    elif (share_map.get("oil", 0.0)) >= 0.22:
        mode = "commodity_oil_sensitive"
    elif market_like >= 0.60 and r2 >= 0.25:
        mode = "beta_or_sector_driven"

    # Sanity fallback for big/liquid names: if fit is terrible, don't over-claim narrative dominance
    mcap = meta.get("marketCap")
    try:
        mcap = float(mcap) if mcap is not None else None
    except Exception:
        mcap = None
    if (mcap is not None and mcap >= 5e10) and r2 < 0.15:
        out["warnings"].append("low_r2_for_large_cap: driver discovery may be unreliable; avoid strong narrative claims.")
        if mode == "narrative_or_idiosyncratic":
            mode = "beta_or_sector_driven_uncertain"

    # Checklist
    checklist: List[str] = []
    if mode == "narrative_or_idiosyncratic":
        checklist += [
            "Prioritize catalysts/narrative (earnings, guidance, product cycle, regulation/geopolitics).",
            "Size for gap risk; technicals are timing tools, not primary drivers.",
            "Use sentiment/topic momentum as confirmation, not as sole signal.",
        ]
    elif mode == "beta_or_sector_driven":
        checklist += [
            "Treat as market/sector exposure; compare vs market+sector; hedge where appropriate.",
            "Avoid over-attributing moves to single headlines when factors explain most variance.",
        ]
    elif mode == "beta_or_sector_driven_uncertain":
        checklist += [
            "Discovery fit quality is low; default to market/sector framing but treat driver claims as uncertain.",
            "Consider extending lookback window or reducing factor set.",
        ]
    elif mode == "rates_sensitive":
        checklist += [
            "Track yield moves/curve and macro event days; interpret returns through rate exposure.",
            "Down-weight RSI/ARIMA; macro regime dominates near-term.",
        ]
    elif mode == "commodity_oil_sensitive":
        checklist += [
            "Track oil/USD/geopolitics; interpret returns through commodity regime.",
            "Earnings often confirm the cycle rather than cause it.",
        ]
    else:
        checklist += [
            "Mixed driver profile; combine macro/sector context with idiosyncratic news.",
            "Avoid single-factor conclusions.",
        ]

    out.update({
        "available": True,
        "n_obs": int(len(df)),
        "ridge_alpha": float(ridge_alpha),
        "r2": float(r2),
        "betas": {nm: float(beta_orig[i]) for i, nm in enumerate(factor_names)},
        "betas_std": {nm: float(betas_std[i]) for i, nm in enumerate(factor_names)},
        "driver_rank": driver_rank,
        "idiosyncratic": {"share": float(idio_share)},
        "market_like_share": float(market_like),
        "mode": mode,
        "analysis_checklist": checklist,
        "selected": {
            "market_proxy": factors.get("market"),
            "sector_proxy": factors.get("sector"),
        },
        "notes": [
            "v2: standardized ridge + pruned factors; driver shares computed on standardized contributions.",
            "Missing factors are dropped automatically if data is unavailable.",
        ],
    })
    return out
