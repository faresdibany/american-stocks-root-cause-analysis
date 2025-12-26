"""stock_driver_analysis.py

Per-stock movement driver analysis for the *American stocks root cause analysis* pipeline.

Goal
----
Given a ticker and a price history window, explain *what drove the move* by:

1) Decomposing price action into components:
   - Trend (slow-moving)  -> usually fundamentals / long-horizon narrative
   - Volatility regimes   -> risk-on vs risk-off conditions
   - Jumps/shocks         -> discrete events / surprises
   - Noise                -> residual

2) Attributing jumps/abnormal returns to event candidates:
   - Structured events (earnings dates, splits/dividends if available)
   - News headlines (optional, via existing RSS fetchers)

Outputs
-------
- JSON per ticker with:
  - detected regimes
  - jump dates and magnitudes
  - trend/volatility summary
  - event attribution evidence

This module is designed to be dependency-light and degrade gracefully.
"""

from __future__ import annotations

import json
import math
import os
import time
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

from providers import (
    NotConfiguredProviderError,
    NullPriceProvider,
    PriceProvider,
    StrictnessPolicy,
)

try:
    import yfinance as yf  # optional fallback for local dev only
except Exception:  # pragma: no cover
    yf = None

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.filters.hp_filter import hpfilter
    from statsmodels.tsa.stattools import grangercausalitytests
except Exception:  # pragma: no cover
    STL = None
    hpfilter = None
    grangercausalitytests = None

try:
    from sklearn.linear_model import Ridge as SklearnRidge
except Exception:  # pragma: no cover
    SklearnRidge = None


# -----------------------------
# Data shapes (lightweight)
# -----------------------------


@dataclass
class Jump:
    date: str
    ret: float
    zscore: float
    direction: str


@dataclass
class VolRegime:
    start: str
    end: str
    label: str
    vol_annualized: float


@dataclass
class DriverReport:
    ticker: str
    start: str
    end: str
    n_obs: int
    trend_summary: Dict[str, Any]
    volatility_regimes: List[VolRegime]
    jumps: List[Jump]
    event_attribution: Dict[str, Any]
    change_points: Dict[str, Any]
    factor_model: Dict[str, Any]  # Multi-factor attribution
    per_jump_drivers: List[Dict[str, Any]]  # Per-jump driver analysis
    granger_causality: Dict[str, Any]  # ENHANCEMENT 1: Temporal causation tests
    causal_dag: Dict[str, Any]  # ENHANCEMENT 2: Causal network structure
    rolling_betas: Dict[str, Any]  # ENHANCEMENT 3: Time-varying factor exposures
    notes: List[str]


# -----------------------------
# Core computations
# -----------------------------


def _require_yfinance() -> None:
    if yf is None:
        raise RuntimeError(
            "yfinance is required for the fallback price loader, but it is not installed. "
            "In institutional deployments you should configure a PriceProvider."
        )


def _normalize_price_series(px: Any, *, ticker: str) -> pd.Series:
    """Normalize various provider/yfinance shapes into a clean 1D float Series."""
    if px is None:
        raise RuntimeError(f"No price data returned for {ticker}")

    # Provider might already return a Series.
    if isinstance(px, pd.Series):
        s = px.copy()
    elif isinstance(px, pd.DataFrame):
        # Prefer common column names
        for col in ("Adj Close", "adj_close", "adjusted_close", "Close", "close"):
            if col in px.columns:
                s = px[col].copy()
                break
        else:
            # best-effort: take first numeric column
            num = px.select_dtypes(include=["number"])
            if num.shape[1] == 0:
                raise RuntimeError(f"Price data for {ticker} had no numeric columns")
            s = num.iloc[:, 0].copy()
    else:
        # Try array-like
        try:
            s = pd.Series(px)
        except Exception as e:  # pragma: no cover
            raise RuntimeError(f"Unrecognized price data type for {ticker}: {type(px)}") from e

    s = s.dropna().astype(float)
    s.name = "price"
    if len(s) == 0:
        raise RuntimeError(f"No usable price data after cleaning for {ticker}")
    return s


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
    *,
    price_provider: Optional[PriceProvider] = None,
    strictness: Optional[StrictnessPolicy] = None,
    allow_yfinance_fallback: bool = False,
) -> pd.Series:
    """Fetch adjusted close prices.

    Institutional default: require an injected PriceProvider.
    Optional dev fallback: allow_yfinance_fallback=True.
    """
    strictness = strictness or StrictnessPolicy()

    if price_provider is not None:
        try:
            px = price_provider.get_adjusted_close(
                ticker=ticker,
                start=start,
                end=end,
                interval=interval,
            )
            return _normalize_price_series(px, ticker=ticker)
        except Exception:
            if strictness.fail_on_missing_prices:
                raise
            # fall through on non-strict mode

    if allow_yfinance_fallback:
        _require_yfinance()
        df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
        if df is None or len(df) == 0:
            if strictness.fail_on_missing_prices:
                raise RuntimeError(f"No data returned for {ticker}")
            return pd.Series(dtype=float, name="price")
        return _normalize_price_series(df, ticker=ticker)

    # No provider and fallback not allowed
    if strictness.fail_on_missing_prices:
        raise NotConfiguredProviderError(
            "Institutional pipeline requires a PriceProvider. "
            "Pass price_provider=... or set allow_yfinance_fallback=True for local/dev only."
        )
    return pd.Series(dtype=float, name="price")


def compute_returns(px: pd.Series) -> pd.Series:
    """Compute log returns (more suitable for factor models)."""
    px = px.dropna()
    rets = np.log(px).diff()
    rets = rets.replace([np.inf, -np.inf], np.nan).dropna()
    rets.name = getattr(px, "name", "ret")
    return rets


def align_series(series: Dict[str, pd.Series]) -> pd.DataFrame:
    """Inner-join multiple return series on date index."""
    df = pd.concat(series, axis=1).dropna(how="any")
    # Flatten columns if needed
    df.columns = [c if isinstance(c, str) else str(c) for c in df.columns]
    return df


def decompose_trend(
    px: pd.Series,
    method: str = "hp",
    hp_lambda: float = 129600.0,
    stl_period: int = 5,
) -> Tuple[pd.Series, pd.Series]:
    """Return (trend, residual) on log(price).

    Notes:
      - For daily equity data, the HP filter λ is often set high; 129600 is sometimes used
        for monthly data, but here we use it as a smooth long-horizon trend proxy.
      - STL requires statsmodels >= 0.14 and a period; we keep period small as a
        generic default.
    """
    logp = np.log(px)

    if method == "stl" and STL is not None:
        stl = STL(logp, period=stl_period, robust=True)
        res = stl.fit()
        trend = res.trend
        resid = res.resid
        return trend.rename("trend"), resid.rename("resid")

    if hpfilter is not None:
        cycle, trend = hpfilter(logp, lamb=hp_lambda)
        resid = cycle
        return trend.rename("trend"), resid.rename("resid")

    # Fallback: simple EWMA trend
    trend = logp.ewm(span=60, adjust=False).mean()
    resid = logp - trend
    return trend.rename("trend"), resid.rename("resid")


def detect_jumps(
    rets: pd.Series,
    z_thresh: float = 3.0,
    min_abs_ret: float = 0.03,
) -> List[Jump]:
    """Detect discrete shocks using robust z-score on returns."""
    if len(rets) < 30:
        return []

    med = float(rets.median())
    mad = float(np.median(np.abs(rets - med)))
    if mad == 0:
        mad = float(rets.std(ddof=0)) or 1e-9

    # 0.6745 makes MAD comparable to std for normal
    z = (rets - med) / (1.4826 * mad)

    jumps = []
    for dt, r in rets.items():
        zr = float(z.loc[dt])
        rr = float(r)
        if abs(zr) >= z_thresh and abs(rr) >= min_abs_ret:
            jumps.append(
                Jump(
                    date=pd.Timestamp(dt).date().isoformat(),
                    ret=rr,
                    zscore=zr,
                    direction="up" if rr > 0 else "down",
                )
            )

    # Most recent first
    jumps.sort(key=lambda j: j.date, reverse=True)
    return jumps


def detect_volatility_regimes(
    rets: pd.Series,
    window: int = 20,
    q_low: float = 0.33,
    q_high: float = 0.66,
) -> List[VolRegime]:
    """Simple volatility regime segmentation using rolling vol quantiles.

    Produces contiguous segments labeled low/medium/high.
    """
    if len(rets) < window * 2:
        return []

    vol = rets.rolling(window).std(ddof=0) * math.sqrt(252)
    vol = vol.dropna()

    lo = float(vol.quantile(q_low))
    hi = float(vol.quantile(q_high))

    def _label(v: float) -> str:
        if v <= lo:
            return "low"
        if v >= hi:
            return "high"
        return "medium"

    labels = vol.apply(lambda x: _label(float(x)))

    # Collapse into segments
    regimes: List[VolRegime] = []
    cur_label: Optional[str] = None
    cur_start: Optional[pd.Timestamp] = None
    cur_vals: List[float] = []

    for dt, lab in labels.items():
        if cur_label is None:
            cur_label = str(lab)
            cur_start = pd.Timestamp(dt)
            cur_vals = [float(vol.loc[dt])]
            continue

        if str(lab) == cur_label:
            cur_vals.append(float(vol.loc[dt]))
            continue

        # close previous
        regimes.append(
            VolRegime(
                start=cur_start.date().isoformat(),
                end=pd.Timestamp(dt).date().isoformat(),
                label=cur_label,
                vol_annualized=float(np.mean(cur_vals)) if cur_vals else float("nan"),
            )
        )

        cur_label = str(lab)
        cur_start = pd.Timestamp(dt)
        cur_vals = [float(vol.loc[dt])]

    # final segment
    if cur_label is not None and cur_start is not None:
        regimes.append(
            VolRegime(
                start=cur_start.date().isoformat(),
                end=pd.Timestamp(labels.index[-1]).date().isoformat(),
                label=cur_label,
                vol_annualized=float(np.mean(cur_vals)) if cur_vals else float("nan"),
            )
        )

    # Only keep a few most recent regimes for readability
    return regimes[-12:]


def detect_changepoints_pelt_like(
    rets: pd.Series,
    window: int = 20,
    z_thresh: float = 2.5,
    min_gap: int = 10,
) -> Dict[str, Any]:
    """Heuristic change-point detection for regime shifts.

    This is a pragmatic, dependency-light alternative to full PELT/Bayesian CPD.
    We flag dates where rolling volatility's first difference is unusually large.

    Returns:
      {"method": "vol_diff_z", "points": [{"date":..., "score":...}], "n": int}
    """
    if len(rets) < window * 3:
        return {"method": "vol_diff_z", "points": [], "n": 0}

    vol = rets.rolling(window).std(ddof=0).dropna()
    dvol = vol.diff().dropna()
    if len(dvol) < 20:
        return {"method": "vol_diff_z", "points": [], "n": 0}

    med = float(dvol.median())
    mad = float(np.median(np.abs(dvol - med)))
    if mad == 0:
        mad = float(dvol.std(ddof=0)) or 1e-9
    z = (dvol - med) / (1.4826 * mad)

    # candidate points
    cand = [(pd.Timestamp(dt), float(abs(z.loc[dt]))) for dt in z.index if abs(float(z.loc[dt])) >= z_thresh]
    cand.sort(key=lambda x: x[0])

    # enforce min_gap spacing, keep highest score per neighborhood
    points: List[Dict[str, Any]] = []
    last_dt: Optional[pd.Timestamp] = None
    pending: Optional[Tuple[pd.Timestamp, float]] = None
    for dt, score in cand:
        if last_dt is None:
            pending = (dt, score)
            last_dt = dt
            continue

        if (dt - last_dt).days <= min_gap:
            # same neighborhood
            if pending is None or score > pending[1]:
                pending = (dt, score)
            last_dt = dt
            continue

        # flush pending
        if pending is not None:
            points.append({"date": pending[0].date().isoformat(), "score": pending[1]})
        pending = (dt, score)
        last_dt = dt

    if pending is not None:
        points.append({"date": pending[0].date().isoformat(), "score": pending[1]})

    # Keep only top few by score for readability
    points_sorted = sorted(points, key=lambda p: float(p.get("score", 0.0)), reverse=True)
    points_sorted = points_sorted[:10]

    return {"method": "vol_diff_z", "points": points_sorted, "n": len(points_sorted)}


def estimate_beta(x: pd.Series, y: pd.Series) -> float:
    """Estimate beta of x vs y using OLS slope with intercept."""
    df = pd.concat([x, y], axis=1).dropna()
    if len(df) < 20:
        return float("nan")
    xr = df.iloc[:, 0].astype(float)
    yr = df.iloc[:, 1].astype(float)
    vx = float(np.var(yr, ddof=0))
    if vx == 0:
        return float("nan")
    cov = float(np.mean((xr - xr.mean()) * (yr - yr.mean())))
    return cov / vx


# -----------------------------
# Multi-factor attribution (NEW)
# -----------------------------

DEFAULT_FACTOR_TICKERS: Dict[str, str] = {
    "market": "SPY",
    "rates": "IEF",   # 7-10y Treasuries
    "usd": "UUP",     # USD index ETF
    "oil": "USO",     # oil proxy ETF
    "vix": "VXX",     # volatility proxy
}


def fetch_factor_returns(
    start: str,
    end: str,
    tickers: Dict[str, str],
) -> Dict[str, pd.Series]:
    """Fetch returns for a dict of factors -> ticker."""
    out: Dict[str, pd.Series] = {}
    for name, tkr in tickers.items():
        try:
            px = fetch_prices(tkr, start=start, end=end)
            out[name] = compute_returns(px)
        except Exception:
            # Skip factors that can't be fetched
            continue
    return out


def ridge_regression_betas(X: np.ndarray, y: np.ndarray, lam: float = 1e-4) -> Tuple[float, np.ndarray]:
    """Ridge regression with intercept. Returns (alpha, betas)."""
    # Add intercept
    X1 = np.column_stack([np.ones(len(X)), X])
    # Closed-form ridge: (X'X + lam*I)^-1 X'y
    I = np.eye(X1.shape[1])
    I[0, 0] = 0.0  # don't penalize intercept
    A = X1.T @ X1 + lam * I
    b = X1.T @ y
    coef = np.linalg.solve(A, b)
    alpha = float(coef[0])
    betas = coef[1:]
    return alpha, betas


def factor_model_fit(
    stock_rets: pd.Series,
    factor_rets: Dict[str, pd.Series],
    ridge_lambda: float = 1e-4,
) -> Dict[str, Any]:
    """Fit a linear factor model: stock ~ factors."""
    if not factor_rets:
        return {"available": False, "reason": "no_factors"}

    df = align_series({"stock": stock_rets, **factor_rets})
    if len(df) < 30:
        return {"available": False, "reason": "insufficient_overlap", "n_obs": int(len(df))}

    y = df["stock"].values.astype(float)
    factor_names = [c for c in df.columns if c != "stock"]
    X = df[factor_names].values.astype(float)

    alpha, betas = ridge_regression_betas(X, y, lam=ridge_lambda)
    # In-sample R^2
    yhat = alpha + X @ betas
    ss_res = float(np.sum((y - yhat) ** 2))
    ss_tot = float(np.sum((y - np.mean(y)) ** 2))
    r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else float("nan")

    return {
        "available": True,
        "n_obs": int(len(df)),
        "alpha": float(alpha),
        "betas": {name: float(b) for name, b in zip(factor_names, betas)},
        "r2": float(r2),
        "factors": factor_names,
    }


def factor_contributions_on_date(
    date: str,
    factor_rets: Dict[str, pd.Series],
    betas: Dict[str, float],
) -> Dict[str, float]:
    """Compute beta_k * r_k on a given date (best-effort)."""
    d = pd.Timestamp(date)
    out: Dict[str, float] = {}
    for k, s in factor_rets.items():
        if k not in betas:
            continue
        # Normalize index to date only
        s2 = s.copy()
        s2.index = pd.to_datetime(s2.index).normalize()
        dn = d.normalize()
        if dn in s2.index:
            rk = float(s2.loc[dn])
            out[k] = float(betas[k] * rk)
    return out


def extract_headline_keywords(
    headlines: List[str],
    top_k: int = 6,
) -> List[str]:
    """Tiny TF-IDF-like keyword extractor (no sklearn dependency)."""
    import re
    stop = {
        "the", "a", "an", "and", "or", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "is", "are", "was", "were", "be", "as", "its", "it", "this", "that",
        "after", "before", "up", "down", "will", "says", "said", "report", "reports",
        "update", "new", "shares", "stock", "company", "inc", "corp", "co"
    }
    docs = []
    for h in headlines:
        toks = [t.lower() for t in re.findall(r"[a-zA-Z0-9]+", h)]
        toks = [t for t in toks if len(t) >= 3 and t not in stop]
        docs.append(toks)

    if not docs:
        return []

    # document frequency
    df: Dict[str, int] = {}
    for toks in docs:
        for t in set(toks):
            df[t] = df.get(t, 0) + 1

    N = len(docs)
    tf: Dict[str, float] = {}
    for toks in docs:
        for t in toks:
            tf[t] = tf.get(t, 0.0) + 1.0

    # TF-IDF-ish
    scores: Dict[str, float] = {}
    for t, tfv in tf.items():
        idf = np.log((N + 1.0) / (df.get(t, 1) + 0.5))
        scores[t] = float(tfv * idf)

    top = sorted(scores.items(), key=lambda kv: kv[1], reverse=True)[:top_k]
    return [t for t, _ in top]


def classify_driver_for_jump(
    jump: Jump,
    near_earnings: bool,
    contrib: Dict[str, float],
    idio: float,
    total_ret: float,
    headline_keywords: List[str],
) -> Dict[str, Any]:
    """Heuristic driver classifier for a single jump day."""
    # shares by absolute contribution
    parts = {**contrib, "idiosyncratic": idio}
    denom = sum(abs(v) for v in parts.values()) or 1.0
    shares = {k: abs(v) / denom for k, v in parts.items()}

    # primary component
    primary = max(shares.items(), key=lambda kv: kv[1])[0]
    primary_share = float(shares[primary])

    # Label logic
    label = "unknown"
    if near_earnings and shares.get("idiosyncratic", 0.0) >= 0.45:
        label = "earnings / company-specific"
    elif primary == "sector":
        label = "sector / industry"
    elif primary == "market":
        label = "broad market"
    elif primary in ("rates", "vix"):
        label = "macro (rates/vol)"
    elif primary in ("usd",):
        label = "macro (fx)"
    elif primary in ("oil",):
        label = "macro (commodities)"
    elif primary == "idiosyncratic":
        label = "company-specific / narrative"

    # confidence proxy
    confidence = min(0.95, max(0.25, primary_share))

    return {
        "date": jump.date,
        "ret": float(total_ret),
        "jump_z": float(jump.zscore),
        "near_earnings": bool(near_earnings),
        "contributions": {k: float(v) for k, v in contrib.items()},
        "idiosyncratic": float(idio),
        "abs_share": {k: float(v) for k, v in shares.items()},
        "primary_component": primary,
        "primary_share": float(primary_share),
        "driver_label": label,
        "confidence_proxy": float(confidence),
        "headline_keywords": headline_keywords[:8],
    }


def _near_date(date: str, candidates: List[str], days: int = 2) -> bool:
    """Check if date is near any candidate dates within N days."""
    try:
        d = pd.Timestamp(date)
    except Exception:
        return False
    for c in candidates:
        try:
            ct = pd.Timestamp(c)
        except Exception:
            continue
        if abs((d - ct).days) <= days:
            return True
    return False


def fetch_benchmark_returns(
    start: str,
    end: str,
    benchmark: str = "SPY",
) -> pd.Series:
    px = fetch_prices(benchmark, start=start, end=end)
    return compute_returns(px)


def abnormal_return_attribution(
    ticker_rets: pd.Series,
    start: str,
    end: str,
    jumps: List[Jump],
    benchmark: str = "SPY",
    window_days: int = 2,
) -> Dict[str, Any]:
    """Event-study style attribution around jump dates.

    Model: r_i,t = alpha + beta * r_m,t + eps_t
    Abnormal return on date d: AR_d = r_i,d - beta*r_m,d (alpha ignored for simplicity)

    Returns a dict with:
      - beta estimate
      - per-jump abnormal returns and cumulative abnormal return in +/- window
    """
    try:
        mkt = fetch_benchmark_returns(start=start, end=end, benchmark=benchmark)
    except Exception:
        return {"available": False, "reason": "benchmark_data_unavailable"}

    # align on same index - ensure ticker_rets has a name
    ticker_rets_named = ticker_rets.copy()
    ticker_rets_named.name = "ret"
    aligned = pd.concat([ticker_rets_named, mkt.rename("mkt")], axis=1).dropna()
    if len(aligned) < 40:
        return {"available": False, "reason": "insufficient_overlap"}

    beta = estimate_beta(aligned["ret"], aligned["mkt"])
    if not np.isfinite(beta):
        return {"available": False, "reason": "beta_estimation_failed"}

    out_jumps = []
    for j in jumps:
        dt = pd.Timestamp(j.date)

        # window in trading days (approx by calendar indexing)
        w = aligned.loc[(aligned.index >= dt - pd.Timedelta(days=window_days)) & (aligned.index <= dt + pd.Timedelta(days=window_days))]
        if len(w) == 0 or dt not in aligned.index:
            continue

        ar_day = float(aligned.loc[dt, "ret"] - beta * aligned.loc[dt, "mkt"])
        ar_series = w["ret"] - beta * w["mkt"]

        out_jumps.append(
            {
                "date": j.date,
                "ret": j.ret,
                "zscore": j.zscore,
                "direction": j.direction,
                "abnormal_return_day": ar_day,
                "car_window": float(ar_series.sum()),
                "window_days": window_days,
            }
        )

    # Simple summary fractions
    abs_ar = [abs(float(x["abnormal_return_day"])) for x in out_jumps]
    abs_r = [abs(float(x["ret"])) for x in out_jumps]
    frac = float(np.mean([a / r for a, r in zip(abs_ar, abs_r) if r > 0])) if out_jumps else float("nan")

    return {
        "available": True,
        "benchmark": benchmark,
        "beta": float(beta),
        "window_days": window_days,
        "jumps": out_jumps,
        "summary": {
            "n_attributed_jumps": len(out_jumps),
            "mean_abs_abnormal_fraction": frac,
        },
    }


def earnings_calendar(ticker: str) -> Dict[str, Any]:
    """Best-effort fetch of earnings dates using yfinance.

    yfinance fields can vary by ticker; returns a dict with any available info.
    """
    if yf is None:
        return {"available": False}

    try:
        tk = yf.Ticker(ticker)
        out: Dict[str, Any] = {"available": True}

        # Newer yfinance versions: get_earnings_dates
        dates = None
        if hasattr(tk, "get_earnings_dates"):
            try:
                dates = tk.get_earnings_dates(limit=12)
            except Exception:
                dates = None

        if isinstance(dates, pd.DataFrame) and len(dates) > 0:
            out["earnings_dates"] = [
                pd.Timestamp(i).date().isoformat() for i in dates.index
            ]
        else:
            out["earnings_dates"] = []

        # Other fields might exist
        for key in ("calendar", "earnings", "splits", "dividends"):
            try:
                val = getattr(tk, key)
                out[key] = str(val)[:1000]
            except Exception:
                pass

        return out

    except Exception:
        return {"available": False}


# -----------------------------
# ENHANCEMENT 1: Granger Causality Testing
# -----------------------------


def estimate_granger_causality(
    ticker_rets: pd.Series,
    factor_rets: Dict[str, pd.Series],
    max_lag: int = 5,
    alpha: float = 0.05,
) -> Dict[str, Any]:
    """Test if factors Granger-cause stock returns.
    
    Granger causality tests whether past values of factors help predict current stock returns
    beyond what the stock's own past returns can predict. This captures *temporal* causation,
    not just correlation.
    
    Parameters
    ----------
    ticker_rets : pd.Series
        Stock return series
    factor_rets : Dict[str, pd.Series]
        Dictionary of factor name -> return series
    max_lag : int, default 5
        Maximum number of lags to test (typically 1-5 for daily data)
    alpha : float, default 0.05
        Significance level for hypothesis tests
        
    Returns
    -------
    Dict with:
        - available: bool
        - results: Dict[factor_name, Dict[str, float]] containing:
            * p_value: minimum p-value across lags (lower = stronger causality)
            * best_lag: lag with strongest causality
            * causal: bool (True if p_value < alpha)
        - summary: Dict with counts of causal factors
        
    Notes
    -----
    Requires statsmodels. Uses F-test for Granger causality.
    Null hypothesis: Factor does NOT Granger-cause stock returns.
    Low p-value -> reject null -> factor predicts stock returns.
    
    Example
    -------
    >>> gc_results = estimate_granger_causality(nvda_rets, {"VIX": vix_rets}, max_lag=3)
    >>> if gc_results["results"]["VIX"]["causal"]:
    ...     print(f"VIX predicts NVDA with lag {gc_results['results']['VIX']['best_lag']}")
    """
    if grangercausalitytests is None:
        return {
            "available": False,
            "reason": "statsmodels.tsa.stattools.grangercausalitytests not available",
        }
    
    if not factor_rets or len(ticker_rets) < max_lag * 10:
        return {
            "available": False,
            "reason": "insufficient_data",
            "n_obs": len(ticker_rets),
        }
    
    results = {}
    for factor_name, factor_series in factor_rets.items():
        try:
            # Align data
            df = pd.concat([ticker_rets.rename("stock"), factor_series.rename("factor")], axis=1).dropna()
            if len(df) < max_lag * 10:
                results[factor_name] = {
                    "p_value": 1.0,
                    "best_lag": None,
                    "causal": False,
                    "error": "insufficient_overlap",
                }
                continue
            
            # Run Granger causality test
            # Data format: [dependent, independent] = [stock, factor]
            # Tests if factor Granger-causes stock
            gc_res = grangercausalitytests(df[["stock", "factor"]], max_lag, verbose=False)
            
            # Extract p-values for each lag (using F-test)
            p_values = {}
            for lag in range(1, max_lag + 1):
                # gc_res[lag] is tuple: (test_results_dict, ...)
                # test_results_dict has keys: 'ssr_ftest', 'ssr_chi2test', 'lrtest', 'params_ftest'
                # Each value is tuple: (statistic, p_value, df)
                p_values[lag] = gc_res[lag][0]["ssr_ftest"][1]  # F-test p-value
            
            # Find best (lowest p-value) lag
            best_lag = min(p_values.items(), key=lambda x: x[1])
            
            results[factor_name] = {
                "p_value": float(best_lag[1]),
                "best_lag": int(best_lag[0]),
                "causal": bool(best_lag[1] < alpha),
                "all_lags_p": {int(k): float(v) for k, v in p_values.items()},
            }
            
        except Exception as e:
            results[factor_name] = {
                "p_value": 1.0,
                "best_lag": None,
                "causal": False,
                "error": str(e)[:200],
            }
    
    # Summary statistics
    causal_factors = [k for k, v in results.items() if v.get("causal", False)]
    
    return {
        "available": True,
        "max_lag": max_lag,
        "alpha": alpha,
        "n_factors_tested": len(factor_rets),
        "results": results,
        "summary": {
            "n_causal_factors": len(causal_factors),
            "causal_factors": causal_factors,
            "strongest_predictor": min(results.items(), key=lambda x: x[1].get("p_value", 1.0))[0] if results else None,
        },
    }


# -----------------------------
# ENHANCEMENT 2: Causal DAG Discovery
# -----------------------------


def build_causal_dag(
    ticker_rets: pd.Series,
    factor_rets: Dict[str, pd.Series],
    alpha: float = 0.05,
    max_cond_vars: int = 3,
) -> Dict[str, Any]:
    """Discover causal structure using PC algorithm (constraint-based causal inference).
    
    The PC algorithm learns a Directed Acyclic Graph (DAG) representing causal relationships
    between the stock and factors. It can reveal:
    - Direct causation: Factor A → Stock
    - Indirect causation: Factor A → Factor B → Stock
    - Common causes: Factor C → Factor A, Factor C → Stock
    - Confounders and mediators
    
    Parameters
    ----------
    ticker_rets : pd.Series
        Stock return series
    factor_rets : Dict[str, pd.Series]
        Dictionary of factor name -> return series
    alpha : float, default 0.05
        Significance level for conditional independence tests
        (lower = more conservative, fewer edges)
    max_cond_vars : int, default 3
        Maximum number of conditioning variables in CI tests
        (higher = more thorough but slower)
        
    Returns
    -------
    Dict with:
        - available: bool
        - nodes: List[str] - all variables in the graph
        - edges: List[Tuple[str, str]] - directed edges (from, to)
        - adjacency_matrix: 2D array - graph structure
        - stock_parents: List[str] - direct causes of stock returns
        - stock_children: List[str] - variables caused by stock returns
        - interpretation: str - human-readable summary
        
    Notes
    -----
    Requires causallearn package: pip install causal-learn
    Uses conditional independence tests (Fisher's Z for continuous data).
    
    Interpretation:
    - A → B means A causes B (A comes before B in causal ordering)
    - If Stock has parents [VIX, Market], these factors directly influence stock
    - If Factor A → Factor B → Stock, Factor A has indirect influence through B
    
    Example
    -------
    >>> dag = build_causal_dag(nvda_rets, {"Market": spy_rets, "VIX": vix_rets})
    >>> print(dag["stock_parents"])  # ['Market', 'VIX']
    >>> print(dag["edges"])  # [('Market', 'NVDA'), ('VIX', 'Market')]
    """
    try:
        from causallearn.search.ConstraintBased.PC import pc
        from causallearn.utils.cit import fisherz
    except ImportError:
        return {
            "available": False,
            "reason": "causal-learn package not installed (pip install causal-learn)",
        }
    
    if not factor_rets or len(ticker_rets) < 100:
        return {
            "available": False,
            "reason": "insufficient_data",
            "n_obs": len(ticker_rets),
        }
    
    try:
        # Align all data
        data_dict = {"stock": ticker_rets, **factor_rets}
        df = pd.concat(data_dict, axis=1).dropna()
        
        if len(df) < 100:
            return {
                "available": False,
                "reason": "insufficient_overlap_after_alignment",
                "n_obs": len(df),
            }
        
        # Run PC algorithm
        # Returns CausalGraph object with:
        # - G.graph: adjacency matrix (0=no edge, 1=edge with →, -1=edge with ←)
        # - G.nodes: node names
        cg = pc(
            df.values,
            alpha=alpha,
            indep_test=fisherz,
            stable=True,  # More stable variable ordering
            uc_rule=0,  # Conservative orientation rules
            uc_priority=2,
            mvpc=False,
            correction_name="MV_Crtn_Fisher_Z",
            background_knowledge=None,
            verbose=False,
            show_progress=False,
        )
        
        # Extract graph structure
        nodes = list(df.columns)
        graph_matrix = cg.G.graph  # n x n array
        
        # Extract directed edges
        edges = []
        for i in range(len(nodes)):
            for j in range(len(nodes)):
                if graph_matrix[i, j] == 1 and graph_matrix[j, i] == -1:
                    # i → j
                    edges.append((nodes[i], nodes[j]))
                elif graph_matrix[i, j] == -1 and graph_matrix[j, i] == 1:
                    # j → i
                    edges.append((nodes[j], nodes[i]))
                elif graph_matrix[i, j] == 1 and graph_matrix[j, i] == 1:
                    # Undirected edge (couldn't orient)
                    edges.append((nodes[i], nodes[j]))
        
        # Find stock's parents and children
        stock_idx = nodes.index("stock")
        stock_parents = []
        stock_children = []
        
        for i, node in enumerate(nodes):
            if i == stock_idx:
                continue
            
            # Check if node → stock
            if graph_matrix[i, stock_idx] == 1 and graph_matrix[stock_idx, i] == -1:
                stock_parents.append(node)
            # Check if stock → node
            elif graph_matrix[stock_idx, i] == 1 and graph_matrix[i, stock_idx] == -1:
                stock_children.append(node)
        
        # Generate interpretation
        if stock_parents:
            interpretation = f"Stock returns are directly influenced by: {', '.join(stock_parents)}. "
        else:
            interpretation = "No direct causal influences detected on stock returns. "
        
        if stock_children:
            interpretation += f"Stock returns directly influence: {', '.join(stock_children)}. "
        
        # Detect indirect paths
        indirect_influences = []
        for parent in stock_parents:
            parent_idx = nodes.index(parent)
            for i, node in enumerate(nodes):
                if i != parent_idx and i != stock_idx and node not in stock_parents:
                    # Check if node → parent
                    if graph_matrix[i, parent_idx] == 1 and graph_matrix[parent_idx, i] == -1:
                        indirect_influences.append(f"{node} → {parent} → stock")
        
        if indirect_influences:
            interpretation += f"Indirect influences: {'; '.join(indirect_influences[:3])}."
        
        return {
            "available": True,
            "algorithm": "PC (constraint-based)",
            "alpha": alpha,
            "n_nodes": len(nodes),
            "n_edges": len(edges),
            "nodes": nodes,
            "edges": edges,
            "adjacency_matrix": graph_matrix.tolist(),
            "stock_parents": stock_parents,
            "stock_children": stock_children,
            "interpretation": interpretation,
        }
        
    except Exception as e:
        return {
            "available": False,
            "reason": f"causal_dag_error: {str(e)[:300]}",
        }


# -----------------------------
# ENHANCEMENT 3: Rolling Factor Betas (Time-Varying Exposures)
# -----------------------------


def rolling_factor_betas(
    ticker_rets: pd.Series,
    factor_rets: Dict[str, pd.Series],
    window: int = 60,
    ridge_alpha: float = 1.0,
    min_periods: int = 30,
) -> Dict[str, Any]:
    """Compute time-varying factor exposures using rolling window regression.
    
    Standard factor models assume constant betas, but in reality:
    - Factor exposures change with business cycles
    - Companies shift strategies (e.g., more/less leverage → different rate sensitivity)
    - Market regimes affect factor loadings
    
    This function estimates how factor betas evolve over time, revealing:
    - Regime-dependent relationships
    - Structural breaks in factor exposure
    - Time periods of high/low factor sensitivity
    
    Parameters
    ----------
    ticker_rets : pd.Series
        Stock return series
    factor_rets : Dict[str, pd.Series]
        Dictionary of factor name -> return series
    window : int, default 60
        Rolling window size in days (60 ≈ 3 months for daily data)
    ridge_alpha : float, default 1.0
        Ridge regression penalty (reduces overfitting with multiple factors)
    min_periods : int, default 30
        Minimum observations required for estimation
        
    Returns
    -------
    Dict with:
        - available: bool
        - window_days: int
        - factor_names: List[str]
        - time_series: List[Dict] - each dict has:
            * date: str
            * betas: Dict[factor_name, float]
            * r2: float (in-sample fit)
        - summary_stats: Dict[factor_name, Dict] containing:
            * mean_beta: float
            * std_beta: float
            * min_beta: float
            * max_beta: float
            * cv: float (coefficient of variation = std/mean)
        - regime_changes: List[Dict] - detected significant beta shifts
        
    Notes
    -----
    Uses Ridge regression to handle multicollinearity when multiple factors are correlated.
    Higher window = smoother estimates but slower to detect changes.
    Lower window = more responsive but noisier.
    
    Interpretation:
    - High CV (std/mean) = factor exposure varies a lot over time
    - Regime changes = dates when betas shift significantly (> 1.5 std)
    - Increasing VIX beta over time = stock becoming more sensitive to volatility
    
    Example
    -------
    >>> rolling = rolling_factor_betas(nvda_rets, {"Market": spy_rets, "VIX": vix_rets}, window=60)
    >>> market_betas = [d["betas"]["Market"] for d in rolling["time_series"]]
    >>> plt.plot(market_betas)  # Visualize how market beta changes over time
    """
    if not factor_rets or len(ticker_rets) < window + min_periods:
        return {
            "available": False,
            "reason": "insufficient_data",
            "n_obs": len(ticker_rets),
            "required": window + min_periods,
        }
    
    # Align data
    data_dict = {"stock": ticker_rets, **factor_rets}
    df = pd.concat(data_dict, axis=1).dropna()
    
    if len(df) < window + min_periods:
        return {
            "available": False,
            "reason": "insufficient_overlap_after_alignment",
            "n_obs": len(df),
        }
    
    factor_names = [col for col in df.columns if col != "stock"]
    
    # Rolling regression
    betas_ts = []
    
    try:
        if SklearnRidge is not None:
            # Use sklearn Ridge if available (faster)
            for i in range(window, len(df)):
                window_data = df.iloc[i - window : i]
                
                if len(window_data) < min_periods:
                    continue
                
                y = window_data["stock"].values
                X = window_data[factor_names].values
                
                # Ridge regression
                model = SklearnRidge(alpha=ridge_alpha)
                model.fit(X, y)
                
                # In-sample R²
                y_pred = model.predict(X)
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                
                betas_ts.append({
                    "date": df.index[i].date().isoformat(),
                    "betas": {name: float(beta) for name, beta in zip(factor_names, model.coef_)},
                    "r2": float(r2),
                })
        else:
            # Fallback to numpy implementation
            for i in range(window, len(df)):
                window_data = df.iloc[i - window : i]
                
                if len(window_data) < min_periods:
                    continue
                
                y = window_data["stock"].values
                X = window_data[factor_names].values
                
                # Ridge regression (closed form)
                _, betas = ridge_regression_betas(X, y, lam=ridge_alpha)
                
                # In-sample R²
                y_pred = X @ betas
                ss_res = np.sum((y - y_pred) ** 2)
                ss_tot = np.sum((y - np.mean(y)) ** 2)
                r2 = 1.0 - ss_res / ss_tot if ss_tot > 0 else 0.0
                
                betas_ts.append({
                    "date": df.index[i].date().isoformat(),
                    "betas": {name: float(beta) for name, beta in zip(factor_names, betas)},
                    "r2": float(r2),
                })
        
        # Summary statistics for each factor
        summary_stats = {}
        for factor_name in factor_names:
            beta_series = np.array([d["betas"][factor_name] for d in betas_ts])
            
            mean_beta = float(np.mean(beta_series))
            std_beta = float(np.std(beta_series, ddof=1))
            min_beta = float(np.min(beta_series))
            max_beta = float(np.max(beta_series))
            cv = float(abs(std_beta / mean_beta)) if mean_beta != 0 else float("inf")
            
            summary_stats[factor_name] = {
                "mean_beta": mean_beta,
                "std_beta": std_beta,
                "min_beta": min_beta,
                "max_beta": max_beta,
                "coefficient_of_variation": cv,
            }
        
        # Detect regime changes (significant beta shifts > 1.5 std)
        regime_changes = []
        for factor_name in factor_names:
            beta_series = np.array([d["betas"][factor_name] for d in betas_ts])
            dates = [d["date"] for d in betas_ts]
            
            mean_beta = summary_stats[factor_name]["mean_beta"]
            std_beta = summary_stats[factor_name]["std_beta"]
            
            # Find points where beta crosses mean ± 1.5*std
            threshold = 1.5 * std_beta
            
            for i in range(1, len(beta_series)):
                prev_beta = beta_series[i - 1]
                curr_beta = beta_series[i]
                
                # Check for significant shift
                if abs(curr_beta - prev_beta) > threshold:
                    regime_changes.append({
                        "date": dates[i],
                        "factor": factor_name,
                        "beta_before": float(prev_beta),
                        "beta_after": float(curr_beta),
                        "shift_magnitude": float(curr_beta - prev_beta),
                        "shift_in_std": float((curr_beta - prev_beta) / std_beta) if std_beta > 0 else 0.0,
                    })
        
        # Sort regime changes by date
        regime_changes.sort(key=lambda x: x["date"], reverse=True)
        
        return {
            "available": True,
            "window_days": window,
            "ridge_alpha": ridge_alpha,
            "n_observations": len(betas_ts),
            "factor_names": factor_names,
            "time_series": betas_ts,  # Full time series (can be large)
            "summary_stats": summary_stats,
            "regime_changes": regime_changes[:20],  # Top 20 most recent changes
            "interpretation": _interpret_rolling_betas(summary_stats, regime_changes),
        }
        
    except Exception as e:
        return {
            "available": False,
            "reason": f"rolling_betas_error: {str(e)[:300]}",
        }


def _interpret_rolling_betas(
    summary_stats: Dict[str, Dict[str, float]],
    regime_changes: List[Dict[str, Any]],
) -> str:
    """Generate human-readable interpretation of rolling beta results."""
    if not summary_stats:
        return "No factor exposures estimated."
    
    # Find most stable and most variable factors
    stable_factors = sorted(
        summary_stats.items(),
        key=lambda x: x[1]["coefficient_of_variation"]
    )[:2]
    
    variable_factors = sorted(
        summary_stats.items(),
        key=lambda x: x[1]["coefficient_of_variation"],
        reverse=True
    )[:2]
    
    interpretation = ""
    
    if stable_factors:
        interpretation += f"Most stable exposures: "
        interpretation += ", ".join([f"{name} (β≈{stats['mean_beta']:.2f}, CV={stats['coefficient_of_variation']:.2f})"
                                    for name, stats in stable_factors])
        interpretation += ". "
    
    if variable_factors and variable_factors[0][1]["coefficient_of_variation"] > 0.3:
        interpretation += f"Time-varying exposures: "
        interpretation += ", ".join([f"{name} (β={stats['mean_beta']:.2f}±{stats['std_beta']:.2f}, CV={stats['coefficient_of_variation']:.2f})"
                                    for name, stats in variable_factors])
        interpretation += ". "
    
    if regime_changes:
        n_changes = len(regime_changes)
        interpretation += f"Detected {n_changes} significant regime shifts in factor exposures. "
        
        # Mention most recent major shift
        if regime_changes:
            recent = regime_changes[0]
            interpretation += f"Most recent: {recent['factor']} beta shifted from {recent['beta_before']:.2f} to {recent['beta_after']:.2f} on {recent['date']}."
    
    return interpretation


def attribute_events(
    ticker: str,
    jumps: List[Jump],
    news_headlines: Optional[Dict[str, List[str]]] = None,
    ticker_rets: Optional[pd.Series] = None,
    start: Optional[str] = None,
    end: Optional[str] = None,
) -> Dict[str, Any]:
    """Lightweight attribution:

    - mark if jump dates are near earnings dates
    - attach headline snippets if provided

    NLP similarity hooks can be added later.
    """
    cal = earnings_calendar(ticker)
    earnings_dates = set(cal.get("earnings_dates", []) or [])

    def _near(d: str, candidates: set[str], days: int = 2) -> bool:
        try:
            dt = pd.Timestamp(d)
        except Exception:
            return False
        for c in candidates:
            try:
                ct = pd.Timestamp(c)
            except Exception:
                continue
            if abs((dt - ct).days) <= days:
                return True
        return False

    jump_attrib = []
    for j in jumps:
        jump_attrib.append(
            {
                **asdict(j),
                "near_earnings": _near(j.date, earnings_dates, days=2),
                "headlines": (news_headlines or {}).get(j.date, [])[:5],
            }
        )

    out = {
        "earnings_calendar": cal,
        "jumps": jump_attrib,
        "abnormal_returns": (
            abnormal_return_attribution(
                ticker_rets=ticker_rets,
                start=start,
                end=end,
                jumps=jumps,
            )
            if ticker_rets is not None and start is not None and end is not None
            else {"available": False, "reason": "missing_returns_or_window"}
        ),
        "summary": {
            "n_jumps": len(jumps),
            "n_jumps_near_earnings": int(
                sum(1 for j in jump_attrib if j.get("near_earnings"))
            ),
        },
    }

    return out


def build_driver_report(
    ticker: str,
    px: pd.Series,
    start: str,
    end: str,
    trend_method: str = "hp",
    factor_tickers: Optional[Dict[str, str]] = None,
    headlines_by_date: Optional[Dict[str, List[str]]] = None,
    ridge_lambda: float = 1e-4,
) -> DriverReport:
    """Build comprehensive driver report with factor attribution.
    
    Parameters
    ----------
    factor_tickers : dict, optional
        Mapping of factor name -> ticker. Recommended keys:
        market, sector, rates, usd, oil, vix
    headlines_by_date : dict, optional
        Mapping of ISO date -> list of headlines
    """
    rets = compute_returns(px)
    trend, resid = decompose_trend(px, method=trend_method)

    jumps = detect_jumps(rets)
    regimes = detect_volatility_regimes(rets)

    # Trend summary
    trend_ret = float(px.iloc[-1] / px.iloc[0] - 1.0) if len(px) >= 2 else float("nan")
    ann_ret = (1.0 + trend_ret) ** (252.0 / max(1.0, float(len(rets)))) - 1.0 if np.isfinite(trend_ret) else float("nan")

    trend_summary = {
        "total_return": trend_ret,
        "annualized_return": ann_ret,
        "trend_method": trend_method,
        "residual_std": float(resid.std(ddof=0)),
    }

    # Change points / regime shifts
    cps = detect_changepoints_pelt_like(rets)

    # Factor model (NEW)
    ft = {**DEFAULT_FACTOR_TICKERS}
    if factor_tickers:
        ft.update({k: v for k, v in factor_tickers.items() if v})
    
    factor_rets: Dict[str, pd.Series] = {}
    try:
        factor_rets = fetch_factor_returns(start=start, end=end, tickers=ft)
    except Exception:
        factor_rets = {}

    fm = factor_model_fit(rets, factor_rets, ridge_lambda=ridge_lambda) if factor_rets else {
        "available": False,
        "reason": "factor_fetch_failed_or_empty"
    }

    # Earnings dates
    cal = earnings_calendar(ticker)
    earnings_dates = cal.get("earnings_dates", []) if cal.get("available") else []

    # Per-jump driver analysis (NEW)
    per_jump: List[Dict[str, Any]] = []
    rets_n = rets.copy()
    rets_n.index = pd.to_datetime(rets_n.index).normalize()

    for j in jumps:
        dn = pd.Timestamp(j.date).normalize()
        if dn not in rets_n.index:
            continue
        r_day = float(rets_n.loc[dn])

        contrib = {}
        idio = float("nan")
        if fm.get("available"):
            betas = fm["betas"]
            # Normalize factor return indices
            fr_norm = {k: s.copy() for k, s in factor_rets.items()}
            for k, s in fr_norm.items():
                s.index = pd.to_datetime(s.index).normalize()
                fr_norm[k] = s
            contrib = factor_contributions_on_date(j.date, fr_norm, betas)
            idio = float(r_day - sum(contrib.values()))

        near_e = _near_date(j.date, earnings_dates, days=2)

        h = (headlines_by_date or {}).get(j.date, [])
        kw = extract_headline_keywords(h, top_k=6) if h else []

        per_jump.append(
            classify_driver_for_jump(
                jump=j,
                near_earnings=near_e,
                contrib=contrib,
                idio=idio if np.isfinite(idio) else float(r_day),
                total_ret=r_day,
                headline_keywords=kw,
            )
        )

    # ENHANCEMENT 1: Granger causality testing (temporal causation)
    gc_results = estimate_granger_causality(
        ticker_rets=rets,
        factor_rets=factor_rets,
        max_lag=5,
        alpha=0.05,
    ) if factor_rets else {"available": False, "reason": "no_factors"}

    # ENHANCEMENT 2: Causal DAG discovery (causal network structure)
    dag_results = build_causal_dag(
        ticker_rets=rets,
        factor_rets=factor_rets,
        alpha=0.05,
        max_cond_vars=3,
    ) if factor_rets and len(rets) >= 100 else {"available": False, "reason": "insufficient_data"}

    # ENHANCEMENT 3: Rolling factor betas (time-varying exposures)
    rolling_results = rolling_factor_betas(
        ticker_rets=rets,
        factor_rets=factor_rets,
        window=60,
        ridge_alpha=1.0,
        min_periods=30,
    ) if factor_rets and len(rets) >= 90 else {"available": False, "reason": "insufficient_data"}

    # Event attribution (original structure preserved for compatibility)
    event_attr = attribute_events(
        ticker,
        jumps,
        news_headlines=headlines_by_date,
        ticker_rets=rets,
        start=start,
        end=end,
    )

    # Notes
    notes = []
    if len(jumps) == 0:
        notes.append("No significant jumps detected under current thresholds.")
    if len(regimes) == 0:
        notes.append("Not enough data to segment volatility regimes.")
    if not fm.get("available"):
        notes.append("Factor model unavailable; per-jump driver labels will be mostly heuristic.")
    if not headlines_by_date:
        notes.append("No headlines provided; narrative linkage limited to earnings proximity + factor attribution.")
    if "sector" not in (factor_tickers or {}):
        notes.append("No sector factor provided; consider passing a sector ETF (e.g., XLK/XLE/XLF) for better attribution.")
    
    # Notes for enhancements
    if gc_results.get("available"):
        causal = gc_results["summary"]["causal_factors"]
        if causal:
            notes.append(f"Granger causality detected: {', '.join(causal)} predict stock returns with temporal lag.")
    
    if dag_results.get("available"):
        parents = dag_results.get("stock_parents", [])
        if parents:
            notes.append(f"Causal DAG analysis: Stock returns directly influenced by {', '.join(parents)}.")
    
    if rolling_results.get("available"):
        n_changes = len(rolling_results.get("regime_changes", []))
        if n_changes > 0:
            notes.append(f"Time-varying factor exposures detected: {n_changes} significant regime shifts in betas.")

    return DriverReport(
        ticker=ticker,
        start=start,
        end=end,
        n_obs=int(len(px)),
        trend_summary=trend_summary,
        volatility_regimes=regimes,
        jumps=jumps,
        event_attribution=event_attr,
        change_points=cps,
        factor_model=fm,
        per_jump_drivers=per_jump,
        granger_causality=gc_results,
        causal_dag=dag_results,
        rolling_betas=rolling_results,
        notes=notes,
    )


def save_report(report: DriverReport, out_dir: str) -> str:
    os.makedirs(out_dir, exist_ok=True)
    ts = time.strftime("%Y%m%d_%H%M%S")
    out_path = os.path.join(out_dir, f"driver_report_{report.ticker}_{ts}.json")

    payload = asdict(report)
    # dataclasses inside dataclasses already converted except VolRegime/Jump lists
    payload["volatility_regimes"] = [asdict(v) for v in report.volatility_regimes]
    payload["jumps"] = [asdict(j) for j in report.jumps]
    # factor_model and per_jump_drivers are already dicts/lists of dicts

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def run_driver_analysis(
    ticker: str,
    start: str,
    end: str,
    out_dir: str,
    trend_method: str = "hp",
    sector_etf: Optional[str] = None,
    headlines_by_date: Optional[Dict[str, List[str]]] = None,
) -> str:
    """Convenience wrapper for CLI usage.
    
    Parameters
    ----------
    sector_etf : str, optional
        If provided, will be used as factor name 'sector' for better attribution
    headlines_by_date : dict, optional
        Mapping of ISO date -> list of headlines for keyword extraction
    """
    px = fetch_prices(ticker, start=start, end=end)
    factors = {"sector": sector_etf} if sector_etf else None
    report = build_driver_report(
        ticker,
        px,
        start=start,
        end=end,
        trend_method=trend_method,
        factor_tickers=factors,
        headlines_by_date=headlines_by_date,
    )
    return save_report(report, out_dir=out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Per-stock movement driver analysis with multi-factor attribution")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-dir", default=os.path.join("..", "outputs"))
    p.add_argument("--trend-method", default="hp", choices=["hp", "stl", "ewm"])
    p.add_argument("--sector-etf", default=None, help="Optional sector ETF ticker (e.g., XLK, XLE, XLF)")
    args = p.parse_args()

    out = run_driver_analysis(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        out_dir=args.out_dir,
        trend_method=args.trend_method,
        sector_etf=args.sector_etf,
    )
    print(out)
