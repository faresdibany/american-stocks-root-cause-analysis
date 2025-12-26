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

try:
    import yfinance as yf
except Exception:  # pragma: no cover
    yf = None

try:
    from statsmodels.tsa.seasonal import STL
    from statsmodels.tsa.filters.hp_filter import hpfilter
except Exception:  # pragma: no cover
    STL = None
    hpfilter = None


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
    factor_model: Dict[str, Any]  # NEW: multi-factor attribution
    per_jump_drivers: List[Dict[str, Any]]  # NEW: per-jump driver analysis
    notes: List[str]


# -----------------------------
# Core computations
# -----------------------------


def _require_yfinance() -> None:
    if yf is None:
        raise RuntimeError(
            "yfinance is required. Install it (see requirements.txt) and try again."
        )


def fetch_prices(
    ticker: str,
    start: str,
    end: str,
    interval: str = "1d",
) -> pd.Series:
    """Fetch adjusted close prices."""
    _require_yfinance()

    df = yf.download(ticker, start=start, end=end, interval=interval, progress=False)
    if df is None or len(df) == 0:
        raise RuntimeError(f"No data returned for {ticker}")

    # Prefer Adj Close when present
    if "Adj Close" in df.columns:
        px = df["Adj Close"].copy()
    else:
        px = df["Close"].copy()

    # yfinance can return a DataFrame with multiple columns (e.g., when using auto_adjust
    # or multi-index columns depending on version). Ensure 1D Series.
    if isinstance(px, pd.DataFrame):
        if px.shape[1] == 1:
            px = px.iloc[:, 0]
        else:
            # best-effort: pick the first numeric column
            px = px.select_dtypes(include=["number"]).iloc[:, 0]

    px = px.dropna().astype(float)
    px.name = "price"
    return px


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

    # align on same index
    aligned = pd.concat([ticker_rets, mkt.rename("mkt")], axis=1).dropna()
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
