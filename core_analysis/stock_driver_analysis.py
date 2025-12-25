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
    rets = px.pct_change().dropna()
    rets.name = "ret"
    return rets


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
) -> DriverReport:
    rets = compute_returns(px)
    trend, resid = decompose_trend(px, method=trend_method)

    jumps = detect_jumps(rets)
    regimes = detect_volatility_regimes(rets)

    # Trend summary (simple)
    trend_ret = float(px.iloc[-1] / px.iloc[0] - 1.0)
    ann_ret = (1.0 + trend_ret) ** (252.0 / max(1.0, float(len(rets)))) - 1.0

    trend_summary = {
        "total_return": trend_ret,
        "annualized_return": ann_ret,
        "trend_method": trend_method,
        "residual_std": float(resid.std(ddof=0)),
    }

    # Change points / regime shifts (heuristic)
    cps = detect_changepoints_pelt_like(rets)

    # Event attribution: structured events + abnormal return study vs SPY
    event_attr = attribute_events(
        ticker,
        jumps,
        news_headlines=None,
        ticker_rets=rets,
        start=start,
        end=end,
    )

    notes = []
    if len(jumps) == 0:
        notes.append("No significant jumps detected under current thresholds.")
    if len(regimes) == 0:
        notes.append("Not enough data to segment volatility regimes.")

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

    with open(out_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)

    return out_path


def run_driver_analysis(
    ticker: str,
    start: str,
    end: str,
    out_dir: str,
    trend_method: str = "hp",
) -> str:
    px = fetch_prices(ticker, start=start, end=end)
    report = build_driver_report(ticker, px, start=start, end=end, trend_method=trend_method)
    return save_report(report, out_dir=out_dir)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser(description="Per-stock movement driver analysis")
    p.add_argument("--ticker", required=True)
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-dir", default=os.path.join("..", "outputs"))
    p.add_argument("--trend-method", default="hp", choices=["hp", "stl", "ewm"])
    args = p.parse_args()

    out = run_driver_analysis(
        ticker=args.ticker,
        start=args.start,
        end=args.end,
        out_dir=args.out_dir,
        trend_method=args.trend_method,
    )
    print(out)
