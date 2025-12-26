# driver_mode_ranking.py
from __future__ import annotations
from typing import Dict, Any, Optional
import numpy as np
import pandas as pd


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    lo = float(np.nanmin(s.values)) if np.isfinite(np.nanmin(s.values)) else 0.0
    hi = float(np.nanmax(s.values)) if np.isfinite(np.nanmax(s.values)) else 1.0
    if hi - lo <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _mode_weights(mode: str) -> Dict[str, float]:
    """
    Mode-aware interpretation:
    - beta_or_sector_driven: trust quant (CAGR/Sharpe), sentiment is secondary
    - narrative_or_idiosyncratic: sentiment/topic matter more; penalize jump-risk
    - rates_sensitive: emphasize risk-adjusted performance; penalize big rate exposure
    - commodity_oil_sensitive: emphasize risk-adjusted performance; penalize big oil exposure
    - mixed: balanced
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


def apply_driver_mode_ranking(
    merged: pd.DataFrame,
    driver_profiles: Dict[str, Dict[str, Any]],
    driver_reports: Optional[Dict[str, Dict[str, Any]]] = None,
) -> pd.DataFrame:
    """
    Adds:
      - driver_mode
      - driver_market_like_share
      - ModeAwareScore
    and sorts by ModeAwareScore descending.

    Assumes merged has (some of): CAGR, Sharpe, CombinedSentiment, TopicMomentum.
    """
    if merged is None or merged.empty:
        return merged

    df = merged.copy()

    # Normalize available columns
    if "CAGR" in df.columns:
        df["CAGR_n"] = _minmax(df["CAGR"])
    else:
        df["CAGR_n"] = 0.0

    if "Sharpe" in df.columns:
        df["Sharpe_n"] = _minmax(df["Sharpe"])
    else:
        df["Sharpe_n"] = 0.0

    if "CombinedSentiment" in df.columns:
        df["Sent_n"] = _minmax(df["CombinedSentiment"])
    else:
        df["Sent_n"] = 0.0

    if "TopicMomentum" in df.columns:
        df["Topic_n"] = _minmax(df["TopicMomentum"])
    else:
        df["Topic_n"] = 0.0

    # Attach driver mode metadata
    modes = {}
    mkt_like = {}
    for t in df.index:
        prof = (driver_profiles or {}).get(t, {}) or {}
        modes[t] = prof.get("mode", "mixed")
        mkt_like[t] = prof.get("market_like_share", np.nan)

    df["driver_mode"] = pd.Series(modes)
    df["driver_market_like_share"] = pd.Series(mkt_like)

    # Optional risk penalties from the full driver report JSON
    # - narrative/idiosyncratic mode: penalize frequent jump regimes (gap-risk)
    # - rates_sensitive: penalize large absolute rates beta share
    jump_pen = pd.Series(0.0, index=df.index)
    rates_pen = pd.Series(0.0, index=df.index)
    oil_pen = pd.Series(0.0, index=df.index)

    if driver_reports:
        for t in df.index:
            rep = driver_reports.get(t, {}) or {}
            per_jump = rep.get("per_jump_drivers") or []
            fm = rep.get("factor_model") or {}
            betas = (fm.get("betas") or {}) if fm.get("available") else {}

            # simple jump-risk proxy
            if isinstance(per_jump, list) and len(per_jump) > 0:
                jump_pen[t] = min(0.25, 0.03 * len(per_jump))  # cap

            # exposure penalties (absolute beta as a rough proxy)
            if isinstance(betas, dict):
                # keys depend on what factors you passed; your default uses:
                # rates / oil etc in stock_driver_analysis.py DEFAULT_FACTOR_TICKERS :contentReference[oaicite:1]{index=1}
                rb = abs(float(betas.get("rates", 0.0))) if "rates" in betas else 0.0
                ob = abs(float(betas.get("oil", 0.0))) if "oil" in betas else 0.0
                rates_pen[t] = min(0.25, 0.05 * rb)
                oil_pen[t] = min(0.25, 0.05 * ob)

    # Compute mode-aware score per ticker
    scores = []
    for t in df.index:
        mode = df.loc[t, "driver_mode"]
        w = _mode_weights(str(mode))

        base = (
            w["CAGR"] * float(df.loc[t, "CAGR_n"]) +
            w["Sharpe"] * float(df.loc[t, "Sharpe_n"]) +
            w["Sent"] * float(df.loc[t, "Sent_n"]) +
            w["Topic"] * float(df.loc[t, "Topic_n"])
        )

        # Apply penalties based on mode
        if mode == "narrative_or_idiosyncratic":
            base -= float(jump_pen[t])
        elif mode == "rates_sensitive":
            base -= float(rates_pen[t])
        elif mode == "commodity_oil_sensitive":
            base -= float(oil_pen[t])

        scores.append(base)

    df["ModeAwareScore"] = pd.Series(scores, index=df.index)

    return df.sort_values("ModeAwareScore", ascending=False)
