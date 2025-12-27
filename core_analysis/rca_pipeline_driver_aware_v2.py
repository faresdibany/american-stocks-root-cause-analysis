"""Root Cause Analysis (RCA) — End-to-end pipeline (driver-aware)

This script is an *independent* pipeline that chains:

1) Driver discovery (outcome-driven: market/sector/macro vs idiosyncratic)
2) Historical price driver analysis (trend/regimes/jumps/changepoints + factor attribution)
3) Quant + AI / world-news ranking
4) Social sentiment ranking (news + Reddit + StockTwits best-effort)
5) Mode-aware consolidation: ranking + explanations that respect driver modes

Design goals
------------
- Keep dependencies optional wherever possible (same style as other pickers).
- Avoid tight coupling: call existing modules and collect their outputs.
- Always write artifacts under ./outputs/.
- Work best-effort for any ticker (including non-US), with robust fallbacks.

Usage
-----
Run from this folder (recommended):
    python .\\core_analysis\\rca_pipeline.py

You can also pass tickers:
    python .\\core_analysis\\rca_pipeline.py --tickers AAPL,MSFT,NVDA

Notes
-----
- Social sentiment may require API keys for Reddit (PRAW). StockTwits often blocks
  automated requests; the social script already degrades gracefully.
- This is not investment advice.
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Any, Optional

import numpy as np
import pandas as pd

# Local imports (same folder)
from stock_driver_analysis import run_driver_analysis

# Restored picker modules
import stock_picker_hybrid_world_news_extension as world_news_picker
import stock_picker_social_sentiment as social_picker

# Optional driver discovery (new; best-effort)
try:
    from driver_discovery_v2_fixed import discover_driver_profile  # type: ignore
except Exception:  # pragma: no cover
    try:
        from driver_discovery import discover_driver_profile  # type: ignore
    except Exception:  # pragma: no cover
        discover_driver_profile = None  # type: ignore


# -----------------------------
# Helpers
# -----------------------------

def _outputs_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
    # Repo-level outputs folder (consistent with other pipelines)
    out = os.path.abspath(os.path.join(here, "..", "outputs"))
    os.makedirs(out, exist_ok=True)
    return out


def _timestamp() -> str:
    return datetime.now().strftime("%Y%m%d_%H%M%S")


def _ensure_indexed_by_ticker(df: pd.DataFrame) -> pd.DataFrame:
    if df is None:
        return pd.DataFrame()
    if not isinstance(df, pd.DataFrame):
        return pd.DataFrame()
    if df.index.name == "Ticker" or (len(df.index) > 0 and isinstance(df.index[0], str)):
        return df
    if "Ticker" in df.columns:
        return df.set_index("Ticker")
    return df


def _safe_read_json(path: str) -> Dict[str, Any]:
    try:
        with open(path, "r", encoding="utf-8") as f:
            obj = json.load(f)
        return obj if isinstance(obj, dict) else {}
    except Exception:
        return {}


def _summarize_driver_report(rep: Dict[str, Any]) -> str:
    """Turn a driver_report_*.json dict into a short human-readable summary."""
    if not rep:
        return "Driver report unavailable."

    parts: List[str] = []

    # Trend
    trend = rep.get("trend_summary") or {}
    try:
        tr = float(trend.get("total_return", 0.0))
        parts.append(f"Total return: {tr*100:.1f}%")
    except Exception:
        pass

    # Vol regimes
    vols = rep.get("volatility_regimes") or []
    if isinstance(vols, list) and vols:
        labels = [v.get("label") for v in vols if isinstance(v, dict) and v.get("label")]
        if labels:
            parts.append("Vol regime(s): " + ", ".join(sorted(set(labels))))

    # Jumps
    jumps = rep.get("jumps") or []
    if isinstance(jumps, list) and jumps:
        try:
            parts.append(f"Jumps: {len(jumps)} (max |ret| {max(abs(float(j.get('ret', 0.0))) for j in jumps)*100:.1f}%)")
        except Exception:
            parts.append(f"Jumps: {len(jumps)}")

    # Change points
    cps = rep.get("change_points") or {}
    pts = cps.get("points") if isinstance(cps, dict) else None
    if isinstance(pts, list) and pts:
        det = []
        for p in pts[:2]:
            if not isinstance(p, dict):
                continue
            d = p.get("date")
            sc = p.get("score")
            try:
                det.append(f"{d} (score {float(sc):.2f})")
            except Exception:
                det.append(str(d))
        parts.append("Change points: " + ", ".join(det))

    # Factor model availability
    fm = rep.get("factor_model") or {}
    if isinstance(fm, dict) and fm.get("available"):
        r2 = fm.get("r2")
        try:
            parts.append(f"Factor R²: {float(r2):.2f}")
        except Exception:
            pass

    return "; ".join(parts) if parts else "No strong driver signals extracted."


def _minmax(s: pd.Series) -> pd.Series:
    s = pd.to_numeric(s, errors="coerce")
    if s.dropna().empty:
        return pd.Series(np.zeros(len(s)), index=s.index)
    lo = float(np.nanmin(s.values))
    hi = float(np.nanmax(s.values))
    if not np.isfinite(lo) or not np.isfinite(hi) or hi - lo <= 1e-12:
        return pd.Series(np.zeros(len(s)), index=s.index)
    return (s - lo) / (hi - lo)


def _mode_weights(mode: str) -> Dict[str, float]:
    """
    Mode-aware interpretation weights:
      - beta_or_sector_driven: trust quant (CAGR/Sharpe); sentiment secondary
      - narrative_or_idiosyncratic: sentiment/topic matter more; penalize jump risk
      - rates_sensitive: emphasize Sharpe; penalize rate exposure
      - commodity_oil_sensitive: emphasize Sharpe; penalize oil exposure
      - mixed: balanced
    """
    mode = (mode or "mixed").strip()
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
    driver_reports_json: Dict[str, Dict[str, Any]],
) -> pd.DataFrame:
    """
    Adds:
      - driver_mode
      - driver_market_like_share
      - ModeAwareScore
    and sorts by ModeAwareScore desc.

    Assumes merged contains some of: CAGR, Sharpe, CombinedSentiment, TopicMomentum.
    """
    if merged is None or merged.empty:
        return merged

    df = merged.copy()

    # Normalized components (safe defaults)
    df["CAGR_n"] = _minmax(df["CAGR"]) if "CAGR" in df.columns else 0.0
    df["Sharpe_n"] = _minmax(df["Sharpe"]) if "Sharpe" in df.columns else 0.0

    # sentiment + topic can exist in either base or social-prefixed versions
    if "CombinedSentiment" in df.columns:
        df["Sent_n"] = _minmax(df["CombinedSentiment"])
    elif "social_CombinedSentiment" in df.columns:
        df["Sent_n"] = _minmax(df["social_CombinedSentiment"])
    else:
        df["Sent_n"] = 0.0

    if "TopicMomentum" in df.columns:
        df["Topic_n"] = _minmax(df["TopicMomentum"])
    elif "social_TopicMomentum" in df.columns:
        df["Topic_n"] = _minmax(df["social_TopicMomentum"])
    else:
        df["Topic_n"] = 0.0

    # Attach driver mode + market-like share
    modes = {}
    mkt_like = {}
    for t in df.index:
        prof = (driver_profiles or {}).get(t, {}) or {}
        modes[t] = prof.get("mode", "mixed")
        mkt_like[t] = prof.get("market_like_share", np.nan)
    df["driver_mode"] = pd.Series(modes)
    df["driver_market_like_share"] = pd.Series(mkt_like)

    # Risk penalties (best-effort)
    jump_pen = pd.Series(0.0, index=df.index)
    rates_pen = pd.Series(0.0, index=df.index)
    oil_pen = pd.Series(0.0, index=df.index)

    for t in df.index:
        rep = (driver_reports_json or {}).get(t, {}) or {}
        per_jump = rep.get("per_jump_drivers") or []
        fm = rep.get("factor_model") or {}
        betas = fm.get("betas") if isinstance(fm, dict) else None

        # simple jump proxy: more jump days => more gap risk
        if isinstance(per_jump, list) and len(per_jump) > 0:
            jump_pen[t] = min(0.25, 0.03 * len(per_jump))

        # penalties depend on factor naming (your driver report uses tickers as factor labels; keep defensive)
        if isinstance(betas, dict):
            # try a few likely keys
            rb = 0.0
            for k in ["rates", "TLT", "IEF", "rates_long", "rates_intermediate"]:
                if k in betas:
                    try:
                        rb = max(rb, abs(float(betas.get(k, 0.0))))
                    except Exception:
                        pass
            ob = 0.0
            for k in ["oil", "USO"]:
                if k in betas:
                    try:
                        ob = max(ob, abs(float(betas.get(k, 0.0))))
                    except Exception:
                        pass
            rates_pen[t] = min(0.25, 0.05 * rb)
            oil_pen[t] = min(0.25, 0.05 * ob)

    # Compute ModeAwareScore
    scores = []
    for t in df.index:
        mode = str(df.loc[t, "driver_mode"] or "mixed")
        w = _mode_weights(mode)

        base = (
            w["CAGR"] * float(df.loc[t, "CAGR_n"]) +
            w["Sharpe"] * float(df.loc[t, "Sharpe_n"]) +
            w["Sent"] * float(df.loc[t, "Sent_n"]) +
            w["Topic"] * float(df.loc[t, "Topic_n"])
        )

        # Apply penalties by mode
        if mode == "narrative_or_idiosyncratic":
            base -= float(jump_pen[t])
        elif mode == "rates_sensitive":
            base -= float(rates_pen[t])
        elif mode == "commodity_oil_sensitive":
            base -= float(oil_pen[t])

        scores.append(base)

    df["ModeAwareScore"] = pd.Series(scores, index=df.index)
    df = df.sort_values("ModeAwareScore", ascending=False)

    return df


# -----------------------------
# Pipeline
# -----------------------------

def run_pipeline(
    tickers: List[str],
    period: str = "12mo",
    interval: str = "1d",
    max_news: int = 12,
    lookback_days: int = 14,
    social_max_news: int = 15,
    social_max_reddit: int = 50,
    social_max_stocktwits: int = 0,
    generate_explanations: bool = True,
    with_advanced_quant: bool = False,
    with_nlg: bool = False,
) -> Dict[str, Any]:
    out_dir = _outputs_dir()
    ts = _timestamp()

    # We may skip certain tickers if upstream data providers return no data
    # (common for dot-class tickers like BRK.B in some environments).
    valid_tickers: List[str] = []

    # Convert yfinance-like period to a concrete [start, end] for the driver analyzer.
    # Note: other picker scripts still use `period`/`interval` internally.
    end = pd.Timestamp.utcnow().tz_localize(None)
    start = end - pd.DateOffset(months=12)
    if period.endswith("mo"):
        try:
            start = end - pd.DateOffset(months=int(period[:-2]))
        except Exception:
            pass
    elif period.endswith("y"):
        try:
            start = end - pd.DateOffset(years=int(period[:-1]))
        except Exception:
            pass
    start_s = start.strftime("%Y-%m-%d")
    end_s = end.strftime("%Y-%m-%d")

    # 1) Driver discovery + driver analysis (per ticker)
    driver_reports: Dict[str, Any] = {}
    driver_reports_json: Dict[str, Dict[str, Any]] = {}
    driver_profiles: Dict[str, Dict[str, Any]] = {}

    for t in tickers:
        prof: Dict[str, Any] = {}
        if callable(discover_driver_profile):
            try:
                prof = discover_driver_profile(ticker=t, start=start_s, end=end_s)  # type: ignore
            except Exception:
                prof = {}
        driver_profiles[t] = prof

        sector_proxy = None
        factor_dict = None
        if isinstance(prof, dict):
            sel = prof.get("selected") or {}
            if isinstance(sel, dict):
                sector_proxy = sel.get("sector_proxy")
            factor_dict = prof.get("factor_tickers")

        # Driver analysis module writes a JSON file and returns its path.
        # We try to pass factor_tickers if the underlying function supports it.
        try:
            try:
                out_path = run_driver_analysis(
                    ticker=t,
                    start=start_s,
                    end=end_s,
                    out_dir=out_dir,
                    trend_method="hp",
                    sector_etf=sector_proxy,
                    factor_tickers=factor_dict,   # may raise TypeError on older versions
                    headlines_by_date=None,
                )
            except TypeError:
                out_path = run_driver_analysis(
                    ticker=t,
                    start=start_s,
                    end=end_s,
                    out_dir=out_dir,
                    trend_method="hp",
                    sector_etf=sector_proxy,
                    headlines_by_date=None,
                )
        except Exception as e:
            # Keep the pipeline best-effort: skip tickers where price data isn't available.
            print(f"[WARN] Skipping {t}: driver analysis failed ({type(e).__name__}: {e})")
            continue

        driver_reports[t] = {"report_path": out_path, "driver_profile": prof}
        rep = _safe_read_json(out_path)
        driver_reports_json[t] = rep
        valid_tickers.append(t)

    # If we skipped anything, ensure downstream stages use the survivors.
    tickers = valid_tickers if valid_tickers else tickers

    # 2+3) Quant + AI/world news picker (already includes ranking)
    world_cfg = world_news_picker.Config(
        tickers=tickers,
        period=period,
        interval=interval,
        top_k=min(5, len(tickers)),
        optimize_weights=True,
        max_news=max_news,
        lookback_days=lookback_days,
    )
    world_out = world_news_picker.run(world_cfg)
    ranked_world = _ensure_indexed_by_ticker(world_out.get("ranked"))
    portfolio_world = world_out.get("portfolio")

    # 4) Social sentiment (already includes ranking)
    social_cfg = social_picker.Config(
        tickers=tickers,
        period=period,
        interval=interval,
        top_k=min(5, len(tickers)),
        max_news=social_max_news,
        max_reddit=social_max_reddit,
        max_stocktwits=social_max_stocktwits,
        lookback_days=lookback_days,
        source_weights={"news": 0.5, "reddit": 0.5, "stocktwits": 0.0},
    )
    social_out = social_picker.run(social_cfg)
    ranked_social = _ensure_indexed_by_ticker(social_out.get("ranked"))
    portfolio_social = social_out.get("portfolio")

    # 5) Merge
    merged = ranked_world.copy() if not ranked_world.empty else pd.DataFrame(index=tickers)

    if not ranked_social.empty:
        # bring in social columns with prefixes to avoid collisions
        for col in ranked_social.columns:
            if col in merged.columns:
                merged[f"social_{col}"] = ranked_social[col]
            else:
                merged[col] = ranked_social[col]

    # Add driver tags based on the *actual* JSON contents (fixes previous bug)
    driver_tags: Dict[str, str] = {}
    for t in tickers:
        rep = driver_reports_json.get(t, {}) or {}
        jumps = rep.get("jumps") or []
        cps = rep.get("change_points") or {}
        pts = cps.get("points") if isinstance(cps, dict) else None

        tag = "stable"
        if isinstance(jumps, list) and len(jumps) >= 2:
            tag = "jump-driven"
        if isinstance(pts, list) and len(pts) >= 2:
            tag = "regime-shift"
        if isinstance(jumps, list) and jumps and isinstance(pts, list) and pts:
            tag = "events+regimes"
        driver_tags[t] = tag

    merged["driver_tag"] = pd.Series(driver_tags)

    # Mode-aware ranking (replacement for Final_Score sorting)
    merged = apply_driver_mode_ranking(
        merged=merged,
        driver_profiles=driver_profiles,
        driver_reports_json=driver_reports_json,
    )

    # Build RCA portfolio from the mode-aware merged ranking
    portfolio_rca = merged.head(min(5, len(merged))).copy()
    portfolio_rca_path = os.path.join(out_dir, f"portfolio_rca_mode_aware_{ts}.csv")
    portfolio_rca.to_csv(portfolio_rca_path)

    # Write artifacts
    report = {
        "timestamp": ts,
        "tickers": tickers,
        "artifacts": {},
        "warnings": [],
        "driver_reports": driver_reports,          # contains report_path + driver_profile
        "driver_reports_json": driver_reports_json, # full parsed JSON reports (useful for downstream)
        "driver_profiles": driver_profiles,
        "world_news": {
            "weights": world_out.get("weights"),
            "backtest": world_out.get("backtest"),
        },
        "social_sentiment": {
            "weights": social_out.get("weights"),
            "backtest": social_out.get("backtest"),
        },
        "top_picks": {
            "world": portfolio_world.to_dict() if hasattr(portfolio_world, "to_dict") else None,
            "social": portfolio_social.to_dict() if hasattr(portfolio_social, "to_dict") else None,
            "rca_mode_aware": portfolio_rca.to_dict() if hasattr(portfolio_rca, "to_dict") else None,
        },
    }

    merged_path = os.path.join(out_dir, f"ranked_signals_rca_{ts}.csv")
    merged.to_csv(merged_path)
    report["artifacts"]["merged_rankings_csv"] = merged_path
    report["artifacts"]["portfolio_rca_mode_aware_csv"] = portfolio_rca_path

    report_path = os.path.join(out_dir, f"rca_pipeline_report_{ts}.json")
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    report["artifacts"]["report_json"] = report_path

    # Optional: run advanced quantitative pipeline and save its outputs into outputs/.
    # This stage can be used independently of NLG.
    if with_advanced_quant:
        try:
            import stock_picker_advanced_quantitative as adv  # local module

            adv_cfg = adv.Config(
                tickers=tickers,
                period=period,
                interval=interval,
                top_k=min(5, len(tickers)),
                include_ai=True,
                max_news=max_news,
                lookback_days=lookback_days,
            )
            adv_out = adv.run(adv_cfg)

            ranked_adv = adv_out.get("ranked")
            portfolio_adv = adv_out.get("portfolio")
            fundamentals_adv = adv_out.get("fundamentals")
            quant_adv = adv_out.get("quant")
            ai_signals_adv = adv_out.get("ai_signals")

            if hasattr(ranked_adv, "to_csv"):
                p = os.path.join(out_dir, f"ranked_signals_adv_quant_{ts}.csv")
                ranked_adv.to_csv(p)
                report["artifacts"]["ranked_signals_adv_quant_csv"] = p
            if hasattr(portfolio_adv, "to_csv"):
                p = os.path.join(out_dir, f"portfolio_adv_quant_{ts}.csv")
                portfolio_adv.to_csv(p)
                report["artifacts"]["portfolio_adv_quant_csv"] = p
            if hasattr(fundamentals_adv, "to_csv"):
                p = os.path.join(out_dir, f"fundamentals_adv_quant_{ts}.csv")
                fundamentals_adv.to_csv(p)
                report["artifacts"]["fundamentals_adv_quant_csv"] = p
            if hasattr(quant_adv, "to_csv"):
                p = os.path.join(out_dir, f"quant_signals_adv_quant_{ts}.csv")
                quant_adv.to_csv(p)
                report["artifacts"]["quant_signals_adv_quant_csv"] = p
            if hasattr(ai_signals_adv, "to_csv"):
                p = os.path.join(out_dir, f"ai_signals_adv_quant_{ts}.csv")
                ai_signals_adv.to_csv(p)
                report["artifacts"]["ai_signals_adv_quant_csv"] = p

            # Update JSON report on disk to include advanced-quant artifacts.
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            report["warnings"].append(f"Advanced-quant step failed/skipped: {e}")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception:
                pass

    # Optional: NLG stage (driver-aware module)
    if with_nlg:
        try:
            import stock_picker_nlg_explanations_driver_aware as nlg  # local module

            nlg_cfg = nlg.Config(
                tickers=tickers,
                period=period,
                interval=interval,
                top_k=min(5, len(tickers)),
                include_ai=True,
                max_news=max_news,
                lookback_days=lookback_days,
                generate_explanations=True,
                explanations_top_n=min(10, len(tickers)),
            )
            nlg_out = nlg.run_with_explanations(nlg_cfg)

            ranked_nlg = nlg_out.get("ranked")
            portfolio_nlg = nlg_out.get("portfolio")
            fundamentals_nlg = nlg_out.get("fundamentals")
            ai_signals_nlg = nlg_out.get("ai_signals")

            if hasattr(ranked_nlg, "to_csv"):
                p = os.path.join(out_dir, f"ranked_signals_nlg_{ts}.csv")
                ranked_nlg.to_csv(p)
                report["artifacts"]["ranked_signals_nlg_csv"] = p
            if hasattr(portfolio_nlg, "to_csv"):
                p = os.path.join(out_dir, f"portfolio_nlg_{ts}.csv")
                portfolio_nlg.to_csv(p)
                report["artifacts"]["portfolio_nlg_csv"] = p
            if hasattr(fundamentals_nlg, "to_csv"):
                p = os.path.join(out_dir, f"fundamentals_nlg_{ts}.csv")
                fundamentals_nlg.to_csv(p)
                report["artifacts"]["fundamentals_nlg_csv"] = p
            if hasattr(ai_signals_nlg, "to_csv"):
                p = os.path.join(out_dir, f"ai_signals_nlg_{ts}.csv")
                ai_signals_nlg.to_csv(p)
                report["artifacts"]["ai_signals_nlg_csv"] = p

            expl_txt = nlg_out.get("explanations")
            if isinstance(expl_txt, str) and expl_txt.strip():
                p = os.path.join(out_dir, f"explanations_nlg_{ts}.txt")
                with open(p, "w", encoding="utf-8") as f:
                    f.write("STOCK ANALYSIS - NATURAL LANGUAGE EXPLANATIONS\n")
                    f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
                    f.write("=" * 80 + "\n\n")
                    f.write(expl_txt)
                report["artifacts"]["explanations_nlg_txt"] = p

            # Update the JSON report on disk to include NLG artifacts.
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            report["warnings"].append(f"NLG step failed/skipped: {e}")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception:
                pass

    # Optional: write a human-readable narrative file.
    if generate_explanations:
        expl_path = os.path.join(out_dir, f"explanations_rca_{ts}.md")
        with open(expl_path, "w", encoding="utf-8") as f:
            f.write(f"# RCA Pipeline Explanations (driver-aware)\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Tickers: {', '.join(tickers)}\n\n")

            # Pull a few useful columns if present
            world_cols = [c for c in ["CAGR", "Sharpe", "CombinedSentiment", "TopicMomentum"] if c in ranked_world.columns]
            social_cols = [c for c in ["CombinedSentiment", "TopicMomentum", "TotalMentions", "NewsCount", "RedditCount", "StockTwitsCount"] if c in ranked_social.columns]

            for rank, t in enumerate(list(merged.index), 1):
                f.write(f"## #{rank} — {t}\n\n")
                if "ModeAwareScore" in merged.columns:
                    try:
                        f.write(f"**ModeAwareScore:** {float(merged.loc[t, 'ModeAwareScore']):.3f}\n\n")
                    except Exception:
                        pass

                # Driver mode block (from discovery)
                prof = driver_profiles.get(t, {}) or {}
                mode = prof.get("mode", "mixed")
                mls = prof.get("market_like_share", None)

                f.write(f"**Driver mode:** `{mode}`\n\n")
                if mls is not None:
                    try:
                        f.write(f"- Market-like share (discovery): {float(mls):.2f}\n")
                    except Exception:
                        pass
                sel = prof.get("selected") if isinstance(prof, dict) else None
                if isinstance(sel, dict):
                    if sel.get("market_proxy"):
                        f.write(f"- Market proxy: {sel.get('market_proxy')}\n")
                    if sel.get("sector_proxy"):
                        f.write(f"- Sector proxy: {sel.get('sector_proxy')}\n")
                checklist = prof.get("analysis_checklist") if isinstance(prof, dict) else None
                if isinstance(checklist, list) and checklist:
                    f.write("\n**How to interpret signals for this stock:**\n\n")
                    for it in checklist[:5]:
                        f.write(f"- {it}\n")
                f.write("\n")

                # Driver summary (actual driver report JSON)
                rep = driver_reports_json.get(t, {}) or {}
                f.write(f"**Driver analysis tag:** {merged.loc[t, 'driver_tag'] if 'driver_tag' in merged.columns else 'n/a'}\n\n")
                d = driver_reports.get(t, {})
                if isinstance(d, dict) and d.get("report_path"):
                    f.write(f"- Driver report: `{d['report_path']}`\n")
                if rep:
                    f.write(f"- Driver highlights: {_summarize_driver_report(rep)}\n")

                # World-news ranking summary
                if t in ranked_world.index and world_cols:
                    s = ranked_world.loc[t, world_cols]
                    f.write("\n**Quant + world news:**\n\n")
                    for c in world_cols:
                        try:
                            f.write(f"- {c}: {float(s[c]):.4f}\n")
                        except Exception:
                            f.write(f"- {c}: {s[c]}\n")

                # Social ranking summary
                if t in ranked_social.index and social_cols:
                    s2 = ranked_social.loc[t, social_cols]
                    f.write("\n**Social sentiment:**\n\n")
                    for c in social_cols:
                        try:
                            f.write(f"- {c}: {float(s2[c]):.4f}\n")
                        except Exception:
                            f.write(f"- {c}: {s2[c]}\n")

                f.write("\n---\n\n")

        report["artifacts"]["explanations_md"] = expl_path
        # Update the JSON report on disk to include explanations artifact.
        try:
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception:
            pass

    return {"report": report, "merged": merged}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA,AMZN,GOOGL")
    p.add_argument("--period", type=str, default="12mo")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--max-news", type=int, default=12)
    p.add_argument("--lookback-days", type=int, default=14)
    p.add_argument("--no-explanations", action="store_true", help="Skip writing explanations_rca_*.md")
    p.add_argument("--with-advanced-quant", action="store_true", help="Also run stock_picker_advanced_quantitative and write its CSV artifacts into outputs/")
    p.add_argument("--with-nlg", action="store_true", help="Also run stock_picker_nlg_explanations and write explanations_nlg_*.txt into outputs/")
    args = p.parse_args()

    tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()]
    out = run_pipeline(
        tickers=tickers,
        period=args.period,
        interval=args.interval,
        max_news=args.max_news,
        lookback_days=args.lookback_days,
        generate_explanations=not args.no_explanations,
        with_advanced_quant=args.with_advanced_quant,
        with_nlg=args.with_nlg,
    )

    merged = out["merged"]
    print("\n=== RCA PIPELINE COMPLETE (driver-aware) ===")
    print("Top 10:")
    cols = [c for c in ["ModeAwareScore", "Final_Score", "CAGR", "Sharpe", "CombinedSentiment", "TopicMomentum", "driver_mode", "driver_tag"] if c in merged.columns]
    print(merged[cols].head(10))


if __name__ == "__main__":
    main()
