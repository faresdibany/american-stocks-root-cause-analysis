"""Root Cause Analysis (RCA) — End-to-end US pipeline

This script makes `american stocks root cause analysis/` an *independent* pipeline
that chains:

1) Historical price driver analysis (trend/regimes/jumps/changepoints + SPY event study)
2) Quantitative ranking
3) AI / world-news analysis
4) Social sentiment (news + Reddit + StockTwits best-effort)
5) One consolidated report output

Design goals
------------
- Keep dependencies optional wherever possible (same style as the other pickers).
- Avoid tight coupling: we call existing modules and collect their outputs.
- Always write artifacts under `../outputs/`.

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
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import datetime
from typing import Dict, List, Any

import pandas as pd

# Local imports (same folder)
from stock_driver_analysis import run_driver_analysis

# Restored picker modules
import stock_picker_hybrid_world_news_extension as world_news_picker
import stock_picker_social_sentiment as social_picker

# Optional: deep NLG (advanced quant script). We import lazily inside run_pipeline
# to avoid heavy imports (statsmodels/arch/talib/transformers/sklearn) unless requested.


def _outputs_dir() -> str:
    here = os.path.dirname(os.path.abspath(__file__))
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
            # majority label
            maj = max(set(labels), key=labels.count)
            parts.append(f"Volatility regime mostly: {maj}")

    # Jumps
    jumps = rep.get("jumps") or []
    if isinstance(jumps, list) and jumps:
        parts.append(f"Jumps detected: {len(jumps)}")
        # Mention top 1–2 jumps
        top = jumps[:2]
        det = []
        for j in top:
            if not isinstance(j, dict):
                continue
            d = j.get("date")
            r = j.get("ret")
            try:
                det.append(f"{d} ({float(r)*100:.1f}%)")
            except Exception:
                det.append(str(d))
        if det:
            parts.append("Top moves: " + ", ".join(det))

    # Abnormal return attribution
    ar = (rep.get("event_attribution") or {}).get("abnormal_returns") or {}
    if isinstance(ar, dict) and ar.get("available"):
        try:
            beta = float(ar.get("beta"))
            parts.append(f"Beta vs {ar.get('benchmark','SPY')}: {beta:.2f}")
        except Exception:
            pass
        summ = ar.get("summary") or {}
        try:
            frac = float(summ.get("mean_abs_abnormal_fraction"))
            if pd.notna(frac):
                parts.append(f"Mean abs abnormal fraction on jumps: {frac:.2f}")
        except Exception:
            pass

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

    return "; ".join(parts) if parts else "No strong driver signals extracted."


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

    # 1) Driver analysis (per ticker)
    driver_reports: Dict[str, Any] = {}
    for t in tickers:
        # Driver analysis module writes a JSON file and returns its path.
        out_path = run_driver_analysis(
            ticker=t,
            start=start_s,
            end=end_s,
            out_dir=out_dir,
            trend_method="hp",
        )
        driver_reports[t] = {"report_path": out_path}

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

    # 5) Merge + explain
    merged = ranked_world.copy() if not ranked_world.empty else pd.DataFrame(index=tickers)
    if not ranked_social.empty:
        # bring in social columns with prefixes to avoid collisions
        for col in ranked_social.columns:
            if col in merged.columns:
                merged[f"social_{col}"] = ranked_social[col]
            else:
                merged[col] = ranked_social[col]

    # Add a simple driver-based narrative tag
    driver_tags = {}
    for t, rep in driver_reports.items():
        jumps = (rep.get("jumps") or []) if isinstance(rep, dict) else []
        cps = (rep.get("change_points") or []) if isinstance(rep, dict) else []
        tag = "stable"
        if jumps and len(jumps) >= 2:
            tag = "jump-driven"
        if cps and len(cps) >= 2:
            tag = "regime-shift"
        if jumps and cps:
            tag = "events+regimes"
        driver_tags[t] = tag

    merged["driver_tag"] = pd.Series(driver_tags)

    # Final score: prefer world picker Final_Score, else fallback to what exists
    if "Final_Score" in merged.columns:
        merged = merged.sort_values("Final_Score", ascending=False)

    # Write artifacts
    report = {
        "timestamp": ts,
        "tickers": tickers,
        "artifacts": {},
        "warnings": [],
        "driver_reports": driver_reports,
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
        },
    }

    merged_path = os.path.join(out_dir, f"ranked_signals_rca_{ts}.csv")
    merged.to_csv(merged_path)
    report["artifacts"]["merged_rankings_csv"] = merged_path

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

            # Update the JSON report on disk to include advanced-quant artifacts.
            with open(report_path, "w", encoding="utf-8") as f:
                json.dump(report, f, indent=2)
        except Exception as e:
            report["warnings"].append(f"Advanced-quant step failed/skipped: {e}")
            try:
                with open(report_path, "w", encoding="utf-8") as f:
                    json.dump(report, f, indent=2)
            except Exception:
                pass

    # Optional: run the deep NLG pipeline and save its outputs into outputs/.
    if with_nlg:
        try:
            import stock_picker_nlg_explanations as nlg  # local module

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

            # Save to outputs/ (no nlg_analysis_* directories).
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
    # This is intentionally dependency-light and complements `stock_picker_nlg_explanations.py`.
    if generate_explanations:
        expl_path = os.path.join(out_dir, f"explanations_rca_{ts}.md")
        with open(expl_path, "w", encoding="utf-8") as f:
            f.write(f"# RCA Pipeline Explanations\n\n")
            f.write(f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"Tickers: {', '.join(tickers)}\n\n")

            # Pull a few useful columns if present
            world_cols = [c for c in ["Final_Score", "CAGR", "Sharpe", "CombinedSentiment", "TopicMomentum"] if c in ranked_world.columns]
            social_cols = [c for c in ["CombinedSentiment", "TopicMomentum", "TotalMentions", "NewsCount", "RedditCount", "StockTwitsCount"] if c in ranked_social.columns]

            for rank, t in enumerate(list(merged.index), 1):
                f.write(f"## #{rank} — {t}\n\n")

                # Driver summary (we currently store just the JSON path)
                d = driver_reports.get(t, {})
                f.write(f"**Driver analysis:** {merged.loc[t, 'driver_tag'] if 'driver_tag' in merged.columns else 'n/a'}\n\n")
                if isinstance(d, dict) and d.get("report_path"):
                    f.write(f"- Driver report: `{d['report_path']}`\n")

                    rep = _safe_read_json(d["report_path"])
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

                # Simple narrative paragraph
                ws = None
                ss = None
                try:
                    ws = float(ranked_world.loc[t, "CombinedSentiment"]) if t in ranked_world.index and "CombinedSentiment" in ranked_world.columns else None
                except Exception:
                    ws = None
                try:
                    ss = float(ranked_social.loc[t, "CombinedSentiment"]) if t in ranked_social.index and "CombinedSentiment" in ranked_social.columns else None
                except Exception:
                    ss = None

                narrative = []
                tag = merged.loc[t, "driver_tag"] if "driver_tag" in merged.columns else "stable"
                if tag == "events+regimes":
                    narrative.append("Price action looks like a mix of event shocks and regime changes (not pure noise).")
                elif tag == "regime-shift":
                    narrative.append("The stock shows regime-change signatures (volatility/trend shifts) rather than steady drift.")
                elif tag == "jump-driven":
                    narrative.append("Large single-day moves (jumps) appear to be a key driver of returns.")
                else:
                    narrative.append("No strong jump/regime flags; movement looks comparatively stable in this window.")

                if ws is not None and abs(ws) >= 0.10:
                    narrative.append(f"News sentiment is {'positive' if ws > 0 else 'negative'} (world+company combined).")
                if ss is not None and abs(ss) >= 0.05:
                    narrative.append(f"Crowd/social tone is {'positive' if ss > 0 else 'negative'} (best-effort sources).")

                f.write("\n**Summary:** ")
                f.write(" ".join(narrative) + "\n\n")

        report["artifacts"]["explanations_md"] = expl_path

    return {"report": report, "merged": merged}


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--tickers", type=str, default="AAPL,MSFT,NVDA,AMZN,GOOGL")
    p.add_argument("--period", type=str, default="12mo")
    p.add_argument("--interval", type=str, default="1d")
    p.add_argument("--max-news", type=int, default=50)
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
    print("\n=== RCA PIPELINE COMPLETE ===")
    print("Top 10:")
    cols = [c for c in ["Final_Score", "CAGR", "Sharpe", "CombinedSentiment", "TopicMomentum", "driver_tag"] if c in merged.columns]
    print(merged[cols].head(10))


if __name__ == "__main__":
    main()
