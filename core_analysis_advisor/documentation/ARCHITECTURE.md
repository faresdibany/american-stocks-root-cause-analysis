# Advisor RCA Pipeline — Architecture

This folder documents the **advisor tooling** fork of the project under `core_analysis_advisor/`.

This fork is designed for:
- advisors running repeatable analyses for client portfolios
- constraint-aware recommendations (risk limits, exclusions, diversification)
- **auditability** (what code ran, when, with what parameters)

It intentionally stays close to the retail `core_analysis/` pipeline, but adds an “advisor layer” around ranking, reporting, and metadata.

## High-level components

### 1) Orchestrator: `core_analysis_advisor/rca_pipeline.py`

This is the main entrypoint. It coordinates:

1. **Per-ticker driver reports** via `stock_driver_analysis.run_driver_analysis()`
2. **World/news + quant ranking** via `stock_picker_hybrid_world_news_extension.run()`
3. **Social sentiment ranking** via `stock_picker_social_sentiment.run()`
4. **Advisor constraint enforcement** (filters and diversification caps)
5. Optional **advanced quant + NLG** enrichment (lazy-imported)
6. Consolidated JSON report and optional CSV artifacts

Key advisor additions in this fork:
- `ClientConstraints` dataclass
- `_apply_constraints_to_ranked()` constraint engine (explainable, best-effort)
- `audit` metadata block embedded into final report (git SHA, python/platform, UTC timestamp)

### 2) Driver analytics: `core_analysis_advisor/stock_driver_analysis.py`

Produces a **per-ticker** JSON report explaining “what drove the move” from prices.

Core analytical layers:
- Trend decomposition (HP filter / STL / fallback EWMA)
- Volatility regime labeling
- Jump detection (robust z-score + absolute return threshold)
- Event-study style abnormal return attribution vs SPY (benchmark)
- Change point detection (heuristic / score-based)
- Multi-factor model attribution (factor betas + per-jump factor contributions)
- Enhancements:
  - Granger causality testing (temporal lead/lag signals)
  - Optional causal DAG discovery (PC algorithm when `causal-learn` installed)
  - Rolling factor betas (time-varying exposures)

Data sources in advisor fork:
- Prices: `yfinance` (best-effort)
- Some event info: derived from price behavior + optional headlines

### 3) Rankers / pickers

These modules already existed in retail and are reused here:
- `stock_picker_hybrid_world_news_extension.py` — merges quant signals + world/news signals
- `stock_picker_social_sentiment.py` — sentiment-driven scoring using news/Reddit/StockTwits (best-effort)

The advisor fork **does not** fundamentally change their internals; instead it:
- collects their outputs
- merges tables
- optionally constrains the resulting candidate list

## Primary data flows

### Inputs

- `tickers`: list of symbols, e.g. `AAPL,MSFT,NVDA`
- `period`, `interval`: used by picker modules (yfinance-like)
- `lookback_days`, `max_news`: controls headline lookback + cap for the pickers
- advisor constraints (optional): max drawdown filter, dividend preferences, sector exclusions, sector cap

### Outputs

All artifacts land in `outputs/` (repo-level folder):

- Per ticker: `driver_report_<TICKER>_<timestamp>.json`
- Pipeline report: `rca_report_<timestamp>.json`
- Optional constrained table: `ranked_world_constrained_<timestamp>.csv`

## Advisor-specific design principles

1. **Explainability over complexity**
   - Constraints are applied with clear “excluded because …” reasons.

2. **Auditability**
   - Final report includes enough metadata to reproduce the run.

3. **Best-effort enrichment**
   - If some columns are missing (e.g., `sector` or `dividend_yield`) constraint enforcement degrades gracefully and records warnings.

## Module map

- `rca_pipeline.py` — end-to-end orchestration + advisor layer (constraints, compliance, audit)
- `stock_driver_analysis.py` — per-ticker RCA (drivers, factor attribution, causal enhancements)
- `stock_picker_hybrid_world_news_extension.py` — ranking + world/news integration
- `stock_picker_social_sentiment.py` — sentiment ranking
- `stock_picker_advanced_quantitative.py` — optional deeper quant features (lazy import)
- `stock_picker_nlg_explanations.py` — optional NLG narrative generation (lazy import)

## Operational notes

- This fork assumes a “practical advisor environment”: internet access is allowed (yfinance/news fetchers), but every run produces an audit trail.
- For regulated environments that **cannot** use public data sources, use the institutional fork under `core_analysis_institutional/`.
