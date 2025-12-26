# Institutional (On‑Prem) RCA Pipeline — `rca_pipeline.py` deep dive

This document explains what `core_analysis_institutional/rca_pipeline.py` does today, and what **institutional-specific behavior** is expected as the provider abstraction is completed.

## What this script is

`rca_pipeline.py` is the end-to-end orchestrator.

It coordinates:
1. Per-ticker driver report generation
2. World/news + quant ranking
3. Social sentiment ranking
4. Consolidation into a single JSON report

## Institutional constraints (why it’s different)

Institutional deployments usually require:
- approved data sources (Bloomberg/Refinitiv/FactSet/internal feeds)
- no public scraping in production
- deterministic runs with explicit configuration

To meet this, the institutional fork introduces `providers.py`.

## Current structure (as implemented)

### Key imports
- Per-ticker RCA: `from stock_driver_analysis import run_driver_analysis`
- Ranking modules:
  - `stock_picker_hybrid_world_news_extension`
  - `stock_picker_social_sentiment`

### `run_pipeline(...)`

The function:

1) Computes a concrete `start` and `end` from `period`.
2) For each ticker:
   - calls `run_driver_analysis(ticker, start, end, out_dir, ...)`
3) Runs world/news picker and social picker.
4) Merges tables.
5) Writes a consolidated JSON report.

Notes:
- In this fork, `max_news` defaults to **50** for the world/news stage.

### Output artifacts

Under the repo-level `outputs/` folder:
- `driver_report_<TICKER>_<timestamp>.json`
- `rca_report_<timestamp>.json`

## Provider integration (what *must* happen in institutional runs)

### PriceProvider (required)

The institutional `stock_driver_analysis.fetch_prices()` now supports:
- `price_provider: Optional[PriceProvider]`
- `strictness: Optional[StrictnessPolicy]`
- `allow_yfinance_fallback: bool = False`

Default institutional behavior:
- If no provider is passed, and fallback is not allowed, it raises `NotConfiguredProviderError`.

This is intentional: it forces on‑prem users to explicitly configure an approved provider.

### News/Fundamentals providers (planned alignment)

The pipeline still calls the retail-derived picker modules, which can involve public endpoints.

Institutional options:
- disable those stages entirely (run only RCA)
- refactor the pickers to accept `NewsProvider` and `FundamentalsProvider`
- re-implement pickers as wrappers around internal research systems

## How to think about “exactly what is happening”

The institutional pipeline conceptually splits into two layers:

### A) Analytics layer (should be identical everywhere)
- trend decomposition
- volatility regimes
- jump detection
- factor attribution
- causal enhancements

### B) Data access layer (must be institution-specific)
- price data
- benchmark data
- factor ETFs data
- headlines/news
- fundamentals

Institutional correctness depends on ensuring **B** uses approved sources.

## Suggested institutional output extensions (recommended)

Many on‑prem customers will want:
- dataset identifiers (vendor, snapshot id)
- timestamps and timezone normalization
- lineage fields per series
- explicit NaN/missingness reporting

Those are best added at the provider level and then echoed into the consolidated report.
