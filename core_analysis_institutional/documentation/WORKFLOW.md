# Institutional (On‑Prem) RCA Pipeline — Workflow

This workflow describes how the institutional fork should be run and what happens step-by-step.

Because institutional requirements vary (different vendors, airgapped networks), this fork separates **analytics** from **data access** using providers.

## 0) Configure providers

Before running the pipeline in a true on‑prem setting, you configure:

- A `PriceProvider` (required)
- Optionally a `NewsProvider` and `FundamentalsProvider`
- A `StrictnessPolicy`

By default:
- missing prices **raise** (`fail_on_missing_prices=True`)

## 1) Entry

The “main” entrypoint is `core_analysis_institutional/rca_pipeline.py`.

The orchestrator currently behaves like retail, but the institutional intent is:
- accept providers and strictness
- pass them through into analytics modules

## 2) Determine analysis date range

The pipeline translates `period` (e.g., `1y`, `12mo`) into explicit `start`/`end` date strings.

This is needed for the driver analyzer.

## 3) Per-ticker RCA driver analysis

For each ticker:

1. Call `stock_driver_analysis.run_driver_analysis(...)`.
2. Inside, price history is loaded by calling:
   - `fetch_prices(..., price_provider=..., strictness=..., allow_yfinance_fallback=False)`

Expected institutional behavior:
- If no provider is passed and fallback is not allowed, the code raises `NotConfiguredProviderError`.

## 4) Signal extraction inside the driver analyzer

After price series are obtained, the analyzer computes:

- log returns
- jump days
- volatility regimes
- trend decomposition
- change points
- factor model attribution
- Granger causality tests
- (optional) causal DAG discovery
- rolling betas

A JSON driver report is written per ticker.

## 5) Ranking / pickers

The current institutional fork still runs the ranking modules derived from retail.

Depending on your on‑prem constraints:
- you may disable or replace the “world news” and “social sentiment” stages
- or you may refactor them to use institutional `NewsProvider` instead of public sources

## 6) Consolidated report

The pipeline writes a consolidated report under `outputs/` combining:
- ranked outputs
- per-ticker driver summaries

## 7) Failure modes (strictness)

`StrictnessPolicy` controls whether missing data should:
- raise immediately (preferred for on‑prem governance)
- or degrade gracefully (useful for exploratory runs)

Recommended institutional defaults:
- `fail_on_missing_prices=True`
- `fail_on_missing_factors` based on your vendor coverage
- `fail_on_missing_news` based on whether news is required for your use case

## 8) Recommended run profiles

### On‑prem / production
- provide an implementation of `PriceProvider`
- keep `allow_yfinance_fallback=False`
- fail closed on required datasets

### Local development
- allow fallback only when explicitly enabled
- use providers when possible to match production behavior
