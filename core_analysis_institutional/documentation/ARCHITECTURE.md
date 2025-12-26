# Institutional (On‑Prem) RCA Pipeline — Architecture

This folder documents the **institutional/on‑prem** fork under `core_analysis_institutional/`.

This fork is designed for:
- environments with **approved market data vendors** (Bloomberg/Refinitiv/FactSet/internal feeds)
- strict governance around internet usage (often *no* direct calls to public endpoints)
- deterministic runs, clear failure modes, and explicit configuration

Compared to the retail/advisor forks, the institutional fork is moving toward a **provider-based architecture**.

## High-level components

### 1) Orchestrator: `core_analysis_institutional/rca_pipeline.py`

The current orchestrator largely matches the retail pipeline structure:

1. Per-ticker RCA driver reports via `stock_driver_analysis.run_driver_analysis()`
2. Ranking via hybrid news/quant picker
3. Ranking via social sentiment picker
4. Consolidation + report writing

Institutional differences (directionally):
- inputs and dependencies should be configurable via **providers**
- pipeline should preferably be **fail-closed** when required data is missing

### 2) Data access contracts: `core_analysis_institutional/providers.py`

This file defines interfaces (Protocols) that separate analytics from data acquisition:

- `PriceProvider.get_adjusted_close(ticker, start, end, interval) -> pd.Series`
- `NewsProvider.get_headlines(ticker, lookback_days, max_items) -> {date -> [headline]}`
- `FundamentalsProvider.get_fundamentals(ticker) -> dict`

And a strictness control:

- `StrictnessPolicy`
  - `fail_on_missing_prices` (default True)
  - `fail_on_missing_factors` (default False)
  - `fail_on_missing_news` (default False)

Default placeholder provider:
- `NullPriceProvider` raises `NotConfiguredProviderError` to force explicit configuration in on‑prem setups.

### 3) Driver analytics: `core_analysis_institutional/stock_driver_analysis.py`

This module computes the per-ticker driver report (trend/regimes/jumps/factor attribution/causal enhancements).

Institutional-specific changes (already implemented):
- `fetch_prices(...)` now supports provider injection:
  - `price_provider: Optional[PriceProvider]`
  - `strictness: Optional[StrictnessPolicy]`
  - `allow_yfinance_fallback: bool = False` (default)

Meaning:
- Production/on‑prem: you pass an approved `PriceProvider`.
- Local dev: you *may* enable yfinance fallback explicitly.

### 4) Pickers / ranking modules

`core_analysis_institutional/` still contains the picker modules copied from retail.

Important note:
- Some picker modules (e.g. `stock_picker.py`) contain their own internal “provider” concepts for public sources.
- These are **not** the same as `core_analysis_institutional/providers.py`.

Long-term goal:
- converge all institutional data access through `providers.py` so the whole pipeline can be run without public endpoints.

## Primary data flows

### Inputs
- ticker universe
- analysis window
- strictness policy (how to behave when data is missing)
- provider implementations (prices, optionally news and fundamentals)

### Outputs
Artifacts are written under repo-level `outputs/`:
- Per ticker driver report JSON
- Pipeline consolidated JSON

## Institutional design principles

1. **Fail closed by default (when it matters)**
   - Missing prices should error early unless explicitly configured otherwise.

2. **Approved sources only**
   - Public internet sources should be disabled by default.

3. **Separation of concerns**
   - Analytics modules should not “know” Bloomberg vs Refinitiv vs internal feeds.

4. **Determinism + lineage**
   - Providers can embed vendor fields, timestamps, revision ids, or snapshot ids.

## Module map

- `providers.py` — provider contracts + strictness policy
- `stock_driver_analysis.py` — driver report generator (now supports PriceProvider injection)
- `rca_pipeline.py` — orchestration (to be extended to accept providers)
- `stock_picker_*` — ranking modules (candidate for further provider alignment)
