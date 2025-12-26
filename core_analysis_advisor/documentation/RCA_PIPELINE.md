# Advisor RCA Pipeline — `rca_pipeline.py` deep dive

This document explains **exactly what happens** inside `core_analysis_advisor/rca_pipeline.py`.

## What this script is

`rca_pipeline.py` is the advisor fork’s orchestration layer.

It does 3 things at once:
1. Runs **root-cause driver analysis** per ticker.
2. Runs two separate **ranking engines** (world/news + social sentiment).
3. Applies **advisor constraints** and writes an auditable, consolidated report.

## Key functions and their roles

### `ClientConstraints` (advisor-specific)

A dataclass representing client-specific portfolio constraints.

Fields (best-effort enforcement):
- `max_drawdown`: hard filter if a drawdown column exists
- `require_dividends` / `min_dividend_yield`: hard filter if dividend yield column exists
- `exclude_sectors`: hard filter if a sector column exists
- `max_sector_fraction_in_top`: soft diversification constraint if sector + `Final_Score` exist

Important behavior:
- If input ranking tables don’t contain the required columns, the pipeline **does not guess**.
- It skips and records warnings into the compliance report.

### `_apply_constraints_to_ranked(ranked, constraints, top_n)`

Takes a ranked DataFrame and returns:
- `ranked_constrained`: possibly filtered / adjusted ranked table
- `compliance`: a structured dict that records
  - what was requested
  - what was excluded
  - what was skipped due to missing data

“Exactly what it does” (in order):
1. If max drawdown constraint is set:
   - find a drawdown column by name (`maxdd`, `max_drawdown`, `maxdd_pct`)
   - drop rows that violate it
2. If dividend constraints are set:
   - find a yield column (`dividend_yield`, `div_yield`, `yield`)
   - drop rows below the threshold
3. If sector exclusions are set:
   - find a sector column (`sector` or `gics_sector`)
   - drop excluded sectors
4. If sector cap is set:
   - only applies if `Final_Score` exists
   - enforces cap within the top N by removing the lowest scored names in the overrepresented sector

### `run_pipeline(...)`

This is the main entrypoint that does the full end-to-end process.

Parameters to know:
- `tickers`: list of tickers
- `period`, `interval`: used by pickers
- world/news knobs: `max_news`, `lookback_days`
- social knobs: `social_max_news`, `social_max_reddit`, `social_max_stocktwits`
- booleans: `with_advanced_quant`, `with_nlg`
- advisor knobs:
  - `client_constraints`
  - `advisor_top_n`

## End-to-end execution sequence

### Step 1 — Determine output directory and timestamps

The script writes all artifacts under the repo-level `outputs/` folder.

It also generates a timestamp used for filenames.

### Step 2 — Expand `period` into explicit dates (for driver analysis)

The driver analyzer is called with `start` and `end`.

So the pipeline:
- computes `end = now (UTC)`
- computes `start` from `period` (typically 12 months)

### Step 3 — Per-ticker driver reports

For each ticker:
- call `run_driver_analysis()` from `stock_driver_analysis.py`
- it writes `driver_report_<TICKER>_<timestamp>.json`

The pipeline stores at least:
- the report path
- a short summary string (via `_summarize_driver_report`)

### Step 4 — World/news + quant ranker

The pipeline calls the hybrid module:
- build `world_news_picker.Config(...)`
- `world_news_picker.run(cfg)`

This provides:
- ranked table
- a suggested portfolio (top-K)

### Step 5 — Social sentiment ranker

Similarly:
- build `social_picker.Config(...)`
- `social_picker.run(cfg)`

Outputs mirror the previous step.

### Step 6 — Merge ranked outputs

The `merged` table starts from world/news ranking:
- social columns are merged in
- collisions are prefixed as `social_<col>`

### Step 7 — Add driver tags

This is a light “advisor UX” feature:
- it looks at jump count and change-point detection output
- assigns a category label like `stable`, `jump-driven`, `regime-shift`

### Step 8 — Apply constraints (advisor fork feature)

If `client_constraints` is provided:
- constraints are applied to the **ranked world/news table**
- a CSV artifact is written for traceability
- a compliance object is embedded into the final report

This is where advisor-specific differentiation lives.

### Step 9 — Optional advanced quant and/or NLG

If requested:
- the pipeline lazy-imports the heavier modules
- it attaches their outputs to the consolidated report

### Step 10 — Write consolidated JSON

A single `rca_report_<timestamp>.json` is written.

The report contains:
- parameter echo / configuration
- driver report references + summaries
- world/news ranking + portfolio
- social ranking + portfolio
- merged table
- compliance report (if any)
- audit metadata

## Audit metadata

This fork includes an `audit` block:
- git SHA (best-effort)
- Python version and OS/platform
- UTC timestamp
- echoed client constraints and compliance decisions

Purpose:
- meet “show your work” expectations
- simplify reproducibility and client file retention

## Known limitations (by design)

- Constraint enforcement is **best-effort** and depends on available columns.
- This fork still uses public data access patterns (yfinance/news fetching) and is not suitable for strict on-prem environments.
- For strict data-lineage and approved-provider enforcement, use `core_analysis_institutional/`.
