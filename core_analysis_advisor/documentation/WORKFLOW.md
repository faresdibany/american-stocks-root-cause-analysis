# Advisor RCA Pipeline — Workflow

This describes what happens when you run the advisor fork `core_analysis_advisor/rca_pipeline.py`.

## 0) Entry

The pipeline is started by calling `run_pipeline(...)` (either via CLI `main()` or imported).

Typical inputs:
- `tickers`: symbols to analyze
- `period` / `interval`: window definition for the picker modules
- `lookback_days` / `max_news`: how much news to fetch and analyze
- advisor tooling additions:
  - `client_constraints: ClientConstraints | None`
  - `advisor_top_n: int` (the “top N” set against which soft caps apply)

## 1) Build a concrete date range for driver analysis

The pipeline converts the `period` argument into `start` and `end` dates.

Why:
- `stock_driver_analysis.run_driver_analysis()` expects explicit `start`/`end`.
- The pickers still use `period`/`interval` natively.

## 2) Per-ticker driver analysis (RCA)

For each ticker:
- Call `stock_driver_analysis.run_driver_analysis(ticker, start, end, out_dir, ...)`
- That function writes `driver_report_<ticker>_<timestamp>.json`

What gets extracted (high level):
- trend summary (total return, trend slope proxies)
- volatility regimes (labeled segments)
- jump/shock list (date/return/zscore)
- change points (regime shifts)
- event attribution (esp. abnormal returns vs benchmark)
- factor model attribution (multi-factor, per-jump contributions)
- causal enhancements (granger/dag/rolling betas)

The pipeline keeps a lightweight index of these reports so it can later add summaries into the consolidated output.

## 3) Quant + world-news ranking

The pipeline runs the hybrid ranking:
- Create `world_news_picker.Config(...)`
- Call `world_news_picker.run(cfg)`

Notes:
- In this fork, `max_news` defaults to **50** for the world/news stage.

Typical outputs:
- `ranked`: a DataFrame-like table of tickers and scores
- `portfolio`: a top-K selection

The pipeline normalizes the ranking to ensure it’s indexed by ticker.

## 4) Social sentiment ranking

The pipeline runs:
- Create `social_picker.Config(...)`
- Call `social_picker.run(cfg)`

This produces another score table and portfolio.

## 5) Merge ranking tables

The pipeline merges ranked tables:
- Start from the world/news ranking as the base
- Add social columns, prefixing collisions with `social_...`

## 6) Add driver tags (quick narrative labeling)

For each ticker, the pipeline reads the driver report summary signals and assigns a tag such as:
- `stable`
- `jump-driven`
- `regime-shift`

These tags are meant to be “advisor friendly” quick categorization.

## 7) Apply advisor constraints (optional)

If `client_constraints` is provided:

### 7.1 Hard filters (best-effort)
- Max drawdown filter
- Dividend yield constraints (only enforced if a yield column exists)
- Sector exclusions (only enforced if a sector column exists)

Every exclusion is written to a compliance object with:
- `ticker`
- `reason`
- the value that triggered exclusion (when available)

### 7.2 Soft diversification cap (best-effort)

If `max_sector_fraction_in_top` is set and the table has:
- sector column, and
- `Final_Score`

…then the pipeline enforces a soft sector diversification cap inside the top N.

If required columns are missing, it skips this step and records a warning.

### 7.3 Constrained artifacts

When constraints are applied, the pipeline writes:
- `ranked_world_constrained_<timestamp>.csv`

This is a practical “advisor output” you can share or archive.

## 8) Optional advanced quant + NLG

If enabled, the pipeline lazily imports and runs:
- `stock_picker_advanced_quantitative` (deeper quant features)
- `stock_picker_nlg_explanations` (human-readable narrative write-up)

These are imported lazily to keep the default run lighter.

## 9) Write consolidated report

The pipeline writes a consolidated JSON report including:
- inputs / parameters
- per-ticker driver report paths and short summaries
- merged ranking tables + portfolios
- compliance report (if constraints were applied)
- audit metadata

## 10) Audit and reproducibility

The advisor fork includes an `audit` block with:
- `git_sha` (best-effort)
- Python version
- platform info
- UTC timestamp

This is intended for:
- internal review
- client file retention
- “show your work” style compliance
