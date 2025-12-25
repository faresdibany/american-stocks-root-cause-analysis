# Root Cause Analysis (per-stock)

This folder adds a **movement driver analysis** layer on top of the existing `american stocks` pipeline.

## What you get (per ticker)

The analysis attempts to explain price movement as:

- **Trend**: slow component (often fundamentals / persistent narrative)
- **Volatility regimes**: low/medium/high risk-on vs risk-off segments
- **Jumps / shocks**: discrete abnormal returns
- **Noise**: residual

Then it performs a lightweight **event attribution** step for jump dates:

- Checks proximity to **earnings dates** (best-effort via `yfinance`)
- (Optional future) attach headlines and compute embedding similarity

## Outputs

Files are written to:

- `american stocks root cause analysis/outputs/driver_report_<TICKER>_<timestamp>.json`

Each JSON includes:

- `trend_summary` (total + annualized return, residual volatility)
- `volatility_regimes` (segments labeled low/medium/high)
- `jumps` (date, return, z-score)
- `event_attribution` (earnings calendar + jump annotations)

## Run

From:

- `american stocks root cause analysis/core_analysis/`

Run batch:

```powershell
python .\driver_analysis_batch.py --tickers AAPL MSFT NVDA --start 2022-01-01 --end 2025-12-25
```

Run single:

```powershell
python .\stock_driver_analysis.py --ticker AAPL --start 2022-01-01 --end 2025-12-25
```

## Notes

- This is a **first iteration** designed to be robust and dependency-light.
- Next upgrades:
  - Change-point detection (PELT) for structural breaks
  - HMM regimes for returns/volatility
  - Event studies with benchmark adjustment (SPY) for abnormal return attribution
  - NLP: headline clustering + embedding similarity vs jump windows
