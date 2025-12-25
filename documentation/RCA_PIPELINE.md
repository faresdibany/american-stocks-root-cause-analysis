# RCA Pipeline Documentation

## Overview

The **Root Cause Analysis (RCA) Pipeline** (`rca_pipeline.py`) is a comprehensive end-to-end stock analysis system that combines multiple analytical approaches to identify and explain stock price movements. It integrates historical price driver analysis, quantitative ranking, AI-powered news analysis, and social sentiment monitoring into a unified workflow.

## Table of Contents

- [Purpose](#purpose)
- [Design Philosophy](#design-philosophy)
- [Pipeline Architecture](#pipeline-architecture)
- [Core Components](#core-components)
- [Data Flow](#data-flow)
- [Output Artifacts](#output-artifacts)
- [Usage](#usage)
- [Configuration Options](#configuration-options)
- [Technical Details](#technical-details)

---

## Purpose

The RCA Pipeline answers the fundamental question: **"What drove this stock's price movement?"**

By analyzing multiple dimensions simultaneously, it provides:
- **Explanatory power**: Understanding why prices moved
- **Predictive signals**: Identifying potential future opportunities
- **Risk assessment**: Characterizing volatility regimes and shock patterns
- **Sentiment analysis**: Gauging market and social media sentiment

---

## Design Philosophy

### Core Principles

1. **Independence**: Runs as a standalone pipeline without tight coupling to other systems
2. **Optional Dependencies**: Gracefully degrades when optional packages are unavailable
3. **Artifact-Centric**: All outputs saved to `../outputs/` directory for reproducibility
4. **Modular Integration**: Orchestrates existing modules rather than reimplementing logic
5. **Comprehensive Coverage**: Combines technical, fundamental, and sentiment analysis

### Design Goals

- Keep dependencies optional wherever possible
- Avoid tight coupling between components
- Always write artifacts to a consistent location
- Provide both machine-readable (JSON/CSV) and human-readable (Markdown) outputs

---

## Pipeline Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                     RCA PIPELINE ENTRY                       │
│                  (rca_pipeline.py::main)                     │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ├──► Input: List of tickers
                         ├──► Config: period, interval, flags
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                    STAGE 1: DRIVER ANALYSIS                  │
│              (stock_driver_analysis.py)                      │
│                                                              │
│  For each ticker:                                            │
│    ├─ Decompose price into components                       │
│    ├─ Detect volatility regimes                             │
│    ├─ Identify jumps/shocks                                 │
│    ├─ Find change points                                    │
│    ├─ Attribute abnormal returns to events                  │
│    └─ Generate driver_report_{ticker}_{timestamp}.json      │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│              STAGE 2: QUANTITATIVE + AI NEWS                 │
│    (stock_picker_hybrid_world_news_extension.py)            │
│                                                              │
│    ├─ Calculate quantitative metrics (Sharpe, CAGR, etc.)   │
│    ├─ Fetch and analyze world/company news                  │
│    ├─ Compute sentiment scores                              │
│    ├─ Run portfolio optimization                            │
│    └─ Generate ranked DataFrame                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                STAGE 3: SOCIAL SENTIMENT                     │
│           (stock_picker_social_sentiment.py)                 │
│                                                              │
│    ├─ Aggregate news sentiment                              │
│    ├─ Scrape Reddit discussions (if PRAW available)         │
│    ├─ Fetch StockTwits data (best-effort)                   │
│    ├─ Combine social signals                                │
│    └─ Generate ranked DataFrame                             │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                STAGE 4: MERGE & CONSOLIDATE                  │
│                                                              │
│    ├─ Merge quantitative and social rankings                │
│    ├─ Add driver tags (stable, jump-driven, etc.)           │
│    ├─ Sort by Final_Score                                   │
│    └─ Create unified DataFrame                              │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│          STAGE 5: OPTIONAL ADVANCED PROCESSING               │
│                                                              │
│    [If --with-advanced-quant]:                              │
│    ├─ Run advanced quantitative analysis                    │
│    ├─ Include GARCH, regime detection, factor models        │
│    └─ Save additional CSV artifacts                         │
│                                                              │
│    [If --with-nlg]:                                         │
│    ├─ Run natural language generation                       │
│    ├─ Generate human-readable explanations                  │
│    └─ Save explanations_nlg_{timestamp}.txt                 │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│            STAGE 6: GENERATE EXPLANATIONS                    │
│                                                              │
│    ├─ Create Markdown explanations file                     │
│    ├─ Summarize driver analysis findings                    │
│    ├─ Include quantitative metrics                          │
│    ├─ Incorporate sentiment scores                          │
│    └─ Generate narrative summaries per ticker               │
└────────────────────────┬────────────────────────────────────┘
                         │
                         ▼
┌─────────────────────────────────────────────────────────────┐
│                   OUTPUT ARTIFACTS                           │
│                    (../outputs/)                             │
│                                                              │
│    ├─ driver_report_{ticker}_{timestamp}.json (per ticker)  │
│    ├─ ranked_signals_rca_{timestamp}.csv                    │
│    ├─ rca_pipeline_report_{timestamp}.json                  │
│    ├─ explanations_rca_{timestamp}.md                       │
│    └─ [Optional advanced/NLG artifacts]                     │
└─────────────────────────────────────────────────────────────┘
```

---

## Core Components

### 1. Driver Analysis (`stock_driver_analysis.py`)

**Purpose**: Decompose price movements into interpretable components

**Key Operations**:
- **Trend Extraction**: Uses Hodrick-Prescott filter or STL decomposition to separate trend from noise
- **Volatility Regimes**: Detects periods of high/medium/low volatility using rolling windows
- **Jump Detection**: Identifies significant single-day price moves (Z-score based)
- **Change Point Detection**: Finds structural breaks in price series
- **Event Attribution**: Links jumps to potential catalysts using SPY benchmark for abnormal returns

**Outputs**:
```json
{
  "ticker": "AAPL",
  "start": "2024-01-01",
  "end": "2024-12-25",
  "trend_summary": {
    "total_return": 0.45,
    "trend_contribution": 0.38,
    "volatility_mean": 0.22
  },
  "volatility_regimes": [
    {
      "start": "2024-01-01",
      "end": "2024-03-15",
      "label": "low",
      "vol_annualized": 0.18
    }
  ],
  "jumps": [
    {
      "date": "2024-02-01",
      "ret": 0.075,
      "zscore": 3.2,
      "direction": "up"
    }
  ],
  "event_attribution": {
    "abnormal_returns": {
      "available": true,
      "beta": 1.15,
      "benchmark": "SPY"
    }
  },
  "change_points": {
    "points": [
      {
        "date": "2024-06-15",
        "score": 2.8
      }
    ]
  }
}
```

### 2. Quantitative + AI News Analysis

**Module**: `stock_picker_hybrid_world_news_extension.py`

**Purpose**: Combine traditional metrics with sentiment analysis

**Features**:
- Calculates CAGR, Sharpe ratio, maximum drawdown
- Fetches recent news via RSS feeds
- Performs sentiment analysis on headlines
- Computes topic momentum scores
- Optimizes portfolio weights
- Runs backtests

**Key Metrics**:
- `Final_Score`: Composite ranking score
- `CAGR`: Compound Annual Growth Rate
- `Sharpe`: Risk-adjusted returns
- `CombinedSentiment`: News sentiment score
- `TopicMomentum`: Trending topic strength

### 3. Social Sentiment Analysis

**Module**: `stock_picker_social_sentiment.py`

**Purpose**: Gauge crowd sentiment from multiple sources

**Data Sources**:
- **News**: RSS feeds (Google News, Yahoo Finance)
- **Reddit**: Via PRAW API (optional, requires credentials)
- **StockTwits**: Web scraping (best-effort, often blocked)

**Aggregation**:
- Configurable source weights (default: 50% news, 50% Reddit, 0% StockTwits)
- Combines sentiment scores across sources
- Counts total mentions, posts, and tweets
- Generates composite social signal

### 4. Merge & Consolidation

**Purpose**: Unify all analytical dimensions

**Process**:
1. Start with world-news ranked DataFrame
2. Merge social sentiment columns (with prefixes to avoid collisions)
3. Add driver tags from Stage 1:
   - `stable`: No strong jump or regime signals
   - `jump-driven`: Multiple significant jumps detected
   - `regime-shift`: Change points indicate regime changes
   - `events+regimes`: Both jumps and change points present
4. Sort by `Final_Score` (descending)

### 5. Optional Advanced Quantitative Analysis

**Module**: `stock_picker_advanced_quantitative.py`

**Purpose**: Deep statistical modeling (optional heavy dependencies)

**Features** (when enabled with `--with-advanced-quant`):
- GARCH models for volatility forecasting
- Regime detection algorithms
- Factor model analysis
- Technical indicators (TALib)
- Advanced statistical tests

### 6. Optional Natural Language Generation

**Module**: `stock_picker_nlg_explanations.py`

**Purpose**: Generate human-readable narratives

**Features** (when enabled with `--with-nlg`):
- Uses transformer models for text generation
- Creates detailed explanations per ticker
- Synthesizes insights across all analytical dimensions
- Outputs plain text summaries

---

## Data Flow

### Input Processing

1. **Ticker List**: Comma-separated string converted to list
2. **Period/Interval**: Parsed into concrete start/end dates
3. **Configuration**: Parameters for news lookback, sentiment weights, etc.

### Inter-Stage Communication

Each stage produces outputs that subsequent stages can consume:

- **Stage 1 → Report Metadata**: Driver JSON paths stored in report dictionary
- **Stage 2 → Merged DataFrame**: Ranked DataFrame becomes merge base
- **Stage 3 → Merged DataFrame**: Social columns added to base DataFrame
- **Stages 4-6 → File System**: All artifacts written to `../outputs/`

### Output Generation

All outputs use timestamped filenames (`{type}_{timestamp}.{ext}`) to prevent overwrites.

---

## Output Artifacts

### Primary Outputs

#### 1. Driver Reports (Per Ticker)
**File**: `driver_report_{ticker}_{timestamp}.json`

Contains detailed decomposition of price movements for a single ticker.

#### 2. Merged Rankings
**File**: `ranked_signals_rca_{timestamp}.csv`

Consolidated DataFrame with all metrics and rankings across tickers.

**Key Columns**:
- `Final_Score`: Overall ranking score
- `CAGR`, `Sharpe`, `MaxDrawdown`: Quantitative metrics
- `CombinedSentiment`, `TopicMomentum`: News sentiment
- `social_CombinedSentiment`, `TotalMentions`: Social metrics
- `driver_tag`: Driver classification

#### 3. Pipeline Report
**File**: `rca_pipeline_report_{timestamp}.json`

Master report containing:
- Timestamp and configuration
- Artifact file paths
- Driver report references
- Portfolio weights and backtest results
- Warnings/errors

#### 4. Explanations (Markdown)
**File**: `explanations_rca_{timestamp}.md`

Human-readable report with:
- Ranked list of tickers
- Driver analysis highlights
- Quantitative metrics summary
- Social sentiment overview
- Narrative interpretation

**Example Structure**:
```markdown
# RCA Pipeline Explanations

Generated: 2024-12-25 14:30:00

Tickers: AAPL, MSFT, NVDA, AMZN, GOOGL

## #1 — AAPL

**Driver analysis:** events+regimes

- Driver report: `../outputs/driver_report_AAPL_20241225_143000.json`
- Driver highlights: Total return: 45.0%; Volatility regime mostly: low; Jumps detected: 3; Top moves: 2024-02-01 (7.5%), 2024-08-15 (5.2%); Beta vs SPY: 1.15

**Quant + world news:**

- Final_Score: 0.8523
- CAGR: 0.4234
- Sharpe: 1.87
- CombinedSentiment: 0.15
- TopicMomentum: 0.67

**Social sentiment:**

- CombinedSentiment: 0.12
- TopicMomentum: 0.58
- TotalMentions: 347
- NewsCount: 45
- RedditCount: 302

**Summary:** Price action looks like a mix of event shocks and regime changes (not pure noise). News sentiment is positive (world+company combined). Crowd/social tone is positive (best-effort sources).
```

### Optional Outputs

#### Advanced Quantitative Artifacts (if `--with-advanced-quant`)
- `ranked_signals_adv_quant_{timestamp}.csv`
- `portfolio_adv_quant_{timestamp}.csv`
- `fundamentals_adv_quant_{timestamp}.csv`
- `quant_signals_adv_quant_{timestamp}.csv`
- `ai_signals_adv_quant_{timestamp}.csv`

#### NLG Artifacts (if `--with-nlg`)
- `ranked_signals_nlg_{timestamp}.csv`
- `portfolio_nlg_{timestamp}.csv`
- `fundamentals_nlg_{timestamp}.csv`
- `ai_signals_nlg_{timestamp}.csv`
- `explanations_nlg_{timestamp}.txt`

---

## Usage

### Basic Usage

Run from the `core_analysis` folder:

```powershell
python .\core_analysis\rca_pipeline.py
```

This uses default tickers: `AAPL,MSFT,NVDA,AMZN,GOOGL`

### Custom Tickers

```powershell
python .\core_analysis\rca_pipeline.py --tickers TSLA,AMD,INTC
```

### With Advanced Features

```powershell
# Include advanced quantitative analysis
python .\core_analysis\rca_pipeline.py --with-advanced-quant

# Include natural language generation
python .\core_analysis\rca_pipeline.py --with-nlg

# Both advanced features
python .\core_analysis\rca_pipeline.py --with-advanced-quant --with-nlg

# Skip Markdown explanations
python .\core_analysis\rca_pipeline.py --no-explanations
```

### Full Example

```powershell
python .\core_analysis\rca_pipeline.py `
  --tickers AAPL,MSFT,GOOGL,TSLA,NVDA,AMD,META,AMZN `
  --period 24mo `
  --interval 1d `
  --max-news 20 `
  --lookback-days 30 `
  --with-advanced-quant `
  --with-nlg
```

---

## Configuration Options

### Command-Line Arguments

| Argument | Type | Default | Description |
|----------|------|---------|-------------|
| `--tickers` | string | `AAPL,MSFT,NVDA,AMZN,GOOGL` | Comma-separated ticker symbols |
| `--period` | string | `12mo` | Historical data window (e.g., `6mo`, `1y`, `24mo`) |
| `--interval` | string | `1d` | Data granularity (`1d`, `1h`, `5m`) |
| `--max-news` | int | `12` | Maximum news articles per ticker |
| `--lookback-days` | int | `14` | Days to look back for news/sentiment |
| `--no-explanations` | flag | False | Skip generating explanations_rca_*.md |
| `--with-advanced-quant` | flag | False | Run advanced quantitative analysis |
| `--with-nlg` | flag | False | Run natural language generation |

### Programmatic Configuration

```python
from core_analysis.rca_pipeline import run_pipeline

result = run_pipeline(
    tickers=["AAPL", "MSFT", "GOOGL"],
    period="12mo",
    interval="1d",
    max_news=15,
    lookback_days=21,
    social_max_news=20,
    social_max_reddit=100,
    social_max_stocktwits=0,
    generate_explanations=True,
    with_advanced_quant=True,
    with_nlg=False,
)

merged_df = result["merged"]
report = result["report"]
```

---

## Technical Details

### Driver Tag Classification

The pipeline assigns a `driver_tag` to each ticker based on detected patterns:

```python
if jumps >= 2 and change_points >= 2:
    tag = "events+regimes"
elif change_points >= 2:
    tag = "regime-shift"
elif jumps >= 2:
    tag = "jump-driven"
else:
    tag = "stable"
```

### Sentiment Threshold Logic

Narratives are generated when sentiment crosses significance thresholds:

- **World News**: `|CombinedSentiment| >= 0.10`
- **Social Media**: `|CombinedSentiment| >= 0.05`

### Error Handling

The pipeline is designed to be resilient:

1. **Optional Dependencies**: Missing packages trigger graceful degradation
2. **API Failures**: Social sentiment degrades if Reddit/StockTwits unavailable
3. **Data Issues**: Empty DataFrames handled with safe fallbacks
4. **Warnings Array**: Non-fatal errors captured in report's `warnings` list

### Performance Considerations

- **Driver Analysis**: O(n × m) where n = tickers, m = observations
- **News Fetching**: Network-bound, benefits from connection pooling
- **Social Scraping**: Rate-limited by source APIs
- **NLG Generation**: GPU-accelerated if CUDA available

### Dependencies

**Core (required)**:
- `pandas`
- `numpy`
- `yfinance`

**Statistical (optional)**:
- `statsmodels` (for HP filter, STL decomposition)

**Advanced Quantitative (optional)**:
- `arch` (GARCH models)
- `ta-lib` (technical indicators)
- `scikit-learn` (factor models)

**NLG (optional)**:
- `transformers` (Hugging Face models)
- `torch` (PyTorch backend)

**Social Sentiment (optional)**:
- `praw` (Reddit API)
- `beautifulsoup4` (web scraping)

---

## Best Practices

### 1. Start Simple

Begin with default settings to understand baseline behavior:

```powershell
python .\core_analysis\rca_pipeline.py --tickers AAPL,MSFT
```

### 2. Incremental Feature Enablement

Add features one at a time to isolate their impact:

```powershell
# First run: baseline
python .\core_analysis\rca_pipeline.py --tickers AAPL

# Second run: add advanced quant
python .\core_analysis\rca_pipeline.py --tickers AAPL --with-advanced-quant

# Third run: add NLG
python .\core_analysis\rca_pipeline.py --tickers AAPL --with-nlg
```

### 3. Optimize for Your Use Case

- **Quick screening**: Use default settings with 5-10 tickers
- **Deep analysis**: Enable `--with-advanced-quant` for 1-3 tickers
- **Client reports**: Enable `--with-nlg` for polished narratives

### 4. Monitor Output Directory

Artifacts accumulate in `../outputs/`. Periodically archive or clean old files.

### 5. API Key Management

For social sentiment, configure environment variables:

```powershell
$env:REDDIT_CLIENT_ID = "your_client_id"
$env:REDDIT_CLIENT_SECRET = "your_secret"
$env:REDDIT_USER_AGENT = "your_app_name"
```

---

## Troubleshooting

### Common Issues

#### "yfinance is required"
**Solution**: Install dependencies
```powershell
pip install -r requirements.txt
```

#### Empty driver reports
**Cause**: Insufficient historical data
**Solution**: Increase `--period` or verify tickers are valid

#### Social sentiment all zeros
**Cause**: Reddit API keys missing or StockTwits blocking
**Solution**: Configure PRAW credentials or ignore social component

#### NLG stage fails
**Cause**: Missing transformers/torch packages
**Solution**: Install optional dependencies
```powershell
pip install transformers torch
```

### Debug Mode

Enable verbose output by modifying the script (no built-in flag yet):

```python
# In rca_pipeline.py, add print statements
print(f"DEBUG: Driver reports: {driver_reports}")
print(f"DEBUG: Merged shape: {merged.shape}")
```

---

## Comparison with Other Modules

| Feature | RCA Pipeline | Basic Stock Picker | Advanced Quant | NLG Explanations |
|---------|--------------|-------------------|----------------|------------------|
| Driver Analysis | ✅ Core feature | ❌ | ❌ | ❌ |
| Quantitative Metrics | ✅ Via world-news | ✅ Basic | ✅ Advanced | ✅ Via advanced |
| News Sentiment | ✅ | ✅ | ✅ | ✅ |
| Social Sentiment | ✅ | ❌ | ❌ | ❌ |
| GARCH/Factor Models | ⚠️ Optional | ❌ | ✅ | ✅ |
| Human Narratives | ✅ Markdown | ❌ | ❌ | ✅ Advanced NLG |
| Portfolio Optimization | ✅ | ❌ | ✅ | ✅ |
| Event Attribution | ✅ Unique | ❌ | ❌ | ❌ |

**Legend**: ✅ Included, ❌ Not included, ⚠️ Optional

---

## Future Enhancements

### Planned Features

1. **Real-time Monitoring**: Webhook support for price alerts
2. **ML Predictions**: Train models on historical driver patterns
3. **Custom Benchmarks**: Support for sector-specific benchmarks beyond SPY
4. **Multi-timeframe Analysis**: Combine daily/hourly/minute data
5. **Database Integration**: Store results in PostgreSQL/MongoDB
6. **Web Dashboard**: Flask/Dash interface for interactive exploration

### Contribution Areas

- Improve social sentiment reliability
- Add more event sources (SEC filings, insider trades)
- Implement regime-switching models
- Enhance NLG with fine-tuned finance models

---

## References

### Related Documentation

- [ARCHITECTURE.md](./ARCHITECTURE.md): Overall system design
- [ROOT_CAUSE_ANALYSIS.md](./ROOT_CAUSE_ANALYSIS.md): Theoretical background
- [WORKFLOW.md](./WORKFLOW.md): Step-by-step operational guide

### External Resources

- [yfinance Documentation](https://pypi.org/project/yfinance/)
- [Statsmodels HP Filter](https://www.statsmodels.org/stable/generated/statsmodels.tsa.filters.hp_filter.hpfilter.html)
- [PRAW Reddit API](https://praw.readthedocs.io/)

---

## License

This pipeline is part of the American Stocks Root Cause Analysis project. Refer to the main repository for licensing terms.

---

## Contact & Support

For questions, issues, or contributions:
- **Repository**: [american-stocks-root-cause-analysis](https://github.com/faresdibany/american-stocks-root-cause-analysis)
- **Owner**: faresdibany

---

*Last Updated: December 25, 2024*
