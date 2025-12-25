# US Stocks Hybrid Picker - Workflow Guide

This document provides step-by-step operational guidance for running the American stocks analysis scripts, understanding their purposes, managing artifacts, and maintaining data quality.

## 1. Folder Overview

The `american stocks root cause analysis` folder contains multiple script variants optimized for US equity markets (NYSE, NASDAQ):

| Script | Primary Purpose | When to Use |
|--------|----------------|-------------|
| `stock_picker_hybrid_american_stocks.py` | **Main US market analyzer** with sector-specific news categories | Daily full analysis of US portfolios |
| `stock_picker_hybrid_world_news_extension.py` | Extended world news integration with comprehensive macro categories | When macro/geopolitical events are significant |
| `stock_picker_daily.py` | Lightweight daily update script (faster, fewer headlines) | Quick daily refresh without full recomputation |
| `stock_picker_hybrid.py` | Generic hybrid picker (flexible market application) | Custom ticker lists or experimental runs |
| `stock_picker.py` | Current production version with all features | Standard production workflow |
| `original_stock_picker.py` | Reference implementation (historical baseline) | Comparison/validation against original methodology |

Generated artifacts: timestamped CSV files for rankings, quant signals, AI signals, and directional warnings.

## 2. Prerequisites & Environment Setup

### Required Python Packages
```bash
pip install pandas numpy yfinance feedparser transformers torch scikit-learn matplotlib
```

### Optional Packages (fallback data sources)
```bash
pip install pandas-datareader alpha_vantage
```

### Environment Variables (Optional)
- `ALPHAVANTAGE_API_KEY`: For Alpha Vantage fallback (if yfinance unavailable).
- None required for FinBERT (downloads automatically on first use).

### System Requirements
- Python ≥ 3.10
- Internet connectivity (for Yahoo Finance API, Google News RSS)
- 4GB+ RAM (FinBERT model loading)
- ~2GB disk space (transformers cache)

## 3. Core Architecture Overview

### Data Flow Sequence
```
1. Configuration (tickers, period, lookback)
   ↓
2. Price Data Acquisition (yfinance → stooq → alphavantage fallback chain)
   ↓
3. Quantitative Metrics (CAGR, Sharpe, Volatility, MaxDD)
   ↓
4. World/Macro News Fetch (8 US-specific categories)
   ↓
5. Ticker-Specific Headlines (Google News RSS)
   ↓
6. Sentiment Analysis (FinBERT)
   ↓
7. Topic Momentum (TF-IDF financial terms)
   ↓
8. Score Fusion (weighted combination of quant + AI)
   ↓
9. Ranking & Portfolio Selection (top N equal-weight)
   ↓
10. Backtest & Performance Metrics
   ↓
11. Directional Warning Assistant (risk signals)
   ↓
12. Artifact Persistence (timestamped CSVs)
```

### US Market-Specific Adaptations
- **News Categories**: economy, geopolitics, energy, finance, technology, consumer, healthcare, industrial
- **Sector Heuristics**: Tech (AAPL, MSFT, GOOGL, etc.), Finance (JPM, BAC, GS), Energy (XOM, CVX, COP), Healthcare (JNJ, PFE, UNH)
- **Sentiment Weighting**: 70% stock-specific, 30% world news (vs. 60/40 for EGX)
- **Exchange Context**: Includes NYSE/NASDAQ in search queries

## 4. Daily Workflow (Recommended)

### Morning Routine (Before Market Open - 9:00 AM ET)
```powershell
# Navigate to folder
cd "C:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"

# Quick daily update (faster, 12 headlines/ticker)
python .\stock_picker_daily.py
```

**Output Files:**
- `daily_ranked_{timestamp}.csv` - Today's rankings
- `daily_news_sentiment.csv` - Appended daily sentiment log

### Full Analysis (Weekly or On-Demand)
```powershell
# Comprehensive analysis (50 headlines/ticker, 14-day lookback)
python .\stock_picker_hybrid_american_stocks.py
```

**Output Files:**
- `rankings_us_{timestamp}.csv` - Full rankings with all metrics
- `quant_signals_us_{timestamp}.csv` - Technical indicators
- `ai_signals_us_{timestamp}.csv` - Sentiment & momentum scores
- `directional_warnings_us.csv` - Risk/opportunity flags

### World News Deep Dive (During Major Events)
```powershell
# Extended world news analysis (120 headlines/ticker)
python .\stock_picker_hybrid_world_news_extension.py
```

**Additional Categories:** More granular geopolitical, commodity, and sector-specific news.

## 5. Step-by-Step Execution Guide

### Step 1: Configure Target Universe
Edit the main section of your chosen script:

```python
cfg = Config(
    tickers=[
        # Modify this list for your universe
        "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",  # Tech
        "JPM", "BAC", "GS",                                  # Finance
        "JNJ", "PFE", "UNH",                                # Healthcare
        "XOM", "CVX", "COP",                                # Energy
        "WMT", "HD", "MCD"                                  # Consumer
    ],
    period="1y",           # Price history window
    interval="1d",         # Daily bars
    top_k=10,             # Portfolio size
    max_news=50,          # Headlines per ticker
    lookback_days=14      # News recency window
)
```

### Step 2: Run Script
```powershell
python .\stock_picker_hybrid_american_stocks.py
```

### Step 3: Monitor Console Output
```
Loading US market price data...
Computing quantitative signals...
Computing AI signals...

[AI] Processing AAPL ...
  headlines found: 48
  stock sentiment: 0.234
  world impact: 0.156
  combined: 0.211

[AI] Processing MSFT ...
...

Ranked US stocks:
         CAGR  Sharpe  CombinedSentiment  Final_Score
NVDA   0.523   1.234          0.345         0.876
AAPL   0.312   0.987          0.211         0.743
...

Results saved to:
- rankings_us_20251125_093022.csv
- quant_signals_us_20251125_093022.csv
- ai_signals_us_20251125_093022.csv

=== Directional Warnings (US, horizon: 5d) ===
   Ticker  Signal  Probability  StopLoss  TakeProfit
0    NVDA     BUY        0.723     145.2       158.9
1    TSLA   WATCH        0.543       N/A         N/A
```

### Step 4: Review Output Files
Open CSV files in Excel or pandas:
```python
import pandas as pd
ranked = pd.read_csv("rankings_us_20251125_093022.csv")
print(ranked.head(10))  # Top 10 picks
```

### Step 5: Generate Portfolio (Optional)
If using for live trading, extract top N tickers:
```python
portfolio = ranked.head(10)
portfolio["Weight"] = 1.0 / len(portfolio)  # Equal weight
portfolio.to_csv("portfolio_live.csv", index=False)
```

## 6. Understanding Output Artifacts

### Quant Signals (`quant_signals_us_*.csv`)
| Column | Description | Interpretation |
|--------|-------------|----------------|
| Ticker | Stock symbol | - |
| CAGR | Compound Annual Growth Rate | Higher is better (momentum proxy) |
| Sharpe | Risk-adjusted return | Higher is better (>1.0 is strong) |
| Volatility | Annualized standard deviation | Lower is better (risk measure) |
| MaxDD | Maximum drawdown | More negative = riskier |

### AI Signals (`ai_signals_us_*.csv`)
| Column | Description | Interpretation |
|--------|-------------|----------------|
| Sentiment | Combined sentiment score | -1 (bearish) to +1 (bullish) |
| StockSentiment | Company-specific news | Direct headline analysis |
| WorldNewsScore | Macro/sector sentiment | Weighted by relevance |
| TopicMomentum | Bullish term density | TF-IDF score [0,1] |
| Headlines | Count of articles | Higher = more coverage |
| Sent_Pos/Neutral/Neg | Polarity counts | Distribution check |
| Sent_Examples_* | Sample headlines | Manual validation |
| World_economy, World_finance, ... | Category scores | Sector drill-down |

### Rankings (`rankings_us_*.csv`)
| Column | Description | Use Case |
|--------|-------------|----------|
| *_n | Normalized metrics | Cross-sectional comparison |
| Final_Score | Weighted composite | Primary ranking criterion |
| All quant + AI columns | Full context | Deep dive analysis |

### Directional Warnings (`directional_warnings_us.csv`)
| Column | Description | Action |
|--------|-------------|--------|
| Signal | BUY/SELL/WATCH | Trading direction |
| Probability | Confidence [0,1] | Threshold: >0.60 for action |
| StopLoss | Risk exit price | Set protective stop |
| TakeProfit | Target exit price | Take profit level |

## 7. Workflow Variants by Use Case

### Use Case 1: Daily Morning Update (5 minutes)
```powershell
# Fast refresh for intraday awareness
python .\stock_picker_daily.py

# Check warnings only
python -c "import pandas as pd; print(pd.read_csv('directional_warnings_us.csv'))"
```

### Use Case 2: Weekly Rebalance (30 minutes)
```powershell
# Full analysis
python .\stock_picker_hybrid_american_stocks.py

# Review top 20
python -c "import pandas as pd; df=pd.read_csv('rankings_us_*'); print(df.head(20))"

# Export portfolio
python -c "import pandas as pd; df=pd.read_csv('rankings_us_*').head(10); df.to_csv('portfolio_week.csv')"
```

### Use Case 3: Market Crisis Deep Dive (1 hour)
```powershell
# Extended world news
python .\stock_picker_hybrid_world_news_extension.py

# Analyze sentiment trends
python -c "
import pandas as pd
daily = pd.read_csv('daily_news_sentiment.csv')
daily['Date'] = pd.to_datetime(daily['Date'])
recent = daily[daily['Date'] >= pd.Timestamp.now() - pd.Timedelta(days=7)]
print(recent.groupby('Ticker')['Sentiment'].mean().sort_values())
"
```

### Use Case 4: Custom Sector Analysis
Modify script ticker list to sector-specific:
```python
# Tech-only analysis
cfg = Config(
    tickers=["AAPL", "MSFT", "GOOGL", "META", "NVDA", "AVGO", "ADBE", "CRM", "ORCL", "NFLX"],
    max_news=100,  # More coverage for focused universe
    lookback_days=7  # Shorter window for fast-moving sector
)
```

## 8. Data Quality & Maintenance

### Daily Checks
- [ ] All tickers returned price data (check console for missing symbols)
- [ ] Headline counts > 0 for major stocks (AAPL, MSFT should have 20+)
- [ ] No exceptions/errors in console output
- [ ] Output files created with today's timestamp
- [ ] `directional_warnings_us.csv` generated (even if empty)

### Weekly Maintenance
- [ ] Archive old CSV files: `mkdir archive\week_XX; mv *_{timestamp}.csv archive\week_XX\`
- [ ] Review sentiment trend: check for persistent negative shifts
- [ ] Validate portfolio performance: compare to benchmark (SPY)

### Monthly Tasks
- [ ] Clear transformers cache if disk space low: `rm -rf ~/.cache/huggingface/`
- [ ] Update ticker universe: add new IPOs, remove delisted
- [ ] Review false positives: inspect `Sent_Examples_Pos/Neg` for noise
- [ ] Recalibrate warning thresholds if excessive false signals

## 9. Troubleshooting Guide

### Issue: No headlines for any ticker
**Cause:** `feedparser` not installed or network blocked  
**Fix:**
```powershell
pip install feedparser
# Test connectivity
python -c "import feedparser; print(feedparser.parse('https://news.google.com/rss?hl=en').entries[0].title)"
```

### Issue: All sentiment scores = 0.0
**Cause:** FinBERT not loaded (transformers/torch missing)  
**Fix:**
```powershell
pip install transformers torch
# Force download
python -c "from transformers import pipeline; pipeline('sentiment-analysis', model='ProsusAI/finbert')"
```

### Issue: Price data missing for some tickers
**Cause:** Delisted, wrong symbol, or yfinance API issue  
**Fix:**
- Verify ticker on Yahoo Finance website
- Check console for "No data found" messages
- Try fallback: set provider in script or use Alpha Vantage

### Issue: Script runs but rankings look random
**Cause:** Insufficient price history or all scores near zero  
**Fix:**
- Increase `period` to "2y" for better Sharpe computation
- Check if market was flat (low volatility = compressed scores)
- Verify `max_news` is adequate (50+ recommended)

### Issue: Slow execution (>30 minutes)
**Cause:** Too many tickers or headlines  
**Fix:**
- Reduce `max_news` to 30
- Limit ticker count to 20-30
- Use `stock_picker_daily.py` instead
- Batch tickers: run 2 separate analyses

### Issue: Warnings file empty every time
**Cause:** No signals met probability thresholds  
**Fix:**
- Lower `prob_buy` / `prob_sell` in `PredictConfig` (e.g., 0.55 instead of 0.60)
- Increase `lookback_days` for more headline data
- Check if sentiment scores are too neutral (need stronger signals)

## 10. Performance Optimization

### Speed Improvements
1. **Use daily script** for routine updates (12x faster than full analysis)
2. **Cache FinBERT model**: First run downloads ~500MB, subsequent runs are instant
3. **Parallel ticker processing** (advanced): Modify script to use `multiprocessing`
4. **Reduce headline count**: 30 headlines often sufficient, saves 40% time

### Memory Management
- Close other applications before running (FinBERT needs 2-3GB)
- If system crashes, reduce ticker count to 10-15
- Use `stock_picker_daily.py` which has lower memory footprint

### Result Quality
- **Minimum 1 year history**: Sharpe needs sufficient samples
- **At least 20 headlines/ticker**: Sentiment needs statistical significance
- **Diversified universe**: 15+ tickers across 3+ sectors
- **Regular updates**: Daily runs capture trend shifts better than weekly

## 11. Integration with Trading Systems

### Export for Manual Trading
```powershell
# Top 10 with weights and key metrics
python -c "
import pandas as pd
df = pd.read_csv('rankings_us_*.csv')
portfolio = df.head(10)[['Ticker', 'CAGR', 'Sharpe', 'Final_Score']]
portfolio['Weight'] = 0.1  # Equal weight
print(portfolio.to_string(index=False))
" > portfolio_today.txt
```

### API Integration Example
```python
# For automated trading systems
import pandas as pd
from datetime import datetime

def get_latest_signals():
    # Find most recent rankings file
    import glob
    files = glob.glob('rankings_us_*.csv')
    latest = max(files, key=lambda x: x.split('_')[-1])
    
    df = pd.read_csv(latest)
    signals = df.head(10)[['Ticker', 'Final_Score', 'Sentiment', 'Sharpe']]
    signals['Timestamp'] = datetime.now()
    
    return signals.to_dict('records')
```

### Risk Management Integration
```python
# Generate stop-loss orders from warnings
import pandas as pd

warnings = pd.read_csv('directional_warnings_us.csv')
stops = warnings[warnings['Signal'] == 'BUY'][['Ticker', 'StopLoss']]

for _, row in stops.iterrows():
    print(f"Set stop for {row['Ticker']} at ${row['StopLoss']:.2f}")
```

## 12. Advanced Usage

### Custom Weight Optimization
Modify `Weights` in script:
```python
# More emphasis on sentiment during earnings season
weights = Weights(
    CAGR=0.35,         # Reduce momentum weight
    Sharpe=0.30,       # Keep risk-adjusted
    Sentiment=0.25,    # Increase from default 0.12
    TopicMomentum=0.10 # Increase from default 0.08
)
```

### Sector Rotation Strategy
```python
# Analyze each sector separately
sectors = {
    'tech': ["AAPL", "MSFT", "GOOGL", "META", "NVDA"],
    'finance': ["JPM", "BAC", "GS", "MS", "C"],
    'healthcare': ["JNJ", "UNH", "PFE", "ABBV", "TMO"],
    'energy': ["XOM", "CVX", "COP", "SLB", "EOG"]
}

for name, tickers in sectors.items():
    cfg.tickers = tickers
    # Run analysis
    # Pick top 2 from each sector
```

### Backtesting Historical Performance
```python
# Use extended period and check historical accuracy
cfg = Config(
    tickers=[...],
    period="5y",  # Long history
    interval="1d"
)
# Compare portfolio performance to SPY benchmark
```

## 13. Output File Management

### Naming Convention
Format: `{type}_{suffix}_{timestamp}.csv`
- Type: rankings, quant_signals, ai_signals
- Suffix: us (for this folder)
- Timestamp: YYYYMMDD_HHMMSS

### Retention Policy
- **Keep forever:** Monthly snapshots for trend analysis
- **Archive after 1 month:** Daily/weekly runs
- **Delete after 6 months:** Experimental runs

### Consolidation Script
```powershell
# Merge all rankings into single historical file
python -c "
import pandas as pd
import glob

files = sorted(glob.glob('rankings_us_*.csv'))
dfs = []
for f in files:
    df = pd.read_csv(f)
    df['Date'] = f.split('_')[-1].split('.')[0][:8]
    dfs.append(df)

merged = pd.concat(dfs, ignore_index=True)
merged.to_csv('rankings_us_historical.csv', index=False)
print(f'Merged {len(files)} files into rankings_us_historical.csv')
"
```

## 14. Monitoring & Alerts

### Set Up Daily Email (Windows Task Scheduler)
```powershell
# Create scheduled task
$action = New-ScheduledTaskAction -Execute "powershell.exe" -Argument "-File C:\path\to\run_and_email.ps1"
$trigger = New-ScheduledTaskTrigger -Daily -At 8:30AM
Register-ScheduledTask -Action $action -Trigger $trigger -TaskName "StockPickerUS"
```

### Simple Alert Script (`run_and_email.ps1`)
```powershell
cd "C:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"
python .\stock_picker_daily.py > daily_log.txt 2>&1

# Check for strong BUY signals
python -c "
import pandas as pd
w = pd.read_csv('directional_warnings_us.csv')
buys = w[(w['Signal'] == 'BUY') & (w['Probability'] > 0.70)]
if len(buys) > 0:
    print('STRONG BUY SIGNALS:')
    print(buys.to_string(index=False))
" | Tee-Object -Variable alerts

if ($alerts) {
    # Send email (requires configured SMTP)
    Send-MailMessage -To "you@example.com" -Subject "Stock Picker Alerts" -Body $alerts -SmtpServer "smtp.gmail.com"
}
```

## 15. Best Practices Summary

✅ **DO:**
- Run daily updates before market open
- Archive old files monthly
- Verify headline counts for key tickers
- Review warnings before trading
- Keep at least 1 year price history
- Use equal weights unless you have strong conviction

❌ **DON'T:**
- Trade based solely on one day's signals
- Ignore risk management (stops/take-profits)
- Run during market hours (stale intraday data)
- Use penny stocks (min price $5 filter exists)
- Over-optimize weights on in-sample data
- Skip validation of top-ranked headline examples

## 16. Script Selection Decision Tree

```
Is this a routine morning update?
├─ Yes → Use stock_picker_daily.py (fast, 5min)
└─ No
   ├─ Is a major news event happening (Fed meeting, war, etc.)?
   │  └─ Yes → Use stock_picker_hybrid_world_news_extension.py (deep macro)
   └─ No
      ├─ Do you need full rebalance analysis?
      │  └─ Yes → Use stock_picker_hybrid_american_stocks.py (standard)
      └─ No → Use stock_picker.py (production default)
```

---

**For detailed architecture and system design, see `ARCHITECTURE.md`.**
