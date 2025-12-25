# Trading Simulation Guide

## Overview

The `trading_simulation.py` script simulates real-world trading by **actually running** your `stock_picker_advanced_quantitative.py` and `stock_picker_nlg_explanations.py` scripts twice daily (morning and afternoon sessions) and making buy/sell decisions based on the rankings.

This is NOT a simplified simulation - it runs the FULL analysis pipeline including:
- Advanced quantitative metrics (CAGR, Sharpe, Volatility, Max Drawdown)
- Technical indicators (MA crossover, MACD, RSI, Bollinger Bands, ADX)
- Statistical forecasting (ARIMA, GARCH, Kalman Filter)
- Mean reversion analysis (Z-Score, Fair Value)
- AI sentiment analysis (FinBERT + World News + Topic Momentum)
- Natural Language Generation explanations for each stock

## How It Works

### Daily Flow

Each trading day:

1. **Morning Session (9:00 AM)**
   - Runs `stock_picker_advanced_quantitative.py` → generates all CSVs
   - Runs `stock_picker_nlg_explanations.py` → generates human-readable explanations
   - Saves all artifacts to `simulation_output/YYYYMMDD_AM/`
   - Gets top 5 ranked stocks
   - Makes buy/sell decisions to align portfolio
   - Executes trades with realistic costs and slippage

2. **Afternoon Session (4:30 PM)**
   - Re-runs FULL analysis with updated data
   - Saves artifacts to `simulation_output/YYYYMMDD_PM/`
   - Re-evaluates top 5 stocks
   - Adjusts positions if rankings changed significantly
   - Records end-of-day portfolio value

3. **End of Day**
   - Tracks portfolio value, cash, holdings
   - Calculates daily return
   - Stores trade log with reasons and costs

### Trading Logic

- **Portfolio Management**: Max 5 positions, 15% capital per position
- **Rebalancing**: Automatic when stocks exit top 5 rankings
- **Costs**: 0.1% transaction fee + 0.05% slippage per trade
- **Realistic**: Uses actual historical prices from Yahoo Finance

## Usage

### Basic Run (Defaults)

```powershell
cd "c:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"
python trading_simulation.py
```

**Defaults:**
- Start Date: 2024-01-01
- End Date: 2024-12-01
- Initial Capital: $100,000
- Max Positions: 5
- Position Size: 15% per stock

### Custom Parameters

```powershell
# Simulate 6 months with $50k capital
python trading_simulation.py --start-date 2024-06-01 --end-date 2024-11-30 --capital 50000

# Aggressive strategy: 3 positions at 25% each
python trading_simulation.py --max-positions 3 --position-size 0.25

# Full year with larger capital
python trading_simulation.py --start-date 2024-01-01 --end-date 2024-12-31 --capital 250000
```

### Parameter Details

| Parameter | Description | Default | Range |
|-----------|-------------|---------|-------|
| `--start-date` | Simulation start (YYYY-MM-DD) | 2024-01-01 | Any valid date |
| `--end-date` | Simulation end (YYYY-MM-DD) | 2024-12-01 | Any valid date |
| `--capital` | Initial investment | $100,000 | Any positive number |
| `--max-positions` | Max stocks in portfolio | 5 | 1-10 recommended |
| `--position-size` | % per position | 0.15 (15%) | 0.1-0.33 (10-33%) |

## Output Structure

```
simulation_output/
├── 20241201_AM/                    # Morning session artifacts
│   ├── ranked_signals_advanced_*.csv
│   ├── portfolio_advanced_*.csv
│   ├── fundamentals_*.csv
│   ├── ai_signals_advanced_*.csv
│   └── nlg_analysis_*/
│       ├── explanations_*.txt
│       └── [all CSVs copied here]
│
├── 20241201_PM/                    # Afternoon session artifacts
│   └── [same structure as AM]
│
├── 20241202_AM/
├── 20241202_PM/
│   ...
│
└── final_results_YYYYMMDD_HHMMSS/  # Final simulation summary
    ├── trade_log_*.csv             # Every buy/sell with prices, costs, reasons
    ├── portfolio_history_*.csv     # Twice-daily portfolio snapshots
    ├── portfolio_chart_*.png       # Visual performance chart
    └── simulation_summary_*.json   # Configuration & metrics
```

## Understanding the Results

### Console Output

During simulation you'll see:
```
==============================================================
Day 1/250: 2024-01-02
==============================================================

🌅 MORNING SESSION (9:00 AM)
  🔄 Running stock_picker_advanced_quantitative.py...
  ✅ Advanced quantitative analysis complete
  🔄 Running stock_picker_nlg_explanations.py...
  ✅ NLG explanations generated
  
Top 5 Morning Rankings:
  1. JPM: Score=0.8234, CAGR=18.5%, Sharpe=1.23
  2. NVDA: Score=0.8156, CAGR=45.2%, Sharpe=1.15
  3. AVGO: Score=0.8089, CAGR=32.1%, Sharpe=1.31
  4. TSLA: Score=0.7945, CAGR=28.7%, Sharpe=0.98
  5. META: Score=0.7821, CAGR=24.3%, Sharpe=1.08

📊 Executed 5 morning trades:
  BUY 150 JPM @ $152.35
  BUY 85 NVDA @ $487.21
  BUY 92 AVGO @ $451.78
  BUY 68 TSLA @ $238.92
  BUY 45 META @ $345.67
  📄 NLG explanations saved to: nlg_analysis_20241002_090000

🌆 AFTERNOON SESSION (4:30 PM)
  ✓ No trades executed (positions aligned)
  
📈 End of Day Summary:
  Portfolio Value: $102,345.67
  Cash: $1,234.56
  Return: +2.35%
  Active Positions: 5
  Holdings: JPM, NVDA, AVGO, TSLA, META
```

### Final Report

```
SIMULATION COMPLETE - FINAL REPORT

💰 PERFORMANCE METRICS:
  Initial Capital:      $      100,000.00
  Final Value:          $      142,567.89
  Total Return:                   42.57%
  Sharpe Ratio:                     1.85
  Max Drawdown:                   -12.34%

📊 TRADING ACTIVITY:
  Total Trades:                      147
  Buy Trades:                         78
  Sell Trades:                        69
  Transaction Costs:    $       1,234.56
  Days Simulated:                    250
```

### Trade Log CSV

Columns:
- `timestamp`: When trade executed (with time)
- `session`: AM or PM
- `ticker`: Stock symbol
- `action`: BUY or SELL
- `shares`: Number of shares
- `price`: Execution price (includes slippage)
- `value`: Total transaction value
- `cost`: Transaction fee
- `reason`: Why trade was made (e.g., "Entered top rankings", "Exited top 5")

### Portfolio History CSV

Columns:
- `timestamp`: When portfolio valued (twice daily)
- `cash`: Available cash
- `holdings_value`: Market value of all positions
- `total_value`: Cash + Holdings
- `positions`: Number of active positions

## Analysis Deep Dive

### NLG Explanations

Each session generates human-readable analysis explaining:
- **Sentiment**: What the news says about each stock
- **Technical Indicators**: MA crossovers, MACD signals, RSI levels
- **Statistical Models**: ARIMA forecasts, GARCH volatility, Kalman trends
- **Mean Reversion**: Z-Score positions, fair value estimates
- **Fundamentals**: P/E ratios, ROE, profitability margins
- **Performance**: CAGR returns, Sharpe ratios, volatility
- **Key Strengths**: Bullet list of positives
- **Key Risks**: Bullet list of warnings
- **Overall Summary**: Tier classification + BUY/HOLD/SELL recommendation

Example excerpt:
```
=================================================================
JPM - JPMorgan Chase & Co. | Overall Score: 0.8234
=================================================================

SENTIMENT ANALYSIS
------------------
JPMorgan Chase shows POSITIVE sentiment (score: 0.72).
The stock-specific sentiment is POSITIVE (0.75), indicating favorable 
news coverage...

TECHNICAL INDICATORS
--------------------
Moving Average: The 50-day MA ($152.45) is ABOVE the 200-day MA ($148.23),
indicating an UPTREND (Golden Cross)...

KEY STRENGTHS
-------------
• Strong positive sentiment with high topic momentum
• Golden Cross - 50-day MA above 200-day MA (bullish trend)
• MACD shows bullish momentum with positive crossover
• ...
```

## Performance Tuning

### Speeding Up Simulation

The simulation runs the FULL analysis scripts twice daily, which can be slow for long periods. Options:

1. **Shorter Period**: Test with 1-3 months first
   ```powershell
   python trading_simulation.py --start-date 2024-10-01 --end-date 2024-12-01
   ```

2. **Reduce Tickers**: Edit the `tickers` list in `main()` function (default: 15 tickers)

3. **Skip AI**: Modify `stock_picker_advanced_quantitative.py` to set `INCLUDE_AI = False` (saves ~60s per run)

### Realistic Expectations

- **15 tickers, 250 trading days, twice daily**: ~500 script runs
- **Each run**: 110-160 seconds (with AI), 50-70 seconds (without AI)
- **Total time**: 15-22 hours (with AI), 7-10 hours (without AI)

For testing, use shorter periods or fewer tickers!

## Tips & Best Practices

### Before Running

1. **Test the scripts independently first**:
   ```powershell
   python stock_picker_advanced_quantitative.py
   python stock_picker_nlg_explanations.py
   ```

2. **Check your virtual environment** has all dependencies:
   - yfinance, pandas, numpy, sklearn
   - transformers, torch (for FinBERT)
   - statsmodels, arch, scipy
   - matplotlib (for charts)

3. **Ensure stable internet** (fetches news and stock data)

### During Simulation

- Don't close the terminal - simulation runs for hours
- Monitor first few days to ensure it's working correctly
- Check `simulation_output/` folder for session artifacts
- Can Ctrl+C to stop early (partial results will be saved)

### After Simulation

1. **Review the final chart** (`portfolio_chart_*.png`)
2. **Analyze the trade log** - why were trades made?
3. **Read NLG explanations** for key trading days
4. **Compare sessions** - did afternoon analysis differ from morning?
5. **Evaluate metrics** - Sharpe ratio, max drawdown, total return

## Troubleshooting

| Issue | Solution |
|-------|----------|
| Script not found | Ensure you're in `american stocks root cause analysis/` directory |
| Import errors | Check all dependencies installed in .venv |
| "No ranked signals file found" | Check if `stock_picker_advanced_quantitative.py` runs standalone |
| Timeout errors | Increase timeout in `run_analysis_scripts()` or reduce ticker count |
| Out of memory | Reduce ticker count or simulation period |
| Slow performance | Disable AI in advanced_quantitative.py or use shorter period |

## Example Scenarios

### Conservative Long-Term

```powershell
# Full year, 5 positions, 15% each
python trading_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --capital 100000 \
  --max-positions 5 \
  --position-size 0.15
```

### Aggressive Momentum

```powershell
# 3 concentrated positions, 30% each
python trading_simulation.py \
  --start-date 2024-01-01 \
  --end-date 2024-12-31 \
  --capital 100000 \
  --max-positions 3 \
  --position-size 0.30
```

### Quick Test

```powershell
# Just 3 months to see how it works
python trading_simulation.py \
  --start-date 2024-09-01 \
  --end-date 2024-11-30 \
  --capital 50000
```

## Next Steps

After running the simulation:

1. **Backtest Different Periods**: How did it perform in bull vs bear markets?
2. **Optimize Parameters**: Try different position sizes, max positions
3. **Compare Strategies**: Run with/without AI, different weighting schemes
4. **Forward Test**: Use recent data to see if it would work today
5. **Paper Trade**: Apply learnings to real-time paper trading

## Questions?

The simulation provides a comprehensive view of how your stock picker would have performed with realistic trading costs and twice-daily rebalancing. Use the NLG explanations to understand WHY the system made each decision!
