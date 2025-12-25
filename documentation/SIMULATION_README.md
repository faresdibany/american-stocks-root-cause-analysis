# Trading Simulation - README

## 🎯 What This Does

This trading simulation **actually runs** your full stock analysis pipeline (`stock_picker_advanced_quantitative.py` + `stock_picker_nlg_explanations.py`) **twice every trading day** and makes realistic buy/sell decisions based on the rankings.

This is NOT a toy simulation - it's a full backtest of your strategy with:
- ✅ Complete quantitative + AI + technical + fundamental analysis
- ✅ Twice-daily execution (morning pre-market + afternoon end-of-day)
- ✅ Realistic transaction costs (0.1%) and slippage (0.05%)
- ✅ Portfolio management with position sizing
- ✅ All analysis artifacts saved per session (CSVs + NLG explanations)
- ✅ Comprehensive performance metrics (Sharpe, drawdown, returns)

## 🚀 Quick Start

### 1. Test Your Scripts First

Make sure these work independently:
```powershell
cd "c:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"
python stock_picker_advanced_quantitative.py
python stock_picker_nlg_explanations.py
```

### 2. Run a Short Simulation (3 months)

```powershell
python trading_simulation.py --start-date 2024-09-01 --end-date 2024-11-30 --capital 50000
```

### 3. Check the Results

```powershell
cd simulation_output
# View final results folder
# Open portfolio_chart_*.png to see performance
# Read trade_log_*.csv to see all trades
```

## 📊 What You'll Get

### During Simulation

Real-time console output showing:
- Each trading day with morning + afternoon sessions
- Top 5 ranked stocks from each analysis
- All buy/sell trades executed
- End-of-day portfolio value and return
- Which artifacts were saved

### After Simulation

**Final Results Folder** with:
- 📈 `portfolio_chart_*.png` - Visual performance over time
- 📝 `trade_log_*.csv` - Every buy/sell with prices, costs, reasons
- 📊 `portfolio_history_*.csv` - Twice-daily snapshots of cash, holdings, value
- 📄 `simulation_summary_*.json` - Configuration and key metrics

**Session Artifacts** (`simulation_output/YYYYMMDD_AM/` and `YYYYMMDD_PM/`):
- All CSVs from advanced_quantitative.py
- NLG explanation folders with human-readable analysis
- Complete audit trail of every analysis run

## 🎛️ Parameters

| Flag | Description | Default |
|------|-------------|---------|
| `--start-date` | Start date (YYYY-MM-DD) | 2024-01-01 |
| `--end-date` | End date (YYYY-MM-DD) | 2024-12-01 |
| `--capital` | Initial investment | $100,000 |
| `--max-positions` | Max stocks held | 5 |
| `--position-size` | % per position | 0.15 (15%) |

## 📖 Full Documentation

See **[SIMULATION_GUIDE.md](./SIMULATION_GUIDE.md)** for:
- Detailed explanation of how it works
- Trading logic and rebalancing rules
- Output structure and file formats
- Performance tuning tips
- Troubleshooting guide
- Example scenarios

## ⚡ Quick Examples

### Conservative Strategy
```powershell
# 5 positions, 15% each, full year
python trading_simulation.py --start-date 2024-01-01 --end-date 2024-12-31
```

### Aggressive Strategy
```powershell
# 3 concentrated positions, 30% each
python trading_simulation.py --max-positions 3 --position-size 0.30
```

### Quick Test
```powershell
# Just 2 months to see results fast
python trading_simulation.py --start-date 2024-10-01 --end-date 2024-11-30
```

## ⏱️ Time Requirements

**Full Year (250 trading days, twice daily = 500 runs)**:
- With AI enabled: 15-22 hours
- Without AI: 7-10 hours

**Pro Tip**: Start with 1-3 months to test, then run longer backtests overnight!

## 🔧 How It Works

1. **Morning (9:00 AM)**
   - Fetches historical data up to today
   - Runs `stock_picker_advanced_quantitative.py`
   - Runs `stock_picker_nlg_explanations.py`
   - Saves all artifacts to `simulation_output/YYYYMMDD_AM/`
   - Gets top 5 stocks
   - Makes buy/sell decisions
   - Executes trades with costs

2. **Afternoon (4:30 PM)**
   - Re-runs full analysis with latest data
   - Saves to `simulation_output/YYYYMMDD_PM/`
   - Adjusts positions if rankings changed
   - Records end-of-day portfolio value

3. **Repeat for every trading day in simulation period**

## 📈 Performance Metrics

The final report shows:

```
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

## 🧠 Why This Is Powerful

Unlike simple backtests, this simulation:

1. **Uses Your ACTUAL Scripts** - Tests the real system you'll run daily
2. **Twice-Daily Analysis** - Captures intraday volatility and ranking changes
3. **Complete Artifacts** - Every CSV and explanation saved for forensic analysis
4. **Realistic Costs** - Transaction fees and slippage like real trading
5. **Human-Readable** - NLG explanations show WHY each trade was made

## 🎓 Learning Opportunities

After running the simulation:

- **Compare Morning vs Afternoon**: How often do rankings change?
- **Analyze Trade Reasons**: Which factors drove buy/sell decisions?
- **Review NLG Explanations**: Understand the AI's reasoning
- **Study Drawdown Periods**: When did the strategy struggle?
- **Optimize Parameters**: Test different position sizes and max positions

## 🚨 Important Notes

- **Requires internet**: Fetches news and stock data during simulation
- **Resource intensive**: Runs full analysis 500+ times
- **Test first**: Run 1-2 months before attempting full year
- **Monitor output**: First few days show if it's working correctly

## 🆘 Troubleshooting

**"Script execution timed out"**
- Increase timeout in `run_analysis_scripts()` method
- Or reduce number of tickers

**"No ranked signals file found"**
- Verify `stock_picker_advanced_quantitative.py` works standalone
- Check for errors in the script output

**Slow performance**
- Disable AI: Set `INCLUDE_AI = False` in advanced_quantitative.py
- Use shorter period: `--start-date 2024-10-01 --end-date 2024-11-30`
- Reduce tickers: Edit `tickers` list in `main()` function

## 🎉 Ready to Start?

```powershell
cd "c:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"

# Quick 2-month test
python trading_simulation.py --start-date 2024-10-01 --end-date 2024-11-30

# Watch the console output
# Check simulation_output/ folder for all artifacts
# Open portfolio_chart_*.png to see results
```

**Happy Backtesting! 📈🚀**
