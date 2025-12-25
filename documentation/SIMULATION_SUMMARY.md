# Trading Simulation - Complete System

## 📦 What You Have

You now have a **complete trading simulation system** that:

1. **Runs your ACTUAL stock picker scripts** (`stock_picker_advanced_quantitative.py` + `stock_picker_nlg_explanations.py`)
2. **Executes twice daily** (morning + afternoon) for realistic intraday rebalancing
3. **Saves all artifacts** (CSVs, NLG explanations) per session for full audit trail
4. **Makes realistic trades** with transaction costs (0.1%) and slippage (0.05%)
5. **Generates comprehensive reports** with performance metrics and visualizations
6. **Compares multiple runs** to find optimal parameters

## 🗂️ Files Created

### Main Scripts

| File | Purpose |
|------|---------|
| **trading_simulation.py** | Main simulation engine - runs scripts twice daily |
| **test_simulation.py** | Quick 5-day test to verify everything works |
| **analyze_simulations.py** | Compare multiple simulation runs |

### Documentation

| File | Content |
|------|---------|
| **SIMULATION_README.md** | Quick start guide and overview |
| **SIMULATION_GUIDE.md** | Complete documentation (how it works, parameters, tips) |

## 🚀 Quick Start Workflow

### Step 1: Test (5 days, ~20 minutes)

```powershell
cd "c:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"
python test_simulation.py
```

This runs a 5-day simulation to ensure:
- Scripts execute without errors
- Data fetching works
- Trades are executed correctly
- Artifacts are saved properly

### Step 2: Short Backtest (3 months, ~2-3 hours)

```powershell
python trading_simulation.py --start-date 2024-09-01 --end-date 2024-11-30
```

Good for:
- Understanding performance in recent market
- Faster iteration on parameters
- Testing before long backtests

### Step 3: Full Year Backtest (250 days, ~15-20 hours)

```powershell
python trading_simulation.py --start-date 2024-01-01 --end-date 2024-12-31
```

Comprehensive analysis:
- Full market cycle (bull, bear, sideways)
- Statistical significance
- Robust performance metrics

### Step 4: Parameter Optimization

Run multiple simulations with different settings:

```powershell
# Conservative (5 positions, 15% each)
python trading_simulation.py --max-positions 5 --position-size 0.15

# Balanced (4 positions, 20% each)
python trading_simulation.py --max-positions 4 --position-size 0.20

# Aggressive (3 positions, 30% each)
python trading_simulation.py --max-positions 3 --position-size 0.30
```

### Step 5: Compare Results

```powershell
python analyze_simulations.py
```

This generates:
- Comparison table of all runs
- Visual charts (return vs Sharpe, drawdown analysis)
- AM vs PM session differences
- Best performers by metric

## 📊 Output Structure

```
american stocks root cause analysis/
├── trading_simulation.py          # Main simulation
├── test_simulation.py             # Quick test
├── analyze_simulations.py         # Results comparison
├── SIMULATION_README.md           # Quick start
├── SIMULATION_GUIDE.md            # Full documentation
│
└── simulation_output/             # All results
    ├── 20241201_AM/               # Morning session
    │   ├── ranked_signals_advanced_*.csv
    │   ├── portfolio_advanced_*.csv
    │   ├── fundamentals_*.csv
    │   ├── ai_signals_advanced_*.csv
    │   └── nlg_analysis_*/
    │       └── explanations_*.txt
    │
    ├── 20241201_PM/               # Afternoon session
    │   └── [same structure]
    │
    ├── final_results_TIMESTAMP/   # Simulation summary
    │   ├── trade_log_*.csv
    │   ├── portfolio_history_*.csv
    │   ├── portfolio_chart_*.png
    │   └── simulation_summary_*.json
    │
    ├── simulation_comparison.csv   # Compare multiple runs
    ├── simulation_comparison.png   # Visual comparison
    └── am_pm_ranking_changes.csv   # Session differences
```

## 🎯 Use Cases

### 1. Backtest Your Strategy

**Goal**: See how your stock picker would have performed historically

```powershell
python trading_simulation.py --start-date 2024-01-01 --end-date 2024-12-31
```

**Review**:
- Final return %
- Sharpe ratio (risk-adjusted return)
- Max drawdown (worst loss period)
- Trade log (why each trade was made)

### 2. Optimize Parameters

**Goal**: Find best position sizing and portfolio composition

```powershell
# Run multiple configurations
python trading_simulation.py --max-positions 5 --position-size 0.15  # Conservative
python trading_simulation.py --max-positions 3 --position-size 0.30  # Aggressive

# Compare results
python analyze_simulations.py
```

**Review**:
- Which parameter set has best Sharpe?
- Which has lowest drawdown?
- Trade-off between return and risk

### 3. Analyze Intraday Volatility

**Goal**: Understand how rankings change throughout the day

```powershell
# Run simulation
python trading_simulation.py --start-date 2024-10-01 --end-date 2024-11-30

# Analyze session differences
python analyze_simulations.py
```

**Review**:
- How often do top 5 stocks change from AM to PM?
- Which stocks are most volatile?
- Is twice-daily rebalancing beneficial?

### 4. Stress Test

**Goal**: See performance during market turbulence

```powershell
# Test specific volatile period (e.g., August 2024 correction)
python trading_simulation.py --start-date 2024-08-01 --end-date 2024-09-30
```

**Review**:
- Max drawdown during correction
- Recovery time after dips
- Which stocks held up best

### 5. Forward Test

**Goal**: Test most recent period to validate current strategy

```powershell
# Last 2 months
python trading_simulation.py --start-date 2024-10-01 --end-date 2024-11-30
```

**Review**:
- Are recent predictions accurate?
- Has market regime changed?
- Do weights need adjustment?

## 💡 Tips & Best Practices

### Performance Optimization

**For Fast Testing**:
- Use 1-3 month periods
- Reduce number of tickers (edit `main()` function)
- Disable AI: Set `INCLUDE_AI = False` in `stock_picker_advanced_quantitative.py`

**For Comprehensive Backtests**:
- Run overnight (15-20 hours for full year)
- Keep all features enabled
- Use full ticker universe (15 stocks)

### Parameter Tuning

**Position Size Guidelines**:
- Conservative: 10-15% (6-7 stocks max)
- Balanced: 15-20% (5-6 stocks)
- Aggressive: 25-33% (3-4 stocks)

**Max Positions Guidelines**:
- More positions = lower volatility, lower return
- Fewer positions = higher volatility, higher potential return
- Sweet spot: 3-5 positions for most strategies

### Interpreting Results

**Good Performance**:
- Sharpe > 1.0 (risk-adjusted return beats market)
- Max drawdown < 20% (acceptable risk)
- Consistent returns across periods

**Warning Signs**:
- Sharpe < 0.5 (poor risk-adjusted return)
- Max drawdown > 30% (excessive risk)
- Negative returns for extended periods

### Next Steps After Simulation

1. **Review NLG Explanations**: Understand WHY trades were made
2. **Check Trade Log**: Look for patterns in buy/sell reasons
3. **Compare Sessions**: See how often AM vs PM rankings differ
4. **Analyze Drawdowns**: Understand worst-case scenarios
5. **Optimize Weights**: Adjust scoring weights in `AdvancedWeights` class
6. **Forward Test**: Apply learnings to recent data
7. **Paper Trade**: Test live with virtual money

## 🎓 Learning Opportunities

### Forensic Analysis

For each trading day you can:
- Open `simulation_output/YYYYMMDD_AM/nlg_analysis_*/explanations_*.txt`
- Read the full reasoning for each stock's ranking
- Compare to afternoon session to see what changed
- Review `trade_log_*.csv` to see which trades were executed

### Pattern Recognition

After multiple simulations:
- Which sectors perform best in different market conditions?
- Which technical indicators are most predictive?
- How accurate are ARIMA forecasts?
- Does sentiment lead or lag price movements?

### Strategy Refinement

Use insights to improve:
- **Scoring weights**: Increase weight on predictive factors
- **Rebalancing frequency**: Maybe daily is enough, not twice-daily?
- **Position sizing**: Dynamic sizing based on conviction?
- **Stop losses**: Add protective stops for drawdown control?

## 🚨 Important Notes

### Data Requirements

- **Internet connection**: Fetches news and stock data
- **Yahoo Finance**: Primary data source (free)
- **Storage**: ~1GB per full year simulation (all artifacts)

### Time Requirements

| Period | With AI | Without AI |
|--------|---------|------------|
| 5 days | 20-30 min | 10-15 min |
| 1 month | 2-3 hours | 1-1.5 hours |
| 3 months | 6-8 hours | 3-4 hours |
| 1 year | 15-22 hours | 7-10 hours |

### System Requirements

- **RAM**: 4GB minimum, 8GB recommended
- **CPU**: Any modern processor (multi-core helps)
- **Storage**: 5GB free for full year with artifacts

## 🆘 Troubleshooting

### Common Issues

**"Script execution timed out"**
- Solution: Increase timeout in `run_analysis_scripts()` (line ~135)
- Or: Reduce ticker count

**"No ranked signals file found"**
- Solution: Verify `stock_picker_advanced_quantitative.py` works standalone
- Check: Script completes without errors

**Slow performance**
- Solution: Disable AI in `stock_picker_advanced_quantitative.py`
- Or: Use shorter simulation period
- Or: Reduce ticker count (edit `main()` in `trading_simulation.py`)

**Memory errors**
- Solution: Reduce ticker count
- Or: Clear old simulation_output/ folders periodically

### Getting Help

1. Check console output for specific error messages
2. Verify standalone scripts work: `python stock_picker_advanced_quantitative.py`
3. Test with short period first: `python test_simulation.py`
4. Check `simulation_output/` for partial results

## 🎉 You're Ready!

Your complete trading simulation system is set up and ready to use. Start with the quick test, then run backtests to validate your strategy!

### Quick Commands

```powershell
cd "c:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"

# Quick test (5 days)
python test_simulation.py

# Short backtest (3 months)
python trading_simulation.py --start-date 2024-09-01 --end-date 2024-11-30

# Full year backtest
python trading_simulation.py --start-date 2024-01-01 --end-date 2024-12-31

# Compare multiple runs
python analyze_simulations.py
```

**Happy Trading! 📈🚀**
