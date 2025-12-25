# Daily Trading Simulation - Complete Documentation

## 📋 Table of Contents
- [Overview](#overview)
- [System Architecture](#system-architecture)
- [Workflow Diagram](#workflow-diagram)
- [Twice-Daily Schedule](#twice-daily-schedule)
- [Simulation Process](#simulation-process)
- [Trading Logic](#trading-logic)
- [Performance Metrics](#performance-metrics)
- [File Outputs](#file-outputs)
- [Automation Setup](#automation-setup)
- [Configuration Options](#configuration-options)
- [Troubleshooting](#troubleshooting)

---

## 🎯 Overview

The **Daily Trading Simulation** is an automated backtesting system that simulates real-world trading by:

1. Running your actual analysis scripts (`stock_picker_advanced_quantitative.py` and `stock_picker_nlg_explanations.py`)
2. Making buy/sell decisions based on the generated rankings
3. Tracking portfolio performance with realistic trading costs
4. Operating on a twice-daily schedule (9:00 AM and 4:30 PM)

### Key Features

✅ **Realistic Trading Environment**
- Transaction costs (0.1% per trade)
- Slippage modeling (0.05%)
- Position sizing (20% per position max)
- Portfolio rebalancing based on rankings

✅ **Comprehensive Analysis**
- Runs complete quantitative analysis daily
- Generates natural language explanations
- Tracks top 5 stocks dynamically
- Adjusts positions as rankings change

✅ **Performance Tracking**
- Portfolio value over time
- Sharpe ratio calculation
- Maximum drawdown monitoring
- Detailed trade log
- Visual performance charts

✅ **Automated Execution**
- Scheduled runs twice daily
- Rolling 30-day simulation window
- Automatic output organization
- Background operation

---

## 🏗️ System Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                   AUTOMATION LAYER                          │
│  Windows Task Scheduler (9:00 AM & 4:30 PM Daily)          │
└────────────────────┬────────────────────────────────────────┘
                     │
                     ▼
┌─────────────────────────────────────────────────────────────┐
│              ORCHESTRATION LAYER                            │
│  daily_trading_simulation.py                                │
│  • Market calendar management                               │
│  • Date range calculation (rolling 30 days)                 │
│  • Session execution (AM/PM)                                │
└────────────┬───────────────────────┬────────────────────────┘
             │                       │
             ▼                       ▼
┌────────────────────────┐  ┌───────────────────────────────┐
│   ANALYSIS LAYER       │  │   TRADING LAYER               │
│                        │  │                               │
│  subprocess.run()      │  │  • Portfolio state tracking   │
│  ├─ advanced_quant.py  │  │  • Position sizing            │
│  │  └─ Rankings        │  │  • Trade execution            │
│  └─ nlg_explanations   │  │  • Cost calculation           │
│     └─ Explanations    │  │  • Value updates              │
└────────────┬───────────┘  └──────────┬────────────────────┘
             │                         │
             └────────┬────────────────┘
                      ▼
┌─────────────────────────────────────────────────────────────┐
│                    OUTPUT LAYER                             │
│  simulation_results/                                        │
│  ├─ trade_log_TIMESTAMP.csv                                 │
│  ├─ portfolio_history_TIMESTAMP.csv                         │
│  └─ performance_chart_TIMESTAMP.png                         │
│                                                             │
│  ranked_signals_advanced_*.csv (each day)                   │
│  nlg_analysis_*/ folders (each day)                         │
└─────────────────────────────────────────────────────────────┘
```

---

## 🔄 Workflow Diagram

### Daily Execution Flow

```
START (Scheduled Task Triggers)
│
├─ 9:00 AM Morning Session
│  │
│  ├─ 1. Load Market Calendar
│  │    └─ Identify trading days in last 30 days
│  │
│  ├─ 2. FOR EACH Trading Day:
│  │    │
│  │    ├─ 2.1. Run Advanced Quantitative Analysis
│  │    │      ├─ Execute: stock_picker_advanced_quantitative.py
│  │    │      ├─ Input: Historical price data up to current sim date
│  │    │      ├─ Process: 
│  │    │      │   • Fundamental metrics (P/E, ROE, etc.)
│  │    │      │   • Technical indicators (MA, MACD, RSI, BB, ADX)
│  │    │      │   • Statistical models (ARIMA, GARCH, Kalman)
│  │    │      │   • Mean reversion (Z-Score, fair value)
│  │    │      │   • AI sentiment (optional - disabled for speed)
│  │    │      └─ Output: ranked_signals_advanced_*.csv
│  │    │
│  │    ├─ 2.2. Run NLG Explanations
│  │    │      ├─ Execute: stock_picker_nlg_explanations.py
│  │    │      ├─ Input: ranked_signals_advanced_*.csv
│  │    │      ├─ Process: Generate human-readable analysis
│  │    │      └─ Output: nlg_analysis_*/ folder
│  │    │
│  │    ├─ 2.3. Load Rankings
│  │    │      └─ Read: ranked_signals_advanced_*.csv
│  │    │
│  │    ├─ 2.4. Get Current Prices
│  │    │      └─ Fetch: Yahoo Finance for simulation date
│  │    │
│  │    ├─ 2.5. Calculate Target Positions
│  │    │      ├─ Select: Top 5 stocks from rankings
│  │    │      ├─ Calculate: 20% of portfolio per position
│  │    │      └─ Determine: Required shares per stock
│  │    │
│  │    ├─ 2.6. Execute Trades
│  │    │      ├─ SELL: Stocks no longer in top 5
│  │    │      ├─ BUY: New stocks entering top 5
│  │    │      ├─ ADJUST: Existing positions (if needed)
│  │    │      └─ Apply: Transaction costs + slippage
│  │    │
│  │    ├─ 2.7. Update Portfolio Value
│  │    │      ├─ Calculate: Cash + Holdings value
│  │    │      └─ Record: timestamp, positions, total value
│  │    │
│  │    └─ 2.8. Display Session Summary
│  │           ├─ Top 5 rankings
│  │           ├─ Trades executed
│  │           └─ Portfolio status
│  │
│  └─ 3. Continue to Next Trading Day
│
├─ 4:30 PM Afternoon Session
│  └─ (Same process as morning - re-analyzes with EOD data)
│
├─ 4. Generate Final Report
│  ├─ Calculate: Total return, Sharpe ratio, max drawdown
│  ├─ Compute: Trading statistics (# trades, costs)
│  └─ Display: Performance summary
│
├─ 5. Save Results
│  ├─ Export: trade_log_*.csv
│  ├─ Export: portfolio_history_*.csv
│  └─ Generate: performance_chart_*.png
│
END
```

---

## ⏰ Twice-Daily Schedule

### Morning Session (9:00 AM)

**Purpose**: Pre-market analysis and positioning

**Timing**: Before market opens (9:30 AM ET)

**Activities**:
1. Simulate past 30 days with morning analysis
2. Run full quantitative + NLG pipeline for each day
3. Make buy/sell decisions based on pre-market data
4. Prepare portfolio for the trading day

**Advantages**:
- Captures overnight news and developments
- Positions portfolio before market opens
- Identifies opportunities early

### Afternoon Session (4:30 PM)

**Purpose**: Post-market analysis and rebalancing

**Timing**: After market closes (4:00 PM ET)

**Activities**:
1. Re-simulate past 30 days with end-of-day data
2. Re-run full analysis with updated prices
3. Rebalance portfolio based on new rankings
4. Capture intraday momentum changes

**Advantages**:
- Incorporates full day's price action
- Adjusts for earnings announcements
- Captures post-market sentiment shifts

---

## 🔬 Simulation Process

### 1. Date Range Calculation

```python
# Rolling 30-day window
START_DATE = TODAY - 30 days
END_DATE = TODAY

# Example on December 1, 2025:
START_DATE = November 1, 2025
END_DATE = December 1, 2025
```

### 2. Market Calendar Loading

```python
# Fetch trading days from Yahoo Finance (SPY proxy)
market_dates = yf.download("SPY", start=START_DATE, end=END_DATE).index

# Filters out:
# - Weekends
# - Market holidays (Thanksgiving, Christmas, etc.)
# - Non-trading days
```

### 3. Day-by-Day Iteration

For each trading day in the simulation period:

**Step 1: Analysis Execution**
```bash
# Run advanced quantitative script
python stock_picker_advanced_quantitative.py
  └─ Uses data up to current simulation date only
  └─ Generates ranked_signals_advanced_TIMESTAMP.csv

# Run NLG explanations script  
python stock_picker_nlg_explanations.py
  └─ Reads latest ranked signals
  └─ Creates nlg_analysis_TIMESTAMP/ folder
```

**Step 2: Ranking Processing**
```python
rankings = pd.read_csv("ranked_signals_advanced_*.csv")
top_5 = rankings.head(5)

# Example rankings:
#   Ticker  Final_Score  CAGR   Sharpe
# 1 NVDA    0.8234      45.2%  1.85
# 2 AAPL    0.7891      32.1%  1.72
# 3 META    0.7654      38.5%  1.68
# 4 MSFT    0.7432      28.9%  1.54
# 5 GOOGL   0.7201      25.3%  1.41
```

**Step 3: Position Targeting**
```python
# Portfolio: $100,000 total
# Position size: 20% = $20,000 per stock
# Top 5 stocks = up to $100,000 invested

capital_per_position = 100000 * 0.20  # $20,000

for stock in top_5:
    price = current_prices[stock]
    shares = int(20000 / price)
    target_positions[stock] = shares

# Example:
# NVDA @ $500 → 40 shares ($20,000)
# AAPL @ $180 → 111 shares ($19,980)
# META @ $350 → 57 shares ($19,950)
# MSFT @ $380 → 52 shares ($19,760)
# GOOGL @ $140 → 142 shares ($19,880)
```

**Step 4: Trade Execution**
```python
# SELL positions not in top 5
for ticker in current_positions:
    if ticker not in target_positions:
        sell_price = price * (1 - slippage)  # 0.05% worse
        proceeds = shares * sell_price
        cost = proceeds * 0.001  # 0.1% transaction fee
        cash += (proceeds - cost)

# BUY new positions or adjust existing
for ticker, target_shares in target_positions.items():
    delta = target_shares - current_positions.get(ticker, 0)
    
    if delta > 0:  # Buy
        buy_price = price * (1 + slippage)  # 0.05% worse
        total_cost = delta * buy_price
        transaction_fee = total_cost * 0.001
        cash -= (total_cost + transaction_fee)
        positions[ticker] += delta
```

**Step 5: Portfolio Update**
```python
holdings_value = sum(
    shares * current_price[ticker]
    for ticker, shares in positions.items()
)

portfolio_value = cash + holdings_value

# Record history
portfolio_history.append({
    'timestamp': current_date,
    'cash': cash,
    'holdings_value': holdings_value,
    'total_value': portfolio_value,
    'num_positions': len(positions)
})
```

### 4. Performance Calculation

After all trading days are simulated:

```python
# Total Return
total_return = (final_value / initial_capital - 1) * 100

# Sharpe Ratio
daily_returns = portfolio_history['total_value'].pct_change()
sharpe = (daily_returns.mean() / daily_returns.std()) * sqrt(252)

# Maximum Drawdown
cumulative = (1 + daily_returns).cumprod()
running_max = cumulative.expanding().max()
drawdown = (cumulative - running_max) / running_max
max_drawdown = drawdown.min() * 100

# Trading Statistics
total_trades = len(trade_log)
buy_trades = len(trade_log[trade_log['action'] == 'BUY'])
sell_trades = len(trade_log[trade_log['action'] == 'SELL'])
total_costs = trade_log['cost'].sum()
```

---

## 💼 Trading Logic

### Position Sizing Strategy

**Maximum Positions**: 5 stocks  
**Position Size**: 20% of portfolio per stock  
**Rebalancing**: Based on ranking changes

```
Portfolio: $100,000
├─ Stock 1: $20,000 (20%)
├─ Stock 2: $20,000 (20%)
├─ Stock 3: $20,000 (20%)
├─ Stock 4: $20,000 (20%)
└─ Stock 5: $20,000 (20%)
```

### Entry Rules

A stock enters the portfolio when:
1. ✅ It ranks in the **top 5** by Final_Score
2. ✅ Current price data is available
3. ✅ Sufficient cash is available for position

### Exit Rules

A stock exits the portfolio when:
1. ❌ It drops **below top 5** rankings
2. ❌ (Automatic full position sell)

### Rebalancing Rules

Positions are adjusted when:
- Stock ranking changes significantly
- Portfolio needs to maintain equal weighting
- New stock enters top 5 (requires capital)

### Cost Structure

**Transaction Costs**: 0.1% per trade
```
Buy $20,000 → Pay $20 fee
Sell $20,000 → Pay $20 fee
```

**Slippage**: 0.05% price impact
```
Buy at $100 → Execute at $100.05 (0.05% worse)
Sell at $100 → Execute at $99.95 (0.05% worse)
```

**Total Round Trip Cost**: ~0.3%
- Buy slippage: 0.05%
- Buy transaction: 0.1%
- Sell slippage: 0.05%
- Sell transaction: 0.1%
- **Total**: 0.3% per complete buy-sell cycle

---

## 📊 Performance Metrics

### Return Metrics

**Total Return (%)**
```
(Final Portfolio Value / Initial Capital - 1) × 100
```

**CAGR (Annualized Return)**
```
((Final Value / Initial Value) ^ (365 / Days)) - 1) × 100
```

### Risk Metrics

**Sharpe Ratio**
```
(Mean Daily Return / Std Dev Daily Return) × √252
```
- Measures risk-adjusted returns
- Higher is better (>1.0 is good, >2.0 is excellent)
- Annualized using 252 trading days

**Maximum Drawdown (%)**
```
Max((Peak Value - Current Value) / Peak Value) × 100
```
- Largest peak-to-trough decline
- Indicates worst-case scenario loss
- Lower is better (closer to 0%)

**Volatility (Annualized)**
```
Std Dev(Daily Returns) × √252 × 100
```

### Trading Metrics

**Win Rate**
```
(Profitable Trades / Total Trades) × 100
```

**Average Trade Cost**
```
Total Transaction Costs / Total Trades
```

**Turnover Rate**
```
(Total Trade Value / Average Portfolio Value) × 100
```

---

## 📁 File Outputs

### Simulation Results Folder

```
simulation_results/
├─ trade_log_20251201_163045.csv
├─ portfolio_history_20251201_163045.csv
└─ performance_chart_20251201_163045.png
```

### Trade Log CSV

**Columns**:
- `timestamp`: When trade occurred
- `session`: "am" or "pm"
- `ticker`: Stock symbol
- `action`: "BUY" or "SELL"
- `shares`: Number of shares traded
- `price`: Execution price (after slippage)
- `value`: Total trade value
- `cost`: Transaction cost paid
- `reason`: Why trade was executed

**Example**:
```csv
timestamp,session,ticker,action,shares,price,value,cost,reason
2024-11-01 09:00:00,am,NVDA,BUY,40,500.25,20010.00,20.01,Entered top rankings
2024-11-01 16:30:00,pm,TSLA,SELL,55,220.45,12124.75,12.12,Exited top rankings
```

### Portfolio History CSV

**Columns**:
- `timestamp`: Date and time
- `cash`: Available cash
- `holdings_value`: Market value of stock holdings
- `total_value`: Cash + holdings
- `num_positions`: Number of stocks held
- `positions`: List of current holdings

**Example**:
```csv
timestamp,cash,holdings_value,total_value,num_positions,positions
2024-11-01 09:00:00,10250.50,89749.50,100000.00,5,"NVDA, AAPL, META, MSFT, GOOGL"
2024-11-01 16:30:00,15320.75,87895.25,103216.00,4,"NVDA, AAPL, META, MSFT"
```

### Performance Chart PNG

Two-panel visualization:

**Panel 1**: Portfolio Value Over Time
- Line chart showing portfolio growth
- Horizontal line at initial capital
- X-axis: Date
- Y-axis: Portfolio value ($)

**Panel 2**: Number of Active Positions
- Line chart showing position count
- X-axis: Date
- Y-axis: Number of stocks (0-5)

### Daily Analysis Outputs

Created for each simulated trading day:

```
ranked_signals_advanced_20241101_090245.csv
ranked_signals_advanced_20241102_090312.csv
...

nlg_analysis_20241101_090250/
├─ explanations_20241101_090250.txt
├─ ranked_signals_advanced_20241101_090245.csv
└─ ... (other analysis files)

nlg_analysis_20241102_090315/
├─ explanations_20241102_090315.txt
└─ ...
```

---

## ⚙️ Automation Setup

### Method 1: Windows Task Scheduler (Current Setup)

**Two Scheduled Tasks**:

1. **DailyTradingSimulation_Morning**
   - Schedule: Daily at 9:00 AM
   - Script: `run_simulation.bat`
   - Priority: Highest
   - Status: ✅ Enabled

2. **DailyTradingSimulation_Afternoon**
   - Schedule: Daily at 4:30 PM
   - Script: `run_simulation.bat`
   - Priority: Highest
   - Status: ✅ Enabled

**Setup Script**: `setup_daily_simulation.ps1`

```powershell
# Run as Administrator
.\setup_daily_simulation.ps1
```

**Verification**:
```powershell
Get-ScheduledTask -TaskName "DailyTradingSimulation_*"
```

### Task Management Commands

**View Tasks**:
```powershell
schtasks /Query /TN "DailyTradingSimulation_Morning"
schtasks /Query /TN "DailyTradingSimulation_Afternoon"
```

**Run Manually**:
```powershell
Start-ScheduledTask -TaskName "DailyTradingSimulation_Morning"
Start-ScheduledTask -TaskName "DailyTradingSimulation_Afternoon"
```

**Disable Tasks**:
```powershell
Disable-ScheduledTask -TaskName "DailyTradingSimulation_Morning"
Disable-ScheduledTask -TaskName "DailyTradingSimulation_Afternoon"
```

**Enable Tasks**:
```powershell
Enable-ScheduledTask -TaskName "DailyTradingSimulation_Morning"
Enable-ScheduledTask -TaskName "DailyTradingSimulation_Afternoon"
```

**Delete Tasks**:
```powershell
Unregister-ScheduledTask -TaskName "DailyTradingSimulation_Morning" -Confirm:$false
Unregister-ScheduledTask -TaskName "DailyTradingSimulation_Afternoon" -Confirm:$false
```

---

## 🎛️ Configuration Options

### Command Line Arguments

```bash
python daily_trading_simulation.py [OPTIONS]
```

**Available Options**:

| Argument | Default | Description |
|----------|---------|-------------|
| `--start-date` | 2024-11-01 | Simulation start date (YYYY-MM-DD) |
| `--end-date` | Today | Simulation end date (YYYY-MM-DD) |
| `--capital` | 100000 | Initial capital ($) |
| `--max-positions` | 5 | Maximum number of stocks to hold |
| `--position-size` | 0.20 | Position size as fraction (0.20 = 20%) |
| `--twice-daily` | flag | Run analysis twice per day (AM + PM) |
| `--python` | sys.executable | Path to Python executable |

**Examples**:

```bash
# Simulate last 60 days with $50k capital
python daily_trading_simulation.py --start-date 2024-10-01 --end-date 2024-12-01 --capital 50000

# Simulate with 3 positions max, 30% each
python daily_trading_simulation.py --max-positions 3 --position-size 0.30

# Run once daily (no twice-daily flag)
python daily_trading_simulation.py --start-date 2024-11-01
```

### Modifying run_simulation.bat

Edit the batch file to change default parameters:

```batch
"C:\...\python.exe" daily_trading_simulation.py ^
    --start-date 2024-10-01 ^
    --end-date %TODAY% ^
    --capital 50000 ^
    --max-positions 3 ^
    --position-size 0.25 ^
    --twice-daily
```

### Ticker Universe

Edit `daily_trading_simulation.py` line 568:

```python
# Default tickers
tickers = [
    "AAPL", "MSFT", "NVDA", "AMZN", "GOOGL",
    "META", "AVGO", "TSLA", "JPM", "V",
    "MA", "WMT", "UNH", "XOM", "JNJ"
]

# Customize to your preference:
tickers = ["AAPL", "MSFT", "GOOGL", "AMZN", "NVDA"]  # Tech-focused
tickers = ["JPM", "BAC", "WFC", "C", "GS"]  # Financial sector
```

### Transaction Costs

Edit `DailyTradingSimulation.__init__()` parameters:

```python
sim = DailyTradingSimulation(
    tickers=tickers,
    transaction_cost_pct=0.001,  # 0.1% (default)
    slippage_pct=0.0005,         # 0.05% (default)
    # Change to:
    transaction_cost_pct=0.0005,  # 0.05% (lower cost broker)
    slippage_pct=0.0002,          # 0.02% (better execution)
)
```

---

## 🔧 Troubleshooting

### Common Issues

#### 1. Simulation Takes Too Long

**Symptom**: Runs exceed 1 hour, task times out

**Solutions**:
- **Reduce date range**: Simulate 15-20 days instead of 30
  ```batch
  --start-date 2024-11-15 --end-date %TODAY%
  ```

- **Reduce ticker count**: Use 5-10 tickers instead of 15
  ```python
  tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "AMZN"]
  ```

- **Disable twice-daily**: Remove `--twice-daily` flag
  ```batch
  REM Remove this line from run_simulation.bat:
  --twice-daily
  ```

- **Disable AI analysis**: Already disabled in simulation for speed
  ```python
  # In run_analysis(), include_ai=False is set
  compute_advanced_quant_signals(..., include_ai=False)
  ```

#### 2. No Output Files Generated

**Symptom**: `simulation_results/` folder is empty

**Checks**:
```powershell
# Verify Python environment
& "C:\...\python.exe" --version

# Check if scripts exist
Test-Path "daily_trading_simulation.py"
Test-Path "stock_picker_advanced_quantitative.py"
Test-Path "stock_picker_nlg_explanations.py"

# Run manually to see errors
python daily_trading_simulation.py --start-date 2024-11-15 --end-date 2024-11-20
```

**Common Causes**:
- Python virtual environment not activated in batch file
- Missing dependencies (yfinance, pandas, numpy, etc.)
- Insufficient permissions to write files
- Script errors (check console output)

#### 3. Task Shows "Access Denied"

**Symptom**: Task won't run, shows access denied error

**Solution**:
```powershell
# Re-create tasks with administrator privileges
# Right-click PowerShell → Run as Administrator
.\setup_daily_simulation.ps1
```

#### 4. Analysis Scripts Fail

**Symptom**: "No ranked signals file generated" errors

**Debugging**:
```powershell
# Run advanced quantitative script manually
cd "american stocks root cause analysis"
python stock_picker_advanced_quantitative.py

# Check for errors in output
# Verify tickers are valid
# Ensure internet connection for Yahoo Finance
```

**Common Issues**:
- Invalid ticker symbols
- No internet connection
- Yahoo Finance rate limiting
- Missing Python libraries

#### 5. Scheduled Tasks Don't Run

**Symptom**: Tasks enabled but not executing at scheduled time

**Checks**:
```powershell
# View task history
Get-ScheduledTask -TaskName "DailyTradingSimulation_Morning" | 
    Get-ScheduledTaskInfo

# Check last run result
schtasks /Query /TN "DailyTradingSimulation_Morning" /V /FO LIST
```

**Solutions**:
- Ensure computer is on at scheduled time
- Check task is enabled: `Enable-ScheduledTask`
- Verify "Run whether user is logged on or not" setting
- Check task triggers are correct (9:00 AM, 4:30 PM)

#### 6. Memory Errors

**Symptom**: "MemoryError" or "Out of memory" errors

**Solutions**:
- Reduce number of tickers
- Shorten simulation period
- Close other memory-intensive applications
- Increase system virtual memory

#### 7. Price Data Missing

**Symptom**: "No price data available" warnings

**Causes**:
- Weekend or holiday (no market data)
- Recently IPO'd stock (limited history)
- Delisted stock
- Yahoo Finance API issues

**Solutions**:
- Ensure tickers are actively traded
- Use stocks with sufficient history (>1 year)
- Check ticker symbols are correct
- Retry later if Yahoo Finance is down

---

## 📈 Expected Results

### Typical Simulation Run

**Duration**: 30-60 minutes (30 days, twice-daily, 15 tickers)

**Console Output**:
```
============================================================
DAILY TRADING SIMULATION
============================================================
Period: 2024-11-01 to 2024-12-01
Initial Capital: $100,000.00
Max Positions: 5
Position Size: 20%
Run Twice Daily: True
============================================================

Fetching market calendar...
Loaded data for 15 tickers

🗓️  Simulating 21 trading days...
Sessions per day: 2 (AM + PM)

============================================================
📅 Day 1/21: 2024-11-01 (Friday)
============================================================

🌅 MORNING SESSION (9:00 AM)

  📊 Running Advanced Quantitative Analysis (am)...
  ✅ Analysis complete: ranked_signals_advanced_20241101_090245.csv
  
  📝 Generating NLG Explanations...
  ✅ NLG explanations complete: nlg_analysis_20241101_090250

  🏆 Top 5 Rankings:
    1. NVDA   | Score: 0.8234 | CAGR:  45.2% | Sharpe:  1.85
    2. AAPL   | Score: 0.7891 | CAGR:  32.1% | Sharpe:  1.72
    3. META   | Score: 0.7654 | CAGR:  38.5% | Sharpe:  1.68
    4. MSFT   | Score: 0.7432 | CAGR:  28.9% | Sharpe:  1.54
    5. GOOGL  | Score: 0.7201 | CAGR:  25.3% | Sharpe:  1.41

  💼 Executed 5 trades:
    BUY   40 NVDA   @ $  500.25
    BUY  111 AAPL   @ $  180.10
    BUY   57 META   @ $  350.15
    BUY   52 MSFT   @ $  380.50
    BUY  142 GOOGL  @ $  140.05

🌆 AFTERNOON SESSION (4:00 PM)
  ✓ No trades needed (portfolio aligned)

📊 End of Day Summary:
  Portfolio Value:  $  100,125.50
  Cash:             $    1,234.25
  Return:                  +0.13%
  Active Positions:             5
  Holdings:         NVDA(40), AAPL(111), META(57), MSFT(52), GOOGL(142)

[... continues for 21 days ...]

============================================================
🎯 SIMULATION COMPLETE - FINAL REPORT
============================================================

💰 PERFORMANCE METRICS:
  Initial Capital:       $    100,000.00
  Final Value:           $    112,450.75
  Total Return:                  +12.45%
  Sharpe Ratio:                    1.82
  Max Drawdown:                   -5.23%

📈 TRADING ACTIVITY:
  Total Trades:                     124
  Buy Orders:                        62
  Sell Orders:                       62
  Transaction Costs:     $      124.51
  Average Cost/Trade:    $        1.00

📊 SIMULATION DETAILS:
  Days Simulated:                    21
  Sessions:                          42
  Max Positions Held:                 5

============================================================

✅ Results saved to 'simulation_results/':
  📄 trade_log_20251201_163045.csv
  📄 portfolio_history_20251201_163045.csv
  📊 performance_chart_20251201_163045.png

🎉 Simulation complete!
```

### Performance Interpretation

**Good Results**:
- ✅ Total Return: >10% (for 30-day period)
- ✅ Sharpe Ratio: >1.5
- ✅ Max Drawdown: <10%
- ✅ Win Rate: >50%

**Typical Results**:
- Total Return: 5-15% (monthly)
- Sharpe Ratio: 1.0-2.0
- Max Drawdown: 5-15%
- Transaction Costs: <1% of returns

**Red Flags**:
- ❌ Negative returns consistently
- ❌ Sharpe Ratio <0.5
- ❌ Max Drawdown >25%
- ❌ Excessive trading (>500 trades/month)

---

## 🎯 Best Practices

### 1. Monitor Daily

Check results after each automated run:
```powershell
cd "simulation_results"
ls -Sort LastWriteTime | select -First 3
```

### 2. Archive Old Results

Monthly archive to prevent clutter:
```powershell
# Create monthly archive
$month = Get-Date -Format "yyyy-MM"
New-Item -ItemType Directory -Path "archive/$month" -Force
Move-Item "simulation_results/*" "archive/$month/"
```

### 3. Review Performance Weekly

Analyze trends:
- Compare week-over-week returns
- Check if strategy consistency
- Identify problematic tickers
- Adjust parameters if needed

### 4. Backup Configurations

Save working configurations:
```powershell
# Backup current setup
Copy-Item "daily_trading_simulation.py" "backups/daily_trading_simulation_$(Get-Date -Format 'yyyyMMdd').py"
Copy-Item "run_simulation.bat" "backups/run_simulation_$(Get-Date -Format 'yyyyMMdd').bat"
```

### 5. Test Changes

Before modifying live automation:
```powershell
# Test with short period
python daily_trading_simulation.py --start-date 2024-11-25 --end-date 2024-11-29
```

### 6. Document Adjustments

Keep notes on parameter changes and their effects:
- Date changed X → Return changed from Y% to Z%
- Ticker list modified → Sharpe improved by N
- Position size adjusted → Drawdown reduced by M%

---

## 📞 Support & Maintenance

### Log Files

Check these if issues occur:
- `simulation_scheduler_log.txt` (if using Python scheduler)
- Task Scheduler history (Windows Event Viewer)
- Console output from manual runs

### Health Checks

Weekly verification:
```powershell
# 1. Check tasks are enabled
Get-ScheduledTask -TaskName "DailyTradingSimulation_*" | Select TaskName, State

# 2. Verify recent outputs
ls simulation_results | Sort-Object LastWriteTime -Desc | Select-Object -First 5

# 3. Check disk space
Get-PSDrive C | Select-Object Used, Free

# 4. Validate Python environment
& "C:\...\python.exe" -c "import pandas; import yfinance; print('OK')"
```

### Performance Optimization

If simulations become slow:
1. **Profile execution time** per component
2. **Cache frequently-used data** (price history)
3. **Parallelize analysis** across tickers
4. **Use faster storage** (SSD vs HDD)
5. **Upgrade hardware** (more RAM, faster CPU)

---

## 📚 Related Documentation

- **`WORKFLOW.md`** - Analysis pipeline workflow
- **`ARCHITECTURE.md`** - System architecture details
- **`SIMULATION_AUTOMATION_GUIDE.md`** - Comprehensive automation guide
- **`stock_picker_advanced_quantitative.py`** - Quantitative analysis implementation
- **`stock_picker_nlg_explanations.py`** - NLG generation implementation

---

## 🔄 Version History

**v1.0** - December 1, 2025
- Initial implementation
- Twice-daily automation
- Windows Task Scheduler integration
- Rolling 30-day simulation window
- Comprehensive output files

---

## 📊 Quick Reference

### Key Files
- `daily_trading_simulation.py` - Main simulation script
- `run_simulation.bat` - Execution wrapper
- `setup_daily_simulation.ps1` - Automation setup
- `simulation_results/` - Output folder

### Key Commands
```powershell
# Run simulation manually
python daily_trading_simulation.py

# Setup automation
.\setup_daily_simulation.ps1  # (as Administrator)

# View tasks
Get-ScheduledTask -TaskName "DailyTradingSimulation_*"

# Run task now
Start-ScheduledTask -TaskName "DailyTradingSimulation_Morning"

# View results
ls simulation_results
```

### Key Metrics
- **Total Return**: (Final - Initial) / Initial × 100%
- **Sharpe Ratio**: (Return / Risk) × √252
- **Max Drawdown**: Worst peak-to-trough decline
- **Win Rate**: Profitable trades / Total trades

---

*Last Updated: December 1, 2025*
