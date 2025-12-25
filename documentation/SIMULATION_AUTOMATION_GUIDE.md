# Trading Simulation Automation Guide

## 📋 Overview

This guide explains how to automate the `daily_trading_simulation.py` script to run on a schedule. The simulation backtests your trading strategy by running the actual `stock_picker_advanced_quantitative.py` and `stock_picker_nlg_explanations.py` scripts for each day in the simulation period.

---

## 🎯 What Gets Automated

When you automate the simulation, it will:

1. ✅ Run the complete trading simulation for a specified period
2. ✅ Execute `stock_picker_advanced_quantitative.py` for each trading day
3. ✅ Execute `stock_picker_nlg_explanations.py` to generate analysis
4. ✅ Make buy/sell decisions based on rankings
5. ✅ Generate performance reports and charts
6. ✅ Save results to `simulation_results/` folder

---

## 🪟 Windows Automation

### Method 1: Windows Task Scheduler (Recommended)

**Step 1: Run PowerShell Setup Script**

Right-click PowerShell and select "Run as Administrator", then:

```powershell
cd "C:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis"
.\setup_simulation_automation.ps1
```

This creates a scheduled task that runs **monthly on the 1st at 6:00 PM**.

**Step 2: Verify Task**

Open Task Scheduler:
```powershell
taskschd.msc
```

Look for "MonthlyTradingSimulation" task.

**Step 3: Customize Schedule (Optional)**

To change when it runs, edit `setup_simulation_automation.ps1`:
- Modify `$TriggerTime = "18:00"` for different time
- Modify `$DayOfMonth = 1` for different day
- Re-run the setup script

**Manual Test:**
```powershell
Start-ScheduledTask -TaskName "MonthlyTradingSimulation"
```

**Disable:**
```powershell
Disable-ScheduledTask -TaskName "MonthlyTradingSimulation"
```

**Delete:**
```powershell
Unregister-ScheduledTask -TaskName "MonthlyTradingSimulation" -Confirm:$false
```

---

### Method 2: Python Scheduler (Alternative)

**Step 1: Install schedule library**

```powershell
pip install schedule
```

**Step 2: Run the scheduler script**

```powershell
python simulation_scheduler.py
```

This keeps running in the background and executes the simulation **monthly on the 1st at 6:00 PM**.

**To run in background:**

Create a shortcut to `simulation_scheduler.py` and place it in:
```
C:\Users\fares\AppData\Roaming\Microsoft\Windows\Start Menu\Programs\Startup
```

Or use Windows Task Scheduler to run `simulation_scheduler.py` at startup.

**View logs:**
```powershell
Get-Content simulation_scheduler_log.txt -Tail 50
```

---

### Method 3: Manual Batch File

Simply double-click `run_simulation.bat` whenever you want to run the simulation.

Or create a desktop shortcut for easy access.

---

## 🐧 Linux/Mac Automation

### Using Cron

**Step 1: Make script executable**

```bash
chmod +x run_simulation.sh
```

**Step 2: Edit crontab**

```bash
crontab -e
```

**Step 3: Add cron job**

For monthly on 1st at 6:00 PM:
```cron
0 18 1 * * /path/to/stock_picker/american_stocks/run_simulation.sh
```

For weekly on Sunday at 6:00 PM:
```cron
0 18 * * 0 /path/to/stock_picker/american_stocks/run_simulation.sh
```

For daily at 6:00 PM:
```cron
0 18 * * * /path/to/stock_picker/american_stocks/run_simulation.sh
```

**Verify cron job:**
```bash
crontab -l
```

**View logs:**
```bash
tail -f automation_log.txt
```

---

## ⚙️ Configuration

### Simulation Parameters

Edit `run_simulation.bat` or `simulation_scheduler.py` to customize:

| Parameter | Default | Description |
|-----------|---------|-------------|
| `--start-date` | 2024-06-01 | Simulation start date |
| `--end-date` | 2024-12-01 | Simulation end date |
| `--capital` | 100000 | Initial capital ($) |
| `--max-positions` | 5 | Maximum stocks to hold |
| `--position-size` | 0.20 | 20% per position |
| `--twice-daily` | flag | Run AM + PM analysis |

### Schedule Timing

**Recommended schedules:**

- **Monthly (1st of month)**: Good for reviewing strategy performance
- **Weekly (Sundays)**: Regular backtesting with fresh data
- **After market close**: Avoid running during market hours

---

## 📊 Output Files

After each simulation run, check `simulation_results/`:

| File | Description |
|------|-------------|
| `trade_log_TIMESTAMP.csv` | Every buy/sell with prices and costs |
| `portfolio_history_TIMESTAMP.csv` | Portfolio value at each session |
| `performance_chart_TIMESTAMP.png` | Visual performance chart |

Plus all the daily outputs:
- `ranked_signals_advanced_*.csv` - Rankings from each day
- `nlg_analysis_*/` - NLG explanations for each day

---

## 🔧 Troubleshooting

### Simulation Fails to Run

**Check Python path:**
```powershell
Get-Command python
# Should point to: C:\Users\fares\OneDrive\Desktop\Stock Picker\.venv\Scripts\python.exe
```

**Verify script location:**
```powershell
Test-Path "C:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis\daily_trading_simulation.py"
# Should return: True
```

### Task Doesn't Execute

**View task history:**
1. Open Task Scheduler
2. Find "MonthlyTradingSimulation"
3. Click "History" tab
4. Look for errors

**Check permissions:**
- Task should run with "Highest privileges"
- Verify the account has access to the script folder

### Simulation Takes Too Long

**Reduce simulation period:**
```powershell
# Instead of 6 months, try 3 months:
--start-date 2024-09-01 --end-date 2024-12-01
```

**Disable twice-daily:**
```powershell
# Remove the --twice-daily flag (runs once per day instead of twice)
```

**Reduce ticker universe:**
Edit `daily_trading_simulation.py` line 568:
```python
tickers = ["AAPL", "MSFT", "NVDA", "GOOGL", "META"]  # Just 5 stocks
```

---

## 📧 Email Notifications (Optional)

To receive email notifications when simulation completes, add to `run_simulation.bat`:

```batch
REM Send email notification (requires configured SMTP)
python send_email_notification.py "Simulation Complete" "simulation_results/"
```

Create `send_email_notification.py`:
```python
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
import sys

def send_notification(subject, message):
    sender = "your_email@gmail.com"
    receiver = "your_email@gmail.com"
    password = "your_app_password"
    
    msg = MIMEMultipart()
    msg['From'] = sender
    msg['To'] = receiver
    msg['Subject'] = subject
    
    msg.attach(MIMEText(message, 'plain'))
    
    with smtplib.SMTP_SSL('smtp.gmail.com', 465) as server:
        server.login(sender, password)
        server.send_message(msg)

if __name__ == "__main__":
    send_notification(sys.argv[1], sys.argv[2])
```

---

## 🌐 Cloud Deployment (Advanced)

### AWS Lambda + EventBridge

Deploy simulation to run in the cloud:

1. Package simulation + dependencies
2. Upload to AWS Lambda
3. Set EventBridge trigger (monthly schedule)
4. Store results in S3

### GitHub Actions

Create `.github/workflows/simulation.yml`:
```yaml
name: Monthly Trading Simulation

on:
  schedule:
    - cron: '0 18 1 * *'  # 6PM on 1st of month
  workflow_dispatch:

jobs:
  simulate:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v2
      - uses: actions/setup-python@v2
      - run: pip install -r requirements.txt
      - run: python american\ stocks/daily_trading_simulation.py
      - uses: actions/upload-artifact@v2
        with:
          name: simulation-results
          path: american\ stocks/simulation_results/
```

---

## 📝 Best Practices

1. **Test First**: Run manually before automating
   ```powershell
   python daily_trading_simulation.py --start-date 2024-11-01 --end-date 2024-11-30
   ```

2. **Monitor Logs**: Check `simulation_scheduler_log.txt` regularly

3. **Backup Results**: Archive old simulation results periodically

4. **Update Dates**: For rolling backtests, use dynamic dates in scripts

5. **Resource Management**: Simulations can be resource-intensive - avoid running during other heavy tasks

---

## ❓ FAQ

**Q: How long does a simulation take?**  
A: Depends on period and tickers. 6 months with 15 tickers, twice-daily: ~30-60 minutes.

**Q: Can I run multiple simulations in parallel?**  
A: Not recommended - can cause file conflicts. Run sequentially.

**Q: Does this use real money?**  
A: No! This is pure simulation/backtesting. No real trades are executed.

**Q: Can I use this for live trading?**  
A: The simulation framework can be adapted for live trading, but requires broker API integration and careful risk management.

**Q: Will it run if my computer is off?**  
A: No (for local automation). Use cloud deployment for 24/7 operation.

---

## 📞 Support

For issues or questions:
1. Check logs in `simulation_scheduler_log.txt`
2. Review output in `simulation_results/`
3. Verify all scripts are in correct location
4. Ensure virtual environment is activated

---

*Last Updated: December 1, 2025*
