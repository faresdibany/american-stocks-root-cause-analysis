# 🚀 Quick Start - American Stocks Analysis

## 📁 New Folder Structure

Your folder is now organized! Everything has been moved to logical folders:

```
american stocks root cause analysis/
├── 📊 core_analysis/       → All stock analysis scripts
├── 🎮 simulation/          → Trading simulation scripts  
├── ⚙️ automation/          → Scheduling & automation
├── 📁 outputs/             → CSV/JSON/log files
└── 📚 documentation/       → All .md documentation
```

---

## ⚡ Quick Commands

### Run Analysis Once

```powershell
cd core_analysis
python stock_picker_advanced_quantitative.py
python stock_picker_nlg_explanations.py
```

### Run Simulation Once

```powershell
cd simulation
python daily_trading_simulation.py --start-date 2024-11-01 --end-date 2024-12-01 --twice-daily
```

### Set Up Twice-Daily Automation

```powershell
cd automation
# Right-click PowerShell → Run as Administrator
.\setup_daily_simulation.ps1
```

### Check Automation Status

```powershell
Get-ScheduledTask -TaskName "DailyTradingSimulation_*"
```

---

## 📖 Documentation

- **README.md** (this folder) - Complete organization guide
- **documentation/SIMULATION_WORKFLOW.md** - Detailed simulation docs
- **documentation/WORKFLOW.md** - Analysis pipeline docs
- **documentation/SIMULATION_AUTOMATION_GUIDE.md** - Automation setup

---

## 🔧 Important Path Updates

### Your Scheduled Tasks Have Been Updated ✅

The automation scripts now point to the correct folders:
- **Script location**: `automation/run_simulation.bat`
- **Working directory**: `simulation/`
- **Outputs**: `simulation/simulation_results/`

### No Action Needed!

Your existing scheduled tasks will continue to work. The paths have been automatically updated.

---

## 📊 Where to Find Things

| What You Need | Where It Is |
|---------------|-------------|
| **Run analysis manually** | `cd core_analysis` |
| **Run simulation** | `cd simulation` |
| **Set up automation** | `cd automation` |
| **View results** | `simulation/simulation_results/` or `outputs/` |
| **Read docs** | `cd documentation` |
| **Check rankings** | `outputs/ranked_signals_*.csv` |

---

## ✅ Everything Still Works!

Your automation is still active:
- ✅ Morning run: Tomorrow 9:00 AM
- ✅ Afternoon run: Tomorrow 4:30 PM
- ✅ All paths updated automatically
- ✅ Outputs will be generated normally

No changes needed on your part! 🎉

---

*For detailed information, see README.md in this folder*
