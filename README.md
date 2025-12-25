# American Stocks — Root Cause Analysis (Independent Pipeline)

This folder is an **independent pipeline** that combines:

- **Historical price driver analysis** (trend/regimes/jumps/changepoints + event study vs SPY)
- **Quantitative analysis**
- **AI / world-news sentiment**
- **Social sentiment** (Google News + Reddit + StockTwits best-effort)
- **Stock ranking + consolidated report**

## 📁 Folder Organization

```
american stocks root cause analysis/
├── 📊 core_analysis/           # Pipeline scripts (RCA + ranking + sentiment)
├── 📁 outputs/                 # Generated artifacts (JSON/CSV)
├── 📚 documentation/           # Documentation
└── 📄 README.md                # This file
```

---

## 📊 Important Scripts

**Location**: `core_analysis/`

| Script | Purpose | Output |
|--------|---------|--------|
| `stock_driver_analysis.py` | Per-ticker movement driver report (trend, regimes, jumps, CPD, event attribution) | `outputs/driver_report_<TICKER>_<timestamp>.json` |
| `driver_analysis_batch.py` | Batch runner for multiple tickers | multiple `driver_report_*.json` |
| `stock_picker_hybrid_world_news_extension.py` | Quant + world-news sentiment ranking (macro-aware) | CSV artifacts (script default) |
| `stock_picker_social_sentiment.py` | Quant + social sentiment ranking (News/Reddit/StockTwits best-effort) | CSV artifacts (script default) |
| `rca_pipeline.py` | **End-to-end orchestrator**: driver → world news → social → merged ranking + JSON report | `outputs/ranked_signals_rca_*.csv`, `outputs/rca_pipeline_report_*.json` |
## 🚀 How to run

### Run the full pipeline (recommended)

```powershell
cd .\core_analysis
python .\rca_pipeline.py --tickers AAPL,MSFT,NVDA --period 6mo --max-news 8 --lookback-days 14
```

Artifacts are written to `outputs/` (this folder is intentionally ignored by git).

### Optional stages

Run advanced quantitative stage:

```powershell
cd .\core_analysis
python .\rca_pipeline.py --tickers AAPL,MSFT --period 6mo --with-advanced-quant
```

Run NLG explanations stage:

```powershell
cd .\core_analysis
python .\rca_pipeline.py --tickers AAPL,MSFT --period 6mo --with-nlg
```

### Run only the driver analysis

Run a single ticker:

```powershell
cd .\core_analysis
python .\stock_driver_analysis.py --ticker AAPL --start 2024-01-01 --end 2024-06-01 --out-dir ..\outputs
```

Run multiple tickers:

```powershell
cd .\core_analysis
python .\driver_analysis_batch.py --tickers AAPL MSFT NVDA --start 2024-01-01 --end 2024-06-01 --out-dir ..\outputs
```

---

## 📁 Outputs Folder

**Location**: `outputs/`

Contains generated root-cause analysis reports:

- `driver_report_<TICKER>_<timestamp>.json`

### Output File Types

| Pattern | Description |
|---------|-------------|
| `driver_report_*.json` | Per-stock driver analysis reports |

### 🧹 Cleanup

Outputs folder can grow large. Archive old files:
```powershell
cd outputs
# Move old files to archive
$archiveDate = Get-Date -Format "yyyy-MM"
New-Item -ItemType Directory -Force -Path "archive/$archiveDate"
Move-Item "*.csv" "archive/$archiveDate/" -ErrorAction SilentlyContinue
```

---

## 📚 Documentation Folder

**Location**: `documentation/`

See:

- `documentation/ROOT_CAUSE_ANALYSIS.md` (driver analysis details + roadmap)
        
Step 4: Review performance
        # Check simulation_results/ folder
        # View performance_chart_*.png
```

---

## 📊 File Path Updates

### Import Paths

Scripts now need to reference the new structure:

**From simulation scripts**:
```python
# Add parent directory to path
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_analysis'))

# Now import works
from stock_picker_advanced_quantitative import compute_advanced_quant_signals
```

**From automation scripts**:
```batch
REM Update paths in batch files
cd /d "%~dp0\..\simulation"
python daily_trading_simulation.py
```

### Output Paths

**Analysis outputs**: Relative to script location (unchanged)
**Simulation outputs**: `simulation_results/` (unchanged)
**Archived outputs**: `outputs/archive/YYYY-MM/`

---

## 🎯 Recommendations

### For Daily Use
1. ✅ Use automated twice-daily simulation
2. ✅ Check `simulation_results/` folder daily
3. ✅ Archive `outputs/` folder monthly
4. ✅ Review `documentation/` as needed

### For Development
1. Test changes in `core_analysis/` first
2. Validate with short simulation runs
3. Update automation scripts if paths change
4. Document new features in `documentation/`

### For Maintenance
1. Clean `outputs/` folder regularly (keep last 30 days)
2. Review `simulation_results/` performance trends
3. Update scheduled tasks if schedule changes
4. Backup `core_analysis/` scripts before major changes

---

## 🔧 Troubleshooting

### Scripts Can't Find Imports

**Issue**: `ModuleNotFoundError` after reorganization

**Fix**: Update sys.path in scripts:
```python
import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..', 'core_analysis'))
```

### Automation Scripts Fail

**Issue**: Tasks run but scripts not found

**Fix**: Update paths in automation scripts:
```batch
REM In run_simulation.bat
cd /d "C:\Users\fares\OneDrive\Desktop\Stock Picker\american stocks root cause analysis\simulation"
```

### Outputs Not Found

**Issue**: Scripts looking for CSV files in wrong location

**Fix**: Check output paths are relative:
```python
# Save to current directory or specify full path
output_path = os.path.join(os.path.dirname(__file__), '..', 'outputs', filename)
```

---

## 📞 Need Help?

**Documentation**: Check `documentation/` folder first  
**Analysis Issues**: See `documentation/WORKFLOW.md`  
**Simulation Issues**: See `documentation/SIMULATION_WORKFLOW.md`  
**Automation Issues**: See `documentation/SIMULATION_AUTOMATION_GUIDE.md`

---

*Last Updated: December 1, 2025*
