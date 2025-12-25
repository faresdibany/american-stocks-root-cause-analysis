# US Stocks Hybrid Picker - System Architecture

This document describes the technical architecture, component design, data structures, and integration patterns for the American stocks analysis system.

## 1. System Overview

### High-Level Architecture
```
┌─────────────────────────────────────────────────────────────┐
│                     Configuration Layer                      │
│  (Config dataclass, Weights, PredictConfig)                 │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                   Data Acquisition Layer                     │
│  ┌─────────────┐  ┌──────────────┐  ┌──────────────────┐  │
│  │  yfinance   │→ │ pandas-reader│→ │ Alpha Vantage   │  │
│  │  (primary)  │  │  (fallback)  │  │  (last resort)  │  │
│  └─────────────┘  └──────────────┘  └──────────────────┘  │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                  Processing Pipeline Layer                   │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │ Quant Module   │  │  AI Module     │  │ Fusion Module│ │
│  │ • CAGR         │  │ • Headlines    │  │ • Normalize  │ │
│  │ • Sharpe       │  │ • Sentiment    │  │ • Weight     │ │
│  │ • Volatility   │  │ • TF-IDF       │  │ • Rank       │ │
│  │ • Max Drawdown │  │ • World News   │  │ • Portfolio  │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
└────────────────┬────────────────────────────────────────────┘
                 │
                 ↓
┌─────────────────────────────────────────────────────────────┐
│                    Analysis & Output Layer                   │
│  ┌────────────────┐  ┌────────────────┐  ┌──────────────┐ │
│  │  Backtest      │  │ Risk Assistant │  │ Persistence  │ │
│  │  • Performance │  │ • Warnings     │  │ • CSV Export │ │
│  │  • Metrics     │  │ • Probabilities│  │ • Logging    │ │
│  └────────────────┘  └────────────────┘  └──────────────┘ │
└─────────────────────────────────────────────────────────────┘
```

## 2. Component Descriptions

### 2.1 Configuration Module
**Purpose:** Centralized parameter management and type safety.

**Key Classes:**
```python
@dataclass
class Config:
    tickers: List[str]           # Universe of stocks
    period: str                  # Price history ("1y", "2y", etc.)
    interval: str                # Bar frequency ("1d", "1h")
    top_k: int                   # Portfolio size
    max_news: int                # Headlines per ticker
    lookback_days: int           # News recency window
    optimize_weights: bool       # Grid search toggle
    target_currency: str | None  # FX conversion (usually None for USD)

@dataclass
class Weights:
    CAGR: float = 0.45           # Momentum weight
    Sharpe: float = 0.35         # Risk-adjusted return weight
    Sentiment: float = 0.12      # AI sentiment weight
    TopicMomentum: float = 0.08  # TF-IDF topic weight

@dataclass
class PredictConfig:
    horizon_days: int = 5        # Warning prediction window
    prob_buy: float = 0.60       # BUY threshold
    prob_sell: float = 0.40      # SELL threshold
    watch_band: float = 0.05     # Neutral zone
    vol_window: int = 20         # Volatility estimation period
    risk_mult_stop: float = 1.25 # Stop-loss multiplier
    risk_mult_tp: float = 2.0    # Take-profit multiplier
    require_headlines: int = 2   # Minimum news threshold
```

**Design Rationale:** Dataclasses provide immutability, type hints, and easy serialization (for logging configs to JSON).

### 2.2 Data Acquisition Layer

#### Price Provider Chain
**Architecture Pattern:** Chain of Responsibility with auto-fallback.

```python
class PriceProvider(ABC):
    @abstractmethod
    def fetch(tickers, start, end) -> pd.DataFrame: pass

class YFinanceProvider(PriceProvider):
    # Primary: Free, reliable, no API key
    
class StooqProvider(PriceProvider):
    # Fallback: Alternative data source
    
class AlphaVantageProvider(PriceProvider):
    # Last resort: Requires API key
```

**Fallback Logic:**
1. Try yfinance → Success? Return data
2. Catch exception → Try Stooq
3. Still fails? → Try Alpha Vantage (if key available)
4. All fail? → Raise aggregated error

**Data Cleaning Pipeline:**
```
Raw Price Data (OHLCV)
    ↓ Extract Close prices
    ↓ Drop columns with >10% NaN
    ↓ Forward fill small gaps
    ↓ Backward fill initial NaN
    ↓ Drop remaining NaN rows
Clean Price DataFrame
```

#### News Provider
**Architecture:** Google News RSS scraping with structured output.

```python
def fetch_headlines_us(ticker, max_items, lookback_days) -> List[str]:
    # Query construction
    query = f"{ticker} stock price NYSE NASDAQ when:{lookback_days}d"
    
    # URL encoding (critical for special chars)
    encoded = quote_plus(query)
    
    # RSS endpoint
    url = f"https://news.google.com/rss/search?q={encoded}&hl=en&gl=US&ceid=US:en"
    
    # Parse and extract titles
    return [entry.title for entry in feedparser.parse(url).entries[:max_items]]
```

**World News Categories (US-Specific):**
- `economy`: GDP, inflation, Fed rates, employment
- `geopolitics`: Trade, sanctions, international tensions
- `energy`: Oil, gas, commodities
- `finance`: Banking, monetary policy, Wall Street
- `technology`: Tech stocks, AI, cybersecurity
- `consumer`: Retail, e-commerce, spending
- `healthcare`: Biotech, pharma, FDA
- `industrial`: Manufacturing, supply chain

### 2.3 Quantitative Analysis Module

**Input:** Price DataFrame (dates × tickers)
**Output:** Quant DataFrame (tickers × metrics)

**Metrics Computation:**
```python
def compute_quant_signals(prices: pd.DataFrame) -> pd.DataFrame:
    returns = prices.pct_change().dropna()
    
    for ticker in prices.columns:
        CAGR = _annualize_return(prices[ticker])
        Sharpe = _sharpe(returns[ticker])
        Volatility = returns[ticker].std() * sqrt(252)
        MaxDD = _max_drawdown((1 + returns[ticker]).cumprod())
```

**Helper Functions:**
- `_annualize_return(series)`: `(final/initial)^(1/years) - 1`
- `_sharpe(returns)`: `(mean * 252) / (std * sqrt(252))`
- `_max_drawdown(equity)`: `min(equity / cummax - 1)`

**Design Notes:**
- Uses population std (ddof=0) for consistency
- Annualization factor = 252 trading days
- CAGR computed on equity curve, not arithmetic average

### 2.4 AI/Sentiment Analysis Module

**Architecture:** Multi-stage pipeline with graceful degradation.

#### Stage 1: Headline Acquisition
```
For each ticker:
    ├─ Fetch stock-specific headlines (15-50 items)
    └─ Fetch world news by category (once per run)
```

#### Stage 2: Sentiment Scoring
```python
class SentimentModel:
    def __init__(self):
        if pipeline and transformers available:
            self.finbert = pipeline("sentiment-analysis", 
                                   model="ProsusAI/finbert")
        else:
            self.finbert = None  # Fallback to neutral
    
    def score(self, texts: List[str]) -> float:
        if self.finbert is None:
            return 0.0  # Neutral if model unavailable
        
        # Batch prediction
        results = self.finbert(texts)
        
        # Convert to [-1, 1]
        mapping = {"positive": 1.0, "negative": -1.0, "neutral": 0.0}
        scores = [mapping[r['label'].lower()] for r in results]
        
        return np.mean(scores)
```

**FinBERT Details:**
- Model: `ProsusAI/finbert` (financial domain fine-tuned BERT)
- Output: 3-class (positive/negative/neutral) with probabilities
- Batch size: Auto (handles 10-50 headlines efficiently)
- Memory: ~1.5GB loaded, ~500MB cache on disk

#### Stage 3: Topic Momentum
```python
def topic_momentum(headlines: List[str]) -> float:
    # Financial bullish terms
    bullish_terms = [
        "outperform", "upgrade", "strong guidance", "beat estimates",
        "dividend", "buyback", "record earnings", "growth", "partnership",
        "expansion", "contract", "approval", "breakthrough"
    ]
    
    # TF-IDF vectorization
    vectorizer = TfidfVectorizer(max_features=2048)
    tfidf_matrix = vectorizer.fit_transform(headlines)
    
    # Extract bullish term weights
    vocab = vectorizer.get_feature_names_out()
    bullish_indices = [i for i, term in enumerate(vocab) 
                      if term in bullish_terms]
    
    # Average weight of bullish terms
    weights = tfidf_matrix[:, bullish_indices].mean()
    
    # Squash to [0, 1] via sigmoid
    return 1 / (1 + exp(-8 * (weights - 0.05)))
```

**Design Rationale:**
- TF-IDF captures term importance relative to corpus
- Sigmoid ensures output in [0,1] range
- Shift/scale parameters tuned empirically

#### Stage 4: World News Integration
```python
def compute_world_news_impact(world_news, sentiment_model):
    impacts = {}
    for category, headlines in world_news:
        sentiment = sentiment_model.score(headlines)
        impacts[category] = sentiment
    return impacts

def weight_world_sentiment(ticker, world_impacts):
    # Sector heuristics (simplified example)
    is_tech = ticker in ["AAPL", "MSFT", "GOOGL", ...]
    is_finance = ticker in ["JPM", "BAC", "GS", ...]
    is_energy = ticker in ["XOM", "CVX", "COP", ...]
    
    weighted_sum = 0.0
    total_weight = 0.0
    
    # Sector-specific weighting
    if is_tech and "technology" in world_impacts:
        weighted_sum += world_impacts["technology"] * 1.0
        total_weight += 1.0
    
    if is_finance and "finance" in world_impacts:
        weighted_sum += world_impacts["finance"] * 1.0
        total_weight += 1.0
    
    # Universal categories (affect all)
    for cat in ["economy", "geopolitics"]:
        weighted_sum += world_impacts.get(cat, 0.0) * 0.5
        total_weight += 0.5
    
    return weighted_sum / total_weight if total_weight > 0 else 0.0
```

#### Stage 5: Sentiment Fusion
```python
def compute_combined_sentiment(stock_sentiment, world_sentiment):
    # US stocks: 70% stock-specific, 30% macro
    # (vs. 60/40 for EGX due to higher idiosyncratic risk in emerging markets)
    return 0.7 * stock_sentiment + 0.3 * world_sentiment
```

### 2.5 Score Fusion Module

**Purpose:** Combine normalized quant + AI signals into unified ranking.

**Normalization Strategy:**
```python
def _min_max(series: pd.Series) -> pd.Series:
    # Cross-sectional normalization to [0, 1]
    return (series - series.min()) / (series.max() - series.min())
```

**Fusion Algorithm:**
```python
def fuse_scores(quant, ai, weights):
    # Join DataFrames on ticker index
    df = quant.join(ai, how="left").fillna(0)
    
    # Normalize to [0, 1] for each metric
    df["CAGR_n"] = _min_max(df["CAGR"])
    df["Sharpe_n"] = _min_max(df["Sharpe"])
    df["CombinedSentiment_n"] = _min_max(df["CombinedSentiment"])
    df["TopicMomentum_n"] = _min_max(df["TopicMomentum"])
    
    # Weighted combination
    df["Final_Score"] = (
        weights.CAGR * df["CAGR_n"] +
        weights.Sharpe * df["Sharpe_n"] +
        weights.Sentiment * df["CombinedSentiment_n"] +
        weights.TopicMomentum * df["TopicMomentum_n"]
    )
    
    # Sort with stable tie-breakers
    return df.sort_values(
        ["Final_Score", "Sharpe", "CAGR"], 
        ascending=[False, False, False]
    )
```

**Weight Optimization (Optional):**
```python
def simple_weight_search(prices, ai, top_k):
    # Brute-force grid search
    grid = []
    for cagr in [0.3, 0.4, 0.5]:
        for sharpe in [0.2, 0.3, 0.4]:
            for sentiment in [0.05, 0.1, 0.2]:
                topic = 1.0 - (cagr + sharpe + sentiment)
                grid.append(Weights(cagr, sharpe, sentiment, topic))
    
    best_sharpe = -inf
    best_weights = Weights()
    
    for w in grid:
        ranked = fuse_scores(quant, ai, w)
        picks = ranked.head(top_k).index
        portfolio_returns = prices[picks].pct_change().mean(axis=1)
        sharpe = _sharpe(portfolio_returns)
        
        if sharpe > best_sharpe:
            best_sharpe = sharpe
            best_weights = w
    
    return best_weights
```

**Optimization Notes:**
- Grid size: ~27 candidates (3×3×3)
- Objective: Maximize in-sample equal-weight portfolio Sharpe
- Risk: Overfitting (use with caution or cross-validate)

### 2.6 Portfolio Construction Module

```python
def build_portfolio(ranked, top_k):
    # Select top N by Final_Score
    picks = ranked.head(top_k)
    
    # Equal weight allocation
    picks["Weight"] = 1.0 / len(picks)
    
    # Return key columns for trading
    return picks[["CAGR", "Sharpe", "CombinedSentiment", 
                  "TopicMomentum", "Final_Score", "Weight"]]
```

**Alternative Weighting Schemes (Not Implemented):**
- Risk parity: `Weight ∝ 1/Volatility`
- Score-weighted: `Weight ∝ Final_Score`
- Markowitz optimization: Min variance for target return

### 2.7 Directional Warning Assistant

**Purpose:** Generate actionable buy/sell/watch signals with risk levels.

```python
@dataclass
class PredictConfig:
    horizon_days: int = 5
    prob_buy: float = 0.60    # Threshold for BUY signal
    prob_sell: float = 0.40   # Threshold for SELL signal
    watch_band: float = 0.05  # Neutral zone around 0.50
    vol_window: int = 20
    risk_mult_stop: float = 1.25   # Stop = price - 1.25*ATR
    risk_mult_tp: float = 2.0      # TP = price + 2.0*ATR
    require_headlines: int = 2

def add_directional_assistant(prices, ai, cfg):
    warnings = []
    
    for ticker in prices.columns:
        # Skip if insufficient news coverage
        if ai.loc[ticker, "Headlines"] < cfg.require_headlines:
            continue
        
        # Extract signals
        sentiment = ai.loc[ticker, "CombinedSentiment"]
        momentum = ai.loc[ticker, "TopicMomentum"]
        
        # Compute probability (heuristic blend)
        prob = 0.5 + 0.3 * sentiment + 0.2 * momentum
        prob = np.clip(prob, 0.0, 1.0)
        
        # Determine signal
        if prob >= cfg.prob_buy:
            signal = "BUY"
        elif prob <= cfg.prob_sell:
            signal = "SELL"
        elif abs(prob - 0.5) > cfg.watch_band:
            signal = "WATCH"
        else:
            continue  # Neutral, skip
        
        # Compute risk levels (ATR-based)
        returns = prices[ticker].pct_change()
        atr = returns.rolling(cfg.vol_window).std().iloc[-1] * prices[ticker].iloc[-1]
        
        stop_loss = prices[ticker].iloc[-1] - cfg.risk_mult_stop * atr
        take_profit = prices[ticker].iloc[-1] + cfg.risk_mult_tp * atr
        
        warnings.append({
            "Ticker": ticker,
            "Signal": signal,
            "Probability": prob,
            "StopLoss": stop_loss if signal == "BUY" else None,
            "TakeProfit": take_profit if signal == "BUY" else None,
            "CurrentPrice": prices[ticker].iloc[-1],
            "ATR": atr
        })
    
    return {"warnings": pd.DataFrame(warnings)}
```

**Probability Model:**
- Base: 50% (neutral)
- Sentiment contribution: ±30% (bounded by [-1, 1])
- Topic momentum: +20% (bounded by [0, 1])
- Final: Clipped to [0, 1]

**Risk Calculation:**
- ATR = 20-day rolling std × current price
- Stop loss = price - 1.25 × ATR (~84% of distribution)
- Take profit = price + 2.0 × ATR (2:1 reward-risk ratio)

## 3. Data Structures & Schemas

### Price DataFrame
```
Index: DatetimeIndex (daily frequency)
Columns: Ticker symbols (str)
Values: Closing prices (float)
Shape: (T dates, N tickers)
```

### Quant DataFrame
```
Index: Ticker (str)
Columns: ["CAGR", "Sharpe", "Volatility", "MaxDD"]
Values: Float (annualized metrics)
Shape: (N tickers, 4 metrics)
```

### AI DataFrame
```
Index: Ticker (str)
Columns: [
    "Sentiment",              # Combined score [-1, 1]
    "StockSentiment",         # Company-specific [-1, 1]
    "WorldNewsScore",         # Macro/sector [-1, 1]
    "TopicMomentum",          # Bullish term density [0, 1]
    "Headlines",              # Count (int)
    "Sent_Pos",               # Positive count (int)
    "Sent_Neutral",           # Neutral count (int)
    "Sent_Neg",               # Negative count (int)
    "Sent_Examples_Pos",      # Sample headlines (str, pipe-separated)
    "Sent_Examples_Neg",      # Sample headlines (str)
    "Sent_Examples_Neutral",  # Sample headlines (str)
    "World_economy",          # Category sentiment [-1, 1]
    "World_finance",          # ...
    "World_technology",       # ...
    # ... other world categories
]
Shape: (N tickers, 10+ columns)
```

### Ranked DataFrame
```
Index: Ticker (str)
Columns: All quant + AI columns, plus:
    "*_n" - Normalized versions [0, 1]
    "Final_Score" - Weighted composite [0, 1]
Sorted by: Final_Score descending
Shape: (N tickers, 20+ columns)
```

### Warnings DataFrame
```
Index: RangeIndex (auto-incrementing)
Columns: [
    "Ticker",        # Stock symbol
    "Signal",        # BUY/SELL/WATCH
    "Probability",   # Float [0, 1]
    "StopLoss",      # Price level (float or None)
    "TakeProfit",    # Price level (float or None)
    "CurrentPrice",  # Latest close
    "ATR"           # Average True Range
]
Shape: (M warnings, 7 columns)
```

## 4. External Dependencies

### Required (Core Functionality)
| Package | Purpose | Version | License |
|---------|---------|---------|---------|
| pandas | Data structures & manipulation | ≥2.0 | BSD |
| numpy | Numerical operations | ≥1.24 | BSD |
| yfinance | Price data source | ≥0.2 | Apache 2.0 |

### Optional (Enhanced Features)
| Package | Purpose | Fallback Behavior |
|---------|---------|-------------------|
| feedparser | RSS news parsing | No headlines (sentiment=0) |
| transformers | FinBERT sentiment | Neutral sentiment (0) |
| torch | Transformer backend | (required by transformers) |
| scikit-learn | TF-IDF topic momentum | Topic score=0 |
| pandas-datareader | Stooq price fallback | Skip to Alpha Vantage |
| requests | Alpha Vantage API | Raise error if all fail |
| matplotlib | Performance charts | Skip plotting |

### Model Downloads (First Run Only)
- `ProsusAI/finbert`: ~500MB (cached in `~/.cache/huggingface/`)
- Tokenizer vocab: ~500KB

## 5. Performance Characteristics

### Execution Time (50 Tickers, 50 Headlines Each)
| Stage | Time | Bottleneck |
|-------|------|-----------|
| Price data load | 5-10s | Network I/O |
| Quant computation | <1s | CPU (vectorized) |
| World news fetch | 10-15s | Network + RSS parse |
| Ticker headlines | 60-90s | Network (2s/ticker) |
| FinBERT inference | 30-45s | GPU if available, else CPU |
| TF-IDF + fusion | 2-3s | CPU |
| Total | **110-160s** | Network-bound |

### Memory Footprint
| Component | RAM |
|-----------|-----|
| Base Python + pandas | 200MB |
| Price data (50 stocks × 252 days) | 10MB |
| FinBERT model | 1.5GB |
| Headlines (2500 total) | 5MB |
| **Peak Usage** | **~1.8GB** |

### Optimization Opportunities
1. **Parallel headline fetching**: Use `concurrent.futures` (3-4× speedup)
2. **Batch FinBERT**: Already implemented (10-20× vs. sequential)
3. **Cache world news**: Reuse across multiple runs same day
4. **GPU acceleration**: PyTorch CUDA (5-10× for sentiment)

## 6. Error Handling & Resilience

### Price Data Fallback Chain
```python
try:
    prices = yfinance_fetch()
except Exception:
    try:
        prices = stooq_fetch()
    except Exception:
        try:
            prices = alphavantage_fetch()
        except Exception:
            raise RuntimeError("All price providers failed")
```

### Graceful AI Degradation
```python
# If FinBERT unavailable
if self.finbert is None:
    return 0.0  # Neutral sentiment

# If feedparser missing
if feedparser is None:
    return []  # No headlines

# If sklearn missing
if TfidfVectorizer is None:
    return 0.0  # No topic momentum
```

### Data Quality Checks
```python
# Remove tickers with excessive missing data
valid_tickers = prices.columns[prices.isna().mean() < 0.1]
prices = prices[valid_tickers]

# Forward/backward fill small gaps
prices = prices.ffill().bfill()

# Drop remaining NaN rows
prices = prices.dropna()
```

## 7. Extension Points

### Adding New Sentiment Models
```python
class CustomSentimentModel(SentimentModel):
    def __init__(self):
        super().__init__()
        self.custom_model = load_your_model()
    
    def score(self, texts):
        # Override with custom logic
        return self.custom_model.predict(texts).mean()
```

### Custom Factor Integration
```python
def compute_custom_factor(prices):
    # Example: RSI indicator
    delta = prices.diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = -delta.where(delta < 0, 0).rolling(14).mean()
    rs = gain / loss
    rsi = 100 - (100 / (1 + rs))
    return rsi

# Add to fusion
df["RSI_n"] = _min_max(df["RSI"])
df["Final_Score"] += weights.RSI * df["RSI_n"]
```

### Alternative News Sources
```python
def fetch_headlines_bloomberg(ticker):
    # Implement Bloomberg Terminal API
    # or web scraping (respect ToS)
    pass

def fetch_headlines_reuters(ticker):
    # Reuters API integration
    pass
```

## 8. Security & Rate Limiting

### API Rate Limits
| Provider | Limit | Mitigation |
|----------|-------|-----------|
| Yahoo Finance | Unofficial (~2000 req/hr) | Batch requests, add jitter |
| Google News RSS | Unknown | 0.05s sleep between requests |
| Alpha Vantage | 5 req/min (free tier) | Last resort only |

### Credential Management
```python
# Alpha Vantage key (if needed)
api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
if not api_key:
    raise EnvironmentError("Set ALPHAVANTAGE_API_KEY")
```

**Never hardcode API keys in scripts.**

### Network Error Handling
```python
import requests
from requests.adapters import HTTPAdapter
from requests.packages.urllib3.util.retry import Retry

session = requests.Session()
retry = Retry(total=3, backoff_factor=1, status_forcelist=[429, 500, 502, 503, 504])
adapter = HTTPAdapter(max_retries=retry)
session.mount("http://", adapter)
session.mount("https://", adapter)
```

## 9. Testing Strategy

### Unit Tests
```python
def test_min_max_normalization():
    s = pd.Series([1, 2, 3, 4, 5])
    normalized = _min_max(s)
    assert normalized.min() == 0.0
    assert normalized.max() == 1.0

def test_sharpe_calculation():
    returns = pd.Series([0.01, -0.01, 0.02, -0.005, 0.015])
    sharpe = _sharpe(returns)
    assert isinstance(sharpe, float)
    assert not np.isnan(sharpe)

def test_sentiment_fallback():
    model = SentimentModel()
    if model.finbert is None:
        assert model.score(["test"]) == 0.0
```

### Integration Tests
```python
def test_end_to_end_pipeline():
    cfg = Config(
        tickers=["AAPL", "MSFT"],
        period="1mo",
        max_news=5
    )
    
    prices = load_prices(cfg.tickers, cfg.period, "1d")
    assert len(prices.columns) > 0
    
    quant = compute_quant_signals(prices)
    assert "CAGR" in quant.columns
    
    ai = compute_ai_signals_us(quant.index.tolist(), max_news=5)
    assert "Sentiment" in ai.columns
    
    ranked = fuse_scores(quant, ai, Weights())
    assert "Final_Score" in ranked.columns
```

### Smoke Tests (Pre-Production)
```bash
# Quick validation (5 tickers, 10 headlines)
python stock_picker_daily.py --tickers "AAPL,MSFT,GOOGL,AMZN,META" --max_news 10

# Check outputs exist
ls -lh rankings_us_*.csv
ls -lh ai_signals_us_*.csv
```

## 10. Logging & Monitoring

### Structured Logging
```python
import logging
logging.basicConfig(
    filename="stock_picker.log",
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

logging.info(f"Started analysis for {len(cfg.tickers)} tickers")
logging.info(f"Loaded {len(prices)} days of price data")
logging.warning(f"Missing headlines for {ticker}")
logging.error(f"FinBERT failed: {e}")
```

### Performance Metrics
```python
import time
start = time.time()

# ... analysis code ...

elapsed = time.time() - start
logging.info(f"Total execution time: {elapsed:.1f}s")
logging.info(f"Headlines/second: {total_headlines / elapsed:.1f}")
```

### Health Checks
- Headline count distribution (should be >5 for most tickers)
- Sentiment distribution (shouldn't be all neutral)
- Price data completeness (>90% coverage)
- Model load success (FinBERT available)

## 11. Deployment Patterns

### Standalone Execution
```bash
# Manual command-line run
python stock_picker_hybrid_american_stocks.py
```

### Scheduled Automation (Windows)
```powershell
# Task Scheduler
schtasks /create /tn "StockPickerUS" /tr "python C:\path\to\script.py" /sc daily /st 08:30
```

### Docker Container (Optional)
```dockerfile
FROM python:3.11-slim
WORKDIR /app
COPY requirements.txt .
RUN pip install -r requirements.txt
COPY *.py .
CMD ["python", "stock_picker_hybrid_american_stocks.py"]
```

### Cloud Function (AWS Lambda Example)
```python
def lambda_handler(event, context):
    # Triggered daily via EventBridge
    cfg = Config(tickers=event["tickers"], ...)
    results = run_analysis(cfg)
    
    # Save to S3
    s3.put_object(Bucket="stock-picks", Key="latest.csv", Body=results.to_csv())
    
    return {"statusCode": 200, "body": "Analysis complete"}
```

## 12. Version Control & Reproducibility

### Config Versioning
```python
# Save config with results for reproducibility
with open("config_snapshot.json", "w") as f:
    json.dump(cfg.__dict__, f, indent=2)
```

### Dependency Pinning
```bash
# Generate requirements.txt
pip freeze > requirements.txt

# Or use pipenv
pipenv lock
```

### Model Versioning
```python
# Pin FinBERT version
model_version = "ProsusAI/finbert@v1.0.0"
```

## 13. Future Architecture Enhancements

### Planned Improvements
1. **Database backend**: Replace CSV with PostgreSQL/SQLite
2. **Real-time streaming**: WebSocket for intraday updates
3. **Multi-asset support**: Add forex, crypto, commodities
4. **Advanced NLP**: Named Entity Recognition, aspect-based sentiment
5. **Ensemble models**: Combine FinBERT + domain-specific models
6. **Explainability**: SHAP values for factor attributions
7. **Reinforcement learning**: Adaptive weight optimization
8. **Portfolio optimization**: Markowitz, Black-Litterman integration

### Scalability Considerations
- Horizontal scaling: Distribute tickers across workers
- Caching layer: Redis for headline/sentiment cache
- Message queue: RabbitMQ for async processing
- Monitoring: Prometheus + Grafana dashboards

---

**For operational procedures, see `WORKFLOW.md`.**
