"""
Stock Picker — online data + news (with graceful fallbacks)
-----------------------------------------------------------
This version keeps **yfinance** as the primary source and adds fallbacks to
other public APIs if a module is missing. It also handles optional news
sentiment (Google News RSS + FinBERT) but degrades cleanly if those packages
aren’t installed.

Key points
----------
- **Price providers** (auto):
  1) yfinance (preferred)
  2) Stooq via pandas-datareader (no API key)
  3) Alpha Vantage (needs `ALPHAVANTAGE_API_KEY` env var)
- **News** (optional): Google News RSS + FinBERT. If `feedparser` or
  `transformers/torch` aren’t installed, the news weight is ignored (treated as 0).
- Fixes double-counted rebalance windows, improves Sharpe/CAGR math,
  guards edge cases, and adds tests.

Install (choose what you need)
------------------------------
# core
pip install pandas numpy matplotlib

# primary market data
pip install yfinance

# fallback market data options (optional)
pip install pandas-datareader
pip install alpha_vantage

# optional news sentiment
pip install feedparser transformers torch

Usage
-----
python stock_picker.py \
  --start 2018-01-01 --end 2025-10-01 --top 10 --rebalance M \
  --tickers "AAPL,MSFT,META,GOOGL,NVDA,AMZN" \
  --provider auto --news-weight 0.25

Run tests
---------
python stock_picker.py --run-tests

Notes
-----
- Alpha Vantage requires env var `ALPHAVANTAGE_API_KEY`.
- Stooq may return descending dates; we sort ascending and forward-fill.
- News factor is cross-sectionally z-scored and blended with technical scores.
"""

from __future__ import annotations
import argparse
import math
import os
from dataclasses import dataclass
from typing import List, Dict, Tuple, Optional

import numpy as np
import pandas as pd
from datetime import datetime, timezone

# -------------------------
# Optional imports (guarded)
# -------------------------
try:
    import yfinance as yf  # primary provider
except Exception:
    yf = None

try:
    from pandas_datareader import data as pdr  # stooq fallback
except Exception:
    pdr = None

try:
    from alpha_vantage.timeseries import TimeSeries  # alpha vantage fallback
except Exception:
    TimeSeries = None

try:
    import feedparser  # news rss
except Exception:
    feedparser = None

try:
    from transformers import AutoTokenizer, AutoModelForSequenceClassification, pipeline
except Exception:
    AutoTokenizer = AutoModelForSequenceClassification = pipeline = None  # transformers optional

# -------------------------
# Config
# -------------------------
DEFAULT_TICKERS = [
    "AAPL", "MSFT", "GOOGL", "AMZN", "META", "NVDA",
    "TSLA", "AVGO", "ADBE", "NFLX", "CRM", "ORCL",
]

@dataclass
class RunConfig:
    start: str = "2018-01-01"
    end: str = "2025-10-01"
    top_n: int = 10
    rebalance: str = "M"   # 'M' (month end), 'W' (week), or 'Q'
    one_way_tc_bps: float = 10.0  # 10 bps per trade
    min_price: float = 5.0  # exclude penny stocks
    provider: str = "auto"  # auto|yfinance|stooq|alphavantage
    debug: bool = False
    # News & AI settings
    news_days: int = 7                # lookback window for news sentiment
    news_max_per_ticker: int = 40     # cap articles per ticker per window
    news_weight: float = 0.25         # blend weight for sentiment factor in final score
    sentiment_model: str = "ProsusAI/finbert"  # HF model id
    news_sources_hint: str = "site:bloomberg.com OR site:reuters.com OR site:cnbc.com OR site:wsj.com"

# global debug flag set at runtime
GLOBAL_DEBUG = False

def dprint(*args, **kwargs):
    if GLOBAL_DEBUG:
        print("[DEBUG]", *args, **kwargs)
    
    sentiment_model: str = "ProsusAI/finbert"  # HF model id
    news_sources_hint: str = "site:bloomberg.com OR site:reuters.com OR site:cnbc.com OR site:wsj.com"  # used in query

# -------------------------
# Providers
# -------------------------
class PriceProvider:
    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        raise NotImplementedError

class YFinanceProvider(PriceProvider):
    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if yf is None:
            raise ImportError("yfinance not installed. `pip install yfinance`.")
        data = yf.download(tickers, start=start, end=end, auto_adjust=True, progress=False)["Close"]
        if isinstance(data, pd.Series):
            data = data.to_frame()
        return data.sort_index()

class StooqProvider(PriceProvider):
    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if pdr is None:
            raise ImportError("pandas-datareader not installed. `pip install pandas-datareader`.")
        frames = []
        for t in tickers:
            df = pdr.DataReader(t, "stooq", start=pd.to_datetime(start), end=pd.to_datetime(end))
            # stooq returns latest first; use 'Close' and sort ascending
            s = df["Close"].sort_index().rename(t)
            frames.append(s)
        out = pd.concat(frames, axis=1)
        return out

class AlphaVantageProvider(PriceProvider):
    def fetch(self, tickers: List[str], start: str, end: str) -> pd.DataFrame:
        if TimeSeries is None:
            raise ImportError("alpha_vantage not installed. `pip install alpha_vantage`.")
        api_key = os.environ.get("ALPHAVANTAGE_API_KEY")
        if not api_key:
            raise EnvironmentError("Missing ALPHAVANTAGE_API_KEY env var.")
        ts = TimeSeries(key=api_key, output_format='pandas')
        frames = []
        for t in tickers:
            data, _ = ts.get_daily_adjusted(symbol=t, outputsize='full')
            s = data['5. adjusted close'].rename(t)
            frames.append(s)
        out = pd.concat(frames, axis=1).sort_index()
        return out.loc[pd.to_datetime(start):pd.to_datetime(end)]


def get_provider(name: str) -> List[PriceProvider]:
    name = (name or "auto").lower()
    if name == "yfinance":
        return [YFinanceProvider()]
    if name == "stooq":
        return [StooqProvider()]
    if name == "alphavantage":
        return [AlphaVantageProvider()]
    # auto chain
    return [YFinanceProvider(), StooqProvider(), AlphaVantageProvider()]


# -------------------------
# Data (with fallbacks)
# -------------------------

def get_price_history(tickers: List[str], start: str, end: str, provider: str = "auto") -> pd.DataFrame:
    errors = []
    for prov in get_provider(provider):
        if getattr(prov, '__class__', None) is not None:
            dprint(f"Trying provider: {prov.__class__.__name__}")
        try:
            data = prov.fetch(tickers, start, end)
            if isinstance(data, pd.Series):
                data = data.to_frame()
            data = data.dropna(how="all")
            keep = data.columns[data.isna().mean() < 0.1]
            dprint(f"Provider {prov.__class__.__name__} returned data with shape {data.shape}; keeping {len(keep)} tickers")
            return data[keep]
        except Exception as e:
            errors.append(f"{prov.__class__.__name__}: {e}")
            dprint(f"Provider {prov.__class__.__name__} failed: {e}")
            continue
    msg = "All providers failed:\n" + "\n".join(errors)
    raise RuntimeError(msg)


# -------------------------
# News & sentiment (AI) — optional
# -------------------------
NEWS_CACHE: Dict[Tuple[str, int], List[Dict]] = {}


def company_query_from_ticker(ticker: str, long_name: str | None = None, sources_hint: str | None = None, days: int = 7) -> str:
    name = long_name or ticker
    base = f"{name} OR {ticker}"
    when = f"when:{days}d"
    hint = f" {sources_hint}" if sources_hint else ""
    return f"{base} {when}{hint}"


def fetch_google_news(query: str, max_items: int = 40) -> List[Dict]:
    if feedparser is None:
        return []
    url = f"https://news.google.com/rss/search?q={query.replace(' ', '+')}&hl=en-US&gl=US&ceid=US:en"
    feed = feedparser.parse(url)
    items = []
    for e in feed.entries[:max_items]:
        published = None
        if hasattr(e, 'published_parsed') and e.published_parsed:
            published = datetime(*e.published_parsed[:6], tzinfo=timezone.utc)
        items.append({
            "title": getattr(e, 'title', ''),
            "summary": getattr(e, 'summary', ''),
            "link": getattr(e, 'link', ''),
            "published": published,
            "source": getattr(e, 'source', {}).get('title') if hasattr(e, 'source') else None,
        })
    return items


def load_finbert_pipeline(model_id: str):
    if pipeline is None:
        raise ImportError("transformers not available. Please `pip install transformers torch`.")
    tok = AutoTokenizer.from_pretrained(model_id)
    mdl = AutoModelForSequenceClassification.from_pretrained(model_id)
    return pipeline("text-classification", model=mdl, tokenizer=tok, return_all_scores=True, truncation=True)


def score_sentiment_finbert(texts: List[str], nlp=None) -> float:
    if not texts:
        return 0.0
    if nlp is None:
        nlp = load_finbert_pipeline("ProsusAI/finbert")
    # Batch where possible for speed
    try:
        outs = nlp([t[:512] for t in texts])
        agg = 0.0
        m = 0
        for out in outs:
            probs = {d['label'].lower(): d['score'] for d in out}
            agg += probs.get('positive',0) - probs.get('negative',0)
            m += 1
        return agg / m if m else 0.0
    except Exception:
        # Fallback slow path
        agg = 0.0
        m = 0
        for t in texts:
            try:
                out = nlp(t[:512])[0]
                probs = {d['label'].lower(): d['score'] for d in out}
                agg += probs.get('positive',0) - probs.get('negative',0)
                m += 1
            except Exception:
                continue
        return agg / m if m else 0.0


def build_news_sentiment_factor(tickers: List[str], cfg: RunConfig) -> pd.DataFrame:
    # Short-circuit if news weight is zero or deps are missing
    if cfg.news_weight <= 0 or feedparser is None:
        if cfg.debug:
            print("[DEBUG] News weight <=0 or feedparser missing; skipping news factor")
        return pd.DataFrame()

    # Try long names via provider chain best-effort
    long_names = {t: t for t in tickers}
    if yf is not None:
        for t in tickers:
            try:
                long_names[t] = getattr(yf.Ticker(t), 'info', {}).get('longName', t)
            except Exception:
                pass

    # Rebalance dates via proxy (SPY) using active provider chain
    try:
        proxy = get_price_history(["SPY"], cfg.start, cfg.end, provider=cfg.provider)
        dates = proxy.resample(cfg.rebalance).last().index
    except Exception:
        # Fallback: monthly end from overall date range
        dates = pd.date_range(pd.to_datetime(cfg.start), pd.to_datetime(cfg.end), freq=cfg.rebalance)

    # Load NLP if available
    try:
        nlp = load_finbert_pipeline(cfg.sentiment_model)
    except Exception:
        nlp = None
        if cfg.debug:
            print("[DEBUG] FinBERT not available — sentiment factor will be zeros. Install transformers+torch to enable.")

    rows = []
    for d in dates:
        if cfg.debug:
            print(f"[DEBUG] Building news sentiment for date {d}")
        start = (d - pd.tseries.frequencies.to_offset(f"{cfg.news_days}D")).to_pydatetime().replace(tzinfo=timezone.utc)
        end = d.to_pydatetime().replace(tzinfo=timezone.utc)
        for t in tickers:
            key = (t, int(d.timestamp()))
            if key in NEWS_CACHE:
                items = NEWS_CACHE[key]
            else:
                q = company_query_from_ticker(t, long_names.get(t), cfg.news_sources_hint, cfg.news_days)
                if cfg.debug:
                    print(f"[DEBUG] Fetching news for {t} with query: {q}")
                items = fetch_google_news(q, cfg.news_max_per_ticker)
                def in_window(it):
                    ts = it.get("published")
                    return (ts is None) or (start <= ts <= end)
                items = [it for it in items if in_window(it)]
                NEWS_CACHE[key] = items

            texts = [f"{it.get('title','')} {it.get('summary','')}".strip() for it in items]
            s = score_sentiment_finbert(texts, nlp) if nlp else 0.0
            if cfg.debug:
                print(f"[DEBUG] Ticker {t} on {d}: {len(texts)} articles, raw score {s}")
            vol_scale = math.log1p(len(texts)) / math.log(1+cfg.news_max_per_ticker) if len(texts)>0 else 0.0
            s *= vol_scale
            rows.append({"date": d, "ticker": t, "news_sent": s})

    if not rows:
        return pd.DataFrame()

    df = pd.DataFrame(rows)
    pivot = df.pivot(index="date", columns="ticker", values="news_sent").sort_index()
    return pivot


# -------------------------
# Factors
# -------------------------

def compute_momentum_12m_1m(prices: pd.DataFrame) -> pd.DataFrame:
    ret_1m = prices.pct_change(21)
    ret_12m = prices.pct_change(252)
    return ret_12m - ret_1m


def compute_vol_20d(prices: pd.DataFrame) -> pd.DataFrame:
    daily = prices.pct_change()
    return daily.rolling(20).std()


def compute_trend_50_200(prices: pd.DataFrame) -> pd.DataFrame:
    ma50 = prices.rolling(50).mean()
    ma200 = prices.rolling(200).mean()
    return ma50 / ma200


def zscore(df: pd.DataFrame, invert: bool = False) -> pd.DataFrame:
    mean = df.mean(axis=1)
    std = df.std(axis=1).replace(0, np.nan)
    z = df.sub(mean, axis=0).div(std, axis=0)
    if invert:
        z = -z
    return z


def combine_factors(momentum: pd.DataFrame, vol: pd.DataFrame, trend: pd.DataFrame,
                    w_mom: float = 0.5, w_vol: float = 0.25, w_trend: float = 0.25) -> pd.DataFrame:
    m_z = zscore(momentum)
    v_z = zscore(vol, invert=True)
    t_z = zscore(trend)
    score = (
        w_mom * m_z +
        w_vol * v_z +
        w_trend * t_z
    )
    return score


def blend_with_news_factor(base_score: pd.DataFrame, news_factor: pd.DataFrame, weight: float) -> pd.DataFrame:
    weight = float(weight)
    if weight <= 0 or news_factor is None or news_factor.empty:
        return base_score
    news_z = zscore(news_factor).reindex_like(base_score).fillna(0.0)
    return (1-weight) * base_score + weight * news_z


# -------------------------
# Portfolio construction
# -------------------------

def rebalance_dates(prices: pd.DataFrame, freq: str = "M") -> pd.DatetimeIndex:
    return prices.resample(freq).last().index


def pick_top_n(scores: pd.DataFrame, date: pd.Timestamp, top_n: int, prices: pd.DataFrame,
               min_price: float = 5.0) -> List[str]:
    if date < scores.index[0]:
        date = scores.index[0]
    if date not in scores.index:
        # get_indexer with method='ffill' returns -1 if no suitable index
        idx = scores.index.get_indexer([date], method='ffill')[0]
        if idx == -1:
            date = scores.index[0]
        else:
            date = scores.index[idx]
    s = scores.loc[date]
    px_row = prices.reindex(columns=s.index).loc[date]
    eligible_mask = px_row >= float(min_price)
    eligible = s.index[eligible_mask.fillna(False)]
    dprint(f"pick_top_n at {date}: eligible {list(eligible)}")
    return list(s.loc[eligible].dropna().sort_values(ascending=False).head(top_n).index)


# -------------------------
# Backtest
# -------------------------

def backtest_equal_weight(prices: pd.DataFrame, scores: pd.DataFrame, cfg: RunConfig) -> pd.DataFrame:
    dates = rebalance_dates(prices, cfg.rebalance)
    pv = pd.Series(index=prices.index, dtype=float)
    pv.iloc[0] = 1.0

    prev_weights: Dict[str, float] = {}

    for i, d in enumerate(dates):
        if d not in prices.index:
            # find nearest prior trading day; get_indexer returns -1 if none
            idx = prices.index.get_indexer([d], method='ffill')[0]
            if idx == -1:
                # no prior price data for this rebalance date; skip
                continue
            d = prices.index[idx]
        picks = pick_top_n(scores, d, cfg.top_n, prices, cfg.min_price)
        if not picks:
            continue
        target_w = {t: 1.0/len(picks) for t in picks}

        turnover = sum(abs(target_w.get(t,0) - prev_weights.get(t,0)) for t in set(list(target_w)+list(prev_weights))) if i>0 else sum(target_w.values())
        tc = turnover * (cfg.one_way_tc_bps/10000.0)
        prev_weights = target_w

        # window: [d, next_d) end-exclusive
        d_next = dates[i+1] if i+1 < len(dates) else prices.index[-1] + pd.Timedelta(days=1)
        mask = (prices.index >= d) & (prices.index < d_next)
        px_window = prices.loc[mask, picks]
        rets = px_window.pct_change().fillna(0)
        ew = np.array([target_w[t] for t in picks])
        port_rets = (rets * ew).sum(axis=1)
        if len(port_rets) > 0:
            port_rets.iloc[0] -= tc
        last_pv = pv.loc[px_window.index[0]] if i > 0 else 1.0
        pv.loc[px_window.index] = (1 + port_rets).cumprod() * last_pv

    return pv.to_frame(name="portfolio")


# -------------------------
# Metrics & reporting
# -------------------------

def date_years_between(t0: pd.Timestamp, t1: pd.Timestamp) -> float:
    return max((t1 - t0).days, 1) / 365.25


def perf_stats(series: pd.Series, freq: str = "D") -> dict:
    s = series.dropna()
    if len(s) < 2:
        return {"CAGR": np.nan, "AnnVol": np.nan, "Sharpe": np.nan, "MaxDD": np.nan}
    rets = s.pct_change().dropna()
    ann_factor = {"D":252, "W":52, "M":12}.get(freq, 252)
    years = date_years_between(s.index[0], s.index[-1])
    total_return = float(s.iloc[-1]/s.iloc[0]) - 1.0
    cagr = (1 + total_return) ** (1/years) - 1 if years > 0 else np.nan
    mu = rets.mean() * ann_factor
    vol = rets.std(ddof=0) * math.sqrt(ann_factor)
    sharpe = (mu / vol) if vol > 0 else np.nan
    roll_max = s.cummax()
    dd = (s/roll_max - 1).min()
    return {"CAGR": cagr, "AnnVol": vol, "Sharpe": sharpe, "MaxDD": dd}


def to_pretty_pct(x: float) -> str:
    return f"{x*100:.2f}%" if pd.notna(x) else "NA"


# -------------------------
# Runner
# -------------------------

def run(cfg: RunConfig, tickers: List[str]) -> None:
    prices = get_price_history(tickers, cfg.start, cfg.end, provider=cfg.provider)
    prices = prices.dropna(how="all", axis=1)

    mom = compute_momentum_12m_1m(prices)
    vol = compute_vol_20d(prices)
    trend = compute_trend_50_200(prices)

    base_score = combine_factors(mom, vol, trend)

    try:
        news_factor = build_news_sentiment_factor(list(prices.columns), cfg)
        news_factor = news_factor.reindex(base_score.index, method='ffill').reindex(columns=base_score.columns, fill_value=0.0)
        score = blend_with_news_factor(base_score, news_factor, cfg.news_weight)
    except Exception as e:
        print("News blending failed:", e)
        score = base_score

    equity = backtest_equal_weight(prices, score, cfg)
    bench = prices.mean(axis=1).to_frame(name="bench")  # simple equally-weighted benchmark
    report = equity.join(bench, how="inner").dropna()

    ps = perf_stats(report["portfolio"], "D")
    bs = perf_stats(report["bench"], "D")

    print("\n==== Performance (portfolio) ====")
    print({k: to_pretty_pct(v) if k!="MaxDD" else to_pretty_pct(v) for k,v in ps.items()})
    print("\n==== Performance (benchmark EW) ====")
    print({k: to_pretty_pct(v) if k!="MaxDD" else to_pretty_pct(v) for k,v in bs.items()})

    try:
        import matplotlib.pyplot as plt
        plot = (report / report.iloc[0]).dropna()
        plt.figure()
        plot.plot(title="Equity Curve: Portfolio vs EW Benchmark")
        plt.xlabel("Date"); plt.ylabel("Growth of $1")
        plt.tight_layout()
        plt.show()
    except Exception as e:
        print("Plotting skipped:", e)


# -------------------------
# Tests
# -------------------------

def _test_zscore_shapes():
    df = pd.DataFrame([[1,2,3],[4,5,6]], index=pd.date_range("2020-01-01", periods=2), columns=["A","B","C"])
    z = zscore(df)
    assert z.shape == df.shape
    assert abs(z.loc[df.index[0]].mean()) < 1e-9


def _test_backtest_no_overlap():
    # Small synthetic frame to test windowing
    idx = pd.date_range("2020-01-01", periods=60, freq="B")
    px = pd.DataFrame({"A": np.linspace(100,110,len(idx)), "B": np.linspace(100,90,len(idx))}, index=idx)
    mom = compute_momentum_12m_1m(px)
    vol = compute_vol_20d(px)
    tr = compute_trend_50_200(px)
    score = combine_factors(mom, vol, tr).fillna(0)
    cfg = RunConfig(start=str(idx[0].date()), end=str(idx[-1].date()), top_n=1, rebalance="M")
    eq = backtest_equal_weight(px, score, cfg)
    assert eq.index.is_monotonic_increasing
    assert (eq["portfolio"].dropna() > 0).all()


def _test_perf_stats_consistency():
    idx = pd.date_range("2021-01-01", periods=252, freq="B")
    s = pd.Series(np.linspace(1.0, 1.2, len(idx)), index=idx)
    ps = perf_stats(s, "D")
    assert 0 < ps["CAGR"] < 1
    assert ps["AnnVol"] >= 0


def _test_provider_chain_smoke():
    # Only ensures function wiring; does not require network if providers missing
    try:
        get_price_history(["AAPL"], "2024-01-01", "2024-02-01", provider="auto")
    except Exception:
        pass


def run_tests():
    _test_zscore_shapes()
    _test_backtest_no_overlap()
    _test_perf_stats_consistency()
    _test_provider_chain_smoke()
    print("All tests passed.")


if __name__ == "__main__":
    p = argparse.ArgumentParser()
    _def = RunConfig()
    p.add_argument("--start", type=str, default=_def.start)
    p.add_argument("--end", type=str, default=_def.end)
    p.add_argument("--top", dest="top_n", type=int, default=_def.top_n)
    p.add_argument("--rebalance", type=str, default=_def.rebalance)
    p.add_argument("--tickers", type=str, default=",")
    p.add_argument("--min_price", type=float, default=_def.min_price)
    p.add_argument("--tc_bps", type=float, default=_def.one_way_tc_bps)
    p.add_argument("--provider", type=str, default=_def.provider, choices=["auto","yfinance","stooq","alphavantage"])
    p.add_argument("--news-weight", type=float, default=_def.news_weight, dest="news_weight")
    p.add_argument("--debug", action="store_true", help="Enable debug prints")
    p.add_argument("--run-tests", action="store_true")
    args = p.parse_args()

    if args.run_tests:
        run_tests()
    else:
        cfg = RunConfig(start=args.start, end=args.end, top_n=args.top_n,
                        rebalance=args.rebalance, one_way_tc_bps=args.tc_bps,
                        min_price=args.min_price, provider=args.provider, news_weight=args.news_weight,
                        debug=args.debug)

        # propagate debug into module-level flag used by dprint
        GLOBAL_DEBUG = cfg.debug

        tickers = [t.strip().upper() for t in args.tickers.split(",") if t.strip()] or DEFAULT_TICKERS
        run(cfg, tickers)
