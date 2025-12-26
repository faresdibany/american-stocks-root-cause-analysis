"""providers.py (institutional on-prem variant)

Institutional customers almost always require:
- approved market data sources (Bloomberg/Refinitiv/FactSet/internal feeds)
- no direct internet scraping (e.g., yfinance, public RSS) from production
- deterministic runs and explicit data lineage

This module defines small provider interfaces that the rest of the institutional
pipeline uses.

The goal is to keep analytics code unchanged while swapping data access.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Dict, List, Optional, Protocol

import pandas as pd


class PriceProvider(Protocol):
    def get_adjusted_close(
        self,
        ticker: str,
        start: str,
        end: str,
        interval: str = "1d",
    ) -> pd.Series:
        """Return a 1D Series of adjusted close prices indexed by datetime."""


class NewsProvider(Protocol):
    def get_headlines(
        self,
        ticker: str,
        lookback_days: int,
        max_items: int,
    ) -> Dict[str, List[str]]:
        """Return mapping of ISO date -> list of headlines."""


class FundamentalsProvider(Protocol):
    def get_fundamentals(self, ticker: str) -> Dict[str, float | str | int | None]:
        """Return a flat dict of fundamentals (PE, market cap, dividend yield, sector, etc)."""


@dataclass
class StrictnessPolicy:
    """Controls failure behavior.

    For institutional environments, it's often preferable to fail closed (raise)
    instead of silently skipping data.
    """

    fail_on_missing_prices: bool = True
    fail_on_missing_factors: bool = False
    fail_on_missing_news: bool = False


class NotConfiguredProviderError(RuntimeError):
    pass


class NullPriceProvider:
    """Default placeholder. Forces explicit configuration in on-prem deployments."""

    def get_adjusted_close(self, ticker: str, start: str, end: str, interval: str = "1d") -> pd.Series:
        raise NotConfiguredProviderError(
            "No PriceProvider configured for institutional pipeline. "
            "Provide an approved implementation (Bloomberg/Refinitiv/internal)."
        )


class StaticHeadlinesProvider:
    """A simple provider for tests and airgapped environments.

    Supply pre-collected headlines (e.g., from an internal news system).
    """

    def __init__(self, headlines_by_ticker: Optional[Dict[str, Dict[str, List[str]]]] = None):
        self._data = headlines_by_ticker or {}

    def get_headlines(self, ticker: str, lookback_days: int, max_items: int) -> Dict[str, List[str]]:
        # Caller can slice further; we return as-is.
        return dict(self._data.get(ticker.upper(), {}))
