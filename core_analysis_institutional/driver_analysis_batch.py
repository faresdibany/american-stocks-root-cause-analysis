"""driver_analysis_batch.py

Batch runner for per-stock driver analysis.

Usage (from this folder):
    python driver_analysis_batch.py --tickers AAPL MSFT NVDA --start 2022-01-01 --end 2025-12-25

Outputs:
    ../outputs/driver_report_<TICKER>_<timestamp>.json
"""

from __future__ import annotations

import os
import sys
from typing import List

from stock_driver_analysis import run_driver_analysis


def parse_args(argv: List[str]):
    import argparse

    p = argparse.ArgumentParser(description="Batch run per-stock movement driver analysis")
    p.add_argument("--tickers", nargs="+", required=True, help="Space-separated tickers")
    p.add_argument("--start", required=True)
    p.add_argument("--end", required=True)
    p.add_argument("--out-dir", default=os.path.join("..", "outputs"))
    p.add_argument("--trend-method", default="hp", choices=["hp", "stl", "ewm"])
    return p.parse_args(argv)


def main(argv: List[str]) -> int:
    args = parse_args(argv)

    outputs = []
    for t in args.tickers:
        try:
            out = run_driver_analysis(
                ticker=t,
                start=args.start,
                end=args.end,
                out_dir=args.out_dir,
                trend_method=args.trend_method,
            )
            outputs.append(out)
            print(f"✅ {t}: {out}")
        except Exception as e:
            print(f"❌ {t}: {e}")

    if outputs:
        print("\nWrote:")
        for p in outputs:
            print(f"- {p}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv[1:]))
