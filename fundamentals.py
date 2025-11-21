# analytics/fundamentals.py
import asyncio
from typing import Dict, Any, List
import pandas as pd
import yfinance as yf
from .utils import safe_ratio
import logging
log = logging.getLogger(__name__)

async def fetch_info_async(ticker: str, loop=None) -> Dict[str, Any]:
    loop = loop or asyncio.get_running_loop()
    return await loop.run_in_executor(None, fetch_info_sync, ticker)

def fetch_info_sync(ticker: str) -> Dict[str, Any]:
    try:
        t = yf.Ticker(ticker)
        info = t.info or {}
    except Exception as e:
        log.warning(f"yfinance info error for {ticker}: {e}")
        info = {}

    market_cap = info.get("marketCap")
    fcf = info.get("freeCashflow")
    fcf_yield = safe_ratio(fcf, market_cap) if market_cap else None

    # Collect many useful metrics; prefer TTM where possible
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "peg": info.get("pegRatio"),
        "div_yield": info.get("dividendYield"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
        "roe": info.get("returnOnEquity"),
        "rev_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        "fcf_yield": fcf_yield,
        "debt_to_equity": info.get("debtToEquity"),
        "interest_coverage": info.get("interestCoverage"),  # may be None
        "gross_margin": info.get("grossMargins"),
        "operating_margin": info.get("operatingMargins"),
        "marketCap": market_cap,
        "sector": info.get("sector", "Unknown"),
        "longName": info.get("longName"),
    }

async def get_bulk_fundamentals(tickers: List[str]) -> pd.DataFrame:
    tasks = [fetch_info_async(t) for t in tickers]
    rows = await asyncio.gather(*tasks, return_exceptions=False)
    return pd.DataFrame(rows)
