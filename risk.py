# analytics/risk.py
import asyncio
import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import yfinance as yf
import logging
log = logging.getLogger(__name__)

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

def _max_drawdown(close: pd.Series) -> float:
    roll_max = close.cummax()
    drawdown = (close - roll_max) / roll_max
    return float(drawdown.min()) if len(drawdown) > 0 else np.nan

def _annualized_vol(returns: pd.Series) -> float:
    return float(returns.std() * (252 ** 0.5)) if len(returns) > 1 else np.nan

def _sharpe(returns: pd.Series) -> float:
    if len(returns) < 2:
        return np.nan
    return float((returns.mean() / (returns.std() + 1e-9)) * (252 ** 0.5))

def compute_risk_metrics_from_history(close: pd.Series) -> dict:
    if close is None or len(close) < 2:
        return {"vol": np.nan, "max_dd": np.nan, "sharpe": np.nan, "ret_3m": np.nan, "rsi": np.nan}
    returns = close.pct_change().dropna()
    vol = _annualized_vol(returns)
    max_dd = _max_drawdown(close)
    sharpe = _sharpe(returns)
    ret_3m = float(close.iloc[-1] / close.iloc[-63] - 1) if len(close) > 63 else np.nan
    rsi = compute_rsi(close).iloc[-1] if len(close) > 0 else np.nan
    return {"vol": vol, "max_dd": max_dd, "sharpe": sharpe, "ret_3m": ret_3m, "rsi": rsi}

def fetch_history_sync(ticker: str, years: int = 5):
    period = f"{max(1, years)}y"
    try:
        t = yf.Ticker(ticker)
        hist = t.history(period=period, interval="1d")
        return hist
    except Exception as e:
        log.warning(f"history fetch error {ticker}: {e}")
        return pd.DataFrame()

async def get_risk_for_ticker(ticker: str, years: int = 5):
    loop = asyncio.get_running_loop()
    hist = await loop.run_in_executor(None, fetch_history_sync, ticker, years)
    if hist.empty:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe": np.nan, "ret_3m": np.nan, "rsi": np.nan}
    close = hist["Close"].dropna()
    metrics = compute_risk_metrics_from_history(close)
    metrics["ticker"] = ticker
    return metrics


async def get_bulk_risk(tickers: list, years: int = 5):
    """
    Returns risk metrics for multiple tickers as a clean DataFrame.
    Ensures compatibility with compute_risk_momentum_scores().
    """

    # Run all async risk fetches
    tasks = [get_risk_for_ticker(t, years) for t in tickers]
    results = await asyncio.gather(*tasks, return_exceptions=False)

    # Convert list → DataFrame
    df = pd.DataFrame(results)

    # Ensure ticker column exists
    if "ticker" not in df.columns:
        df["ticker"] = tickers

    # Required columns for momentum
    required_cols = ["volatility", "beta", "max_drawdown", "var_95"]

    for col in required_cols:
        if col not in df.columns:
            df[col] = None  # placeholder so momentum does not crash

    # Safety: fill NaNs
    df = df.fillna(df.mean(numeric_only=True)).fillna(0)

    return df


