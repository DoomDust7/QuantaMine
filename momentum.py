import pandas as pd
import numpy as np
import yfinance as yf
import asyncio


# ---------------------------------------------------------
#  BASIC MOMENTUM FEATURE FUNCTIONS
# ---------------------------------------------------------

def distance_from_52w_high(close: pd.Series):
    if close.empty:
        return np.nan
    high52 = close.rolling(252, min_periods=1).max().iloc[-1]
    if high52 == 0:
        return np.nan
    return float((high52 - close.iloc[-1]) / high52)


def sma_distance(close: pd.Series, period: int):
    if len(close) < period:
        return np.nan
    sma = close.rolling(period).mean().iloc[-1]
    if sma == 0 or np.isnan(sma):
        return np.nan
    return float((close.iloc[-1] - sma) / sma)


def compute_momentum_features(close: pd.Series):
    return {
        "dist_52w_high": distance_from_52w_high(close),
        "sma50_dist": sma_distance(close, 50),
        "sma200_dist": sma_distance(close, 200),
    }


# ---------------------------------------------------------
#   COMBINED RISK + MOMENTUM SCORER
# ---------------------------------------------------------

def compute_risk_momentum_scores(risk_df: pd.DataFrame, momentum_df: pd.DataFrame = None) -> pd.DataFrame:
    """
    Takes:
        risk_df     = risk metrics per ticker
        momentum_df = momentum metrics per ticker

    Returns:
        DataFrame with columns:
            ticker, risk_score, momentum_score, <other risk columns...>
    """

    df = risk_df.copy()

    # Ensure required risk columns always exist
    required_risk_cols = ["volatility", "max_drawdown", "sharpe"]
    for col in required_risk_cols:
        if col not in df.columns:
            df[col] = 0.0

    # Compute normalized 0–5 risk score
    df["risk_score"] = (
        0.4 * (1 - df["volatility"].rank(pct=True, ascending=True)) +
        0.4 * (1 - df["max_drawdown"].rank(pct=True, ascending=True)) +
        0.2 * (df["sharpe"].rank(pct=True, ascending=True))
    ) * 5.0

    # Handle momentum integration
    if momentum_df is not None:
        mom = momentum_df.copy()

        # Ensure required momentum columns exist
        for col in ["dist_52w_high", "sma50_dist", "sma200_dist"]:
            if col not in mom.columns:
                mom[col] = 0.0

        mom["momentum_score"] = (
            0.5 * (1 - mom["dist_52w_high"].rank(pct=True, ascending=True)) +
            0.25 * (mom["sma50_dist"].rank(pct=True, ascending=True)) +
            0.25 * (mom["sma200_dist"].rank(pct=True, ascending=True))
        ) * 5.0

        df = df.merge(mom[["ticker", "momentum_score"]], on="ticker", how="left")

    else:
        df["momentum_score"] = 2.5  # neutral default

    # Final cleanup
    df["risk_score"] = df["risk_score"].fillna(2.5).clip(0, 5).round(2)
    df["momentum_score"] = df["momentum_score"].fillna(2.5).clip(0, 5).round(2)

    return df


# ---------------------------------------------------------
#   ASYNC MOMENTUM FETCHERS (USED BY BACKTESTER)
# ---------------------------------------------------------

async def get_momentum_for_ticker(ticker: str, end_date: str = None):
    """
    Fetch price history and compute SMA distances + 52-week high distance.
    Used for both live scoring and backtesting.
    """

    try:
        if end_date:
            # backtesting: simulate history only up to end_date
            data = yf.download(ticker, start="2000-01-01", end=end_date, progress=False)
        else:
            # live scoring: look back 1 year
            data = yf.download(ticker, period="1y", progress=False)

        if data.empty or "Close" not in data.columns:
            return {"ticker": ticker, "dist_52w_high": 0, "sma50_dist": 0, "sma200_dist": 0}

        close = data["Close"]
        feats = compute_momentum_features(close)
        feats["ticker"] = ticker

        return feats

    except Exception:
        return {"ticker": ticker, "dist_52w_high": 0, "sma50_dist": 0, "sma200_dist": 0}


async def get_bulk_momentum(tickers: list, end_date: str = None):
    """
    Fetch momentum data for many tickers at once.
    Returns a pandas DataFrame with 1 row per ticker.
    """

    tasks = [
        get_momentum_for_ticker(t, end_date=end_date)
        for t in tickers
    ]

    rows = await asyncio.gather(*tasks)

    df = pd.DataFrame(rows)

    # Ensure all required momentum columns exist
    for col in ["dist_52w_high", "sma50_dist", "sma200_dist"]:
        if col not in df.columns:
            df[col] = 0.0

    return df
