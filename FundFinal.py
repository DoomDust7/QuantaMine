"""
Unified Buy Score Model
Combines: Value, Quality & Growth, Volatility & Risk, Price & Momentum
Author: Sashank Kocherlakota
"""

import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm

# =====================================================
# Helper Functions
# =====================================================

def normalize(series, invert=False):
    """Normalize a pandas Series between 0 and 1."""
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.fillna(series.median())
    ranks = series.rank(pct=True)
    return 1 - ranks if invert else ranks

def safe_ratio(a, b):
    """Avoid division by zero."""
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return a / b

def compute_rsi(prices, period=14):
    """Compute RSI indicator."""
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    rsi = 100 - (100 / (1 + rs))
    return rsi

def compute_beta(stock_returns, market_returns):
    """Compute beta vs market (SPY)."""
    if len(stock_returns) < 2 or len(market_returns) < 2:
        return np.nan
    cov = np.cov(stock_returns[1:], market_returns[1:])[0][1]
    var = np.var(market_returns[1:])
    return cov / var if var > 0 else np.nan

def buy_rating(score):
    """Assign qualitative rating based on score."""
    if score >= 0.8:
        return "Strong Buy"
    elif score >= 0.65:
        return "Buy"
    elif score >= 0.45:
        return "Hold"
    else:
        return "Avoid"

# =====================================================
# VALUE MODEL
# =====================================================

def get_value_metrics(ticker):
    """Fetch value-based fundamentals."""
    t = yf.Ticker(ticker)
    info = t.info
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE", np.nan),
        "pb": info.get("priceToBook", np.nan),
        "peg": info.get("pegRatio", np.nan),
        "div_yield": info.get("dividendYield", np.nan),
        "ev_ebitda": info.get("enterpriseToEbitda", np.nan)
    }

def value_score(df):
    """Compute normalized Value Buy Score."""
    df["pe_score"] = normalize(df["pe"], invert=True)
    df["pb_score"] = normalize(df["pb"], invert=True)
    df["peg_score"] = normalize(df["peg"], invert=True)
    df["div_yield_score"] = normalize(df["div_yield"], invert=False)
    df["ev_ebitda_score"] = normalize(df["ev_ebitda"], invert=True)
    df["value_score"] = df[
        ["pe_score","pb_score","peg_score","div_yield_score","ev_ebitda_score"]
    ].mean(axis=1, skipna=True)
    return df

# =====================================================
# QUALITY & GROWTH MODEL
# =====================================================

def get_quality_metrics(ticker):
    """Fetch Quality & Growth metrics."""
    t = yf.Ticker(ticker)
    info = t.info
    return {
        "ticker": ticker,
        "roe": info.get("returnOnEquity", np.nan),
        "rev_growth": info.get("revenueGrowth", np.nan),
        "eps_growth": info.get("earningsQuarterlyGrowth", np.nan),
        "fcf_yield": info.get("freeCashflow", np.nan),
        "debt_to_equity": info.get("debtToEquity", np.nan)
    }

def quality_score(df):
    """Compute normalized Quality & Growth Buy Score."""
    df["roe_score"] = normalize(df["roe"], invert=False)
    df["rev_growth_score"] = normalize(df["rev_growth"], invert=False)
    df["eps_growth_score"] = normalize(df["eps_growth"], invert=False)
    df["fcf_yield_score"] = normalize(df["fcf_yield"], invert=False)
    df["debt_to_equity_score"] = normalize(df["debt_to_equity"], invert=True)
    df["quality_score"] = df[
        ["roe_score","rev_growth_score","eps_growth_score","fcf_yield_score","debt_to_equity_score"]
    ].mean(axis=1, skipna=True)
    return df

# =====================================================
# VOLATILITY & RISK MODEL
# =====================================================

def get_vol_risk_metrics(ticker):
    """Fetch Volatility and Risk metrics."""
    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d")
    if hist.empty:
        return {"ticker": ticker, "vol_1m": np.nan, "vol_3m": np.nan,
                "vol_6m": np.nan, "max_drawdown": np.nan, "sharpe_proxy": np.nan}

    returns = hist["Close"].pct_change().dropna()
    vol_1m = returns[-21:].std() if len(returns) >= 21 else np.nan
    vol_3m = returns[-63:].std() if len(returns) >= 63 else np.nan
    vol_6m = returns[-126:].std() if len(returns) >= 126 else np.nan

    rolling_max = hist["Close"].cummax()
    drawdown = (hist["Close"] - rolling_max) / rolling_max
    max_drawdown = drawdown.min()

    sharpe_proxy = returns.mean() / (returns.std() + 1e-9)

    return {
        "ticker": ticker,
        "vol_1m": vol_1m,
        "vol_3m": vol_3m,
        "vol_6m": vol_6m,
        "max_drawdown": max_drawdown,
        "sharpe_proxy": sharpe_proxy
    }

def vol_risk_score(df):
    """Compute normalized Volatility & Risk Buy Score."""
    df["vol_1m_score"] = normalize(df["vol_1m"], invert=True)
    df["vol_3m_score"] = normalize(df["vol_3m"], invert=True)
    df["vol_6m_score"] = normalize(df["vol_6m"], invert=True)
    df["max_drawdown_score"] = normalize(df["max_drawdown"], invert=True)
    df["sharpe_score"] = normalize(df["sharpe_proxy"], invert=False)
    df["risk_score"] = df[
        ["vol_1m_score","vol_3m_score","vol_6m_score","max_drawdown_score","sharpe_score"]
    ].mean(axis=1, skipna=True)
    return df

# =====================================================
# PRICE & MOMENTUM MODEL
# =====================================================

def get_price_momentum_metrics(ticker, market_ticker="SPY"):
    """Fetch price & momentum metrics."""
    stock = yf.Ticker(ticker)
    market = yf.Ticker(market_ticker)
    hist = stock.history(period="1y", interval="1d")
    market_hist = market.history(period="1y", interval="1d")
    if hist.empty or market_hist.empty:
        return {"ticker": ticker, "ret_1m": np.nan, "ret_3m": np.nan, "ret_6m": np.nan,
                "ret_12m": np.nan, "sma_ratio": np.nan, "rsi": np.nan, "beta": np.nan}

    prices = hist["Close"]
    market_prices = market_hist["Close"]

    # Returns
    ret_1m = safe_ratio(prices.iloc[-1], prices.iloc[-21]) - 1 if len(prices) > 21 else np.nan
    ret_3m = safe_ratio(prices.iloc[-1], prices.iloc[-63]) - 1 if len(prices) > 63 else np.nan
    ret_6m = safe_ratio(prices.iloc[-1], prices.iloc[-126]) - 1 if len(prices) > 126 else np.nan
    ret_12m = safe_ratio(prices.iloc[-1], prices.iloc[0]) - 1 if len(prices) >= 252 else np.nan

    # SMA ratio
    sma50 = prices.rolling(50, min_periods=1).mean().iloc[-1]
    sma200 = prices.rolling(200, min_periods=1).mean().iloc[-1]
    sma_ratio = safe_ratio(sma50, sma200)

    # RSI and Beta
    rsi = compute_rsi(prices).iloc[-1]
    stock_returns = prices.pct_change().dropna().values
    market_returns = market_prices.pct_change().dropna().values
    beta = compute_beta(stock_returns, market_returns)

    return {"ticker": ticker, "ret_1m": ret_1m, "ret_3m": ret_3m, "ret_6m": ret_6m,
            "ret_12m": ret_12m, "sma_ratio": sma_ratio, "rsi": rsi, "beta": beta}

def price_momentum_score(df):
    """Compute normalized Price & Momentum Buy Score."""
    df["ret_1m_score"] = normalize(df["ret_1m"], invert=False)
    df["ret_3m_score"] = normalize(df["ret_3m"], invert=False)
    df["ret_6m_score"] = normalize(df["ret_6m"], invert=False)
    df["ret_12m_score"] = normalize(df["ret_12m"], invert=False)
    df["sma_ratio_score"] = normalize(df["sma_ratio"], invert=False)
    df["rsi_score"] = 1 - abs(df["rsi"] - 50)/50
    df["rsi_score"] = df["rsi_score"].clip(0, 1)
    df["beta_score"] = normalize(df["beta"], invert=True)
    df["momentum_score"] = df[
        ["ret_1m_score","ret_3m_score","ret_6m_score","ret_12m_score","sma_ratio_score","rsi_score","beta_score"]
    ].mean(axis=1, skipna=True)
    return df

# =====================================================
# MASTER COMBINER
# =====================================================

def unified_buy_score(tickers):
    print("\nFetching Value Data...")
    value_data = [get_value_metrics(t) for t in tqdm(tickers)]
    df_val = value_score(pd.DataFrame(value_data))

    print("\nFetching Quality & Growth Data...")
    qual_data = [get_quality_metrics(t) for t in tqdm(tickers)]
    df_qual = quality_score(pd.DataFrame(qual_data))

    print("\nFetching Volatility & Risk Data...")
    risk_data = [get_vol_risk_metrics(t) for t in tqdm(tickers)]
    df_risk = vol_risk_score(pd.DataFrame(risk_data))

    print("\nFetching Price & Momentum Data...")
    mom_data = [get_price_momentum_metrics(t) for t in tqdm(tickers)]
    df_mom = price_momentum_score(pd.DataFrame(mom_data))

    # Merge all models
    merged = df_val.merge(df_qual, on="ticker", how="outer")
    merged = merged.merge(df_risk, on="ticker", how="outer")
    merged = merged.merge(df_mom, on="ticker", how="outer")

    # Weighted final buy score (equal weighting)
    merged["buy_score"] = merged[["value_score", "quality_score", "risk_score", "momentum_score"]].mean(axis=1, skipna=True)
    merged["buy_rating"] = merged["buy_score"].apply(buy_rating)

    merged = merged.sort_values("buy_score", ascending=False).reset_index(drop=True)
    return merged[["ticker","buy_score","buy_rating","value_score","quality_score","risk_score","momentum_score"]]

# =====================================================
# EXECUTION
# =====================================================

if __name__ == "__main__":
    tickers = ["AAPL","MSFT","GOOGL","TSLA","JPM","XOM","NVDA"]
    results = unified_buy_score(tickers)
    print("\nUnified Top Picks:\n", results.head(10))
