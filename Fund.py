import yfinance as yf
import pandas as pd
import numpy as np
from tqdm import tqdm  # for progress bar

# ---- STEP 1: Define normalization function ----
def normalize(series, invert=False):
    """
    Normalize a pandas Series between 0 and 1.
    If invert=True, lower values are better (e.g. PE, PB, EV/EBITDA).
    """
    series = series.replace([np.inf, -np.inf], np.nan)  # drop bad values
    series = series.fillna(series.median())  # impute with median
    ranks = series.rank(pct=True)  # percentile rank
    if invert:
        return 1 - ranks
    return ranks

# ---- STEP 2: Pull financial data ----
def get_fundamentals(ticker):
    stock = yf.Ticker(ticker)
    info = stock.info
    
    # Extract key ratios (yfinance naming can vary!)
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "peg": info.get("pegRatio"),
        "div_yield": info.get("dividendYield"),
        "ev_ebitda": info.get("enterpriseToEbitda")
    }

# ---- STEP 3: Compute Buy Scores ----
def compute_buy_scores(tickers):
    # Gather data
    data = []
    for ticker in tqdm(tickers, desc="Fetching fundamentals"):
        try:
            fundamentals = get_fundamentals(ticker)
            data.append(fundamentals)
        except Exception as e:
            print(f"Error fetching {ticker}: {e}")
    
    df = pd.DataFrame(data)
    
    # Normalize each metric
    df["pe_score"] = normalize(df["pe"], invert=True)
    df["pb_score"] = normalize(df["pb"], invert=True)
    df["peg_score"] = normalize(df["peg"], invert=True)
    df["div_yield_score"] = normalize(df["div_yield"], invert=False)
    df["ev_ebitda_score"] = normalize(df["ev_ebitda"], invert=True)
    
    # Final Buy Score = average of available metrics
    score_cols = ["pe_score", "pb_score", "peg_score", "div_yield_score", "ev_ebitda_score"]
    df["buy_score"] = df[score_cols].mean(axis=1, skipna=True)
    
    # Sort by Buy Score
    df = df.sort_values("buy_score", ascending=False).reset_index(drop=True)
    
    return df[["ticker", "buy_score"] + score_cols]

# ---- STEP 4: Run Example ----
if __name__ == "__main__":
    tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "XOM", "NVDA"]  # sample list
    results = compute_buy_scores(tickers)
    print("\nTop Picks:\n", results.head())
