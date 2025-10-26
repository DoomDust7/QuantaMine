import asyncio
import pandas as pd
import yfinance as yf
from tqdm import tqdm
from openai import AsyncOpenAI
import os
import nest_asyncio

# Allow async inside interactive environments (VSCode, Jupyter, etc.)
nest_asyncio.apply()

# ----------------------------
# CONFIG
# ----------------------------
os.environ["OPENAI_API_KEY"] = "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"  # Replace this
client = AsyncOpenAI()
tickers = ["AAPL", "MSFT", "NVDA", "TSLA", "JPM"]

# ----------------------------
# HELPERS
# ----------------------------
def normalize(series):
    min_val, max_val = series.min(), series.max()
    if min_val == max_val:
        return series.fillna(0.5)
    return (series - min_val) / (max_val - min_val)

def buy_rating(score):
    if score >= 0.8: return "Strong Buy"
    elif score >= 0.65: return "Buy"
    elif score >= 0.45: return "Hold"
    else: return "Sell"

# ----------------------------
# METRIC COMPUTATION FUNCTIONS
# ----------------------------
def compute_fundamentals(tickers):
    df = pd.DataFrame(columns=["ticker","pe","pb","peg","roe","rev_growth"])
    for t in tqdm(tickers, desc="Fetching Fundamentals"):
        try:
            info = yf.Ticker(t).info
            df.loc[len(df)] = [
                t,
                info.get("trailingPE", None),
                info.get("priceToBook", None),
                info.get("pegRatio", None),
                info.get("returnOnEquity", None),
                info.get("revenueGrowth", None)
            ]
        except Exception:
            df.loc[len(df)] = [t,None,None,None,None,None]
    for col in ["pe","pb","peg","roe","rev_growth"]:
        df[col] = normalize(df[col])
    df["fundamentals_score"] = df[["pe","pb","peg","roe","rev_growth"]].mean(axis=1)
    return df[["ticker","fundamentals_score"]]

def compute_volatility(tickers):
    df = pd.DataFrame(columns=["ticker","volatility_score"])
    for t in tqdm(tickers, desc="Fetching Volatility"):
        try:
            hist = yf.Ticker(t).history(period="6mo")["Close"]
            daily_vol = hist.pct_change().std()
            df.loc[len(df)] = [t, daily_vol]
        except Exception:
            df.loc[len(df)] = [t, 0.5]
    df["volatility_score"] = 1 - normalize(df["volatility_score"])  # lower vol = better
    return df

def compute_momentum(tickers):
    df = pd.DataFrame(columns=["ticker","momentum_score"])
    for t in tqdm(tickers, desc="Fetching Momentum"):
        try:
            hist = yf.Ticker(t).history(period="6mo")["Close"]
            returns = hist.pct_change(21).iloc[-1]  # 1-month momentum
            df.loc[len(df)] = [t, returns]
        except Exception:
            df.loc[len(df)] = [t, 0.5]
    df["momentum_score"] = normalize(df["momentum_score"])
    return df

# ----------------------------
# ASYNC LLM CONFIDENCE MODULE
# ----------------------------
async def get_llm_confidence(ticker, fscore, vscore, mscore):
    prompt = f"""
    You are an expert quantitative analyst. You are given a stock and its numerical metrics:
    - Fundamentals Score: {fscore:.2f}
    - Volatility Score: {vscore:.2f}
    - Momentum Score: {mscore:.2f}

    Each score is on a 0–1 scale where higher = better.
    Based on this data alone, estimate how confident you are that this stock is a good BUY right now.
    Respond with ONLY a number between 0.0 and 1.0 representing your confidence level.
    """
    try:
        response = await client.chat.completions.create(
            model="gpt-4o-mini",
            messages=[{"role":"user","content":prompt}],
            temperature=0.2,
        )
        text = response.choices[0].message.content.strip()
        try:
            val = float(text)
            return max(0.0, min(1.0, val))  # enforce valid range
        except:
            return 0.5
    except Exception:
        return 0.5

async def evaluate_all_llm(df):
    tasks = []
    for _, row in df.iterrows():
        tasks.append(
            get_llm_confidence(
                row["ticker"],
                row["fundamentals_score"],
                row["volatility_score"],
                row["momentum_score"]
            )
        )
    results = await asyncio.gather(*tasks)
    return results

# ----------------------------
# MAIN PIPELINE (ASYNC)
# ----------------------------
async def main():
    fund_df = compute_fundamentals(tickers)
    vol_df = compute_volatility(tickers)
    mom_df = compute_momentum(tickers)
    df = fund_df.merge(vol_df, on="ticker").merge(mom_df, on="ticker")

    print("\nFetching LLM Confidence for each stock...")
    df["llm_confidence"] = await evaluate_all_llm(df)

    df["final_buy_score"] = (
        0.3 * df["fundamentals_score"] +
        0.3 * df["momentum_score"] +
        0.2 * df["volatility_score"] +
        0.2 * df["llm_confidence"]
    )
    df["buy_rating"] = df["final_buy_score"].apply(buy_rating)

    print("\n🏆 Final Results:")
    print(df[["ticker","final_buy_score","buy_rating","llm_confidence"]])

# Run async
if __name__ == "__main__":
    asyncio.run(main())
