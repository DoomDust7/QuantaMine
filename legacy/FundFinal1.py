"""
Unified AI-Augmented Buy Score Model
Combines: Value, Quality & Growth, Volatility & Risk, Price & Momentum, Sentiment, Gemini LLM Confidence
Author: Sashank Kocherlakota
"""

import asyncio
import os
import time
import json
import random
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from tqdm import tqdm
import nest_asyncio
import google.generativeai as genai
from typing import Tuple, Dict, Any

nest_asyncio.apply()

# =====================================================
# CONFIGURATION
# =====================================================

# --- API keys ---
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HUGGINGFACE_API_KEY = "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"
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = "9b2a1413d6344c34ab7fb9a694b73ede"

# Gemini
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCWf3DGL_n7TBf_B2YGM5kKUBi-iWbx6XI")
genai.configure(api_key=GEMINI_API_KEY)
_GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"  # fast, cheap text model

# Stock universe
tickers = ["AAPL", "MSFT", "GOOGL", "TSLA", "JPM", "XOM", "NVDA"]

# =====================================================
# HELPER FUNCTIONS
# =====================================================

def normalize(series, invert=False):
    series = series.replace([np.inf, -np.inf], np.nan)
    series = series.fillna(series.median())
    ranks = series.rank(pct=True)
    return 1 - ranks if invert else ranks

def buy_rating(score):
    if score >= 0.8: return "Strong Buy"
    elif score >= 0.65: return "Buy"
    elif score >= 0.45: return "Hold"
    else: return "Avoid"

def safe_ratio(a, b):
    if b == 0 or pd.isna(a) or pd.isna(b):
        return np.nan
    return a / b

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# =====================================================
# FUNDAMENTAL, RISK & MOMENTUM MODELS
# =====================================================

def get_value_metrics(ticker):
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
    df["pe_score"] = normalize(df["pe"], invert=True)
    df["pb_score"] = normalize(df["pb"], invert=True)
    df["peg_score"] = normalize(df["peg"], invert=True)
    df["div_yield_score"] = normalize(df["div_yield"])
    df["ev_ebitda_score"] = normalize(df["ev_ebitda"], invert=True)
    df["value_score"] = df[
        ["pe_score","pb_score","peg_score","div_yield_score","ev_ebitda_score"]
    ].mean(axis=1)
    return df

def get_quality_metrics(ticker):
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
    df["roe_score"] = normalize(df["roe"])
    df["rev_growth_score"] = normalize(df["rev_growth"])
    df["eps_growth_score"] = normalize(df["eps_growth"])
    df["fcf_yield_score"] = normalize(df["fcf_yield"])
    df["debt_to_equity_score"] = normalize(df["debt_to_equity"], invert=True)
    df["quality_score"] = df[
        ["roe_score","rev_growth_score","eps_growth_score","fcf_yield_score","debt_to_equity_score"]
    ].mean(axis=1)
    return df

def get_vol_risk_metrics(ticker):
    t = yf.Ticker(ticker)
    hist = t.history(period="6mo", interval="1d")
    if hist.empty:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe_proxy": np.nan}
    returns = hist["Close"].pct_change().dropna()
    vol = returns.std()
    rolling_max = hist["Close"].cummax()
    drawdown = (hist["Close"] - rolling_max) / rolling_max
    max_dd = drawdown.min()
    sharpe_proxy = returns.mean() / (returns.std() + 1e-9)
    return {"ticker": ticker, "vol": vol, "max_dd": max_dd, "sharpe_proxy": sharpe_proxy}

def vol_risk_score(df):
    df["vol_score"] = normalize(df["vol"], invert=True)
    df["dd_score"] = normalize(df["max_dd"], invert=True)
    df["sharpe_score"] = normalize(df["sharpe_proxy"])
    df["risk_score"] = df[["vol_score","dd_score","sharpe_score"]].mean(axis=1)
    return df

def get_price_momentum_metrics(ticker):
    stock = yf.Ticker(ticker)
    hist = stock.history(period="1y", interval="1d")
    if hist.empty:
        return {"ticker": ticker, "ret_3m": np.nan, "rsi": np.nan}
    prices = hist["Close"]
    ret_3m = safe_ratio(prices.iloc[-1], prices.iloc[-63]) - 1 if len(prices) > 63 else np.nan
    rsi = compute_rsi(prices).iloc[-1]
    return {"ticker": ticker, "ret_3m": ret_3m, "rsi": rsi}

def momentum_score(df):
    df["ret_score"] = normalize(df["ret_3m"])
    df["rsi_score"] = 1 - abs(df["rsi"] - 50)/50
    df["rsi_score"] = df["rsi_score"].clip(0, 1)
    df["momentum_score"] = df[["ret_score","rsi_score"]].mean(axis=1)
    return df

# =====================================================
# SENTIMENT ANALYSIS (FinBERT)
# =====================================================
def get_sentiment(text):
    try:
        r = requests.post(
            HUGGINGFACE_API_URL,
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": text}
        )
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            best = max(data[0], key=lambda x: x['score'])
            lbl = best['label'].lower()
            sc = best['score']
            if lbl == "positive": return sc
            elif lbl == "negative": return 1 - sc
            return 0.5
        return 0.5
    except Exception:
        return 0.5

def fetch_news(ticker):
    try:
        params = {"q": ticker, "language": "en", "pageSize": 5, "apiKey": NEWSAPI_KEY}
        resp = requests.get(NEWSAPI_URL, params=params)
        arts = resp.json().get("articles", [])
        texts = []
        for a in arts:
            title = a.get("title", "")
            desc  = a.get("description", "")
            text  = f"{title}. {desc}".strip()
            if len(text) > 20:
                texts.append(text)
        if not texts:
            texts = [f"Recent financial updates about {ticker}"]
        return texts
    except Exception:
        return [f"Recent financial updates about {ticker}"]

def sentiment_scores(tickers):
    results = []
    for t in tqdm(tickers, desc="Fetching Sentiment Data"):
        news = fetch_news(t)
        scs = [get_sentiment(txt) for txt in news]
        avg = sum(scs) / len(scs)
        results.append({"ticker": t, "sentiment_score": avg})
    return pd.DataFrame(results)

# =====================================================
# GEMINI LLM CONFIDENCE + EXPLANATION
# =====================================================

def _build_prompt(row: pd.Series) -> str:
    return f"""
You are an expert equity analyst. You are given normalized 0–1 scores for a stock:

Ticker: {row['ticker']}
Value Score: {row['value_score']:.2f}
Quality Score: {row['quality_score']:.2f}
Risk Score: {row['risk_score']:.2f}
Momentum Score: {row['momentum_score']:.2f}
Sentiment Score: {row['sentiment_score']:.2f}

Task:
1) Output a BUY confidence between 0 and 1 (0 = strong sell, 0.5 = neutral, 1 = strong buy).
2) Provide a short 1–2 sentence explanation referencing the above scores.

Respond ONLY as valid JSON with fields:
{{
  "confidence": <number 0..1>,
  "reason": "<short explanation>"
}}
"""

def _safe_parse_json(text: str) -> Dict[str, Any]:
    try:
        data = json.loads(text)
        if "confidence" in data and "reason" in data:
            c = max(0.0, min(1.0, float(data["confidence"])))
            r = str(data["reason"]).strip()
            return {"confidence": c, "reason": r or "No explanation provided."}
    except Exception:
        pass
    import re
    m = re.search(r"(\d+\.\d+|\d+)", text)
    conf = float(m.group(1)) if m else 0.5
    conf = max(0, min(1, conf))
    return {"confidence": conf, "reason": text.strip() or "No explanation."}

def _gemini_once(prompt: str) -> Tuple[float, str]:
    try:
        model = genai.GenerativeModel(_GEMINI_MODEL_NAME)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        text = getattr(resp, "text", "") or ""
        if not text and getattr(resp, "candidates", None):
            cand = resp.candidates[0]
            if cand.content and cand.content.parts:
                text = "".join(p.text for p in cand.content.parts if hasattr(p, "text"))
        if not text:
            return 0.5, "No output from model."
        data = _safe_parse_json(text)
        return data["confidence"], data["reason"]
    except Exception as e:
        return 0.5, f"Gemini error: {e}"

def get_gemini_confidence_and_reason(row: pd.Series, retries: int = 2) -> Tuple[float, str]:
    prompt = _build_prompt(row)
    for attempt in range(retries + 1):
        conf, reason = _gemini_once(prompt)
        if "error" not in reason.lower() and "no output" not in reason.lower():
            return conf, reason
        time.sleep(min(2**attempt, 10) + random.uniform(0, 0.5))
    return conf, reason

async def evaluate_gemini_async(df: pd.DataFrame, max_concurrency: int = 2) -> pd.DataFrame:
    sem = asyncio.Semaphore(max_concurrency)
    async def _one(i):
        async with sem:
            row = df.iloc[i]
            conf, reason = await asyncio.to_thread(get_gemini_confidence_and_reason, row)
            return i, conf, reason
    tasks = [asyncio.create_task(_one(i)) for i in range(len(df))]
    for f in tqdm(asyncio.as_completed(tasks), total=len(tasks), desc="Gemini LLM Confidence"):
        i, conf, reason = await f
        df.at[i, "llm_confidence"] = conf
        df.at[i, "llm_reason"] = reason
    return df

# =====================================================
# MASTER PIPELINE
# =====================================================

async def main():
    print("\nFetching Value Data...")
    df_val = value_score(pd.DataFrame([get_value_metrics(t) for t in tqdm(tickers)]))
    print("\nFetching Quality & Growth Data...")
    df_qual = quality_score(pd.DataFrame([get_quality_metrics(t) for t in tqdm(tickers)]))
    print("\nFetching Risk Data...")
    df_risk = vol_risk_score(pd.DataFrame([get_vol_risk_metrics(t) for t in tqdm(tickers)]))
    print("\nFetching Momentum Data...")
    df_mom = momentum_score(pd.DataFrame([get_price_momentum_metrics(t) for t in tqdm(tickers)]))
    print("\nFetching Sentiment Data...")
    df_sent = sentiment_scores(tickers)

    df = df_val.merge(df_qual, on="ticker", how="outer")
    df = df.merge(df_risk, on="ticker", how="outer")
    df = df.merge(df_mom, on="ticker", how="outer")
    df = df.merge(df_sent, on="ticker", how="outer")

    df["buy_score"] = df[["value_score","quality_score","risk_score","momentum_score"]].mean(axis=1)
    df["buy_rating"] = df["buy_score"].apply(buy_rating)

    print("\nFetching Gemini LLM Confidence and Explanations...")
    df["llm_confidence"], df["llm_reason"] = 0.5, "Pending"
    df = await evaluate_gemini_async(df)

    df["final_buy_score"] = (
        0.5*df["buy_score"] +
        0.2*df["sentiment_score"] +
        0.3*df["llm_confidence"]
    )
    df["final_rating"] = df["final_buy_score"].apply(buy_rating)

    print("\n🏆 Final Results:")
    cols = [
        "ticker","final_buy_score","final_rating",
        "buy_score","sentiment_score","llm_confidence",
        "value_score","quality_score","risk_score","momentum_score","llm_reason"
    ]
    print(df[cols].sort_values("final_buy_score", ascending=False).to_string(index=False))

if __name__ == "__main__":
    asyncio.run(main())
