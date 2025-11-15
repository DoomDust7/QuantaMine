# ai_backend.py
import asyncio
import os
import time
import json
import random
import numpy as np
import pandas as pd
import yfinance as yf
import requests
from typing import Tuple, Dict, Any
import google.generativeai as genai

# ========= CONFIG =========
HUGGINGFACE_API_URL = "https://api-inference.huggingface.co/models/ProsusAI/finbert"
HUGGINGFACE_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "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")
NEWSAPI_URL = "https://newsapi.org/v2/everything"
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "9b2a1413d6344c34ab7fb9a694b73ede")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "AIzaSyCWf3DGL_n7TBf_B2YGM5kKUBi-iWbx6XI")
genai.configure(api_key=GEMINI_API_KEY)
_GEMINI_MODEL_NAME = "models/gemini-2.5-flash-lite"

# ========= HELPERS =========
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
    if b == 0 or pd.isna(a) or pd.isna(b): return np.nan
    return a / b

def compute_rsi(prices, period=14):
    delta = prices.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.rolling(period, min_periods=1).mean()
    avg_loss = loss.rolling(period, min_periods=1).mean()
    rs = avg_gain / (avg_loss + 1e-9)
    return 100 - (100 / (1 + rs))

# ========= DATA FUNCTIONS =========
def get_value_metrics(ticker):
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE"), "pb": info.get("priceToBook"),
        "peg": info.get("pegRatio"), "div_yield": info.get("dividendYield"),
        "ev_ebitda": info.get("enterpriseToEbitda")
    }

def value_score(df):
    df["pe_score"] = normalize(df["pe"], invert=True)
    df["pb_score"] = normalize(df["pb"], invert=True)
    df["peg_score"] = normalize(df["peg"], invert=True)
    df["div_yield_score"] = normalize(df["div_yield"])
    df["ev_ebitda_score"] = normalize(df["ev_ebitda"], invert=True)
    df["value_score"] = df[["pe_score","pb_score","peg_score","div_yield_score","ev_ebitda_score"]].mean(axis=1)
    return df

def get_quality_metrics(ticker):
    info = yf.Ticker(ticker).info
    return {
        "ticker": ticker,
        "roe": info.get("returnOnEquity"),
        "rev_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        "fcf_yield": info.get("freeCashflow"),
        "debt_to_equity": info.get("debtToEquity")
    }

def quality_score(df):
    df["roe_score"] = normalize(df["roe"])
    df["rev_growth_score"] = normalize(df["rev_growth"])
    df["eps_growth_score"] = normalize(df["eps_growth"])
    df["fcf_yield_score"] = normalize(df["fcf_yield"])
    df["debt_to_equity_score"] = normalize(df["debt_to_equity"], invert=True)
    df["quality_score"] = df[["roe_score","rev_growth_score","eps_growth_score","fcf_yield_score","debt_to_equity_score"]].mean(axis=1)
    return df

def get_vol_risk_metrics(ticker):
    hist = yf.Ticker(ticker).history(period="6mo", interval="1d")
    if hist.empty:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe_proxy": np.nan}
    returns = hist["Close"].pct_change().dropna()
    vol = returns.std()
    rolling_max = hist["Close"].cummax()
    drawdown = (hist["Close"] - rolling_max) / rolling_max
    return {"ticker": ticker, "vol": vol, "max_dd": drawdown.min(),
            "sharpe_proxy": returns.mean() / (returns.std() + 1e-9)}

def vol_risk_score(df):
    df["vol_score"] = normalize(df["vol"], invert=True)
    df["dd_score"] = normalize(df["max_dd"], invert=True)
    df["sharpe_score"] = normalize(df["sharpe_proxy"])
    df["risk_score"] = df[["vol_score","dd_score","sharpe_score"]].mean(axis=1)
    return df

def get_price_momentum_metrics(ticker):
    hist = yf.Ticker(ticker).history(period="1y", interval="1d")
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

# ========= SENTIMENT =========
def get_sentiment(text):
    try:
        r = requests.post(HUGGINGFACE_API_URL,
            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
            json={"inputs": text})
        data = r.json()
        if isinstance(data, list) and len(data) > 0:
            best = max(data[0], key=lambda x: x['score'])
            lbl, sc = best['label'].lower(), best['score']
            return sc if lbl == "positive" else (1 - sc if lbl == "negative" else 0.5)
        return 0.5
    except Exception:
        return 0.5

def fetch_news(ticker):
    try:
        r = requests.get(NEWSAPI_URL, params={"q": ticker, "language": "en", "pageSize": 5, "apiKey": NEWSAPI_KEY})
        articles = r.json().get("articles", [])
        return [f"{a.get('title','')}. {a.get('description','')}".strip() for a in articles if a.get("title")] or [f"Recent updates about {ticker}"]
    except Exception:
        return [f"Recent updates about {ticker}"]

def sentiment_scores(tickers):
    out = []
    for t in tickers:
        news = fetch_news(t)
        scores = [get_sentiment(n) for n in news]
        out.append({"ticker": t, "sentiment_score": np.mean(scores)})
    return pd.DataFrame(out)

# ========= GEMINI LLM =========
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

Respond ONLY as valid JSON:
{{"confidence": <0–1>, "reason": "<short explanation>"}}
"""

def _safe_parse_json(txt):
    try:
        d = json.loads(txt)
        c = max(0,min(1,float(d.get("confidence",0.5))))
        return {"confidence": c, "reason": d.get("reason","No explanation")}
    except Exception:
        return {"confidence": 0.5, "reason": txt[:200]}

def _gemini_call(prompt):
    try:
        model = genai.GenerativeModel(_GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2, max_output_tokens=256, response_mime_type="application/json"))
        txt = getattr(resp, "text", "") or ""
        if not txt and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            txt = "".join(p.text for p in parts if hasattr(p,"text"))
        return _safe_parse_json(txt or "{}")
    except Exception as e:
        return {"confidence":0.5,"reason":f"Gemini error: {e}"}

async def evaluate_gemini(df):
    for i,row in df.iterrows():
        prompt=_build_prompt(row)
        result=_gemini_call(prompt)
        df.at[i,"llm_confidence"]=result["confidence"]
        df.at[i,"llm_reason"]=result["reason"]
    return df

# ========= MASTER PIPELINE =========
async def run_analysis(tickers):
    df_val = value_score(pd.DataFrame([get_value_metrics(t) for t in tickers]))
    df_val = df_val.apply(pd.to_numeric, errors="coerce")
    df_qual = quality_score(pd.DataFrame([get_quality_metrics(t) for t in tickers]))
    df_qual = df_qual.apply(pd.to_numeric, errors="coerce")
    df_risk = vol_risk_score(pd.DataFrame([get_vol_risk_metrics(t) for t in tickers]))
    df_risk = df_risk.apply(pd.to_numeric, errors="coerce")
    df_mom = momentum_score(pd.DataFrame([get_price_momentum_metrics(t) for t in tickers]))
    df_mom = df_mom.apply(pd.to_numeric, errors="coerce")
    df_sent = sentiment_scores(tickers)
    df_sent = df_sent.apply(pd.to_numeric, errors="coerce")

    df = df_val.merge(df_qual,on="ticker").merge(df_risk,on="ticker").merge(df_mom,on="ticker").merge(df_sent,on="ticker")
    df["buy_score"]=df[["value_score","quality_score","risk_score","momentum_score"]].mean(axis=1)
    df["buy_rating"]=df["buy_score"].apply(buy_rating)

    df["llm_confidence"]=0.5; df["llm_reason"]="Pending"
    df=await evaluate_gemini(df)

    df["final_buy_score"]=0.5*df["buy_score"]+0.2*df["sentiment_score"]+0.3*df["llm_confidence"]
    df["final_rating"]=df["final_buy_score"].apply(buy_rating)
    return df
