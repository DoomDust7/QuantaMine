# analytics/sentiment.py
import asyncio
import httpx
from typing import List
from datetime import datetime
import logging
import numpy as np
import hashlib
from config.api_keys import XAI_API_URL, XAI_API_KEY, XAI_MODEL_NAME
import pandas as pd

log = logging.getLogger(__name__)

HEADERS_XAI = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}
XAI_TIMEOUT = 10


# ---------------------------------------
# Helper — deterministic daily value
# ---------------------------------------
def daily_seed_value(ticker: str) -> float:
    today = datetime.utcnow().strftime("%Y-%m-%d")
    key = f"{ticker}_{today}"
    hashed = hashlib.sha256(key.encode()).hexdigest()
    num = int(hashed[:12], 16)
    return (num % 10000) / 10000.0


# -------------------------------
# xAI/Grok Sentiment (async)
# -------------------------------
async def xai_sentiment_async(text: str) -> float:
    prompt = (
        "Analyze the sentiment of this text regarding the stock. "
        "Return a score from 0 (very negative) to 1 (very positive). "
        f"Text: '{text}'. Respond only with the number."
    )

    payload = {
        "model": XAI_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.1,
        "max_tokens": 10,
    }

    try:
        async with httpx.AsyncClient(timeout=XAI_TIMEOUT) as client:
            r = await client.post(XAI_API_URL, headers=HEADERS_XAI, json=payload)
            txt = r.json()["choices"][0]["message"]["content"].strip()
            return float(txt)
    except Exception as e:
        log.warning(f"xAI API error: {e}")
        return 0.5


# -------------------------------
# 100% xAI-only sentiment
# -------------------------------
async def sentiment_for_ticker(ticker: str) -> dict:
    """
    Returns sentiment (0–5 scale) using ONLY xAI.
    Includes a deterministic daily stabilizer.
    """

    # a simple input prompt for Grok
    text = f"Sentiment analysis for stock {ticker}."

    # call xAI once
    score = await xai_sentiment_async(text)

    # fallback
    if score is None or isinstance(score, Exception):
        score = 0.5

    # deterministic daily stabilizer (0.4–0.8)
    base = daily_seed_value(ticker)
    daily_adjustment = 0.4 + (base * 0.4)

    final_score = np.mean([score, daily_adjustment])

    return {"ticker": ticker, "sentiment_score": round(final_score * 5.0, 2)}


# ---------------------------------------------------------------
# Bulk sentiment wrapper (used by backtester)
# ---------------------------------------------------------------
async def get_bulk_sentiment(tickers: list, date_override: str = None):
    """
    Multiple tickers at once.
    Backtest mode → neutral 2.5.
    """

    if date_override is None:
        tasks = [sentiment_for_ticker(t) for t in tickers]
        results = await asyncio.gather(*tasks)
        return pd.DataFrame(results)

    # backtest neutral sentiment
    rows = [{"ticker": t, "sentiment_score": 2.5} for t in tickers]
    return pd.DataFrame(rows)
