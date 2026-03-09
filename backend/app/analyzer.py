"""
QuantaMine Analyzer - Modernized Backend
Combines: Value, Quality, Risk, Momentum, Sentiment, and Gemini LLM reasoning.
"""

import os
import json
import re
import time
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from functools import lru_cache
from typing import Any, Dict, Generator, List, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import google.generativeai as genai
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

LOG = logging.getLogger(__name__)

# ── Config from environment ──────────────────────────────────────────────────
HF_API_URL = os.getenv(
    "HUGGINGFACE_API_URL",
    "https://api-inference.huggingface.co/models/ProsusAI/finbert",
)
HF_API_KEY = os.getenv("HUGGINGFACE_API_KEY", "")
NEWSAPI_URL = os.getenv("NEWSAPI_URL", "https://newsapi.org/v2/everything")
NEWSAPI_KEY = os.getenv("NEWSAPI_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "models/gemini-2.5-flash-lite")

WEIGHTS = {
    "algo": float(os.getenv("WEIGHT_ALGO", "0.5")),
    "sentiment": float(os.getenv("WEIGHT_SENT", "0.2")),
    "llm": float(os.getenv("WEIGHT_LLM", "0.3")),
}

if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

# ── HTTP session with retries ─────────────────────────────────────────────────
SESSION = requests.Session()
_retry = Retry(total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
SESSION.mount("https://", HTTPAdapter(max_retries=_retry))
HTTP_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "12"))


# ── Utilities ─────────────────────────────────────────────────────────────────
def safe_float(v) -> Optional[float]:
    try:
        if v is None or (isinstance(v, float) and np.isnan(v)):
            return None
        return float(v)
    except Exception:
        return None


def safe_ratio(a, b) -> float:
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan


def clamp01(x) -> float:
    try:
        if x is None or pd.isna(x):
            return 0.5
        return max(0.0, min(1.0, float(x)))
    except Exception:
        return 0.5


def normalize(series: pd.Series, invert: bool = False, min_valid: int = 2) -> pd.Series:
    s = series.replace([np.inf, -np.inf], np.nan).astype(float)
    if s.dropna().shape[0] < min_valid:
        return pd.Series(0.5, index=s.index)
    filled = s.fillna(s.median())
    ranks = filled.rank(method="average", pct=True)
    return (1 - ranks) if invert else ranks


def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    delta = prices.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = (-delta.clip(upper=0)).fillna(0.0)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    return 100.0 - (100.0 / (1.0 + rs))


def extract_json(text: str) -> str:
    m = re.search(r"\{.*\}", text, flags=re.DOTALL)
    return m.group(0) if m else text


def buy_rating(score: float) -> str:
    s = clamp01(score)
    if s >= 0.8:
        return "Strong Buy"
    elif s >= 0.65:
        return "Buy"
    elif s >= 0.45:
        return "Hold"
    return "Avoid"


# ── yfinance helpers ──────────────────────────────────────────────────────────
@lru_cache(maxsize=256)
def _get_info(ticker: str) -> Dict[str, Any]:
    try:
        return yf.Ticker(ticker).info or {}
    except Exception as e:
        LOG.warning("yfinance info error %s: %s", ticker, e)
        return {}


def _get_history(ticker: str, period: str = "1y") -> pd.DataFrame:
    try:
        hist = yf.Ticker(ticker).history(period=period, interval="1d")
        return hist if not hist.empty else pd.DataFrame()
    except Exception as e:
        LOG.warning("yfinance history error %s: %s", ticker, e)
        return pd.DataFrame()


# ── Metric extractors ─────────────────────────────────────────────────────────
def _value_row(ticker: str, info: Dict) -> Dict:
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "peg": info.get("pegRatio"),
        "div_yield": info.get("dividendYield"),
        "ev_ebitda": info.get("enterpriseToEbitda"),
    }


def _quality_row(ticker: str, info: Dict) -> Dict:
    return {
        "ticker": ticker,
        "roe": info.get("returnOnEquity"),
        "rev_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        "fcf_yield": safe_ratio(info.get("freeCashflow"), info.get("marketCap")),
        "debt_to_equity": info.get("debtToEquity"),
    }


def _risk_row(ticker: str, hist: pd.DataFrame) -> Dict:
    if hist.empty or "Close" not in hist.columns:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe_proxy": np.nan}
    prices = hist["Close"].dropna()
    if len(prices) < 2:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe_proxy": np.nan}
    ret = prices.pct_change().dropna()
    vol = float(ret.std())
    rolling_max = prices.cummax()
    dd = ((prices - rolling_max) / rolling_max).min()
    sharpe = float(ret.mean() / (ret.std() + 1e-12))
    return {"ticker": ticker, "vol": vol, "max_dd": float(dd), "sharpe_proxy": sharpe}


def _momentum_row(ticker: str, hist: pd.DataFrame) -> Dict:
    if hist.empty or "Close" not in hist.columns:
        return {"ticker": ticker, "ret_3m": np.nan, "rsi": np.nan}
    prices = hist["Close"].dropna()
    ret_3m = safe_ratio(prices.iloc[-1], prices.iloc[-63]) - 1 if len(prices) > 63 else np.nan
    rsi = float(compute_rsi(prices).iloc[-1]) if len(prices) > 0 else np.nan
    return {"ticker": ticker, "ret_3m": ret_3m, "rsi": rsi}


# ── Scoring ───────────────────────────────────────────────────────────────────
def _score_value(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pe_s"] = normalize(df["pe"], invert=True)
    df["pb_s"] = normalize(df["pb"], invert=True)
    df["peg_s"] = normalize(df["peg"], invert=True)
    df["div_s"] = normalize(df["div_yield"])
    df["ev_s"] = normalize(df["ev_ebitda"], invert=True)
    df["value_score"] = df[["pe_s", "pb_s", "peg_s", "div_s", "ev_s"]].mean(axis=1)
    return df


def _score_quality(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roe_s"] = normalize(df["roe"])
    df["rev_s"] = normalize(df["rev_growth"])
    df["eps_s"] = normalize(df["eps_growth"])
    df["fcf_s"] = normalize(df["fcf_yield"])
    df["de_s"] = normalize(df["debt_to_equity"], invert=True)
    df["quality_score"] = df[["roe_s", "rev_s", "eps_s", "fcf_s", "de_s"]].mean(axis=1)
    return df


def _score_risk(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["dd_mag"] = df["max_dd"].abs()
    df["vol_s"] = normalize(df["vol"], invert=True)
    df["dd_s"] = normalize(df["dd_mag"], invert=True)
    df["sharpe_s"] = normalize(df["sharpe_proxy"])
    df["risk_score"] = df[["vol_s", "dd_s", "sharpe_s"]].mean(axis=1)
    return df


def _score_momentum(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_s"] = normalize(df["ret_3m"])
    df["rsi_s"] = (1.0 - (df["rsi"] - 50.0).abs() / 50.0).clip(0, 1).fillna(0.5)
    df["momentum_score"] = df[["ret_s", "rsi_s"]].mean(axis=1)
    return df


# ── Sentiment ─────────────────────────────────────────────────────────────────
def _sentiment_score(text: str) -> float:
    if not HF_API_KEY:
        return 0.5
    try:
        resp = SESSION.post(
            HF_API_URL,
            headers={"Authorization": f"Bearer {HF_API_KEY}"},
            json={"inputs": text},
            timeout=HTTP_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if isinstance(data, list) and data:
            items = data[0] if isinstance(data[0], list) else data
            best = max(items, key=lambda x: x.get("score", 0))
            lbl = str(best.get("label", "")).lower()
            sc = float(best.get("score", 0.5))
            if "positive" in lbl:
                return clamp01(sc)
            if "negative" in lbl:
                return clamp01(1 - sc)
        return 0.5
    except Exception as e:
        LOG.debug("HF sentiment error: %s", e)
        return 0.5


def _fetch_news(ticker: str) -> List[str]:
    if not NEWSAPI_KEY:
        return []
    try:
        r = SESSION.get(
            NEWSAPI_URL,
            params={"q": ticker, "language": "en", "pageSize": 5, "apiKey": NEWSAPI_KEY},
            timeout=HTTP_TIMEOUT,
        )
        r.raise_for_status()
        articles = r.json().get("articles", []) or []
        return [
            f"{a.get('title', '')}. {a.get('description', '')}".strip()
            for a in articles
            if a.get("title")
        ]
    except Exception as e:
        LOG.debug("NewsAPI error for %s: %s", ticker, e)
        return []


def _compute_sentiment(ticker: str) -> float:
    texts = _fetch_news(ticker)
    if not texts:
        return 0.5
    scores = [_sentiment_score(t) for t in texts]
    return float(np.mean(scores))


# ── Gemini LLM ────────────────────────────────────────────────────────────────
def _build_prompt(row: pd.Series) -> str:
    return (
        "You are an expert equity analyst. Normalized scores (0=worst, 1=best):\n\n"
        f"Ticker: {row['ticker']}\n"
        f"Value:     {row.get('value_score', 0.5):.2f}\n"
        f"Quality:   {row.get('quality_score', 0.5):.2f}\n"
        f"Risk:      {row.get('risk_score', 0.5):.2f}\n"
        f"Momentum:  {row.get('momentum_score', 0.5):.2f}\n"
        f"Sentiment: {row.get('sentiment_score', 0.5):.2f}\n\n"
        "Respond ONLY as valid JSON:\n"
        '{"confidence": <0.0-1.0>, "reason": "<1-2 sentence analysis>"}'
    )


def _gemini_call(prompt: str) -> Dict[str, Any]:
    if not GEMINI_API_KEY:
        return {"confidence": 0.5, "reason": "Gemini API key not configured."}
    try:
        model = genai.GenerativeModel(GEMINI_MODEL)
        resp = model.generate_content(
            prompt,
            generation_config=genai.types.GenerationConfig(
                temperature=0.2,
                max_output_tokens=256,
                response_mime_type="application/json",
            ),
        )
        txt = getattr(resp, "text", "") or ""
        if not txt and getattr(resp, "candidates", None):
            parts = resp.candidates[0].content.parts
            txt = "".join(p.text for p in parts if hasattr(p, "text"))
        block = extract_json(txt or "{}")
        d = json.loads(block)
        return {
            "confidence": clamp01(d.get("confidence", 0.5)),
            "reason": str(d.get("reason", ""))[:500],
        }
    except Exception as e:
        LOG.debug("Gemini error: %s", e)
        return {"confidence": 0.5, "reason": f"Analysis unavailable: {e}"}


# ── Streaming pipeline ────────────────────────────────────────────────────────
def run_analysis_stream(tickers: List[str]) -> Generator[Dict, None, None]:
    """
    Yields progress dicts with keys: type, ticker (optional), stage, progress, data (optional).
    Consumers (FastAPI SSE endpoint) serialize these to JSON.
    """
    n = len(tickers)

    yield {"type": "progress", "stage": "init", "progress": 0, "message": f"Starting analysis for {n} ticker(s)…"}

    # Step 1: fetch info & history per ticker
    infos: Dict[str, Dict] = {}
    hists: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers):
        yield {"type": "progress", "stage": "fetch", "ticker": t, "progress": int(5 + (i / n) * 25), "message": f"Fetching data for {t}…"}
        infos[t] = _get_info(t)
        hists[t] = _get_history(t)

    yield {"type": "progress", "stage": "compute", "progress": 30, "message": "Computing financial metrics…"}

    # Step 2: build metric DataFrames
    df_val = _score_value(
        pd.DataFrame([_value_row(t, infos[t]) for t in tickers]).apply(
            lambda c: pd.to_numeric(c, errors="coerce") if c.name != "ticker" else c
        )
    )
    df_qual = _score_quality(
        pd.DataFrame([_quality_row(t, infos[t]) for t in tickers]).apply(
            lambda c: pd.to_numeric(c, errors="coerce") if c.name != "ticker" else c
        )
    )
    df_risk = _score_risk(
        pd.DataFrame([_risk_row(t, hists[t]) for t in tickers]).apply(
            lambda c: pd.to_numeric(c, errors="coerce") if c.name != "ticker" else c
        )
    )
    df_mom = _score_momentum(
        pd.DataFrame([_momentum_row(t, hists[t]) for t in tickers]).apply(
            lambda c: pd.to_numeric(c, errors="coerce") if c.name != "ticker" else c
        )
    )

    yield {"type": "progress", "stage": "sentiment", "progress": 45, "message": "Analyzing news sentiment…"}

    # Step 3: sentiment per ticker
    sent_rows = []
    for i, t in enumerate(tickers):
        yield {"type": "progress", "stage": "sentiment", "ticker": t, "progress": int(45 + (i / n) * 15), "message": f"Sentiment for {t}…"}
        sent_rows.append({"ticker": t, "sentiment_score": _compute_sentiment(t)})
    df_sent = pd.DataFrame(sent_rows)

    # Step 4: merge
    df = (
        df_val.merge(df_qual, on="ticker", how="outer")
        .merge(df_risk, on="ticker", how="outer")
        .merge(df_mom, on="ticker", how="outer")
        .merge(df_sent, on="ticker", how="left")
    )
    df["buy_score"] = df[["value_score", "quality_score", "risk_score", "momentum_score"]].mean(axis=1)
    df["buy_rating"] = df["buy_score"].apply(lambda x: buy_rating(clamp01(x)))
    df["sentiment_score"] = df["sentiment_score"].fillna(0.5)

    yield {"type": "progress", "stage": "llm", "progress": 62, "message": "Running Gemini AI analysis…"}

    # Step 5: Gemini concurrent
    with ThreadPoolExecutor(max_workers=min(8, n)) as pool:
        futures = {pool.submit(_gemini_call, _build_prompt(row)): i for i, row in df.iterrows()}
        completed = 0
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                res = {"confidence": 0.5, "reason": str(e)}
            df.at[i, "llm_confidence"] = res["confidence"]
            df.at[i, "llm_reason"] = res["reason"]
            completed += 1
            t = df.at[i, "ticker"]
            yield {
                "type": "progress",
                "stage": "llm",
                "ticker": t,
                "progress": int(62 + (completed / n) * 28),
                "message": f"Gemini reasoning for {t}…",
            }

    # Step 6: final scores
    df["final_buy_score"] = (
        WEIGHTS["algo"] * df["buy_score"].fillna(0.5)
        + WEIGHTS["sentiment"] * df["sentiment_score"]
        + WEIGHTS["llm"] * df["llm_confidence"].apply(clamp01)
    )
    df["final_rating"] = df["final_buy_score"].apply(lambda x: buy_rating(clamp01(x)))
    df = df.sort_values("final_buy_score", ascending=False).reset_index(drop=True)

    yield {"type": "progress", "stage": "done", "progress": 98, "message": "Finalizing results…"}

    # Step 7: emit per-ticker results
    SCORE_COLS = [
        "ticker", "final_buy_score", "final_rating",
        "buy_score", "buy_rating",
        "value_score", "quality_score", "risk_score", "momentum_score",
        "sentiment_score", "llm_confidence", "llm_reason",
        "pe", "pb", "peg", "div_yield", "ev_ebitda",
        "roe", "rev_growth", "ret_3m", "rsi", "vol", "sharpe_proxy",
    ]
    available = [c for c in SCORE_COLS if c in df.columns]
    for _, row in df[available].iterrows():
        record = {}
        for k, v in row.items():
            if isinstance(v, float) and np.isnan(v):
                record[k] = None
            elif hasattr(v, "item"):
                record[k] = v.item()
            else:
                record[k] = v
        yield {"type": "result", "ticker": row["ticker"], "data": record}

    yield {"type": "done", "progress": 100, "message": "Analysis complete!"}


def run_analysis(tickers: List[str]) -> pd.DataFrame:
    """Synchronous version — collects all streaming events and returns final DataFrame."""
    results = []
    for event in run_analysis_stream(tickers):
        if event.get("type") == "result":
            results.append(event["data"])
    return pd.DataFrame(results)
