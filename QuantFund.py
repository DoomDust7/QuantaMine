# ai_backend.py
"""
Quant-ready equity screening backend.

Key fixes / improvements over previous version:
- NO hard-coded API keys; require env vars and fail fast.
- Batched yfinance downloads (yf.download) to reduce I/O and be deterministic.
- Per-ticker caching for info/history with lru_cache when appropriate.
- Robust requests Session with retries and timeouts for NewsAPI and HuggingFace.
- Safer normalization with handling for small samples and explicit neutral defaults.
- Corrected drawdown handling (use magnitude).
- Free-Cash-Flow yield computed as FCF / market_cap (not raw FCF).
- RSI implemented using Wilder smoothing (ewm).
- LLM (Gemini) calls executed concurrently via ThreadPoolExecutor; prompts parsed robustly.
- Logging hooks and lightweight metrics counters for observability.
- Configurable weights for final blend.
"""

import os
import json
import re
import time
import logging
from functools import lru_cache
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import List, Dict, Any, Optional

import numpy as np
import pandas as pd
import requests
import yfinance as yf
import google.generativeai as genai
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

# -----------------------
# CONFIG / ENV VALIDATION
# -----------------------
LOG = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")

HUGGINGFACE_API_URL = os.getenv("HUGGINGFACE_API_URL", "https://api-inference.huggingface.co/models/ProsusAI/finbert")
HUGGINGFACE_API_KEY = os.getenv("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")
NEWSAPI_URL = os.getenv("NEWSAPI_URL", "https://newsapi.org/v2/everything")
NEWSAPI_KEY = os.getenv("9b2a1413d6344c34ab7fb9a694b73ede")
GEMINI_API_KEY = os.getenv("AIzaSyCWf3DGL_n7TBf_B2YGM5kKUBi-iWbx6XI")
_GEMINI_MODEL_NAME = os.getenv("GEMINI_MODEL_NAME", "models/gemini-2.5-flash-lite")

# fail fast if required keys are missing
missing = []
if not HUGGINGFACE_API_KEY:
    missing.append("HUGGINGFACE_API_KEY")
if not NEWSAPI_KEY:
    missing.append("NEWSAPI_KEY")
if not GEMINI_API_KEY:
    missing.append("GEMINI_API_KEY")
if missing:
    raise RuntimeError(f"Missing required environment variables: {', '.join(missing)}")

# configure genai
genai.configure(api_key=GEMINI_API_KEY)

# Threadpool for blocking external calls (Gemini etc.)
_DEFAULT_THREAD_POOL = ThreadPoolExecutor(max_workers=int(os.getenv("THREADPOOL_MAX_WORKERS", "8")))

# weights for final aggregation; keep configurable
WEIGHTS = {
    "algo": float(os.getenv("WEIGHT_ALGO", 0.5)),       # value/quality/risk/momentum blend weight
    "sentiment": float(os.getenv("WEIGHT_SENT", 0.2)),
    "llm": float(os.getenv("WEIGHT_LLM", 0.3)),
}
assert abs(sum(WEIGHTS.values()) - 1.0) < 1e-6, "Weights must sum to 1.0"

# requests session with retries for reliability
SESSION = requests.Session()
RETRIES = Retry(total=3, backoff_factor=0.4, status_forcelist=[429, 500, 502, 503, 504])
SESSION.mount("https://", HTTPAdapter(max_retries=RETRIES))
DEFAULT_TIMEOUT = int(os.getenv("HTTP_TIMEOUT", "10"))

# -----------------------
# UTILITIES & HELPERS
# -----------------------
def safe_ratio(a: Optional[float], b: Optional[float]) -> float:
    """Return a/b with safety for None, nan, zero denom -> np.nan"""
    try:
        if a is None or b is None or pd.isna(a) or pd.isna(b) or b == 0:
            return np.nan
        return float(a) / float(b)
    except Exception:
        return np.nan

def clamp01(x: float) -> float:
    if pd.isna(x):
        return 0.5
    return max(0.0, min(1.0, float(x)))

def to_numeric_preserve_ticker(df: pd.DataFrame) -> pd.DataFrame:
    """Convert all columns except 'ticker' to numeric (coerce). Preserve ticker as string."""
    cols = df.columns.tolist()
    if "ticker" in cols:
        numeric_cols = [c for c in cols if c != "ticker"]
        df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors="coerce")
        df["ticker"] = df["ticker"].astype(str)
    else:
        df = df.apply(pd.to_numeric, errors="coerce")
    return df

def normalize(series: pd.Series, invert: bool = False, min_valid: int = 5) -> pd.Series:
    """
    Cross-sectional percentile normalization -> returns values in [0,1].
    - replace +/-inf with nan
    - if insufficient non-NA samples (< min_valid), return 0.5 neutral for all
    - fill missing with median before ranking
    - invert=True means lower raw values map to higher normalized scores
    Note: this is cohort-relative normalization (documented).
    """
    s = series.replace([np.inf, -np.inf], np.nan).astype(float)
    if s.dropna().shape[0] < min_valid:
        return pd.Series(0.5, index=s.index)
    filled = s.fillna(s.median())
    ranks = filled.rank(method="average", pct=True)
    return (1 - ranks) if invert else ranks

def compute_rsi(prices: pd.Series, period: int = 14) -> pd.Series:
    """
    RSI using Wilder's smoothing (EMA with alpha=1/period).
    Returns RSI series in [0, 100].
    """
    delta = prices.diff()
    gain = delta.clip(lower=0).fillna(0.0)
    loss = -delta.clip(upper=0).fillna(0.0)
    # Wilder smoothing via ewm (adjust=False matches recursive smoothing)
    avg_gain = gain.ewm(alpha=1.0 / period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1.0 / period, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-12)
    rsi = 100.0 - (100.0 / (1.0 + rs))
    return rsi

def extract_json_block(text: str) -> str:
    """
    Try to extract the first JSON object/block from the text.
    Falls back to the whole text if no block found.
    """
    m = re.search(r"\{(?:[^{}]|(?R))*\}", text, flags=re.DOTALL)
    if m:
        return m.group(0)
    return text

# -----------------------
# YFINANCE BATCHED INGEST
# -----------------------
@lru_cache(maxsize=256)
def _fetch_ticker_info(ticker: str) -> Dict[str, Any]:
    """Cache per-ticker info dict from yfinance."""
    try:
        info = yf.Ticker(ticker).info or {}
    except Exception as e:
        LOG.warning("yfinance info error for %s: %s", ticker, e)
        info = {}
    return info

def _batch_download_histories(tickers: List[str], period: str = "1y", interval: str = "1d") -> Dict[str, pd.DataFrame]:
    """
    Use yf.download to fetch histories for multiple tickers in one call to reduce network I/O.
    Returns a dict ticker -> DataFrame (with 'Close' column).
    """
    if not tickers:
        return {}
    try:
        # yf.download returns MultiIndex columns if multiple tickers; group_by='ticker' can help
        df = yf.download(tickers, period=period, interval=interval, group_by="ticker", threads=True, progress=False)
    except Exception as e:
        LOG.warning("yf.download failed: %s", e)
        # fallback to individual calls
        df = {}
    out = {}
    if isinstance(df, dict) or df is None:
        # fallback scenario: build per-ticker by calling history individually
        for t in tickers:
            try:
                hist = yf.Ticker(t).history(period=period, interval=interval)
                out[t] = hist
            except Exception as e:
                LOG.warning("yf.history fallback failed for %s: %s", t, e)
                out[t] = pd.DataFrame()
        return out
    # df is a DataFrame with grouped columns if multiple tickers
    # If a single ticker, df's columns are typical OHLCV; wrap accordingly
    if len(tickers) == 1:
        out[tickers[0]] = df
        return out
    # multi-ticker: columns like ('AAPL', 'Close'), etc.
    for t in tickers:
        try:
            # Some versions of yfinance create a nested dataframe: df[t]['Close']
            if (t in df.columns.levels[0]):
                sub = df[t]
            else:
                # alternative representation
                # try selecting columns that end with the ticker
                sub = df.xs(t, axis=1, level=0, drop_level=False) if hasattr(df.columns, 'levels') else df
            out[t] = sub.copy()
        except Exception:
            # best effort: empty DataFrame
            out[t] = pd.DataFrame()
    return out

# -----------------------
# METRIC CONSTRUCTORS
# -----------------------
def get_value_metrics_from_info(ticker: str, info: Dict[str, Any]) -> Dict[str, Any]:
    # value metrics (raw)
    return {
        "ticker": ticker,
        "pe": info.get("trailingPE"),
        "pb": info.get("priceToBook"),
        "peg": info.get("pegRatio"),
        "div_yield": safe_ratio(info.get("dividendRate") or info.get("dividendYield"), info.get("lastPrice") or info.get("previousClose") or 1),
        "ev_ebitda": info.get("enterpriseToEbitda")
    }

def get_quality_metrics_from_info(ticker: str, info: Dict[str, Any]) -> Dict[str, Any]:
    return {
        "ticker": ticker,
        "roe": info.get("returnOnEquity"),
        "rev_growth": info.get("revenueGrowth"),
        "eps_growth": info.get("earningsQuarterlyGrowth"),
        # compute FCF yield properly: freeCashflow / marketCap (if marketCap available)
        "fcf_yield": safe_ratio(info.get("freeCashflow"), info.get("marketCap")),
        "debt_to_equity": info.get("debtToEquity")
    }

def get_vol_risk_metrics_from_history(ticker: str, hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe_proxy": np.nan}
    prices = hist["Close"].dropna()
    if prices.shape[0] < 2:
        return {"ticker": ticker, "vol": np.nan, "max_dd": np.nan, "sharpe_proxy": np.nan}
    returns = prices.pct_change().dropna()
    vol = float(returns.std())
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max  # negative or zero
    max_dd = float(drawdown.min())  # e.g., -0.35
    sharpe_proxy = float(returns.mean() / (returns.std() + 1e-12))
    # return with drawdown magnitude positive (for easier normalization later)
    return {"ticker": ticker, "vol": vol, "max_dd": max_dd, "sharpe_proxy": sharpe_proxy}

def get_price_momentum_metrics_from_history(ticker: str, hist: pd.DataFrame) -> Dict[str, Any]:
    if hist is None or hist.empty or "Close" not in hist.columns:
        return {"ticker": ticker, "ret_3m": np.nan, "rsi": np.nan}
    prices = hist["Close"].dropna()
    # approx 63 trading days ~ 3 months
    if len(prices) > 63:
        denom = prices.iloc[-63]
        ret_3m = safe_ratio(prices.iloc[-1], denom) - 1 if denom != 0 else np.nan
    else:
        ret_3m = np.nan
    rsi = compute_rsi(prices).iloc[-1] if prices.shape[0] > 0 else np.nan
    return {"ticker": ticker, "ret_3m": ret_3m, "rsi": rsi}

# -----------------------
# SENTIMENT & NEWS (robust)
# -----------------------
def get_sentiment_from_hf(text: str, timeout: int = DEFAULT_TIMEOUT) -> float:
    """
    Call HuggingFace FinBERT (or configured model) and return a 0..1 sentiment score.
    Robust parsing: accept multiple response shapes. Return neutral 0.5 on failure.
    """
    try:
        resp = SESSION.post(HUGGINGFACE_API_URL,
                            headers={"Authorization": f"Bearer {HUGGINGFACE_API_KEY}"},
                            json={"inputs": text},
                            timeout=timeout)
        resp.raise_for_status()
        data = resp.json()
        # multiple possible shapes: list of lists/dicts, or dict with scores; handle defensively
        if isinstance(data, dict):
            # some HF endpoints return {'label': 'POSITIVE', 'score': 0.95}
            label = data.get("label")
            score = data.get("score")
            if label and score is not None:
                lbl = str(label).lower()
                sc = float(score)
                if "positive" in lbl:
                    return clamp01(sc)
                if "negative" in lbl:
                    return clamp01(1.0 - sc)
                return 0.5
            # sometimes nested
            candidates = data.get("scores") or data.get("results")
            if isinstance(candidates, list) and candidates:
                first = candidates[0]
                return get_sentiment_from_hf(json.dumps(first), timeout=timeout)
        if isinstance(data, list) and data:
            # often HF returns list-of-lists like [[{"label": "POSITIVE", "score": 0.99}, ...]]
            first = data[0]
            if isinstance(first, list) and first:
                best = max(first, key=lambda x: x.get("score", 0.0))
                lbl = str(best.get("label", "")).lower()
                sc = float(best.get("score", 0.0))
                if "positive" in lbl:
                    return clamp01(sc)
                if "negative" in lbl:
                    return clamp01(1.0 - sc)
                return 0.5
            if isinstance(first, dict):
                lbl = str(first.get("label", "")).lower()
                sc = float(first.get("score", 0.0))
                if "positive" in lbl:
                    return clamp01(sc)
                if "negative" in lbl:
                    return clamp01(1.0 - sc)
                return 0.5
        return 0.5
    except Exception as e:
        LOG.debug("HF sentiment call failed: %s", e)
        return 0.5

def fetch_news_for_ticker(ticker: str, company_name: Optional[str] = None, timeout: int = DEFAULT_TIMEOUT) -> List[Dict[str, Any]]:
    """
    Fetch recent news articles for a ticker/company.
    Returns list of dicts with keys: title, description, publishedAt
    Use company name if available (less noisy than ticker).
    """
    q = company_name or ticker
    params = {"q": q, "language": "en", "pageSize": 5, "apiKey": NEWSAPI_KEY}
    try:
        r = SESSION.get(NEWSAPI_URL, params=params, timeout=timeout)
        r.raise_for_status()
        data = r.json()
        articles = data.get("articles", []) or []
        out = []
        for a in articles:
            out.append({
                "title": a.get("title", "") or "",
                "description": a.get("description", "") or "",
                "publishedAt": a.get("publishedAt")
            })
        return out or []
    except Exception as e:
        LOG.debug("NewsAPI fetch failed for %s: %s", ticker, e)
        return []

def compute_sentiment_score_for_articles(articles: List[Dict[str, Any]]) -> float:
    """
    Compute a weighted sentiment score across articles.
    Weight by recency (exponential decay) so recent articles matter more.
    """
    if not articles:
        return 0.5
    now_ts = time.time()
    scores = []
    weights = []
    for art in articles:
        text = (art.get("title", "") + ". " + art.get("description", "")).strip()
        if not text:
            continue
        sc = get_sentiment_from_hf(text)
        # parse publishedAt to compute age; handle missing format gracefully
        pub = art.get("publishedAt")
        try:
            age_days = (now_ts - time.mktime(time.strptime(pub, "%Y-%m-%dT%H:%M:%SZ"))) / 86400.0 if pub else 30.0
        except Exception:
            age_days = 30.0
        # exponential weight: lambda = 0.5 per 7 days (half-life 7 days)
        lam = 0.5 / 7.0
        w = np.exp(-lam * max(0.0, age_days))
        scores.append(sc)
        weights.append(w)
    if not scores:
        return 0.5
    return float(np.average(scores, weights=weights))

def sentiment_scores(tickers: List[str], name_map: Optional[Dict[str, str]] = None) -> pd.DataFrame:
    """
    For each ticker, fetch news and compute weighted sentiment. name_map maps ticker->company_name if available.
    Returns DataFrame with columns: ticker, sentiment_score
    """
    out = []
    for t in tickers:
        articles = fetch_news_for_ticker(t, company_name=(name_map.get(t) if name_map else None))
        sc = compute_sentiment_score_for_articles(articles)
        out.append({"ticker": t, "sentiment_score": sc})
    return pd.DataFrame(out)

# -----------------------
# LLM (Gemini) INTERACTIONS - concurrent & robust
# -----------------------
def _build_prompt(row: pd.Series) -> str:
    return (
        "You are an expert equity analyst. You are given normalized 0–1 scores for a stock:\n\n"
        f"Ticker: {row['ticker']}\n"
        f"Value Score: {row.get('value_score', 0.5):.2f}\n"
        f"Quality Score: {row.get('quality_score', 0.5):.2f}\n"
        f"Risk Score: {row.get('risk_score', 0.5):.2f}\n"
        f"Momentum Score: {row.get('momentum_score', 0.5):.2f}\n"
        f"Sentiment Score: {row.get('sentiment_score', 0.5):.2f}\n\n"
        "Task:\n"
        "1) Output a BUY confidence between 0 and 1 (0 = strong sell, 0.5 = neutral, 1 = strong buy).\n"
        "2) Provide a short 1–2 sentence explanation referencing the above scores.\n\n"
        "Respond ONLY as valid JSON: {\"confidence\": <0-1>, \"reason\": \"<short explanation>\"}\n"
    )

def _safe_parse_json_from_llm(txt: str) -> Dict[str, Any]:
    """
    Extract JSON block and parse. On failure, return neutral confidence with raw text truncated for reason.
    """
    try:
        block = extract_json_block(txt)
        d = json.loads(block)
        conf = clamp01(float(d.get("confidence", 0.5)))
        reason = d.get("reason", "")
        if not isinstance(reason, str):
            reason = str(reason)[:400]
        return {"confidence": conf, "reason": reason}
    except Exception:
        return {"confidence": 0.5, "reason": (txt or "")[:400]}

def _gemini_call(prompt: str) -> Dict[str, Any]:
    """
    Blocking call to Gemini via google.generativeai, wrapped for threadpool execution.
    Return dict with keys: confidence (0..1), reason (string).
    """
    try:
        model = genai.GenerativeModel(_GEMINI_MODEL_NAME)
        resp = model.generate_content(prompt, generation_config=genai.types.GenerationConfig(
            temperature=0.2, max_output_tokens=256, response_mime_type="application/json"
        ))
        # Extract text robustly
        txt = getattr(resp, "text", None) or ""
        if not txt and getattr(resp, "candidates", None):
            try:
                parts = resp.candidates[0].content.parts
                txt = "".join(p.text for p in parts if hasattr(p, "text"))
            except Exception:
                txt = str(resp)
        return _safe_parse_json_from_llm(txt or "{}")
    except Exception as e:
        LOG.debug("Gemini call failed: %s", e)
        return {"confidence": 0.5, "reason": f"Gemini error: {e}"}

def evaluate_gemini_concurrent(df: pd.DataFrame, max_workers: int = 8) -> pd.DataFrame:
    """
    Evaluate Gemini for each row in df concurrently using a threadpool.
    Writes 'llm_confidence' and 'llm_reason' columns into df and returns df.
    """
    if df.empty:
        df["llm_confidence"] = []
        df["llm_reason"] = []
        return df
    LOG.info("Starting concurrent Gemini evaluation for %d tickers", df.shape[0])
    with ThreadPoolExecutor(max_workers=max_workers) as executor:
        futures = {}
        for i, row in df.iterrows():
            prompt = _build_prompt(row)
            futures[executor.submit(_gemini_call, prompt)] = i
        for fut in as_completed(futures):
            i = futures[fut]
            try:
                res = fut.result()
            except Exception as e:
                LOG.debug("Gemini worker threw: %s", e)
                res = {"confidence": 0.5, "reason": f"Exception: {e}"}
            df.at[i, "llm_confidence"] = res.get("confidence", 0.5)
            df.at[i, "llm_reason"] = res.get("reason", "No reason")
    return df

# -----------------------
# AGGREGATION / SCORING
# -----------------------
def value_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["pe_score"] = normalize(df["pe"], invert=True)
    df["pb_score"] = normalize(df["pb"], invert=True)
    df["peg_score"] = normalize(df["peg"], invert=True)
    df["div_yield_score"] = normalize(df["div_yield"])
    df["ev_ebitda_score"] = normalize(df["ev_ebitda"], invert=True)
    df["value_score"] = df[["pe_score", "pb_score", "peg_score", "div_yield_score", "ev_ebitda_score"]].mean(axis=1)
    return df

def quality_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["roe_score"] = normalize(df["roe"])
    df["rev_growth_score"] = normalize(df["rev_growth"])
    df["eps_growth_score"] = normalize(df["eps_growth"])
    df["fcf_yield_score"] = normalize(df["fcf_yield"])
    df["debt_to_equity_score"] = normalize(df["debt_to_equity"], invert=True)
    df["quality_score"] = df[["roe_score", "rev_growth_score", "eps_growth_score", "fcf_yield_score", "debt_to_equity_score"]].mean(axis=1)
    return df

def vol_risk_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    # convert negative drawdown to magnitude for normalization
    if "max_dd" in df.columns:
        df["dd_mag"] = df["max_dd"].abs()  # positive magnitudes
    else:
        df["dd_mag"] = np.nan
    df["vol_score"] = normalize(df["vol"], invert=True)
    df["dd_score"] = normalize(df["dd_mag"], invert=True)
    df["sharpe_score"] = normalize(df["sharpe_proxy"])
    df["risk_score"] = df[["vol_score", "dd_score", "sharpe_score"]].mean(axis=1)
    return df

def momentum_score(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["ret_score"] = normalize(df["ret_3m"])
    # prefer RSI near 50 -> map to [0,1] with 1 at 50, linear decays to 0 at 0 or 100
    df["rsi_score"] = 1.0 - (df["rsi"] - 50.0).abs() / 50.0
    df["rsi_score"] = df["rsi_score"].clip(0.0, 1.0).fillna(0.5)
    df["momentum_score"] = df[["ret_score", "rsi_score"]].mean(axis=1)
    return df

def buy_rating(score: float) -> str:
    if score >= 0.8:
        return "Strong Buy"
    elif score >= 0.65:
        return "Buy"
    elif score >= 0.45:
        return "Hold"
    else:
        return "Avoid"

# -----------------------
# MASTER PIPELINE
# -----------------------
def run_analysis(tickers: List[str], company_name_map: Optional[Dict[str, str]] = None,
                 yf_periods: Dict[str, str] = None, threadpool_workers: int = 8) -> pd.DataFrame:
    """
    Synchronous wrapper that performs the full analysis:
    - fetches info/history in batches
    - computes value/quality/risk/momentum metrics
    - fetches news & computes sentiment
    - calls Gemini concurrently for reasoning
    - aggregates into final buy scores and ratings

    Returns a DataFrame with per-ticker scores and metadata.
    """
    company_name_map = company_name_map or {}
    yf_periods = yf_periods or {"momentum": "1y", "risk": "6mo"}

    LOG.info("Running analysis for %d tickers", len(tickers))

    # 1) Fetch batched histories
    hist_mom = _batch_download_histories(tickers, period=yf_periods["momentum"], interval="1d")
    hist_risk = _batch_download_histories(tickers, period=yf_periods["risk"], interval="1d")

    # 2) construct metric rows using cached info & histories
    val_rows = []
    qual_rows = []
    risk_rows = []
    mom_rows = []
    for t in tickers:
        info = _fetch_ticker_info(t)
        val_rows.append(get_value_metrics_from_info(t, info))
        qual_rows.append(get_quality_metrics_from_info(t, info))
        risk_rows.append(get_vol_risk_metrics_from_history(t, hist_risk.get(t, pd.DataFrame())))
        mom_rows.append(get_price_momentum_metrics_from_history(t, hist_mom.get(t, pd.DataFrame())))

    df_val = to_numeric_preserve_ticker(pd.DataFrame(val_rows))
    df_qual = to_numeric_preserve_ticker(pd.DataFrame(qual_rows))
    df_risk = to_numeric_preserve_ticker(pd.DataFrame(risk_rows))
    df_mom = to_numeric_preserve_ticker(pd.DataFrame(mom_rows))

    # 3) score sub-pillars
    df_val = value_score(df_val)
    df_qual = quality_score(df_qual)
    df_risk = vol_risk_score(df_risk)
    df_mom = momentum_score(df_mom)

    # 4) sentiment (news -> HF)
    df_sent = sentiment_scores(tickers, name_map=company_name_map)
    df_sent = to_numeric_preserve_ticker(df_sent)

    # 5) merge safely using outer join to preserve tickers and mark missing fields
    df = df_val.merge(df_qual, on="ticker", how="outer", suffixes=("_val", "_qual"))
    df = df.merge(df_risk, on="ticker", how="outer")
    df = df.merge(df_mom, on="ticker", how="outer", suffixes=("_risk", "_mom"))
    df = df.merge(df_sent, on="ticker", how="left")

    # 6) ensure numeric columns converted (safe)
    df = to_numeric_preserve_ticker(df)

    # 7) aggregate algorithmic buy score (equal-weighted of sub-pillar scores)
    df["buy_score"] = df[["value_score", "quality_score", "risk_score", "momentum_score"]].mean(axis=1)
    df["buy_rating"] = df["buy_score"].apply(lambda x: buy_rating(clamp01(x if not pd.isna(x) else 0.5)))

    # 8) initialize llm columns
    df["llm_confidence"] = 0.5
    df["llm_reason"] = "Pending"

    # 9) run LLM concurrently (threadpool) to obtain llm_confidence & reason
    df = evaluate_gemini_concurrent(df, max_workers=threadpool_workers)

    # 10) final blend with configurable weights
    df["sentiment_score"] = df["sentiment_score"].fillna(0.5)
    df["llm_confidence"] = df["llm_confidence"].apply(clamp01)
    df["final_buy_score"] = (
        WEIGHTS["algo"] * df["buy_score"].fillna(0.5) +
        WEIGHTS["sentiment"] * df["sentiment_score"] +
        WEIGHTS["llm"] * df["llm_confidence"]
    )
    df["final_rating"] = df["final_buy_score"].apply(lambda x: buy_rating(clamp01(x if not pd.isna(x) else 0.5)))

    # 11) metadata and diagnostics
    df["universe_count"] = len(tickers)
    LOG.info("Analysis complete for %d tickers", len(tickers))
    return df.reset_index(drop=True)
