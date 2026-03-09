"""
QuantaMine Portfolio Allocator
Given a budget and a list of tickers, computes optimal long-term allocation
using 5-year historical data: CAGR, Sharpe ratio, and max drawdown.
"""

import logging
import math
import time
from typing import Dict, Generator, List, Optional

import numpy as np
import pandas as pd

from app.analyzer import (
    _get_history,
    _gemini_call,
    clamp01,
    normalize,
    _YF_DELAY,
)

LOG = logging.getLogger(__name__)

RISK_FREE_ANNUAL = 0.04  # 4% annualised risk-free rate


# ── Per-ticker 5-year metrics ─────────────────────────────────────────────────

def _compute_5y_metrics(ticker: str, hist: pd.DataFrame) -> Dict:
    base = {
        "ticker": ticker,
        "cagr": None,
        "annualized_vol": None,
        "sharpe_5y": None,
        "max_dd": None,
    }
    if hist.empty or "Close" not in hist.columns:
        return base

    prices = hist["Close"].dropna()
    if len(prices) < 50:  # need enough data for meaningful stats
        return base

    # CAGR
    start, end = float(prices.iloc[0]), float(prices.iloc[-1])
    n_years = len(prices) / 252.0
    try:
        cagr = (end / start) ** (1.0 / n_years) - 1.0
    except Exception:
        cagr = None

    # Daily returns
    daily_ret = prices.pct_change().dropna()

    # Annualized volatility
    ann_vol = float(daily_ret.std() * math.sqrt(252)) if len(daily_ret) > 1 else None

    # Annualized Sharpe
    rf_daily = RISK_FREE_ANNUAL / 252.0
    excess = daily_ret - rf_daily
    if len(excess) > 1 and excess.std() > 1e-12:
        sharpe = float((excess.mean() / excess.std()) * math.sqrt(252))
    else:
        sharpe = None

    # Max drawdown
    rolling_max = prices.cummax()
    drawdown = (prices - rolling_max) / rolling_max
    max_dd = float(drawdown.min())

    return {
        "ticker": ticker,
        "cagr": cagr,
        "annualized_vol": ann_vol,
        "sharpe_5y": sharpe,
        "max_dd": max_dd,
    }


# ── Allocation engine ─────────────────────────────────────────────────────────

def _score_and_allocate(metrics: List[Dict], budget: float) -> List[Dict]:
    df = pd.DataFrame(metrics)

    # Normalize each metric — percentile rank approach (same as analyzer.py)
    df["cagr_rank"] = normalize(df["cagr"].apply(
        lambda x: float(x) if x is not None else np.nan
    ))
    df["sharpe_rank"] = normalize(df["sharpe_5y"].apply(
        lambda x: float(x) if x is not None else np.nan
    ))
    df["dd_rank"] = normalize(df["max_dd"].apply(
        lambda x: abs(float(x)) if x is not None else np.nan
    ), invert=True)  # lower drawdown magnitude = better rank

    # Composite long-term score
    df["composite_score"] = (
        0.40 * df["cagr_rank"].fillna(0.5)
        + 0.40 * df["sharpe_rank"].fillna(0.5)
        + 0.20 * df["dd_rank"].fillna(0.5)
    )

    # Softmax allocation (temperature=4 for meaningful differentiation)
    scores = df["composite_score"].values.astype(float)
    temp = 4.0
    exp_scores = np.exp(scores * temp - np.max(scores * temp))  # numerically stable
    raw_alloc = exp_scores / exp_scores.sum()

    # Clamp: min 5%, max 60%, iteratively renormalize until constraints hold
    n = len(df)
    min_alloc = 0.05
    max_alloc = 0.60
    w = raw_alloc.copy()
    for _ in range(50):
        w = np.clip(w, min_alloc, max_alloc)
        s = w.sum()
        if s < 1e-9:
            w = np.ones(n) / n
            break
        w = w / s
        if np.all(w >= min_alloc - 1e-9) and np.all(w <= max_alloc + 1e-9):
            break
    clamped = w

    df["allocation_pct"] = clamped
    df["allocation_usd"] = clamped * budget

    results = []
    for _, row in df.sort_values("allocation_pct", ascending=False).iterrows():
        results.append({
            "ticker": row["ticker"],
            "allocation_pct": float(row["allocation_pct"]),
            "allocation_usd": float(row["allocation_usd"]),
            "cagr_5y": row["cagr"] if row["cagr"] is not None else None,
            "sharpe_5y": row["sharpe_5y"] if row["sharpe_5y"] is not None else None,
            "max_dd_5y": row["max_dd"] if row["max_dd"] is not None else None,
            "annualized_vol": row["annualized_vol"] if row["annualized_vol"] is not None else None,
            "composite_score": float(row["composite_score"]),
        })
    return results


# ── Gemini portfolio summary ──────────────────────────────────────────────────

def _gemini_portfolio_summary(allocations: List[Dict], budget: float) -> str:
    lines = []
    for a in allocations:
        cagr_str = f"{a['cagr_5y']*100:.1f}%" if a["cagr_5y"] is not None else "N/A"
        sharpe_str = f"{a['sharpe_5y']:.2f}" if a["sharpe_5y"] is not None else "N/A"
        dd_str = f"{a['max_dd_5y']*100:.1f}%" if a["max_dd_5y"] is not None else "N/A"
        lines.append(
            f"  {a['ticker']}: {a['allocation_pct']*100:.1f}% (${a['allocation_usd']:,.0f})"
            f" | 5Y CAGR {cagr_str} | Sharpe {sharpe_str} | Max DD {dd_str}"
        )
    table = "\n".join(lines)

    prompt = (
        f"You are a friendly financial educator helping a new investor understand their portfolio.\n\n"
        f"Budget: ${budget:,.0f}\n"
        f"Proposed allocation based on 5-year historical performance:\n{table}\n\n"
        "Write a 2-3 sentence plain-English summary that:\n"
        "1. Describes the overall portfolio strategy (growth-oriented, diversified, etc.)\n"
        "2. Highlights the strongest holding and why\n"
        "3. Mentions one key risk the investor should know\n\n"
        'Respond ONLY as valid JSON: {"confidence": 0.0, "reason": "<your 2-3 sentence summary here>"}'
    )

    result = _gemini_call(prompt)
    reason = result.get("reason", "")
    if reason and len(reason) > 40 and "unavailable" not in reason.lower():
        return reason
    return (
        f"This portfolio of {len(allocations)} stocks is allocated based on 5-year historical "
        f"CAGR and risk-adjusted returns (Sharpe ratio). The largest position reflects the best "
        f"combination of growth and stability over the past 5 years. As with all investments, "
        f"past performance does not guarantee future results."
    )


# ── Main streaming generator ──────────────────────────────────────────────────

def run_portfolio_stream(tickers: List[str], budget: float) -> Generator[Dict, None, None]:
    """Yields SSE-style event dicts as portfolio allocation progresses."""
    n = len(tickers)
    yield {
        "type": "progress", "stage": "init", "progress": 2,
        "message": f"Starting portfolio analysis for {n} ticker(s)…",
    }

    # Fetch 5-year history with rate-limit delay
    hists: Dict[str, pd.DataFrame] = {}
    for i, t in enumerate(tickers):
        yield {
            "type": "progress", "stage": "fetch", "ticker": t,
            "progress": int(5 + (i / n) * 50),
            "message": f"Fetching 5-year history for {t}…",
        }
        if i > 0:
            time.sleep(_YF_DELAY)
        hists[t] = _get_history(t, period="5y")
        LOG.info("Fetched 5Y history %s: %d rows", t, len(hists[t]))

    yield {"type": "progress", "stage": "compute", "progress": 57, "message": "Computing 5-year metrics…"}

    metrics = [_compute_5y_metrics(t, hists[t]) for t in tickers]

    yield {"type": "progress", "stage": "allocate", "progress": 70, "message": "Optimizing budget allocation…"}

    allocations = _score_and_allocate(metrics, budget)

    yield {"type": "progress", "stage": "llm", "progress": 78, "message": "Generating portfolio summary…"}

    summary = _gemini_portfolio_summary(allocations, budget)

    yield {"type": "progress", "stage": "done", "progress": 95, "message": "Finalizing allocations…"}

    for alloc in allocations:
        yield {"type": "allocation", "ticker": alloc["ticker"], "data": alloc}

    yield {"type": "summary", "text": summary}
    yield {"type": "done", "progress": 100, "message": "Portfolio allocation complete!"}
