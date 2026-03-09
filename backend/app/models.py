from pydantic import BaseModel, field_validator
from typing import List, Optional


class AnalysisRequest(BaseModel):
    tickers: List[str]

    @field_validator("tickers")
    @classmethod
    def validate_tickers(cls, v: List[str]) -> List[str]:
        if not v:
            raise ValueError("At least one ticker required")
        if len(v) > 20:
            raise ValueError("Maximum 20 tickers per request")
        cleaned = [t.strip().upper() for t in v if t.strip()]
        if not cleaned:
            raise ValueError("No valid tickers provided")
        return cleaned


class ScoreBreakdown(BaseModel):
    value_score: float
    quality_score: float
    risk_score: float
    momentum_score: float
    sentiment_score: float
    buy_score: float
    llm_confidence: float
    final_buy_score: float


class StockResult(BaseModel):
    ticker: str
    final_buy_score: float
    final_rating: str
    buy_score: float
    buy_rating: str
    value_score: float
    quality_score: float
    risk_score: float
    momentum_score: float
    sentiment_score: float
    llm_confidence: float
    llm_reason: str
    # raw metrics
    pe: Optional[float] = None
    pb: Optional[float] = None
    peg: Optional[float] = None
    div_yield: Optional[float] = None
    ev_ebitda: Optional[float] = None
    roe: Optional[float] = None
    rev_growth: Optional[float] = None
    ret_3m: Optional[float] = None
    rsi: Optional[float] = None
    vol: Optional[float] = None
    sharpe_proxy: Optional[float] = None
