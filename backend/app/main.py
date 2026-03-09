import json
import os
import asyncio
from typing import AsyncGenerator
from dotenv import load_dotenv

load_dotenv(dotenv_path=os.path.join(os.path.dirname(__file__), "../../.env"))

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
import yfinance as yf

from app.models import AnalysisRequest
from app.analyzer import run_analysis_stream

app = FastAPI(title="QuantaMine API", version="2.0.0", description="AI-Powered Stock Analyzer")

FRONTEND_URL = os.getenv("FRONTEND_URL", "http://localhost:3000")

app.add_middleware(
    CORSMiddleware,
    allow_origins=[FRONTEND_URL, "http://localhost:3000", "http://localhost:3001"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


@app.get("/health")
async def health():
    return {"status": "ok", "version": "2.0.0"}


@app.post("/api/validate")
async def validate_tickers(req: AnalysisRequest):
    """Quick ticker validation via yfinance — returns valid/invalid lists."""
    valid, invalid = [], []
    for ticker in req.tickers:
        try:
            info = yf.Ticker(ticker).info
            if info and info.get("regularMarketPrice") is not None or info.get("previousClose") is not None:
                valid.append(ticker)
            else:
                # Try history as fallback
                hist = yf.Ticker(ticker).history(period="5d")
                if not hist.empty:
                    valid.append(ticker)
                else:
                    invalid.append(ticker)
        except Exception:
            invalid.append(ticker)
    return {"valid": valid, "invalid": invalid}


@app.post("/api/analyze/stream")
async def analyze_stream(req: AnalysisRequest):
    """
    Server-Sent Events endpoint.
    Streams progress updates and per-ticker results as they are computed.
    """

    async def event_generator() -> AsyncGenerator[str, None]:
        loop = asyncio.get_event_loop()

        def run_sync():
            events = []
            for event in run_analysis_stream(req.tickers):
                events.append(event)
            return events

        # Run the synchronous generator in a thread to avoid blocking
        events = await loop.run_in_executor(None, run_sync)

        for event in events:
            data = json.dumps(event, default=str)
            yield f"data: {data}\n\n"

        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={
            "Cache-Control": "no-cache",
            "X-Accel-Buffering": "no",
        },
    )


@app.post("/api/analyze/stream/live")
async def analyze_stream_live(req: AnalysisRequest):
    """
    True streaming SSE endpoint — emits events as they are produced.
    Uses asyncio.Queue bridging the sync generator to async SSE.
    """
    queue: asyncio.Queue = asyncio.Queue()

    async def producer():
        loop = asyncio.get_event_loop()
        def _run():
            for event in run_analysis_stream(req.tickers):
                loop.call_soon_threadsafe(queue.put_nowait, event)
            loop.call_soon_threadsafe(queue.put_nowait, None)  # sentinel

        await loop.run_in_executor(None, _run)

    async def event_generator() -> AsyncGenerator[str, None]:
        asyncio.create_task(producer())
        while True:
            event = await queue.get()
            if event is None:
                break
            yield f"data: {json.dumps(event, default=str)}\n\n"
        yield "data: [DONE]\n\n"

    return StreamingResponse(
        event_generator(),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )
