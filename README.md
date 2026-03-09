# QuantaMine v2 — AI-Powered Stock Analyzer

> Combines fundamental metrics, technical indicators, news sentiment, and Gemini LLM reasoning into a composite investment score.

## Architecture

```
QuantaMine/
├── backend/          FastAPI (Python 3.11)
│   └── app/
│       ├── main.py       SSE streaming API
│       ├── analyzer.py   Scoring engine
│       └── models.py     Pydantic models
├── frontend/         Next.js 15 + TailwindCSS
│   └── app/
│       ├── page.tsx      Main dashboard
│       └── components/   StockCard, ScoreBar, etc.
├── docker-compose.yml
└── .env.example
```

## Scoring Model

| Component | Weight | Source |
|-----------|--------|--------|
| Algo Score | 50% | Value + Quality + Risk + Momentum |
| Sentiment | 20% | FinBERT on NewsAPI headlines |
| LLM Confidence | 30% | Gemini reasoning |

**Ratings:** Strong Buy (≥80) · Buy (≥65) · Hold (≥45) · Avoid (<45)

## Quick Start

### 1. Configure environment

```bash
cp .env.example .env
# Fill in GEMINI_API_KEY, NEWSAPI_KEY, HUGGINGFACE_API_KEY
```

### 2. Run with Docker

```bash
docker compose up --build
```

Open [http://localhost:3000](http://localhost:3000)

### 3. Run locally (development)

**Backend:**
```bash
cd backend
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
uvicorn app.main:app --reload --port 8000
```

**Frontend:**
```bash
cd frontend
npm install
npm run dev
```

## API

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/health` | GET | Health check |
| `/api/validate` | POST | Validate ticker symbols |
| `/api/analyze/stream/live` | POST | SSE streaming analysis |

### Example

```bash
curl -X POST http://localhost:8000/api/analyze/stream/live \
  -H "Content-Type: application/json" \
  -d '{"tickers": ["AAPL", "MSFT", "NVDA"]}'
```

## Deployment

### Vercel (frontend) + Railway/Render (backend)

1. Deploy backend to Railway or Render
2. Set `NEXT_PUBLIC_API_URL` to your backend URL
3. Deploy frontend to Vercel

## Features

- **Real-time streaming** — results appear as each ticker is analyzed via SSE
- **6-pillar scoring** — Value, Quality, Risk, Momentum, Sentiment, AI
- **Gemini reasoning** — natural language explanation per stock
- **Interactive UI** — expandable cards, sortable table, CSV export
- **Docker ready** — single `docker compose up` to run everything
