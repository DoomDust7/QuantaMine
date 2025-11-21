# analytics/llm.py
import asyncio
import json
import logging
from typing import List, Dict, Any
import httpx
from config.api_keys import XAI_API_URL, XAI_API_KEY, XAI_MODEL_NAME

log = logging.getLogger(__name__)

HEADERS = {"Authorization": f"Bearer {XAI_API_KEY}", "Content-Type": "application/json"}

def _build_prompt_batch(rows: List[Dict[str, Any]]) -> str:
    # Build a single batched prompt requesting JSON mapping
    # Modified to let Grok reason independently without pillar scores
    prompt_lines = []
    prompt_lines.append("You are Grok built by xAI, acting as a senior equity analyst. For each ticker, using your up-to-date knowledge, provide a comprehensive 7-day performance score out of 100 (higher = better potential for market outperformance), based on factors like financial strength, profitability, growth, valuation, and momentum. Output as a short JSON object with keys 'confidence' (score/100 as 0-1 float) and 'reason' (1-2 sentences, and at the end include the expected movement in the next 7 days).")
    prompt_lines.append("Return a single JSON object mapping ticker -> {confidence, reason}. Example:")
    prompt_lines.append('{"AAPL": {"confidence": 0.94, "reason": "Short reason, and expected movement in the next 7 days: +2%."}, "MSFT": {...}}')
    prompt_lines.append("")
    for r in rows:
        prompt_lines.append(f"Ticker: {r.get('ticker')}")
        prompt_lines.append("---")
    prompt_lines.append("")
    prompt_lines.append("Respond ONLY with valid JSON and nothing else.")
    return "\n".join(prompt_lines)

def _safe_parse_batch(txt: str) -> Dict[str, Any]:
    try:
        d = json.loads(txt)
        out = {}
        for k, v in d.items():
            try:
                conf = float(v.get("confidence", 0.5))
                conf = max(0.0, min(1.0, conf))
                reason = str(v.get("reason", "No reason"))
                out[k] = {"confidence": conf, "reason": reason}
            except Exception:
                out[k] = {"confidence": 0.5, "reason": str(v)[:200]}
        return out
    except Exception as e:
        log.warning(f"Failed to parse LLM response JSON: {e}")
        return {}

def _xai_call_sync(prompt: str, timeout: int = 25) -> Dict[str, Any]:
    payload = {
        "model": XAI_MODEL_NAME,
        "messages": [{"role": "user", "content": prompt}],
        "temperature": 0.3,
        "max_tokens": 800,
    }
    try:
        r = httpx.post(XAI_API_URL, headers=HEADERS, json=payload, timeout=timeout)
        r.raise_for_status()
        txt = r.json()["choices"][0]["message"]["content"]
        return _safe_parse_batch(txt)
    except Exception as e:
        log.warning(f"LLM call failed: {e}")
        return {}

async def run_batched_llm(rows: List[Dict[str, Any]]) -> Dict[str, Dict[str, Any]]:
    # Batch size guard — keep prompt reasonably sized
    batch_size = 15
    results = {}
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        prompt = _build_prompt_batch(batch)
        loop = asyncio.get_running_loop()
        parsed = await loop.run_in_executor(None, _xai_call_sync, prompt)
        results.update(parsed)
    return results