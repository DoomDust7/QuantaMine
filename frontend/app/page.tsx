"use client";

import { useState, useRef, useCallback } from "react";
import {
  Search,
  Zap,
  TrendingUp,
  Brain,
  Shield,
  X,
  RotateCcw,
  AlertCircle,
} from "lucide-react";
import { cn, type StockResult, type SSEEvent } from "@/lib/utils";
import { StockCard } from "@/components/StockCard";
import { ProgressBar, SkeletonCard } from "@/components/ProgressBar";
import { ResultsTable } from "@/components/ResultsTable";

const API_URL = process.env.NEXT_PUBLIC_API_URL || "http://localhost:8000";

const EXAMPLE_TICKERS = [
  ["AAPL", "MSFT", "NVDA", "TSLA"],
  ["JPM", "BAC", "GS"],
  ["SPY", "QQQ", "AMZN", "META"],
];

export default function Home() {
  const [input, setInput] = useState("");
  const [loading, setLoading] = useState(false);
  const [progress, setProgress] = useState(0);
  const [progressMsg, setProgressMsg] = useState("");
  const [progressTicker, setProgressTicker] = useState<string | undefined>();
  const [results, setResults] = useState<StockResult[]>([]);
  const [error, setError] = useState<string | null>(null);
  const [pendingTickers, setPendingTickers] = useState<string[]>([]);

  const abortRef = useRef<AbortController | null>(null);

  const parseTickers = (raw: string): string[] =>
    raw
      .split(/[\s,]+/)
      .map((t) => t.trim().toUpperCase())
      .filter(Boolean);

  const handleAnalyze = useCallback(async () => {
    const tickers = parseTickers(input);
    if (!tickers.length) return;

    // Reset state
    setError(null);
    setResults([]);
    setLoading(true);
    setProgress(0);
    setProgressMsg("Initializing…");
    setProgressTicker(undefined);
    setPendingTickers(tickers);

    // Cancel any existing request
    abortRef.current?.abort();
    const controller = new AbortController();
    abortRef.current = controller;

    try {
      const response = await fetch(`${API_URL}/api/analyze/stream/live`, {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify({ tickers }),
        signal: controller.signal,
      });

      if (!response.ok) {
        throw new Error(`Server error: ${response.status}`);
      }

      const reader = response.body?.getReader();
      const decoder = new TextDecoder();
      if (!reader) throw new Error("No response body");

      let buffer = "";
      while (true) {
        const { done, value } = await reader.read();
        if (done) break;

        buffer += decoder.decode(value, { stream: true });
        const lines = buffer.split("\n");
        buffer = lines.pop() ?? "";

        for (const line of lines) {
          if (line.startsWith("data: ")) {
            const raw = line.slice(6).trim();
            if (raw === "[DONE]") {
              setLoading(false);
              setPendingTickers([]);
              break;
            }
            try {
              const event: SSEEvent = JSON.parse(raw);
              if (event.type === "progress") {
                setProgress(event.progress);
                setProgressMsg(event.message);
                setProgressTicker(event.ticker);
              } else if (event.type === "result") {
                setResults((prev) => [...prev, event.data]);
                setPendingTickers((prev) => prev.filter((t) => t !== event.ticker));
              } else if (event.type === "done") {
                setLoading(false);
                setPendingTickers([]);
              } else if (event.type === "error") {
                setError(event.message);
                setLoading(false);
              }
            } catch {
              // ignore parse errors for partial lines
            }
          }
        }
      }
    } catch (err: unknown) {
      if (err instanceof Error && err.name === "AbortError") return;
      setError(err instanceof Error ? err.message : "Analysis failed. Is the backend running?");
      setLoading(false);
    }
  }, [input]);

  const handleReset = () => {
    abortRef.current?.abort();
    setLoading(false);
    setResults([]);
    setError(null);
    setProgress(0);
    setPendingTickers([]);
    setInput("");
  };

  const hasResults = results.length > 0;

  return (
    <div className="min-h-screen bg-[#0a0f0e]">
      {/* Nav */}
      <nav className="sticky top-0 z-50 border-b border-[#1e3330] bg-[#0a0f0e]/90 backdrop-blur-md">
        <div className="mx-auto flex max-w-7xl items-center justify-between px-6 py-3">
          <div className="flex items-center gap-2.5">
            <div className="flex h-8 w-8 items-center justify-center rounded-lg bg-teal-500/20 border border-teal-500/30">
              <Zap size={15} className="text-teal-400" />
            </div>
            <span className="font-bold text-white tracking-tight">QuantaMine</span>
            <span className="rounded-full bg-teal-500/10 border border-teal-500/20 px-2 py-0.5 text-[10px] font-medium text-teal-400">
              v2
            </span>
          </div>
          <div className="hidden sm:flex items-center gap-5 text-xs text-gray-500">
            <span className="flex items-center gap-1.5"><TrendingUp size={12} />Fundamental</span>
            <span className="flex items-center gap-1.5"><Brain size={12} />Gemini AI</span>
            <span className="flex items-center gap-1.5"><Shield size={12} />Sentiment</span>
          </div>
        </div>
      </nav>

      <main className="mx-auto max-w-7xl px-4 py-10 sm:px-6">
        {/* Hero */}
        <div className="mb-10 text-center">
          <h1 className="text-4xl font-bold text-white sm:text-5xl">
            AI-Powered{" "}
            <span className="bg-gradient-to-r from-teal-400 to-emerald-400 bg-clip-text text-transparent">
              Stock Analysis
            </span>
          </h1>
          <p className="mt-3 text-gray-400 max-w-xl mx-auto text-sm sm:text-base">
            Combines fundamental metrics, technical indicators, news sentiment, and Gemini LLM reasoning into a single investment score.
          </p>
        </div>

        {/* Input section */}
        <div className="mx-auto max-w-2xl mb-10">
          <div className="rounded-2xl border border-[#1e3330] bg-[#0f1a18] p-5 space-y-4">
            <div className="flex gap-2">
              <div className="relative flex-1">
                <Search size={14} className="absolute left-3 top-1/2 -translate-y-1/2 text-gray-500" />
                <input
                  type="text"
                  placeholder="AAPL, MSFT, NVDA, TSLA…"
                  value={input}
                  onChange={(e) => setInput(e.target.value)}
                  onKeyDown={(e) => e.key === "Enter" && !loading && handleAnalyze()}
                  className="w-full rounded-xl border border-[#1e3330] bg-[#0a0f0e] pl-9 pr-4 py-2.5 text-sm text-white placeholder-gray-600 outline-none focus:border-teal-700 focus:ring-1 focus:ring-teal-700/50 transition-colors font-mono"
                  disabled={loading}
                />
              </div>
              {hasResults || loading ? (
                <button
                  onClick={handleReset}
                  className="flex items-center gap-1.5 rounded-xl border border-[#1e3330] bg-[#0a0f0e] px-4 py-2.5 text-sm text-gray-400 hover:text-white hover:border-gray-600 transition-colors"
                >
                  <RotateCcw size={14} />
                  Reset
                </button>
              ) : null}
              <button
                onClick={handleAnalyze}
                disabled={loading || !input.trim()}
                className={cn(
                  "flex items-center gap-1.5 rounded-xl px-5 py-2.5 text-sm font-medium transition-all",
                  loading || !input.trim()
                    ? "cursor-not-allowed bg-teal-900/30 text-teal-700"
                    : "bg-teal-600 text-white hover:bg-teal-500 shadow-lg shadow-teal-900/30"
                )}
              >
                {loading ? (
                  <>
                    <span className="h-3.5 w-3.5 rounded-full border-2 border-teal-400/30 border-t-teal-400 animate-spin" />
                    Analyzing…
                  </>
                ) : (
                  <>
                    <Zap size={14} />
                    Analyze
                  </>
                )}
              </button>
            </div>

            {/* Example tickers */}
            <div className="flex flex-wrap gap-2">
              {EXAMPLE_TICKERS.map((group, gi) => (
                <button
                  key={gi}
                  onClick={() => setInput(group.join(", "))}
                  disabled={loading}
                  className="rounded-lg border border-[#1e3330] bg-[#0a0f0e] px-2.5 py-1 text-xs text-gray-500 hover:text-teal-400 hover:border-teal-800 transition-colors font-mono disabled:opacity-40"
                >
                  {group.join(", ")}
                </button>
              ))}
            </div>
          </div>
        </div>

        {/* Error */}
        {error && (
          <div className="mx-auto max-w-2xl mb-6 flex items-start gap-3 rounded-xl border border-rose-800/40 bg-rose-900/10 px-4 py-3 text-sm text-rose-400">
            <AlertCircle size={16} className="mt-0.5 shrink-0" />
            <div>
              <p className="font-medium">Analysis failed</p>
              <p className="text-rose-500 text-xs mt-0.5">{error}</p>
            </div>
            <button onClick={() => setError(null)} className="ml-auto shrink-0 hover:text-rose-300">
              <X size={14} />
            </button>
          </div>
        )}

        {/* Progress */}
        {loading && (
          <div className="mx-auto max-w-2xl mb-8">
            <ProgressBar progress={progress} message={progressMsg} ticker={progressTicker} />
          </div>
        )}

        {/* Results grid */}
        {(hasResults || (loading && pendingTickers.length > 0)) && (
          <div className="space-y-8">
            {/* Summary */}
            {hasResults && (
              <div className="grid grid-cols-2 sm:grid-cols-4 gap-3">
                {[
                  { label: "Stocks Analyzed", value: results.length },
                  { label: "Strong Buy / Buy", value: results.filter((r) => r.final_rating === "Strong Buy" || r.final_rating === "Buy").length },
                  { label: "Avg Score", value: `${Math.round((results.reduce((s, r) => s + r.final_buy_score, 0) / results.length) * 100)}` },
                  { label: "Top Pick", value: results[0]?.ticker ?? "—" },
                ].map((s) => (
                  <div key={s.label} className="rounded-xl border border-[#1e3330] bg-[#0f1a18] p-4 text-center">
                    <div className="font-mono text-xl font-bold text-teal-400">{s.value}</div>
                    <div className="text-xs text-gray-500 mt-0.5">{s.label}</div>
                  </div>
                ))}
              </div>
            )}

            {/* Cards grid */}
            <div>
              <h2 className="text-sm font-medium text-gray-400 mb-4">
                {loading ? `Analyzing ${pendingTickers.join(", ")}…` : "Results"}{" "}
                <span className="text-gray-600">({results.length} complete)</span>
              </h2>
              <div className="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-3 gap-4">
                {results.map((r, i) => (
                  <StockCard key={r.ticker} result={r} rank={i + 1} />
                ))}
                {/* Skeleton cards for pending */}
                {loading && pendingTickers.slice(0, 3).map((t) => (
                  <SkeletonCard key={t} />
                ))}
              </div>
            </div>

            {/* Full table */}
            {hasResults && <ResultsTable results={results} />}
          </div>
        )}

        {/* Empty state */}
        {!hasResults && !loading && !error && (
          <div className="flex flex-col items-center justify-center py-20 text-center">
            <div className="flex h-16 w-16 items-center justify-center rounded-2xl bg-teal-500/10 border border-teal-500/20 mb-4">
              <TrendingUp size={28} className="text-teal-500" />
            </div>
            <p className="text-gray-400 font-medium">Enter tickers above to start analysis</p>
            <p className="text-gray-600 text-sm mt-1">Supports US stocks, ETFs, and international tickers</p>
          </div>
        )}
      </main>

      {/* Footer */}
      <footer className="border-t border-[#1e3330] mt-20 py-6 text-center text-xs text-gray-600">
        QuantaMine v2 · Powered by yfinance, FinBERT & Gemini AI · Not financial advice
      </footer>
    </div>
  );
}
