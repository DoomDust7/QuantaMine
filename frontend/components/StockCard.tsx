"use client";

import { useState } from "react";
import { ChevronDown, ChevronUp, Brain, TrendingUp, TrendingDown } from "lucide-react";
import { cn, fmt, fmtPct, getRatingConfig, type StockResult } from "@/lib/utils";
import { ScoreBar } from "./ScoreBar";
import {
  RadialBarChart,
  RadialBar,
  PolarAngleAxis,
  ResponsiveContainer,
} from "recharts";

interface StockCardProps {
  result: StockResult;
  rank: number;
}

const SCORE_LABELS: { key: keyof StockResult; label: string }[] = [
  { key: "value_score", label: "Value" },
  { key: "quality_score", label: "Quality" },
  { key: "risk_score", label: "Risk" },
  { key: "momentum_score", label: "Momentum" },
  { key: "sentiment_score", label: "Sentiment" },
  { key: "llm_confidence", label: "AI Confidence" },
];

export function StockCard({ result, rank }: StockCardProps) {
  const [expanded, setExpanded] = useState(false);
  const cfg = getRatingConfig(result.final_rating);
  const score = Math.round((result.final_buy_score ?? 0) * 100);
  const chartData = [{ name: "score", value: score, fill: getChartColor(result.final_buy_score) }];
  const isPositive = (result.ret_3m ?? 0) >= 0;

  return (
    <div
      className={cn(
        "rounded-2xl border bg-[#0f1a18] p-5 flex flex-col gap-4 transition-all duration-300",
        "hover:border-teal-800/60 hover:shadow-lg hover:shadow-teal-900/20",
        result.final_buy_score >= 0.7 ? "border-teal-800/40 glow-green" : "border-[#1e3330]",
        "animate-slide-up"
      )}
    >
      {/* Header */}
      <div className="flex items-start justify-between gap-3">
        <div className="flex items-center gap-3">
          <div className="flex h-9 w-9 items-center justify-center rounded-xl bg-[#162320] text-xs font-bold text-gray-400 border border-[#1e3330]">
            #{rank}
          </div>
          <div>
            <div className="flex items-center gap-2">
              <span className="font-mono text-xl font-bold text-white">{result.ticker}</span>
              {result.ret_3m != null && (
                <span className={cn("flex items-center gap-0.5 text-xs font-medium", isPositive ? "text-emerald-400" : "text-rose-400")}>
                  {isPositive ? <TrendingUp size={12} /> : <TrendingDown size={12} />}
                  {fmtPct(result.ret_3m)}
                </span>
              )}
            </div>
            <span className={cn("mt-0.5 inline-flex items-center gap-1.5 rounded-full border px-2 py-0.5 text-xs font-medium", cfg.bg, cfg.color)}>
              <span className={cn("h-1.5 w-1.5 rounded-full", cfg.dot)} />
              {result.final_rating}
            </span>
          </div>
        </div>

        {/* Radial score gauge */}
        <div className="h-16 w-16 shrink-0">
          <ResponsiveContainer width="100%" height="100%">
            <RadialBarChart
              cx="50%"
              cy="50%"
              innerRadius="70%"
              outerRadius="100%"
              startAngle={90}
              endAngle={-270}
              data={chartData}
            >
              <PolarAngleAxis type="number" domain={[0, 100]} tick={false} />
              <RadialBar dataKey="value" cornerRadius={6} background={{ fill: "#1e3330" }} />
              <text
                x="50%"
                y="50%"
                textAnchor="middle"
                dominantBaseline="middle"
                className="fill-white font-mono font-bold"
                fontSize={14}
              >
                {score}
              </text>
            </RadialBarChart>
          </ResponsiveContainer>
        </div>
      </div>

      {/* Score bars */}
      <div className="grid grid-cols-2 gap-x-4 gap-y-2.5">
        {SCORE_LABELS.map(({ key, label }) => (
          <ScoreBar key={key} label={label} value={result[key] as number} />
        ))}
      </div>

      {/* Expand toggle */}
      <button
        onClick={() => setExpanded((e) => !e)}
        className="flex items-center justify-between text-xs text-gray-500 hover:text-gray-300 transition-colors pt-1 border-t border-[#1e3330]"
      >
        <span className="flex items-center gap-1.5">
          <Brain size={12} />
          AI Reasoning & Details
        </span>
        {expanded ? <ChevronUp size={14} /> : <ChevronDown size={14} />}
      </button>

      {/* Expanded section */}
      {expanded && (
        <div className="space-y-4 animate-fade-in">
          {/* AI reasoning */}
          <div className="rounded-xl bg-[#162320] border border-[#1e3330] p-3">
            <p className="text-xs font-medium text-teal-400 mb-1.5">Gemini Analysis</p>
            <p className="text-sm text-gray-300 leading-relaxed">{result.llm_reason || "No reasoning available."}</p>
          </div>

          {/* Raw metrics grid */}
          <div>
            <p className="text-xs font-medium text-gray-500 mb-2">Raw Metrics</p>
            <div className="grid grid-cols-3 gap-2">
              {[
                { label: "P/E", value: fmt(result.pe) },
                { label: "P/B", value: fmt(result.pb) },
                { label: "PEG", value: fmt(result.peg) },
                { label: "Div Yield", value: fmtPct(result.div_yield) },
                { label: "EV/EBITDA", value: fmt(result.ev_ebitda) },
                { label: "ROE", value: fmtPct(result.roe) },
                { label: "Rev Growth", value: fmtPct(result.rev_growth) },
                { label: "RSI", value: fmt(result.rsi, 1) },
                { label: "Volatility", value: fmtPct(result.vol) },
                { label: "Sharpe", value: fmt(result.sharpe_proxy, 3) },
              ].map((m) => (
                <div key={m.label} className="rounded-lg bg-[#0a0f0e] border border-[#1e3330] p-2 text-center">
                  <div className="text-[10px] text-gray-500 mb-0.5">{m.label}</div>
                  <div className="font-mono text-xs font-medium text-gray-200">{m.value}</div>
                </div>
              ))}
            </div>
          </div>
        </div>
      )}
    </div>
  );
}

function getChartColor(score: number): string {
  if (score >= 0.75) return "#34d399";
  if (score >= 0.55) return "#14b8a6";
  if (score >= 0.4) return "#fbbf24";
  return "#f87171";
}
