"use client";

import { cn, fmt, fmtPct, type PortfolioResult } from "@/lib/utils";
import { ScoreBar } from "@/components/ScoreBar";

interface AllocationCardProps {
  result: PortfolioResult;
  rank: number;
}

function metricColor(value: number | null, good: "high" | "low"): string {
  if (value == null) return "text-gray-500";
  const isGood = good === "high" ? value > 0 : value < 0;
  if (isGood) return "text-emerald-400";
  return "text-rose-400";
}

export function AllocationCard({ result, rank }: AllocationCardProps) {
  const pct = (result.allocation_pct * 100).toFixed(1);
  const usd = result.allocation_usd.toLocaleString("en-US", {
    style: "currency",
    currency: "USD",
    maximumFractionDigits: 0,
  });

  const cagr = result.cagr_5y;
  const sharpe = result.sharpe_5y;
  const maxDd = result.max_dd_5y;
  const vol = result.annualized_vol;

  // Allocation badge color based on weight
  const badgeColor =
    result.allocation_pct >= 0.3
      ? "text-emerald-400 border-emerald-400/30 bg-emerald-400/10"
      : result.allocation_pct >= 0.15
      ? "text-teal-400 border-teal-400/30 bg-teal-400/10"
      : "text-amber-400 border-amber-400/30 bg-amber-400/10";

  return (
    <div className="rounded-2xl border border-[#1e3330] bg-[#0f1a18] p-5 flex flex-col gap-4">
      {/* Header */}
      <div className="flex items-start justify-between">
        <div className="flex items-center gap-3">
          <span className="flex h-7 w-7 items-center justify-center rounded-full bg-white/5 text-xs font-mono text-gray-500 border border-white/10">
            #{rank}
          </span>
          <div>
            <span className="font-bold text-white text-lg font-mono tracking-tight">
              {result.ticker}
            </span>
            <div className="text-xs text-gray-500 mt-0.5">Long-term position</div>
          </div>
        </div>

        {/* Allocation badge */}
        <div className={cn("rounded-xl border px-3 py-2 text-center", badgeColor)}>
          <div className="text-2xl font-bold font-mono leading-none">{pct}%</div>
          <div className="text-xs font-medium mt-0.5">{usd}</div>
        </div>
      </div>

      {/* Composite score bar */}
      <ScoreBar label="Long-term Score" value={result.composite_score} />

      {/* Metrics grid */}
      <div className="grid grid-cols-2 gap-2">
        {/* 5Y CAGR */}
        <div className="rounded-lg bg-white/[0.03] border border-white/5 px-3 py-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wide">5Y CAGR</div>
          <div className={cn("font-mono font-semibold text-sm mt-0.5", metricColor(cagr, "high"))}>
            {cagr != null ? `${(cagr * 100).toFixed(1)}%` : "—"}
          </div>
        </div>

        {/* Sharpe */}
        <div className="rounded-lg bg-white/[0.03] border border-white/5 px-3 py-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wide">Sharpe</div>
          <div className={cn("font-mono font-semibold text-sm mt-0.5", metricColor(sharpe, "high"))}>
            {sharpe != null ? fmt(sharpe, 2) : "—"}
          </div>
        </div>

        {/* Max Drawdown */}
        <div className="rounded-lg bg-white/[0.03] border border-white/5 px-3 py-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wide">Max DD</div>
          <div className={cn("font-mono font-semibold text-sm mt-0.5", maxDd != null ? "text-rose-400" : "text-gray-500")}>
            {maxDd != null ? `${(maxDd * 100).toFixed(1)}%` : "—"}
          </div>
        </div>

        {/* Annualized Vol */}
        <div className="rounded-lg bg-white/[0.03] border border-white/5 px-3 py-2">
          <div className="text-[10px] text-gray-500 uppercase tracking-wide">Ann. Vol</div>
          <div className={cn("font-mono font-semibold text-sm mt-0.5 text-gray-300")}>
            {vol != null ? `${(vol * 100).toFixed(1)}%` : "—"}
          </div>
        </div>
      </div>
    </div>
  );
}
