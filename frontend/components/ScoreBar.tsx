"use client";

import { cn } from "@/lib/utils";

interface ScoreBarProps {
  label: string;
  value: number | null | undefined;
  className?: string;
}

function getBarColor(v: number): string {
  if (v >= 0.75) return "bg-emerald-400";
  if (v >= 0.55) return "bg-teal-400";
  if (v >= 0.4) return "bg-amber-400";
  return "bg-rose-400";
}

export function ScoreBar({ label, value, className }: ScoreBarProps) {
  const pct = value != null && !isNaN(value) ? Math.round(value * 100) : null;

  return (
    <div className={cn("space-y-1", className)}>
      <div className="flex items-center justify-between text-xs">
        <span className="text-gray-400">{label}</span>
        <span className={cn("font-mono font-medium", pct == null ? "text-gray-500" : pct >= 65 ? "text-emerald-400" : pct >= 45 ? "text-amber-400" : "text-rose-400")}>
          {pct != null ? `${pct}` : "—"}
        </span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
        {pct != null && (
          <div
            className={cn("h-full rounded-full transition-all duration-700 ease-out", getBarColor(value!))}
            style={{ width: `${pct}%` }}
          />
        )}
      </div>
    </div>
  );
}
