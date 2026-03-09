import { clsx, type ClassValue } from "clsx";
import { twMerge } from "tailwind-merge";

export function cn(...inputs: ClassValue[]) {
  return twMerge(clsx(inputs));
}

export function fmt(v: number | null | undefined, decimals = 2): string {
  if (v == null || isNaN(v)) return "—";
  return v.toFixed(decimals);
}

export function fmtPct(v: number | null | undefined): string {
  if (v == null || isNaN(v)) return "—";
  return `${(v * 100).toFixed(1)}%`;
}

export const RATING_CONFIG = {
  "Strong Buy": { color: "text-emerald-400", bg: "bg-emerald-400/15 border-emerald-400/30", dot: "bg-emerald-400" },
  Buy: { color: "text-teal-400", bg: "bg-teal-400/15 border-teal-400/30", dot: "bg-teal-400" },
  Hold: { color: "text-amber-400", bg: "bg-amber-400/15 border-amber-400/30", dot: "bg-amber-400" },
  Avoid: { color: "text-rose-400", bg: "bg-rose-400/15 border-rose-400/30", dot: "bg-rose-400" },
} as const;

export type Rating = keyof typeof RATING_CONFIG;

export function getRatingConfig(rating: string) {
  return RATING_CONFIG[rating as Rating] ?? RATING_CONFIG["Hold"];
}

export type StockResult = {
  ticker: string;
  final_buy_score: number;
  final_rating: string;
  buy_score: number;
  buy_rating: string;
  value_score: number;
  quality_score: number;
  risk_score: number;
  momentum_score: number;
  sentiment_score: number;
  llm_confidence: number;
  llm_reason: string;
  pe?: number | null;
  pb?: number | null;
  peg?: number | null;
  div_yield?: number | null;
  ev_ebitda?: number | null;
  roe?: number | null;
  rev_growth?: number | null;
  ret_3m?: number | null;
  rsi?: number | null;
  vol?: number | null;
  sharpe_proxy?: number | null;
};

export type SSEEvent =
  | { type: "progress"; stage: string; ticker?: string; progress: number; message: string }
  | { type: "result"; ticker: string; data: StockResult }
  | { type: "done"; progress: number; message: string }
  | { type: "error"; message: string };
