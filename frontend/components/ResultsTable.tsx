"use client";

import { Download, ArrowUpDown } from "lucide-react";
import { cn, fmt, fmtPct, getRatingConfig, type StockResult } from "@/lib/utils";
import { useState } from "react";

interface ResultsTableProps {
  results: StockResult[];
}

export function ResultsTable({ results }: ResultsTableProps) {
  const [sortKey, setSortKey] = useState<keyof StockResult>("final_buy_score");
  const [sortDesc, setSortDesc] = useState(true);

  const sorted = [...results].sort((a, b) => {
    const av = a[sortKey] as number ?? 0;
    const bv = b[sortKey] as number ?? 0;
    return sortDesc ? bv - av : av - bv;
  });

  const toggle = (key: keyof StockResult) => {
    if (sortKey === key) setSortDesc((d) => !d);
    else { setSortKey(key); setSortDesc(true); }
  };

  const downloadCSV = () => {
    const cols: (keyof StockResult)[] = [
      "ticker", "final_rating", "final_buy_score",
      "value_score", "quality_score", "risk_score", "momentum_score",
      "sentiment_score", "llm_confidence", "pe", "pb", "roe", "rev_growth", "ret_3m",
    ];
    const header = cols.join(",");
    const rows = results.map((r) =>
      cols.map((c) => {
        const v = r[c];
        return typeof v === "string" ? `"${v}"` : (v ?? "");
      }).join(",")
    );
    const csv = [header, ...rows].join("\n");
    const blob = new Blob([csv], { type: "text/csv" });
    const url = URL.createObjectURL(blob);
    const a = document.createElement("a");
    a.href = url;
    a.download = "quantamine_analysis.csv";
    a.click();
    URL.revokeObjectURL(url);
  };

  const Th = ({ label, col }: { label: string; col: keyof StockResult }) => (
    <th
      className="px-3 py-2.5 text-left text-xs font-medium text-gray-500 cursor-pointer hover:text-gray-300 select-none whitespace-nowrap"
      onClick={() => toggle(col)}
    >
      <span className="flex items-center gap-1">
        {label}
        <ArrowUpDown size={10} className={cn(sortKey === col ? "text-teal-400" : "opacity-30")} />
      </span>
    </th>
  );

  return (
    <div className="space-y-3">
      <div className="flex items-center justify-between">
        <h2 className="text-sm font-medium text-gray-400">Full Results</h2>
        <button
          onClick={downloadCSV}
          className="flex items-center gap-1.5 rounded-lg border border-[#1e3330] bg-[#0f1a18] px-3 py-1.5 text-xs text-gray-400 hover:text-teal-400 hover:border-teal-800 transition-colors"
        >
          <Download size={12} />
          Export CSV
        </button>
      </div>

      <div className="rounded-xl border border-[#1e3330] overflow-hidden">
        <div className="overflow-x-auto">
          <table className="w-full text-sm">
            <thead className="bg-[#0f1a18] border-b border-[#1e3330]">
              <tr>
                <Th label="Ticker" col="ticker" />
                <Th label="Rating" col="final_rating" />
                <Th label="Score" col="final_buy_score" />
                <Th label="Value" col="value_score" />
                <Th label="Quality" col="quality_score" />
                <Th label="Risk" col="risk_score" />
                <Th label="Momentum" col="momentum_score" />
                <Th label="Sentiment" col="sentiment_score" />
                <Th label="AI Conf." col="llm_confidence" />
                <Th label="P/E" col="pe" />
                <Th label="ROE" col="roe" />
                <Th label="3M Ret." col="ret_3m" />
              </tr>
            </thead>
            <tbody>
              {sorted.map((r, i) => {
                const cfg = getRatingConfig(r.final_rating);
                return (
                  <tr
                    key={r.ticker}
                    className={cn(
                      "border-b border-[#1e3330]/50 transition-colors hover:bg-white/[0.02]",
                      i % 2 === 0 ? "bg-[#0a0f0e]" : "bg-[#0d1614]"
                    )}
                  >
                    <td className="px-3 py-2.5 font-mono font-bold text-white">{r.ticker}</td>
                    <td className="px-3 py-2.5">
                      <span className={cn("rounded-full border px-2 py-0.5 text-xs font-medium", cfg.bg, cfg.color)}>
                        {r.final_rating}
                      </span>
                    </td>
                    <td className="px-3 py-2.5 font-mono font-bold text-teal-400">
                      {Math.round((r.final_buy_score ?? 0) * 100)}
                    </td>
                    {(["value_score", "quality_score", "risk_score", "momentum_score", "sentiment_score", "llm_confidence"] as const).map((k) => (
                      <td key={k} className="px-3 py-2.5 font-mono text-xs text-gray-300">
                        {r[k] != null ? Math.round((r[k] as number) * 100) : "—"}
                      </td>
                    ))}
                    <td className="px-3 py-2.5 font-mono text-xs text-gray-300">{fmt(r.pe)}</td>
                    <td className="px-3 py-2.5 font-mono text-xs text-gray-300">{fmtPct(r.roe)}</td>
                    <td className={cn("px-3 py-2.5 font-mono text-xs", (r.ret_3m ?? 0) >= 0 ? "text-emerald-400" : "text-rose-400")}>
                      {fmtPct(r.ret_3m)}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        </div>
      </div>
    </div>
  );
}
