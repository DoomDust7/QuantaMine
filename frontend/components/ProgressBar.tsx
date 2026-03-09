"use client";

import { cn } from "@/lib/utils";

interface ProgressBarProps {
  progress: number;
  message: string;
  ticker?: string;
}

export function ProgressBar({ progress, message, ticker }: ProgressBarProps) {
  return (
    <div className="w-full space-y-2">
      <div className="flex items-center justify-between text-sm">
        <span className="text-gray-300">{message}</span>
        <span className="font-mono text-teal-400">{progress}%</span>
      </div>
      <div className="h-1.5 w-full rounded-full bg-white/5 overflow-hidden">
        <div
          className="h-full rounded-full bg-gradient-to-r from-teal-600 to-emerald-400 transition-all duration-500 ease-out"
          style={{ width: `${progress}%` }}
        />
      </div>
      {ticker && (
        <p className="text-xs text-gray-500">
          Processing: <span className="font-mono text-teal-500">{ticker}</span>
        </p>
      )}
    </div>
  );
}

export function SkeletonCard() {
  return (
    <div className="rounded-2xl border border-[#1e3330] bg-[#0f1a18] p-5 space-y-4">
      <div className="flex items-start justify-between">
        <div className="space-y-2">
          <div className="h-6 w-20 rounded-lg skeleton" />
          <div className="h-4 w-16 rounded-full skeleton" />
        </div>
        <div className="h-16 w-16 rounded-full skeleton" />
      </div>
      <div className="grid grid-cols-2 gap-x-4 gap-y-3">
        {Array.from({ length: 6 }).map((_, i) => (
          <div key={i} className="space-y-1.5">
            <div className="flex justify-between">
              <div className="h-3 w-16 rounded skeleton" />
              <div className="h-3 w-6 rounded skeleton" />
            </div>
            <div className="h-1.5 w-full rounded-full skeleton" />
          </div>
        ))}
      </div>
    </div>
  );
}
