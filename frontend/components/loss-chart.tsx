"use client";

import { useMemo, useEffect, useState } from "react";
import {
  LineChart,
  Line,
  XAxis,
  YAxis,
  CartesianGrid,
  Tooltip,
  Legend,
  ResponsiveContainer,
} from "recharts";

interface LossChartProps {
  logs: Record<string, unknown>[];
  height?: number;
}

export function LossChart({ logs, height = 300 }: LossChartProps) {
  // Avoid SSR rendering issues with recharts (needs window/document)
  const [mounted, setMounted] = useState(false);
  useEffect(() => {
    setMounted(true);
  }, []);

  const chartData = useMemo(() => {
    const map = new Map<number, { step: number; loss?: number; eval_loss?: number }>();

    for (const entry of logs) {
      const step = entry.step as number | undefined;
      if (step == null) continue;

      const existing = map.get(step);
      if (existing) {
        if (entry.loss != null) existing.loss = entry.loss as number;
        if (entry.eval_loss != null) existing.eval_loss = entry.eval_loss as number;
      } else {
        map.set(step, {
          step,
          loss: entry.loss as number | undefined,
          eval_loss: entry.eval_loss as number | undefined,
        });
      }
    }

    return Array.from(map.values()).sort((a, b) => a.step - b.step);
  }, [logs]);

  if (!mounted || chartData.length === 0) {
    return (
      <div className="flex items-center justify-center rounded-lg border text-sm text-muted-foreground" style={{ height }}>
        {chartData.length === 0
          ? "No training data yet. Loss curve will appear here."
          : "Loading chart..."}
      </div>
    );
  }

  const hasEval = chartData.some((d) => d.eval_loss != null);

  return (
    <div style={{ width: "100%", height }}>
      <ResponsiveContainer width="100%" height="100%">
        <LineChart data={chartData}>
          <CartesianGrid strokeDasharray="3 3" className="stroke-border" />
          <XAxis
            dataKey="step"
            tick={{ fontSize: 12 }}
            label={{ value: "Step", position: "insideBottom", offset: -5 }}
          />
          <YAxis
            tick={{ fontSize: 12 }}
            label={{
              value: "Loss",
              angle: -90,
              position: "insideLeft",
            }}
          />
          <Tooltip
            contentStyle={{
              borderRadius: "0.375rem",
              fontSize: "0.75rem",
            }}
          />
          <Legend />
          <Line
            type="monotone"
            dataKey="loss"
            stroke="hsl(221.2, 83.2%, 53.3%)"
            name="Train Loss"
            dot={false}
            strokeWidth={2}
            isAnimationActive={false}
          />
          {hasEval && (
            <Line
              type="monotone"
              dataKey="eval_loss"
              stroke="hsl(0, 84.2%, 60.2%)"
              name="Eval Loss"
              dot={false}
              strokeWidth={2}
              strokeDasharray="5 5"
              isAnimationActive={false}
            />
          )}
        </LineChart>
      </ResponsiveContainer>
    </div>
  );
}
