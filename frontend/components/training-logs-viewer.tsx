"use client";

import { useEffect, useRef } from "react";

interface TrainingLogsViewerProps {
  logs: Record<string, unknown>[];
  maxHeight?: number;
}

export function TrainingLogsViewer({
  logs,
  maxHeight = 400,
}: TrainingLogsViewerProps) {
  const endRef = useRef<HTMLDivElement>(null);

  useEffect(() => {
    endRef.current?.scrollIntoView({ behavior: "smooth" });
  }, [logs.length]);

  return (
    <div
      className="overflow-auto rounded-lg border bg-gray-950 p-4 font-mono text-xs text-gray-300"
      style={{ maxHeight }}
    >
      {logs.length === 0 && (
        <p className="text-gray-500">Waiting for training logs...</p>
      )}
      {logs.map((entry, i) => {
        const step = entry.step ?? "";
        const totalSteps = entry.total_steps ?? "";
        const event = entry.event ?? "log";

        if (event === "complete") {
          return (
            <div key={i} className="text-green-400">
              [COMPLETE] Training finished. Best metric:{" "}
              {String(entry.best_metric ?? "N/A")}
            </div>
          );
        }

        if (event === "eval") {
          return (
            <div key={i} className="text-yellow-300">
              [EVAL] Step {String(step)}/{String(totalSteps)} — eval_loss:{" "}
              {formatNum(entry.eval_loss)}
              {entry.eval_runtime != null && (
                <> — runtime: {formatNum(entry.eval_runtime)}s</>
              )}
            </div>
          );
        }

        if (event === "checkpoint") {
          return (
            <div key={i} className="text-cyan-400">
              [CHECKPOINT] Step {String(step)} — Checkpoint saved
            </div>
          );
        }

        // Default: training log
        return (
          <div key={i}>
            <span className="text-gray-500">
              [{String(step)}/{String(totalSteps)}]
            </span>{" "}
            loss: {formatNum(entry.loss)}
            {entry.learning_rate != null && (
              <> lr: {Number(entry.learning_rate).toExponential(2)}</>
            )}
            {entry.epoch != null && <> epoch: {formatNum(entry.epoch)}</>}
          </div>
        );
      })}
      <div ref={endRef} />
    </div>
  );
}

function formatNum(val: unknown): string {
  if (val == null) return "—";
  const n = Number(val);
  if (isNaN(n)) return String(val);
  return n.toFixed(4);
}
