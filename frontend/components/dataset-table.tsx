"use client";

import type { DatasetExample, DatasetStats, TaskTypeConfig } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

interface DatasetTableProps {
  examples: DatasetExample[];
  stats?: DatasetStats | null;
  taskTypeConfig?: TaskTypeConfig | null;
}

/** Render a cell value — handles strings, objects, arrays. */
function CellValue({ value }: { value: unknown }) {
  if (value === null || value === undefined) return <span className="text-muted-foreground">—</span>;
  if (typeof value === "string") return <>{value}</>;
  if (Array.isArray(value)) {
    // Messages array (chat)
    if (value.length > 0 && typeof value[0] === "object" && "role" in value[0]) {
      return (
        <div className="space-y-1">
          {value.map((msg, i) => (
            <div key={i} className="text-xs">
              <span className="font-medium capitalize">{msg.role}:</span>{" "}
              <span>{msg.content}</span>
            </div>
          ))}
        </div>
      );
    }
    return <>{value.join(", ")}</>;
  }
  if (typeof value === "object") {
    return (
      <pre className="whitespace-pre-wrap text-xs">
        {JSON.stringify(value, null, 1)}
      </pre>
    );
  }
  return <>{String(value)}</>;
}

export function DatasetTable({ examples, stats, taskTypeConfig }: DatasetTableProps) {
  // Determine columns from task config or fallback to instruction/output
  const columns = taskTypeConfig?.display_columns ?? [
    { key: "instruction", label: "Instruction", maxWidth: 300 },
    { key: "output", label: "Output", maxWidth: 300 },
  ];

  return (
    <div className="space-y-4">
      {stats && (
        <div className="flex flex-wrap gap-3 text-sm">
          <Badge variant="secondary">Total: {stats.total}</Badge>
          <Badge variant="success">Valid: {stats.valid}</Badge>
          {stats.invalid > 0 && (
            <Badge variant="destructive">Invalid: {stats.invalid}</Badge>
          )}
          <Badge variant="outline">Train: {stats.train}</Badge>
          <Badge variant="outline">Eval: {stats.eval}</Badge>
          {stats.avg_token_count != null && (
            <Badge variant="outline">
              Avg tokens: {Math.round(stats.avg_token_count)}
            </Badge>
          )}
        </div>
      )}

      <div className="overflow-auto rounded-lg border">
        <table className="w-full text-sm">
          <thead className="bg-muted/50">
            <tr>
              <th className="px-4 py-2 text-left font-medium">Split</th>
              {columns.map((col) => (
                <th key={col.key} className="px-4 py-2 text-left font-medium">
                  {col.label}
                </th>
              ))}
              <th className="px-4 py-2 text-left font-medium">Tokens</th>
              <th className="px-4 py-2 text-left font-medium">Valid</th>
            </tr>
          </thead>
          <tbody>
            {examples.map((ex) => (
              <tr key={ex.id} className="border-t hover:bg-muted/30">
                <td className="px-4 py-2 whitespace-nowrap">
                  <Badge
                    variant={
                      ex.split === "train"
                        ? "default"
                        : ex.split === "eval"
                        ? "secondary"
                        : "outline"
                    }
                  >
                    {ex.split}
                  </Badge>
                </td>
                {columns.map((col) => (
                  <td
                    key={col.key}
                    className="px-4 py-2"
                    style={{ maxWidth: col.maxWidth ?? 300, overflow: "hidden", textOverflow: "ellipsis" }}
                  >
                    <div className="max-h-24 overflow-hidden">
                      <CellValue value={ex.data?.[col.key]} />
                    </div>
                  </td>
                ))}
                <td className="px-4 py-2 tabular-nums">
                  {ex.token_count ?? "—"}
                </td>
                <td className="px-4 py-2">
                  {ex.is_valid ? (
                    <span className="text-green-600">✓</span>
                  ) : (
                    <span className="text-red-500">✗</span>
                  )}
                </td>
              </tr>
            ))}
            {examples.length === 0 && (
              <tr>
                <td
                  colSpan={columns.length + 3}
                  className="px-4 py-8 text-center text-muted-foreground"
                >
                  No examples yet. Generate a preview or full dataset.
                </td>
              </tr>
            )}
          </tbody>
        </table>
      </div>
    </div>
  );
}
