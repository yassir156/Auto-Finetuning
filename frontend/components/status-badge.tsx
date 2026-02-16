"use client";

import type { JobStatus, ProjectStatus } from "@/lib/types";
import { Badge } from "@/components/ui/badge";

const STATUS_VARIANT: Record<string, "default" | "secondary" | "destructive" | "outline" | "success"> = {
  draft: "outline",
  uploading: "secondary",
  chunking: "secondary",
  generating: "secondary",
  ready_to_train: "default",
  training: "default",
  evaluating: "default",
  completed: "success",
  success: "success",
  failed: "destructive",
  queued: "outline",
  running: "default",
  retrying: "secondary",
  cancelled: "secondary",
};

interface StatusBadgeProps {
  status: JobStatus | ProjectStatus | string;
}

export function StatusBadge({ status }: StatusBadgeProps) {
  return (
    <Badge variant={STATUS_VARIANT[status] ?? "outline"}>
      {status.replace(/_/g, " ")}
    </Badge>
  );
}
