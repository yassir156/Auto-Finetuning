/**
 * FineTuneFlow — API Client.
 *
 * Thin fetch wrapper around backend REST endpoints.
 * All methods return typed data or throw an ApiError.
 */

import type {
  ApiError,
  Project,
  ProjectListResponse,
  ModelResolveResponse,
  FileInfo,
  Chunk,
  DatasetExample,
  DatasetStats,
  HardwareCheck,
  TrainStatus,
  Job,
  ExportFileInfo,
  JobEnqueuedResponse,
  TrainEnqueuedResponse,
  TaskTypeConfig,
  DatasetUploadResponse,
  PlaygroundStatusResponse,
  PlaygroundGenerateRequest,
  PlaygroundGenerateResponse,
} from "./types";

// Client-side: use Next.js rewrite proxy (/api/*) to avoid CORS
// Server-side: call backend container directly
const BASE_URL =
  typeof window === "undefined"
    ? (process.env.INTERNAL_API_URL || "http://backend:8000")
    : "/api";

// ── Helpers ──────────────────────────────────

async function request<T>(
  path: string,
  init?: RequestInit
): Promise<T> {
  const url = `${BASE_URL}${path}`;
  const res = await fetch(url, {
    headers: { "Content-Type": "application/json", ...init?.headers },
    ...init,
  });

  if (!res.ok) {
    const body: ApiError = await res.json().catch(() => ({
      detail: res.statusText,
      error_code: "UNKNOWN",
    }));
    throw body;
  }

  if (res.status === 204) return undefined as T;
  return res.json();
}

// ── Projects ─────────────────────────────────

export async function listProjects(
  params?: { status?: string; limit?: number; offset?: number }
): Promise<ProjectListResponse> {
  const q = new URLSearchParams();
  if (params?.status) q.set("status", params.status);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return request(`/projects${qs ? `?${qs}` : ""}`);
}

export async function getProject(id: string): Promise<Project> {
  return request(`/projects/${id}`);
}

export async function createProject(data: {
  name: string;
  description?: string;
  task_type: string;
  config?: Record<string, unknown>;
}): Promise<Project> {
  return request("/projects", {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function updateProject(
  id: string,
  data: Record<string, unknown>
): Promise<Project> {
  return request(`/projects/${id}`, {
    method: "PATCH",
    body: JSON.stringify(data),
  });
}

export async function deleteProject(id: string): Promise<void> {
  return request(`/projects/${id}`, { method: "DELETE" });
}

// ── Model Resolution ─────────────────────────

export async function resolveModel(
  projectId: string,
  modelId: string
): Promise<ModelResolveResponse> {
  return request(`/projects/${projectId}/model/resolve`, {
    method: "POST",
    body: JSON.stringify({ model_id: modelId }),
  });
}

// ── Files ────────────────────────────────────

export async function uploadFiles(
  projectId: string,
  files: File[],
  kind: string
): Promise<{ files: FileInfo[] }> {
  const form = new FormData();
  form.set("kind", kind);
  files.forEach((f) => form.append("files", f));

  const res = await fetch(`${BASE_URL}/projects/${projectId}/files/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const body: ApiError = await res.json().catch(() => ({
      detail: res.statusText,
      error_code: "UNKNOWN",
    }));
    throw body;
  }
  return res.json();
}

export async function listFiles(
  projectId: string,
  kind?: string
): Promise<{ files: FileInfo[] }> {
  const q = kind ? `?kind=${kind}` : "";
  return request(`/projects/${projectId}/files${q}`);
}

export async function deleteFile(
  projectId: string,
  fileId: string
): Promise<void> {
  return request(`/projects/${projectId}/files/${fileId}`, {
    method: "DELETE",
  });
}

// ── Chunks ───────────────────────────────────

export async function generateChunks(
  projectId: string,
  opts?: { chunk_size_tokens?: number; chunk_overlap_tokens?: number }
): Promise<JobEnqueuedResponse> {
  return request(`/projects/${projectId}/chunks/generate`, {
    method: "POST",
    body: JSON.stringify(opts ?? {}),
  });
}

export async function listChunks(
  projectId: string,
  params?: { source_file_id?: string; limit?: number; offset?: number }
): Promise<{ items: Chunk[]; total: number }> {
  const q = new URLSearchParams();
  if (params?.source_file_id) q.set("source_file_id", params.source_file_id);
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return request(`/projects/${projectId}/chunks${qs ? `?${qs}` : ""}`);
}

// ── Dataset ──────────────────────────────────

export async function datasetPreview(
  projectId: string,
  opts?: { num_examples?: number; ollama_model?: string }
): Promise<JobEnqueuedResponse> {
  return request(`/projects/${projectId}/dataset/preview`, {
    method: "POST",
    body: JSON.stringify(opts ?? {}),
  });
}

export async function datasetGenerate(
  projectId: string,
  opts?: {
    num_examples_target?: number;
    examples_per_chunk?: number;
    ollama_model?: string;
  }
): Promise<JobEnqueuedResponse> {
  return request(`/projects/${projectId}/dataset/generate`, {
    method: "POST",
    body: JSON.stringify(opts ?? {}),
  });
}

export async function listDatasetExamples(
  projectId: string,
  params?: {
    split?: string;
    is_valid?: boolean;
    limit?: number;
    offset?: number;
  }
): Promise<{
  items: DatasetExample[];
  total: number;
  stats: DatasetStats;
}> {
  const q = new URLSearchParams();
  if (params?.split) q.set("split", params.split);
  if (params?.is_valid !== undefined)
    q.set("is_valid", String(params.is_valid));
  if (params?.limit) q.set("limit", String(params.limit));
  if (params?.offset) q.set("offset", String(params.offset));
  const qs = q.toString();
  return request(
    `/projects/${projectId}/dataset/examples${qs ? `?${qs}` : ""}`
  );
}

export async function getDatasetStats(
  projectId: string
): Promise<DatasetStats> {
  return request(`/projects/${projectId}/dataset/stats`);
}

export async function deleteDatasetExample(
  projectId: string,
  exampleId: string
): Promise<void> {
  return request(`/projects/${projectId}/dataset/examples/${exampleId}`, {
    method: "DELETE",
  });
}

// ── Hardware ─────────────────────────────────

export async function checkHardware(): Promise<HardwareCheck> {
  return request("/hardware/check");
}

// ── Training ─────────────────────────────────

export async function startTraining(
  projectId: string,
  data: { method: string; hyperparams?: Record<string, unknown> }
): Promise<TrainEnqueuedResponse> {
  return request(`/projects/${projectId}/train/start`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}

export async function getTrainStatus(
  projectId: string
): Promise<TrainStatus> {
  return request(`/projects/${projectId}/train/status`);
}

export async function cancelTraining(
  projectId: string
): Promise<{ status: string; message: string }> {
  return request(`/projects/${projectId}/train/cancel`, { method: "POST" });
}

/**
 * Subscribe to training SSE logs.
 * Returns a function to close the stream.
 */
export function streamTrainLogs(
  projectId: string,
  onEvent: (event: string, data: Record<string, unknown>) => void,
  onError?: (err: Event) => void
): () => void {
  const source = new EventSource(
    `${BASE_URL}/projects/${projectId}/train/logs/stream`
  );

  const handler = (evt: MessageEvent) => {
    try {
      onEvent(evt.type, JSON.parse(evt.data));
    } catch {
      /* ignore parse errors */
    }
  };

  for (const type of ["log", "progress", "eval", "checkpoint", "complete", "error"]) {
    source.addEventListener(type, handler);
  }

  source.onerror = (err) => {
    source.close();
    onError?.(err);
  };

  return () => source.close();
}

// ── Export ────────────────────────────────────

export async function triggerExport(
  projectId: string
): Promise<JobEnqueuedResponse> {
  return request(`/projects/${projectId}/export`, { method: "POST" });
}

export async function listExportFiles(
  projectId: string
): Promise<{ files: ExportFileInfo[] }> {
  return request(`/projects/${projectId}/export/files`);
}

export function getExportDownloadUrl(
  projectId: string,
  filename?: string
): string {
  if (filename) {
    return `${BASE_URL}/projects/${projectId}/export/download/${filename}`;
  }
  return `${BASE_URL}/projects/${projectId}/export/download`;
}

// ── Jobs ─────────────────────────────────────

export async function listJobs(
  projectId: string,
  params?: { type?: string; status?: string }
): Promise<{ jobs: Job[] }> {
  const q = new URLSearchParams();
  if (params?.type) q.set("type", params.type);
  if (params?.status) q.set("status", params.status);
  const qs = q.toString();
  return request(`/projects/${projectId}/jobs${qs ? `?${qs}` : ""}`);
}

export async function getJob(jobId: string): Promise<Job> {
  return request(`/jobs/${jobId}`);
}

export async function cancelJob(
  jobId: string
): Promise<{ status: string }> {
  return request(`/jobs/${jobId}/cancel`, { method: "POST" });
}

// ── Task Types ──────────────────────────────

export async function getTaskTypes(): Promise<{ task_types: TaskTypeConfig[] }> {
  return request("/task-types");
}

// ── Dataset Upload (JSONL) ───────────────

export async function uploadDatasetJsonl(
  projectId: string,
  file: File
): Promise<DatasetUploadResponse> {
  const form = new FormData();
  form.append("file", file);

  const res = await fetch(`${BASE_URL}/projects/${projectId}/dataset/upload`, {
    method: "POST",
    body: form,
  });

  if (!res.ok) {
    const body: ApiError = await res.json().catch(() => ({
      detail: res.statusText,
      error_code: "UNKNOWN",
    }));
    throw body;
  }
  return res.json();
}

// ── Playground / Inference ─────────────────────────

export async function getPlaygroundStatus(
  projectId: string
): Promise<PlaygroundStatusResponse> {
  return request(`/projects/${projectId}/playground/status`);
}

export async function playgroundGenerate(
  projectId: string,
  data: PlaygroundGenerateRequest
): Promise<PlaygroundGenerateResponse> {
  return request(`/projects/${projectId}/playground/generate`, {
    method: "POST",
    body: JSON.stringify(data),
  });
}
