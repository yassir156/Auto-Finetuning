/**
 * FineTuneFlow — TypeScript types matching the backend Pydantic schemas.
 */

// ── Enums ───────────────────────────────────
export type TaskType =
  | "instruction_tuning"
  | "qa_grounded"
  | "summarization"
  | "report_generation"
  | "information_extraction"
  | "classification"
  | "chat_dialogue_sft";
export type ProjectStatus =
  | "draft"
  | "uploading"
  | "chunking"
  | "generating"
  | "ready_to_train"
  | "training"
  | "evaluating"
  | "completed"
  | "failed";
export type FileKind = "raw_doc" | "dataset_upload" | "dataset_generated" | "artifact" | "export" | "log";
export type FileStatus = "uploading" | "ready" | "processing" | "failed";
export type JobType =
  | "chunking"
  | "dataset_preview"
  | "dataset_generate"
  | "train"
  | "eval"
  | "export";
export type JobStatus =
  | "queued"
  | "running"
  | "success"
  | "failed"
  | "retrying"
  | "cancelled";
export type FinetuneMethod = "lora" | "qlora" | "dora" | "ia3" | "prefix" | "full";
export type DatasetSplit = "preview" | "train" | "eval";

// ── Error ───────────────────────────────────
export interface ApiError {
  detail: string;
  error_code: string;
  context?: Record<string, unknown>;
}

// ── Project ─────────────────────────────────
export interface ProjectConfig {
  num_examples_target: number;
  max_seq_length: number;
  train_eval_split: number;
}

export interface ProjectStats {
  file_count: number;
  chunk_count: number;
  example_count: number;
  last_job_status: string | null;
}

export interface Project {
  id: string;
  name: string;
  description: string | null;
  task_type: TaskType;
  base_model_id: string | null;
  model_info: Record<string, unknown> | null;
  status: ProjectStatus;
  config: ProjectConfig | null;
  created_at: string;
  updated_at: string;
  stats?: ProjectStats;
}

export interface ProjectListResponse {
  items: Project[];
  total: number;
  limit: number;
  offset: number;
}

// ── Model Resolution ────────────────────────
export interface ModelResolveResponse {
  model_id: string;
  model_type: string | null;
  num_parameters: number | null;
  estimated_vram_fp16_gb: number | null;
  estimated_vram_4bit_gb: number | null;
  architecture: string | null;
  vocab_size: number | null;
  max_position_embeddings: number | null;
  license: string | null;
  valid: boolean;
  warnings: string[];
}

// ── File ────────────────────────────────────
export interface FileInfo {
  id: string;
  filename: string;
  kind: FileKind;
  status: FileStatus;
  mime_type: string | null;
  size_bytes: number;
  created_at: string;
}

// ── Chunk ───────────────────────────────────
export interface Chunk {
  id: string;
  source_file_id: string;
  chunk_index: number;
  content: string;
  token_count: number | null;
  metadata: Record<string, unknown> | null;
}

// ── Dataset ─────────────────────────────────
export interface DatasetExample {
  id: string;
  split: DatasetSplit;
  data: Record<string, string>;
  is_valid: boolean;
  token_count: number | null;
  created_at: string;
}

export interface DatasetStats {
  total: number;
  valid: number;
  invalid: number;
  train: number;
  eval: number;
  avg_token_count: number | null;
}

// ── Hardware ────────────────────────────────
export interface HardwareCheck {
  has_nvidia_smi: boolean;
  gpu_name: string | null;
  gpu_count: number;
  vram_total_gb: number | null;
  vram_free_gb: number | null;
  driver_version: string | null;
  cuda_runtime: string | null;
  torch_version: string | null;
  torch_cuda: string | null;
  cuda_available: boolean;
  bnb_available: boolean;
  recommended_method: FinetuneMethod | null;
  recommendation_reason: string | null;
  notes: string[];
  warnings: string[];
}

// ── Training ────────────────────────────────
export interface TrainHyperparams {
  num_epochs: number;
  learning_rate: number;
  per_device_batch_size: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
  warmup_ratio: number;
  weight_decay: number;
}

export interface TrainStatus {
  run_id: string | null;
  job_id: string | null;
  status: JobStatus;
  method: FinetuneMethod | null;
  progress_pct: number;
  current_metrics: Record<string, unknown> | null;
  started_at: string | null;
  elapsed_seconds: number | null;
}

// ── Job ─────────────────────────────────────
export interface Job {
  id: string;
  type: JobType;
  status: JobStatus;
  progress_pct: number;
  result_summary: Record<string, unknown> | null;
  error_message: string | null;
  error_code: string | null;
  created_at: string;
  started_at: string | null;
  finished_at: string | null;
}

// ── Export ───────────────────────────────────
export interface ExportFileInfo {
  filename: string;
  size_bytes: number;
}

// ── JobEnqueued (generic async response) ────
export interface JobEnqueuedResponse {
  job_id: string;
  status: string;
  message: string;
}

export interface TrainEnqueuedResponse extends JobEnqueuedResponse {
  run_id: string;
}

// \u2500\u2500 Task Type Config (from /task-types endpoint) \u2500\u2500
export interface TaskTypeConfig {
  key: TaskType;
  label: string;
  description: string;
  required_fields: string[];
  optional_fields: string[];
  sample_example: Record<string, unknown>;
  display_columns: { key: string; label: string; maxWidth?: number }[];
}

// \u2500\u2500 Dataset Upload Response \u2500\u2500
export interface DatasetUploadResponse {
  total_lines: number;
  valid: number;
  invalid: number;
  train: number;
  eval: number;
  errors: { line: number; error: string }[];
}
// ── Playground / Inference ──
export interface PlaygroundStatusResponse {
  available: boolean;
  base_model_id: string | null;
  adapter_dir: string | null;
  run_id: string | null;
  method: string | null;
  metrics: Record<string, unknown> | null;
  message: string;
}

export interface PlaygroundGenerateRequest {
  prompt: string;
  max_new_tokens?: number;
  temperature?: number;
  top_p?: number;
  top_k?: number;
  repetition_penalty?: number;
  do_sample?: boolean;
}

export interface PlaygroundGenerateResponse {
  generated_text: string;
  prompt: string;
  num_tokens_prompt: number;
  num_tokens_generated: number;
}
