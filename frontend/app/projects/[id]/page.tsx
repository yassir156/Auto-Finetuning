"use client";

import { useCallback, useEffect, useRef, useState } from "react";
import { useParams, useRouter } from "next/navigation";
import {
  getProject,
  updateProject,
  resolveModel,
  uploadFiles,
  listFiles,
  generateChunks,
  datasetPreview,
  datasetGenerate,
  listDatasetExamples,
  getDatasetStats,
  checkHardware,
  startTraining,
  getTrainStatus,
  cancelTraining,
  streamTrainLogs,
  triggerExport,
  listExportFiles,
  getExportDownloadUrl,
  getJob,
  getTaskTypes,
  uploadDatasetJsonl,
} from "@/lib/api";
import type {
  Project,
  FileInfo,
  DatasetExample,
  DatasetStats,
  HardwareCheck,
  TrainStatus,
  ExportFileInfo,
  ModelResolveResponse,
  TaskTypeConfig,
  DatasetUploadResponse,
  FinetuneMethod,
} from "@/lib/types";
import { WIZARD_STEPS, WizardStep } from "@/lib/store";
import { Button } from "@/components/ui/button";
import { Input } from "@/components/ui/input";
import { Card, CardContent, CardHeader, CardTitle, CardDescription } from "@/components/ui/card";
import { Badge } from "@/components/ui/badge";
import { Progress } from "@/components/ui/progress";
import { Select } from "@/components/ui/select";
import { WizardSteps } from "@/components/wizard-steps";
import { FileUploadDropzone } from "@/components/file-upload-dropzone";
import { DatasetTable } from "@/components/dataset-table";
import { LossChart } from "@/components/loss-chart";
import { TrainingLogsViewer } from "@/components/training-logs-viewer";
import { StatusBadge } from "@/components/status-badge";

export default function ProjectPage() {
  const params = useParams();
  const router = useRouter();
  const projectId = params.id as string;

  // State
  const [project, setProject] = useState<Project | null>(null);
  const [currentStep, setCurrentStep] = useState<WizardStep>("model");
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  // Model
  const [modelId, setModelId] = useState("");
  const [modelInfo, setModelInfo] = useState<ModelResolveResponse | null>(null);
  const [resolving, setResolving] = useState(false);

  // Files
  const [files, setFiles] = useState<FileInfo[]>([]);
  const [uploading, setUploading] = useState(false);

  // Dataset
  const [examples, setExamples] = useState<DatasetExample[]>([]);
  const [datasetStats, setDatasetStats] = useState<DatasetStats | null>(null);
  const [generating, setGenerating] = useState(false);

  // Hardware
  const [hardware, setHardware] = useState<HardwareCheck | null>(null);
  const [checkingHw, setCheckingHw] = useState(false);

  // Training
  const [method, setMethod] = useState<FinetuneMethod>("qlora");
  const [trainStatus, setTrainStatus] = useState<TrainStatus | null>(null);
  const [trainLogs, setTrainLogs] = useState<Record<string, unknown>[]>([]);
  const [isTraining, setIsTraining] = useState(false);

  // Export
  const [exportFiles, setExportFiles] = useState<ExportFileInfo[]>([]);
  const [exporting, setExporting] = useState(false);

  // Task types config
  const [taskTypes, setTaskTypes] = useState<TaskTypeConfig[]>([]);
  // Data source choice: "upload_jsonl" = user has dataset, "generate" = generate from PDFs
  const [dataSource, setDataSource] = useState<"upload_jsonl" | "generate" | null>(null);
  const [uploadResult, setUploadResult] = useState<DatasetUploadResponse | null>(null);

  // SSE cleanup ref
  const sseCloseRef = useRef<(() => void) | null>(null);
  // Polling interval refs
  const pollRef = useRef<ReturnType<typeof setInterval> | null>(null);
  // Keep a ref to trainLogs to avoid stale closures in callbacks/intervals
  const trainLogsRef = useRef<Record<string, unknown>[]>([]);

  // Sync trainLogs state → ref
  useEffect(() => {
    trainLogsRef.current = trainLogs;
  }, [trainLogs]);

  // Cleanup SSE and polls on unmount
  useEffect(() => {
    return () => {
      sseCloseRef.current?.();
      if (pollRef.current) clearInterval(pollRef.current);
    };
  }, []);

  // ── Load project ──────────────────────────
  const fetchProject = useCallback(async () => {
    try {
      setLoading(true);
      const [p, ttRes] = await Promise.all([
        getProject(projectId),
        getTaskTypes(),
      ]);
      setProject(p);
      setTaskTypes(ttRes.task_types);
      if (p.base_model_id) setModelId(p.base_model_id);

      // Determine initial step based on project state
      if (p.status === "completed") {
        setCurrentStep("export");
      } else if (p.status === "training" || p.status === "evaluating") {
        setCurrentStep("train");
      } else if (p.status === "ready_to_train") {
        setCurrentStep("hardware");
      } else if (p.base_model_id) {
        setCurrentStep("data");
      }
    } catch {
      setError("Project not found");
    } finally {
      setLoading(false);
    }
  }, [projectId]);

  useEffect(() => {
    fetchProject();
  }, [fetchProject]);

  // ── Step: Model ───────────────────────────
  const handleResolveModel = async () => {
    if (!modelId.trim()) return;
    try {
      setResolving(true);
      setError(null);
      const info = await resolveModel(projectId, modelId.trim());
      setModelInfo(info);
      if (info.valid) {
        await updateProject(projectId, {
          base_model_id: modelId.trim(),
          model_info: info as unknown as Record<string, unknown>,
        });
        await fetchProject();
      }
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Model resolution failed");
    } finally {
      setResolving(false);
    }
  };

  // ── Step: Data (Files) ────────────────────
  const handleUploadFiles = async (newFiles: File[]) => {
    try {
      setUploading(true);
      setError(null);
      await uploadFiles(projectId, newFiles, "raw_doc");
      const res = await listFiles(projectId);
      setFiles(res.files);
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Upload failed");
    } finally {
      setUploading(false);
    }
  };

  const handleChunk = async () => {
    try {
      setGenerating(true);
      setError(null);
      const { job_id } = await generateChunks(projectId);
      // Poll job until done
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const job = await getJob(job_id);
          if (job.status === "success") {
            if (pollRef.current) clearInterval(pollRef.current);
            setGenerating(false);
          } else if (job.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setGenerating(false);
            setError(job.error_message ?? "Chunking failed");
          }
        } catch { /* retry at next interval */ }
      }, 2000);
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Chunking failed");
      setGenerating(false);
    }
  };

  // ── Step: Preview ─────────────────────────
  const handlePreview = async () => {
    try {
      setGenerating(true);
      setError(null);
      const { job_id } = await datasetPreview(projectId);
      // Poll job until done
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const job = await getJob(job_id);
          if (job.status === "success") {
            if (pollRef.current) clearInterval(pollRef.current);
            const res = await listDatasetExamples(projectId, { split: "preview", limit: 20 });
            setExamples(res.items);
            setGenerating(false);
          } else if (job.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setError(job.error_message ?? "Preview generation failed");
            setGenerating(false);
          }
        } catch { /* retry at next interval */ }
      }, 2000);
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Preview failed");
      setGenerating(false);
    }
  };

  // ── Step: Review (full generation) ────────
  const handleGenerate = async () => {
    try {
      setGenerating(true);
      setError(null);
      const { job_id } = await datasetGenerate(projectId);
      // Poll job until done
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const job = await getJob(job_id);
          if (job.status === "success") {
            if (pollRef.current) clearInterval(pollRef.current);
            await refreshDataset();
            setGenerating(false);
          } else if (job.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setError(job.error_message ?? "Dataset generation failed");
            setGenerating(false);
          }
        } catch { /* retry at next interval */ }
      }, 3000);
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Generation failed");
      setGenerating(false);
    }
  };

  const refreshDataset = async () => {
    try {
      const [exRes, statsRes] = await Promise.all([
        listDatasetExamples(projectId, { limit: 50 }),
        getDatasetStats(projectId),
      ]);
      setExamples(exRes.items);
      setDatasetStats(statsRes);
    } catch { /* ignore */ }
  };

  // ── Step: Hardware ────────────────────────
  const handleCheckHardware = async () => {
    try {
      setCheckingHw(true);
      setError(null);
      const hw = await checkHardware();
      setHardware(hw);
      if (hw.recommended_method) {
        setMethod(hw.recommended_method as FinetuneMethod);
      }
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Hardware check failed");
    } finally {
      setCheckingHw(false);
    }
  };

  // ── Step: Train ───────────────────────────

  // Polling fallback when SSE is unavailable (e.g. HTTP/2 proxies like Codespaces)
  const startTrainPolling = useCallback(() => {
    if (pollRef.current) clearInterval(pollRef.current);
    pollRef.current = setInterval(async () => {
      try {
        const status = await getTrainStatus(projectId);
        setTrainStatus(status);

        // Rebuild loss curve from stored metrics
        if (status.current_metrics) {
          const rebuilt: Record<string, unknown>[] = [];
          const trainCurve = status.current_metrics.train_loss_curve as
            | { step: number; loss: number }[]
            | undefined;
          const evalCurve = status.current_metrics.eval_loss_curve as
            | { step: number; eval_loss: number }[]
            | undefined;
          if (trainCurve) {
            for (const pt of trainCurve) {
              rebuilt.push({ event: "log", step: pt.step, loss: pt.loss });
            }
          }
          if (evalCurve) {
            for (const pt of evalCurve) {
              rebuilt.push({ event: "eval", step: pt.step, eval_loss: pt.eval_loss });
            }
          }
          if (rebuilt.length > 0) {
            setTrainLogs(rebuilt);
          }
        }

        if (status.status === "success" || status.status === "failed") {
          if (pollRef.current) clearInterval(pollRef.current);
          setIsTraining(false);
          // Final refresh to get complete metrics
          refreshTrainStatus();
        }
      } catch { /* retry next interval */ }
    }, 4000);
  }, [projectId]);

  const handleStartTraining = async () => {
    try {
      setIsTraining(true);
      setError(null);
      setTrainLogs([]);
      await startTraining(projectId, { method });

      // Try SSE first, fall back to polling if it fails
      let sseFailed = false;
      const close = streamTrainLogs(
        projectId,
        (event, data) => {
          setTrainLogs((prev) => [...prev, { event, ...data }]);
          // Update progress from SSE log events
          if (data.progress_pct != null) {
            setTrainStatus((prev) =>
              prev
                ? { ...prev, progress_pct: data.progress_pct as number }
                : prev
            );
          }
          if (event === "complete") {
            setIsTraining(false);
            close();
            sseCloseRef.current = null;
            refreshTrainStatus();
          }
        },
        () => {
          // SSE failed (HTTP/2 proxy, network error, etc.) — switch to polling
          sseCloseRef.current = null;
          if (!sseFailed) {
            sseFailed = true;
            startTrainPolling();
          }
        }
      );
      sseCloseRef.current = close;
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Training start failed");
      setIsTraining(false);
    }
  };

  const handleCancelTraining = async () => {
    try {
      await cancelTraining(projectId);
      setIsTraining(false);
      refreshTrainStatus();
    } catch { /* ignore */ }
  };

  const refreshTrainStatus = async () => {
    try {
      const status = await getTrainStatus(projectId);
      setTrainStatus(status);
      if (status.status === "running" || status.status === "queued") {
        setIsTraining(true);
      } else {
        setIsTraining(false);
      }

      // Rebuild trainLogs from stored metrics (loss curves) when not streaming live
      // Uses ref to avoid stale closure when called from interval callbacks
      if (
        status.current_metrics &&
        trainLogsRef.current.length === 0
      ) {
        const rebuilt: Record<string, unknown>[] = [];
        const trainCurve = status.current_metrics.train_loss_curve as
          | { step: number; loss: number }[]
          | undefined;
        const evalCurve = status.current_metrics.eval_loss_curve as
          | { step: number; eval_loss: number }[]
          | undefined;

        if (trainCurve) {
          for (const pt of trainCurve) {
            rebuilt.push({ event: "log", step: pt.step, loss: pt.loss });
          }
        }
        if (evalCurve) {
          for (const pt of evalCurve) {
            rebuilt.push({ event: "eval", step: pt.step, eval_loss: pt.eval_loss });
          }
        }
        if (rebuilt.length > 0) {
          setTrainLogs(rebuilt);
        }
      }
    } catch { /* ignore */ }
  };

  // ── Step: Export ──────────────────────────
  const handleExport = async () => {
    try {
      setExporting(true);
      setError(null);
      const { job_id } = await triggerExport(projectId);
      // Poll job until done
      if (pollRef.current) clearInterval(pollRef.current);
      pollRef.current = setInterval(async () => {
        try {
          const job = await getJob(job_id);
          if (job.status === "success") {
            if (pollRef.current) clearInterval(pollRef.current);
            await refreshExportFiles();
            setExporting(false);
          } else if (job.status === "failed") {
            if (pollRef.current) clearInterval(pollRef.current);
            setError(job.error_message ?? "Export failed");
            setExporting(false);
          }
        } catch { /* retry at next interval */ }
      }, 2000);
    } catch (e: unknown) {
      setError(e && typeof e === "object" && "detail" in e ? (e as { detail: string }).detail : "Export failed");
      setExporting(false);
    }
  };

  const refreshExportFiles = async () => {
    try {
      const res = await listExportFiles(projectId);
      setExportFiles(res.files);
    } catch { /* ignore */ }
  };

  // ── Step change effects ───────────────────
  useEffect(() => {
    if (!projectId) return;
    if (currentStep === "data") {
      listFiles(projectId).then((r) => setFiles(r.files)).catch(() => {});
    } else if (currentStep === "preview" || currentStep === "review") {
      refreshDataset();
    } else if (currentStep === "train") {
      refreshTrainStatus();
    } else if (currentStep === "export") {
      refreshExportFiles();
    }
  // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [currentStep, projectId]);

  // ── Navigation ────────────────────────────
  const stepIdx = WIZARD_STEPS.indexOf(currentStep);
  const canGoNext = stepIdx < WIZARD_STEPS.length - 1;
  const canGoPrev = stepIdx > 0;
  const goNext = () => canGoNext && setCurrentStep(WIZARD_STEPS[stepIdx + 1]);
  const goPrev = () => canGoPrev && setCurrentStep(WIZARD_STEPS[stepIdx - 1]);

  // ── Loading / Error ───────────────────────
  if (loading) {
    return <div className="py-20 text-center text-muted-foreground">Loading project...</div>;
  }
  if (!project) {
    return (
      <div className="py-20 text-center">
        <p className="text-destructive">{error || "Project not found"}</p>
        <Button variant="outline" className="mt-4" onClick={() => router.push("/")}>
          Back to Dashboard
        </Button>
      </div>
    );
  }

  return (
    <div className="mx-auto max-w-4xl space-y-6">
      {/* Header */}
      <div className="flex items-center justify-between">
        <div>
          <h1 className="text-2xl font-bold">{project.name}</h1>
          {project.description && (
            <p className="text-sm text-muted-foreground">{project.description}</p>
          )}
        </div>
        <StatusBadge status={project.status} />
      </div>

      {/* Wizard Steps */}
      <WizardSteps
        currentStep={currentStep}
        onStepClick={setCurrentStep}
      />

      {/* Error banner */}
      {error && (
        <div className="rounded-lg border border-destructive/50 bg-destructive/10 p-3 text-sm text-destructive">
          {error}
          <button className="ml-2 underline" onClick={() => setError(null)}>
            Dismiss
          </button>
        </div>
      )}

      {/* ═══ Step Content ═══ */}

      {/* MODEL STEP */}
      {currentStep === "model" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Select Base Model</CardTitle>
            <CardDescription>
              Enter a HuggingFace model ID (e.g. meta-llama/Llama-3.1-8B)
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Input
                value={modelId}
                onChange={(e) => setModelId(e.target.value)}
                placeholder="meta-llama/Llama-3.1-8B"
                className="flex-1"
              />
              <Button onClick={handleResolveModel} disabled={resolving || !modelId.trim()}>
                {resolving ? "Resolving..." : "Resolve"}
              </Button>
            </div>
            {modelInfo && (
              <div className="rounded-lg border p-4 text-sm space-y-2">
                <p><strong>Model:</strong> {modelInfo.model_id}</p>
                {modelInfo.num_parameters && (
                  <p><strong>Parameters:</strong> {(modelInfo.num_parameters / 1e9).toFixed(1)}B</p>
                )}
                {modelInfo.architecture && (
                  <p><strong>Architecture:</strong> {modelInfo.architecture}</p>
                )}
                {modelInfo.estimated_vram_fp16_gb && (
                  <p><strong>Est. VRAM (FP16):</strong> {modelInfo.estimated_vram_fp16_gb.toFixed(1)} GB</p>
                )}
                {modelInfo.estimated_vram_4bit_gb && (
                  <p><strong>Est. VRAM (4-bit):</strong> {modelInfo.estimated_vram_4bit_gb.toFixed(1)} GB</p>
                )}
                {modelInfo.warnings.length > 0 && (
                  <div className="mt-2">
                    {modelInfo.warnings.map((w, i) => (
                      <p key={i} className="text-yellow-600 text-xs">⚠ {w}</p>
                    ))}
                  </div>
                )}
              </div>
            )}
          </CardContent>
        </Card>
      )}

      {/* TASK STEP */}
      {currentStep === "task" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Task Configuration</CardTitle>
            <CardDescription>
              Choose the type of fine-tuning task for your model.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="grid gap-3 sm:grid-cols-2">
              {taskTypes.map((tt) => {
                const isSelected = project.task_type === tt.key;
                return (
                  <button
                    key={tt.key}
                    onClick={async () => {
                      await updateProject(projectId, { task_type: tt.key });
                      setDataSource(null);
                      setUploadResult(null);
                      fetchProject();
                    }}
                    className={`rounded-lg border p-4 text-left transition-colors ${
                      isSelected
                        ? "border-primary bg-primary/5 ring-2 ring-primary"
                        : "border-border hover:border-primary/50 hover:bg-muted/50"
                    }`}
                  >
                    <div className="font-medium text-sm">{tt.label}</div>
                    <p className="mt-1 text-xs text-muted-foreground">
                      {tt.description}
                    </p>
                  </button>
                );
              })}
            </div>

            {/* Show selected task info + data format */}
            {(() => {
              const selectedConfig = taskTypes.find((t) => t.key === project.task_type);
              if (!selectedConfig) return null;
              return (
                <div className="mt-4 rounded-lg border bg-muted/30 p-4 space-y-3">
                  <h4 className="text-sm font-medium">
                    Expected JSONL format for &quot;{selectedConfig.label}&quot;:
                  </h4>
                  <pre className="overflow-x-auto rounded bg-muted p-3 text-xs">
                    {JSON.stringify(selectedConfig.sample_example, null, 2)}
                  </pre>
                  <p className="text-xs text-muted-foreground">
                    Required fields: {selectedConfig.required_fields.join(", ")}
                    {selectedConfig.optional_fields.length > 0 &&
                      ` · Optional: ${selectedConfig.optional_fields.join(", ")}`}
                  </p>
                </div>
              );
            })()}

            {/* Dataset source question */}
            <div className="mt-4 space-y-3">
              <h4 className="text-sm font-medium">
                Do you already have a dataset in JSONL format?
              </h4>
              <div className="flex gap-3">
                <Button
                  variant={dataSource === "upload_jsonl" ? "default" : "outline"}
                  onClick={() => setDataSource("upload_jsonl")}
                  size="sm"
                >
                  Yes, upload my JSONL
                </Button>
                <Button
                  variant={dataSource === "generate" ? "default" : "outline"}
                  onClick={() => setDataSource("generate")}
                  size="sm"
                >
                  No, generate from documents
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* DATA STEP */}
      {currentStep === "data" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">
              {dataSource === "upload_jsonl" ? "Upload Dataset (JSONL)" : "Upload Documents"}
            </CardTitle>
            <CardDescription>
              {dataSource === "upload_jsonl"
                ? "Upload your pre-formatted JSONL dataset file."
                : "Upload your documents to be chunked and used for dataset generation."}
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            {/* JSONL Upload Mode */}
            {dataSource === "upload_jsonl" && (
              <div className="space-y-4">
                {/* Show expected format reminder */}
                {(() => {
                  const cfg = taskTypes.find((t) => t.key === project.task_type);
                  if (!cfg) return null;
                  return (
                    <div className="rounded-lg border bg-muted/30 p-4">
                      <h4 className="text-sm font-medium mb-2">
                        Expected format for &quot;{cfg.label}&quot;:
                      </h4>
                      <pre className="overflow-x-auto rounded bg-muted p-3 text-xs">
                        {JSON.stringify(cfg.sample_example, null, 0)}
                      </pre>
                      <p className="mt-2 text-xs text-muted-foreground">
                        One JSON object per line. Required fields: {cfg.required_fields.join(", ")}
                      </p>
                    </div>
                  );
                })()}

                <div>
                  <label className="block text-sm font-medium mb-2">
                    Select JSONL file
                  </label>
                  <input
                    type="file"
                    accept=".jsonl,.json"
                    onChange={async (e) => {
                      const file = e.target.files?.[0];
                      if (!file) return;
                      try {
                        setUploading(true);
                        setError(null);
                        const res = await uploadDatasetJsonl(projectId, file);
                        setUploadResult(res);
                        if (res.valid > 0) {
                          await fetchProject();
                        }
                      } catch (err: unknown) {
                        setError(
                          err && typeof err === "object" && "detail" in err
                            ? (err as { detail: string }).detail
                            : "Upload failed"
                        );
                      } finally {
                        setUploading(false);
                      }
                    }}
                    className="block w-full text-sm file:mr-4 file:rounded-md file:border-0 file:bg-primary file:px-4 file:py-2 file:text-sm file:font-medium file:text-primary-foreground hover:file:bg-primary/90"
                  />
                </div>

                {uploading && <p className="text-sm text-muted-foreground">Uploading & validating...</p>}

                {uploadResult && (
                  <div className="rounded-lg border p-4 space-y-2 text-sm">
                    <p>
                      <Badge variant="success">Valid: {uploadResult.valid}</Badge>{" "}
                      {uploadResult.invalid > 0 && (
                        <Badge variant="destructive">Invalid: {uploadResult.invalid}</Badge>
                      )}
                    </p>
                    <p>Train: {uploadResult.train} · Eval: {uploadResult.eval}</p>
                    {uploadResult.errors.length > 0 && (
                      <div className="mt-2">
                        <p className="text-xs font-medium text-destructive">Errors (first 10):</p>
                        {uploadResult.errors.map((err, i) => (
                          <p key={i} className="text-xs text-destructive">
                            Line {err.line}: {err.error}
                          </p>
                        ))}
                      </div>
                    )}
                    {uploadResult.valid > 0 && (
                      <p className="text-green-600 text-xs mt-2">
                        ✓ Dataset uploaded successfully! You can proceed to the next step.
                      </p>
                    )}
                  </div>
                )}
              </div>
            )}

            {/* Document Upload + Generate Mode */}
            {dataSource !== "upload_jsonl" && (
              <>
                <FileUploadDropzone
                  onFiles={handleUploadFiles}
                  disabled={uploading}
                />
                {uploading && <p className="text-sm text-muted-foreground">Uploading...</p>}

                {files.length > 0 && (
                  <div className="space-y-2">
                    <h3 className="text-sm font-medium">Uploaded Files ({files.length})</h3>
                    <div className="space-y-1">
                      {files.map((f) => (
                        <div
                          key={f.id}
                          className="flex items-center justify-between rounded border px-3 py-2 text-sm"
                        >
                          <span>{f.filename}</span>
                          <span className="text-xs text-muted-foreground">
                            {(f.size_bytes / 1024).toFixed(1)} KB
                          </span>
                        </div>
                      ))}
                    </div>
                    <Button onClick={handleChunk} disabled={generating} variant="secondary">
                      {generating ? "Chunking..." : "Generate Chunks"}
                    </Button>
                  </div>
                )}
              </>
            )}
          </CardContent>
        </Card>
      )}

      {/* PREVIEW STEP */}
      {currentStep === "preview" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Dataset Preview</CardTitle>
            <CardDescription>
              Generate a small preview to verify quality before full generation.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={handlePreview} disabled={generating}>
              {generating ? "Generating preview..." : "Generate Preview (10 examples)"}
            </Button>
            <DatasetTable examples={examples} taskTypeConfig={taskTypes.find(t => t.key === project.task_type) ?? null} />
          </CardContent>
        </Card>
      )}

      {/* REVIEW STEP */}
      {currentStep === "review" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Full Dataset Generation</CardTitle>
            <CardDescription>
              Generate the complete dataset from all document chunks.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Button onClick={handleGenerate} disabled={generating}>
                {generating ? "Generating..." : "Generate Full Dataset"}
              </Button>
              <Button variant="outline" onClick={refreshDataset}>
                Refresh
              </Button>
            </div>
            <DatasetTable examples={examples} stats={datasetStats} taskTypeConfig={taskTypes.find(t => t.key === project.task_type) ?? null} />
          </CardContent>
        </Card>
      )}

      {/* HARDWARE STEP */}
      {currentStep === "hardware" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Hardware Check</CardTitle>
            <CardDescription>
              Probe your GPU hardware for training recommendations.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <Button onClick={handleCheckHardware} disabled={checkingHw}>
              {checkingHw ? "Checking..." : "Check Hardware"}
            </Button>
            {hardware && (
              <div className="rounded-lg border p-4 text-sm space-y-2">
                <p>
                  <strong>GPU:</strong>{" "}
                  {hardware.cuda_available ? hardware.gpu_name : "No CUDA GPU detected"}
                </p>
                {hardware.vram_total_gb != null && (
                  <p><strong>VRAM:</strong> {hardware.vram_total_gb} GB</p>
                )}
                {hardware.recommended_method && (
                  <p>
                    <strong>Recommended:</strong>{" "}
                    <Badge>{hardware.recommended_method.toUpperCase()}</Badge>
                  </p>
                )}
                {hardware.recommendation_reason && (
                  <p className="text-muted-foreground">{hardware.recommendation_reason}</p>
                )}
                {hardware.warnings.map((w, i) => (
                  <p key={i} className="text-yellow-600 text-xs">⚠ {w}</p>
                ))}
              </div>
            )}

            <div className="pt-4">
              <label className="mb-1 block text-sm font-medium">Fine-tuning Method</label>
              <Select
                value={method}
                onValueChange={(v) => setMethod(v as FinetuneMethod)}
                options={[
                  { value: "qlora", label: "QLoRA — 4-bit quantized LoRA" },
                  { value: "lora", label: "LoRA — Low-Rank Adaptation" },
                  { value: "dora", label: "DoRA — Weight-Decomposed LoRA" },
                  { value: "ia3", label: "IA³ — Few-param Activation Scaling" },
                  { value: "prefix", label: "Prefix Tuning — Virtual Tokens" },
                  { value: "full", label: "Full Fine-tuning — All Parameters" },
                ]}
              />
            </div>
          </CardContent>
        </Card>
      )}

      {/* TRAIN STEP */}
      {currentStep === "train" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Training</CardTitle>
          </CardHeader>
          <CardContent className="space-y-4">
            {trainStatus && (
              <div className="flex items-center gap-4 text-sm">
                <StatusBadge status={trainStatus.status} />
                <Progress value={trainStatus.progress_pct} />
                <span className="tabular-nums">{trainStatus.progress_pct}%</span>
                {trainStatus.elapsed_seconds != null && (
                  <span className="text-muted-foreground">
                    {Math.round(trainStatus.elapsed_seconds)}s elapsed
                  </span>
                )}
              </div>
            )}

            <div className="flex gap-2">
              {!isTraining && (
                <Button onClick={handleStartTraining}>
                  Start Training ({method.toUpperCase()})
                </Button>
              )}
              {isTraining && (
                <Button variant="destructive" onClick={handleCancelTraining}>
                  Cancel Training
                </Button>
              )}
              <Button variant="outline" onClick={refreshTrainStatus}>
                Refresh Status
              </Button>
            </div>

            <LossChart logs={trainLogs} />
            <TrainingLogsViewer logs={trainLogs} />
          </CardContent>
        </Card>
      )}

      {/* EXPORT STEP */}
      {currentStep === "export" && (
        <Card>
          <CardHeader>
            <CardTitle className="text-lg">Export Artifacts</CardTitle>
            <CardDescription>
              Package your fine-tuned adapter for download.
            </CardDescription>
          </CardHeader>
          <CardContent className="space-y-4">
            <div className="flex gap-2">
              <Button onClick={handleExport} disabled={exporting}>
                {exporting ? "Exporting..." : "Generate Export Package"}
              </Button>
              <Button variant="outline" onClick={refreshExportFiles}>
                Refresh
              </Button>
            </div>

            {exportFiles.length > 0 && (
              <div className="space-y-2">
                <h3 className="text-sm font-medium">Export Files</h3>
                {exportFiles.map((f) => (
                  <div
                    key={f.filename}
                    className="flex items-center justify-between rounded border px-3 py-2 text-sm"
                  >
                    <span>{f.filename}</span>
                    <div className="flex items-center gap-2">
                      <span className="text-xs text-muted-foreground">
                        {(f.size_bytes / 1024).toFixed(1)} KB
                      </span>
                      <a
                        href={getExportDownloadUrl(projectId, f.filename)}
                        className="text-xs text-primary hover:underline"
                        target="_blank"
                        rel="noopener noreferrer"
                      >
                        Download
                      </a>
                    </div>
                  </div>
                ))}

                <div className="pt-2">
                  <a href={getExportDownloadUrl(projectId)}>
                    <Button variant="outline">Download All (ZIP)</Button>
                  </a>
                </div>
              </div>
            )}

            {/* Show training metrics if available */}
            {trainStatus?.current_metrics && (
              <div className="rounded-lg border p-4 text-sm space-y-1">
                <h3 className="font-medium">Training Results</h3>
                {Object.entries(trainStatus.current_metrics)
                  .filter(([k]) => !k.includes("curve") && !k.includes("samples"))
                  .map(([key, val]) => (
                    <p key={key}>
                      <span className="text-muted-foreground">{key}:</span>{" "}
                      {typeof val === "number" ? val.toFixed(4) : String(val)}
                    </p>
                  ))}
              </div>
            )}

            {/* Playground Link */}
            <div className="rounded-lg border border-primary/20 bg-primary/5 p-4">
              <div className="flex items-center justify-between">
                <div>
                  <h3 className="font-medium">🧪 Test your model</h3>
                  <p className="text-sm text-muted-foreground">
                    Try the Playground to test your fine-tuned model with custom prompts.
                  </p>
                </div>
                <Button
                  onClick={() => router.push(`/projects/${projectId}/playground`)}
                >
                  Open Playground →
                </Button>
              </div>
            </div>
          </CardContent>
        </Card>
      )}

      {/* Navigation */}
      <div className="flex items-center justify-between pt-4">
        <Button variant="outline" onClick={goPrev} disabled={!canGoPrev}>
          ← Previous
        </Button>
        <Button onClick={goNext} disabled={!canGoNext}>
          Next →
        </Button>
      </div>
    </div>
  );
}
