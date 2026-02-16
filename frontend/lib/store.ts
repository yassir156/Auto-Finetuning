/**
 * FineTuneFlow — Zustand Store.
 *
 * Global client-side state for the active project wizard,
 * project list, and UI flags.
 */

import { create } from "zustand";
import type {
  Project,
  FileInfo,
  Chunk,
  DatasetExample,
  DatasetStats,
  HardwareCheck,
  TrainStatus,
  TrainHyperparams,
  FinetuneMethod,
  Job,
  ExportFileInfo,
  ModelResolveResponse,
} from "./types";

// ── Wizard Steps ─────────────────────────────
export const WIZARD_STEPS = [
  "model",
  "task",
  "data",
  "preview",
  "review",
  "hardware",
  "train",
  "export",
] as const;

export type WizardStep = (typeof WIZARD_STEPS)[number];

// ── Store Interface ──────────────────────────
interface AppState {
  // Project list (dashboard)
  projects: Project[];
  projectsLoading: boolean;
  setProjects: (projects: Project[]) => void;
  setProjectsLoading: (v: boolean) => void;

  // Active project (wizard)
  activeProject: Project | null;
  setActiveProject: (p: Project | null) => void;

  // Wizard navigation
  currentStep: WizardStep;
  setCurrentStep: (step: WizardStep) => void;

  // Model resolution
  modelInfo: ModelResolveResponse | null;
  setModelInfo: (info: ModelResolveResponse | null) => void;

  // Files
  files: FileInfo[];
  setFiles: (files: FileInfo[]) => void;

  // Chunks
  chunks: Chunk[];
  chunksTotal: number;
  setChunks: (chunks: Chunk[], total: number) => void;

  // Dataset
  datasetExamples: DatasetExample[];
  datasetStats: DatasetStats | null;
  setDatasetExamples: (examples: DatasetExample[]) => void;
  setDatasetStats: (stats: DatasetStats | null) => void;

  // Hardware
  hardware: HardwareCheck | null;
  setHardware: (hw: HardwareCheck | null) => void;

  // Training config
  finetuneMethod: FinetuneMethod;
  hyperparams: TrainHyperparams;
  setFinetuneMethod: (m: FinetuneMethod) => void;
  setHyperparams: (h: Partial<TrainHyperparams>) => void;

  // Training status
  trainStatus: TrainStatus | null;
  setTrainStatus: (s: TrainStatus | null) => void;

  // Training log entries (streamed)
  trainLogs: Record<string, unknown>[];
  addTrainLog: (entry: Record<string, unknown>) => void;
  clearTrainLogs: () => void;

  // Jobs
  jobs: Job[];
  setJobs: (jobs: Job[]) => void;

  // Export
  exportFiles: ExportFileInfo[];
  setExportFiles: (files: ExportFileInfo[]) => void;

  // Error toast
  errorMessage: string | null;
  setErrorMessage: (msg: string | null) => void;

  // Reset wizard state
  resetWizard: () => void;
}

// ── Default Hyperparams ──────────────────────
const DEFAULT_HYPERPARAMS: TrainHyperparams = {
  num_epochs: 3,
  learning_rate: 2e-4,
  per_device_batch_size: 4,
  gradient_accumulation_steps: 4,
  max_seq_length: 2048,
  lora_r: 16,
  lora_alpha: 32,
  lora_dropout: 0.05,
  warmup_ratio: 0.03,
  weight_decay: 0.01,
};

// ── Store ────────────────────────────────────
export const useAppStore = create<AppState>((set) => ({
  // Projects
  projects: [],
  projectsLoading: false,
  setProjects: (projects) => set({ projects }),
  setProjectsLoading: (v) => set({ projectsLoading: v }),

  // Active project
  activeProject: null,
  setActiveProject: (p) => set({ activeProject: p }),

  // Wizard
  currentStep: "model",
  setCurrentStep: (step) => set({ currentStep: step }),

  // Model
  modelInfo: null,
  setModelInfo: (info) => set({ modelInfo: info }),

  // Files
  files: [],
  setFiles: (files) => set({ files }),

  // Chunks
  chunks: [],
  chunksTotal: 0,
  setChunks: (chunks, total) => set({ chunks, chunksTotal: total }),

  // Dataset
  datasetExamples: [],
  datasetStats: null,
  setDatasetExamples: (examples) => set({ datasetExamples: examples }),
  setDatasetStats: (stats) => set({ datasetStats: stats }),

  // Hardware
  hardware: null,
  setHardware: (hw) => set({ hardware: hw }),

  // Training config
  finetuneMethod: "qlora",
  hyperparams: { ...DEFAULT_HYPERPARAMS },
  setFinetuneMethod: (m) => set({ finetuneMethod: m }),
  setHyperparams: (h) =>
    set((state) => ({
      hyperparams: { ...state.hyperparams, ...h },
    })),

  // Training status
  trainStatus: null,
  setTrainStatus: (s) => set({ trainStatus: s }),

  // Train logs
  trainLogs: [],
  addTrainLog: (entry) =>
    set((state) => ({ trainLogs: [...state.trainLogs, entry] })),
  clearTrainLogs: () => set({ trainLogs: [] }),

  // Jobs
  jobs: [],
  setJobs: (jobs) => set({ jobs }),

  // Export
  exportFiles: [],
  setExportFiles: (files) => set({ exportFiles: files }),

  // Error
  errorMessage: null,
  setErrorMessage: (msg) => set({ errorMessage: msg }),

  // Reset
  resetWizard: () =>
    set({
      activeProject: null,
      currentStep: "model",
      modelInfo: null,
      files: [],
      chunks: [],
      chunksTotal: 0,
      datasetExamples: [],
      datasetStats: null,
      hardware: null,
      finetuneMethod: "qlora",
      hyperparams: { ...DEFAULT_HYPERPARAMS },
      trainStatus: null,
      trainLogs: [],
      jobs: [],
      exportFiles: [],
    }),
}));
