# FineTuneFlow — Spécifications Frontend

## 1. Stack technique

| Outil | Usage |
|-------|-------|
| Next.js 14 (App Router) | Framework React |
| React 18 | UI |
| TypeScript | Type safety |
| Tailwind CSS | Styling |
| shadcn/ui | Composants UI (Button, Card, Dialog, Table, etc.) |
| Zustand | State management global |
| recharts | Graphiques (loss curves) |
| react-dropzone | Upload drag & drop |
| EventSource (natif) | SSE pour logs temps réel |

## 2. Pages et routes

```
/                                    → Dashboard (liste projets)
/projects/new                        → Création de projet
/projects/[id]                       → Détail projet (redirect vers wizard)
/projects/[id]/wizard                → Wizard multi-étapes
/projects/[id]/wizard?step=model     → Step spécifique
```

## 3. Dashboard (`/`)

### Layout
```
┌──────────────────────────────────────────────────────────┐
│  FineTuneFlow                              [+ New Project]│
├──────────────────────────────────────────────────────────┤
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 📄 Mon projet Q&A                                  │  │
│  │ Task: Q&A | Model: Llama-3.1-8B | Status: Training │  │
│  │ 3 docs | 1800 examples | Created 2h ago            │  │
│  │                                          [Open] [🗑] │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  ┌────────────────────────────────────────────────────┐  │
│  │ 📄 Instruction Tuning v2                           │  │
│  │ Task: Instruction | Model: — | Status: Draft       │  │
│  │ 0 docs | 0 examples | Created 1d ago              │  │
│  │                                          [Open] [🗑] │  │
│  └────────────────────────────────────────────────────┘  │
│                                                          │
│  No more projects.                                       │
└──────────────────────────────────────────────────────────┘
```

### Données affichées par projet
- Nom
- Task type (badge)
- Modèle de base (si choisi)
- Statut (badge coloré)
- Nombre de fichiers
- Nombre d'exemples
- Date de création (relative)
- Actions : Open, Delete

## 4. Wizard — Vue d'ensemble

### Navigation
```
[1. Model] → [2. Task] → [3. Data] → [4. Preview] → [5. Review] → [6. Hardware] → [7. Train] → [8. Export]
```

- Chaque step est une section de la page wizard
- Progress bar en haut (steps complétés en vert)
- Boutons Prev / Next en bas
- Un step peut être disabled si les prérequis ne sont pas remplis
- L'état est persité dans Zustand + API (le projet en DB est la source de vérité)

### Contraintes de navigation

| Step | Prérequis pour accéder |
|------|----------------------|
| 1. Model | — |
| 2. Task | — |
| 3. Data | Modèle choisi + task choisie |
| 4. Preview | Au moins 1 fichier uploadé OU 1 dataset uploadé |
| 5. Review | Preview terminée OU dataset uploadé |
| 6. Hardware | Dataset prêt (>= 10 exemples valides) |
| 7. Train | Hardware OK + dataset prêt |
| 8. Export | Training terminé (success) |

## 5. Wizard Steps — Détail

### 5.1 Model Step

```
┌──────────────────────────────────────────────────────┐
│  Step 1: Choose Base Model                           │
│                                                      │
│  HuggingFace Model ID:                               │
│  ┌──────────────────────────────────────────┐        │
│  │ meta-llama/Llama-3.1-8B                  │ [Resolve]│
│  └──────────────────────────────────────────┘        │
│                                                      │
│  ✅ Model found: LlamaForCausalLM                    │
│  Parameters: 8.03B                                   │
│  VRAM (FP16): ~16.1 GB                               │
│  VRAM (4-bit): ~5.5 GB                               │
│  License: llama3.1                                   │
│                                                      │
│                                         [Next →]     │
└──────────────────────────────────────────────────────┘
```

**Interactions :**
- Input texte pour le model ID
- Bouton "Resolve" → POST `/model/resolve`
- Affiche les infos du modèle si trouvé
- Erreur si modèle non trouvé
- "Next" sauvegarde dans le projet

### 5.2 Task Step

```
┌──────────────────────────────────────────────────────┐
│  Step 2: Select Task Type                            │
│                                                      │
│  ┌─────────────────────┐  ┌─────────────────────┐   │
│  │  📝 Instruction      │  │  ❓ Q&A              │   │
│  │  Tuning              │  │                     │   │
│  │                      │  │  Generate question- │   │
│  │  Generate diverse    │  │  answer pairs from  │   │
│  │  instruction/output  │  │  your documents     │   │
│  │  pairs from your     │  │  with context       │   │
│  │  documents           │  │                     │   │
│  │  ✅ Selected         │  │                     │   │
│  └─────────────────────┘  └─────────────────────┘   │
│                                                      │
│                                  [← Prev] [Next →]   │
└──────────────────────────────────────────────────────┘
```

**Interactions :**
- 2 cartes cliquables (radio)
- Description de chaque type
- PATCH projet avec le task_type sélectionné

### 5.3 Data Step

```
┌──────────────────────────────────────────────────────┐
│  Step 3: Upload Data                                 │
│                                                      │
│  ┌─ Upload Documents ─────────────────────────────┐  │
│  │                                                │  │
│  │  ╔═══════════════════════════════════════════╗  │  │
│  │  ║  📁 Drag & drop files here               ║  │  │
│  │  ║     or click to browse                    ║  │  │
│  │  ║                                           ║  │  │
│  │  ║  PDF, DOCX, TXT, MD (max 50 MB each)     ║  │  │
│  │  ╚═══════════════════════════════════════════╝  │  │
│  │                                                │  │
│  │  Uploaded files:                               │  │
│  │  📄 rapport.pdf (2.3 MB)              [🗑]     │  │
│  │  📄 notes.docx (456 KB)              [🗑]     │  │
│  │  📄 data.txt (123 KB)                [🗑]     │  │
│  │                                                │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│  ── OR ──                                            │
│                                                      │
│  ┌─ Upload Dataset ───────────────────────────────┐  │
│  │  Upload a pre-made dataset (JSONL, CSV, JSON)  │  │
│  │  [Choose file...]                              │  │
│  └────────────────────────────────────────────────┘  │
│                                                      │
│                                  [← Prev] [Next →]   │
└──────────────────────────────────────────────────────┘
```

**Interactions :**
- Zone drag & drop (react-dropzone)
- Liste des fichiers uploadés avec bouton supprimer
- OU upload de dataset direct
- Validation côté client (extension, taille) avant envoi
- Progress bar pendant upload

### 5.4 Preview Step

```
┌──────────────────────────────────────────────────────┐
│  Step 4: Dataset Preview                             │
│                                                      │
│  Generate a preview of 10 examples to check quality. │
│                                                      │
│  Ollama Model: [llama3.1:70b        ▼]               │
│                                                      │
│  [Generate Preview]                                  │
│                                                      │
│  ⏳ Generating... (Job running, 60%)                  │
│  ████████████░░░░░░░░                                │
│                                                      │
│  Preview Results (8/10 valid):                       │
│  ┌────┬──────────────────────┬──────────────────┐    │
│  │ #  │ Instruction          │ Output (truncated)│    │
│  ├────┼──────────────────────┼──────────────────┤    │
│  │ 1  │ Explain gradient...  │ Gradient descent  │    │
│  │ 2  │ What is the role...  │ The optimizer...  │    │
│  │ .. │ ...                  │ ...              │    │
│  └────┴──────────────────────┴──────────────────┘    │
│                                                      │
│  ⚠️ 2 examples were invalid and filtered out.         │
│                                                      │
│  Looks good? [Generate Full Dataset (target: 2000)]  │
│                                                      │
│                                  [← Prev] [Next →]   │
└──────────────────────────────────────────────────────┘
```

**Interactions :**
- Bouton "Generate Preview" → POST `/dataset/preview`
- Polling du job status (GET `/jobs/{id}`) toutes les 2s
- Affichage des exemples une fois terminé
- Bouton "Generate Full Dataset" → POST `/dataset/generate`
- Possibilité de changer le modèle Ollama

### 5.5 Review Step

```
┌──────────────────────────────────────────────────────┐
│  Step 5: Review Dataset                              │
│                                                      │
│  ┌─ Stats ─────────────────────────────────────────┐ │
│  │ Total: 2000 | Valid: 1800 | Train: 1620 | Eval: 180 │
│  │ Avg tokens: 312 | Duplicates removed: 35        │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  Filter: [All ▼] [Valid only ☑] Search: [________]   │
│                                                      │
│  ┌────┬──────────────┬────────┬──────────┬────────┐  │
│  │ ☐  │ Instruction  │ Output │ Tokens   │ Valid  │  │
│  ├────┼──────────────┼────────┼──────────┼────────┤  │
│  │ ☐  │ Explain...   │ The... │ 245      │ ✅     │  │
│  │ ☐  │ What is...   │ It...  │ 189      │ ✅     │  │
│  │ ☐  │ Describe...  │ Too..  │ 8        │ ❌     │  │
│  └────┴──────────────┴────────┴──────────┴────────┘  │
│                                                      │
│  Page: [< 1 2 3 ... 36 >]   [Delete Selected]       │
│                                                      │
│                                  [← Prev] [Next →]   │
└──────────────────────────────────────────────────────┘
```

**Interactions :**
- Tableau paginé (50 par page)
- Filtres : split (train/eval), valid/invalid, recherche texte
- Sélection multiple + suppression batch
- Click sur une ligne → modal avec l'exemple complet
- Stats en haut avec badges

### 5.6 Hardware Step

```
┌──────────────────────────────────────────────────────┐
│  Step 6: Hardware Check                              │
│                                                      │
│  [Check Hardware]                                    │
│                                                      │
│  ┌─ GPU Info ──────────────────────────────────────┐ │
│  │ GPU: NVIDIA RTX 4090                            │ │
│  │ VRAM: 24.0 GB (22.5 GB free)                    │ │
│  │ CUDA: 12.4 | PyTorch: 2.5.0                    │ │
│  │ bitsandbytes: ✅ Available                       │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  ┌─ Recommendation ───────────────────────────────┐  │
│  │ ✅ QLoRA recommended                            │  │
│  │ 24GB VRAM with 8B model → QLoRA 4-bit fits     │  │
│  │ comfortably. Estimated VRAM usage: ~7.2 GB      │  │
│  └─────────────────────────────────────────────────┘  │
│                                                      │
│                                  [← Prev] [Next →]   │
└──────────────────────────────────────────────────────┘
```

### 5.7 Train Step

```
┌──────────────────────────────────────────────────────┐
│  Step 7: Training                                    │
│                                                      │
│  Method: [QLoRA ▼]                                   │
│                                                      │
│  ┌─ Hyperparameters (defaults applied) ────────────┐ │
│  │ Epochs: [3]    LR: [2e-4]    Batch: [4]        │ │
│  │ Grad Accum: [4]  Max Seq Len: [2048]            │ │
│  │ LoRA r: [16]    Alpha: [32]   Dropout: [0.05]  │ │
│  │                          [Reset to defaults]     │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  [▶ Start Training]                                  │
│                                                      │
│  ── Training Logs ──                                 │
│  ████████████████████░░░░░░░░░░ 65% (650/1000)      │
│                                                      │
│  ┌─────────────────────────────────────────────────┐ │
│  │  📈 Loss Curve                                  │ │
│  │  train ── eval --                               │ │
│  │  2.5│\                                          │ │
│  │  2.0│ \                                         │ │
│  │  1.5│  \___                                     │ │
│  │  1.0│      \___                                 │ │
│  │  0.5│          \_______                         │ │
│  │     └─────────────────── steps                  │ │
│  └─────────────────────────────────────────────────┘ │
│                                                      │
│  Step 650 | Loss: 0.42 | LR: 1.2e-4 | ETA: 25min   │
│                                                      │
│  [Cancel Training]                                   │
│                                                      │
│                                  [← Prev] [Next →]   │
└──────────────────────────────────────────────────────┘
```

**Interactions :**
- Formulaire hyperparams avec defaults pré-remplis
- Bouton "Start Training" → POST `/train/start`
- SSE `EventSource` vers `/train/logs/stream`
- Graphique loss mis à jour en temps réel
- Progress bar
- Métriques courantes
- Bouton Cancel

### 5.8 Export Step

```
┌──────────────────────────────────────────────────────┐
│  Step 8: Export & Results                            │
│                                                      │
│  ┌─ Training Summary ─────────────────────────────┐  │
│  │ Model: Llama-3.1-8B + QLoRA adapter            │  │
│  │ Duration: 2h 15min                              │  │
│  │ Final train loss: 0.32                          │  │
│  │ Final eval loss: 0.48                           │  │
│  │ Perplexity: 1.62                                │  │
│  └─────────────────────────────────────────────────┘  │
│                                                      │
│  ┌─ Sample Outputs ───────────────────────────────┐  │
│  │ Q: Explain gradient descent                     │  │
│  │ Expected: Gradient descent is...                │  │
│  │ Model: Gradient descent is an optimization...   │  │
│  │ ─────────────────────────                       │  │
│  │ Q: What is backpropagation?                     │  │
│  │ Expected: Backprop...                           │  │
│  │ Model: Backpropagation is the algorithm...      │  │
│  └─────────────────────────────────────────────────┘  │
│                                                      │
│  [Export Artifacts]                                   │
│                                                      │
│  Files:                                              │
│  📦 adapter_model.safetensors (123 KB)    [⬇]       │
│  📄 report.md (5 KB)                     [⬇]       │
│  📄 metrics.json (1 KB)                  [⬇]       │
│  📦 finetuneflow_export.zip (130 KB)     [⬇ All]   │
│                                                      │
│                                  [← Prev] [🏠 Home]  │
└──────────────────────────────────────────────────────┘
```

## 6. Zustand Store

```typescript
// frontend/lib/store.ts

interface WizardState {
  // Project data
  projectId: string | null;
  project: Project | null;
  
  // Wizard navigation
  currentStep: number;
  completedSteps: Set<number>;
  
  // Step data
  modelInfo: ModelInfo | null;
  taskType: 'instruction' | 'qa' | null;
  uploadedFiles: FileInfo[];
  previewExamples: DatasetExample[];
  datasetStats: DatasetStats | null;
  hardwareInfo: HardwareInfo | null;
  trainingConfig: TrainingConfig;
  trainingStatus: TrainingStatus | null;
  exportFiles: ExportFile[];
  
  // Actions
  setProject: (project: Project) => void;
  setStep: (step: number) => void;
  completeStep: (step: number) => void;
  setModelInfo: (info: ModelInfo) => void;
  setTaskType: (type: 'instruction' | 'qa') => void;
  addFile: (file: FileInfo) => void;
  removeFile: (fileId: string) => void;
  setPreviewExamples: (examples: DatasetExample[]) => void;
  setDatasetStats: (stats: DatasetStats) => void;
  setHardwareInfo: (info: HardwareInfo) => void;
  updateTrainingConfig: (config: Partial<TrainingConfig>) => void;
  setTrainingStatus: (status: TrainingStatus) => void;
  reset: () => void;
}
```

## 7. Types TypeScript

```typescript
// frontend/lib/types.ts

interface Project {
  id: string;
  name: string;
  description?: string;
  task_type: 'instruction' | 'qa';
  base_model_id?: string;
  model_info?: ModelInfo;
  status: ProjectStatus;
  config: Record<string, any>;
  created_at: string;
  updated_at: string;
}

type ProjectStatus = 
  | 'draft' | 'uploading' | 'chunking' | 'generating' 
  | 'ready_to_train' | 'training' | 'evaluating' 
  | 'completed' | 'failed';

interface ModelInfo {
  model_id: string;
  model_type: string;
  num_parameters: number;
  estimated_vram_fp16_gb: number;
  estimated_vram_4bit_gb: number;
  architecture: string;
  vocab_size: number;
  license: string;
}

interface FileInfo {
  id: string;
  filename: string;
  kind: 'raw_doc' | 'dataset_upload';
  status: 'uploading' | 'ready' | 'failed';
  mime_type: string;
  size_bytes: number;
  created_at: string;
}

interface DatasetExample {
  id: string;
  split: 'preview' | 'train' | 'eval';
  data: {
    instruction: string;
    input: string;
    output: string;
  };
  is_valid: boolean;
  token_count: number;
  validation_error?: string;
}

interface DatasetStats {
  total: number;
  valid: number;
  invalid: number;
  by_split: {
    train: SplitStats;
    eval: SplitStats;
  };
}

interface SplitStats {
  count: number;
  avg_tokens: number;
  min_tokens: number;
  max_tokens: number;
}

interface HardwareInfo {
  gpu_name: string | null;
  vram_total_gb: number;
  vram_free_gb: number;
  cuda_available: boolean;
  bnb_available: boolean;
  recommended_method: 'lora' | 'qlora' | null;
  recommendation_reason: string;
  warnings: string[];
}

interface TrainingConfig {
  method: 'lora' | 'qlora';
  num_epochs: number;
  learning_rate: number;
  per_device_batch_size: number;
  gradient_accumulation_steps: number;
  max_seq_length: number;
  lora_r: number;
  lora_alpha: number;
  lora_dropout: number;
}

interface TrainingStatus {
  run_id: string;
  job_id: string;
  status: 'queued' | 'running' | 'success' | 'failed' | 'cancelled';
  progress_pct: number;
  current_metrics?: {
    step: number;
    total_steps: number;
    train_loss: number;
    eval_loss?: number;
    learning_rate: number;
    epoch: number;
  };
  started_at?: string;
  elapsed_seconds?: number;
}

interface Job {
  id: string;
  type: string;
  status: 'queued' | 'running' | 'success' | 'failed' | 'retrying' | 'cancelled';
  progress_pct: number;
  error_message?: string;
  created_at: string;
  started_at?: string;
  finished_at?: string;
}
```

## 8. Client API

```typescript
// frontend/lib/api.ts

const API_BASE = process.env.NEXT_PUBLIC_API_BASE || 'http://localhost:8000';

class ApiClient {
  private base: string;

  constructor(base: string = API_BASE) {
    this.base = base;
  }

  private async request<T>(path: string, options?: RequestInit): Promise<T> {
    const res = await fetch(`${this.base}${path}`, {
      headers: { 'Content-Type': 'application/json', ...options?.headers },
      ...options,
    });
    if (!res.ok) {
      const error = await res.json().catch(() => ({ detail: 'Unknown error' }));
      throw new ApiError(res.status, error.detail, error.error_code);
    }
    return res.json();
  }

  // Projects
  createProject(data: CreateProjectRequest) { return this.request<Project>('/projects', { method: 'POST', body: JSON.stringify(data) }); }
  listProjects() { return this.request<{ items: Project[]; total: number }>('/projects'); }
  getProject(id: string) { return this.request<Project>(`/projects/${id}`); }
  updateProject(id: string, data: Partial<Project>) { return this.request<Project>(`/projects/${id}`, { method: 'PATCH', body: JSON.stringify(data) }); }
  deleteProject(id: string) { return this.request<void>(`/projects/${id}`, { method: 'DELETE' }); }

  // Model
  resolveModel(projectId: string, modelId: string) { return this.request<ModelInfo>(`/projects/${projectId}/model/resolve`, { method: 'POST', body: JSON.stringify({ model_id: modelId }) }); }

  // Files
  async uploadFiles(projectId: string, files: File[], kind: string) {
    const formData = new FormData();
    files.forEach(f => formData.append('files', f));
    formData.append('kind', kind);
    const res = await fetch(`${this.base}/projects/${projectId}/files/upload`, { method: 'POST', body: formData });
    if (!res.ok) throw new ApiError(res.status, 'Upload failed');
    return res.json();
  }
  listFiles(projectId: string) { return this.request<{ files: FileInfo[] }>(`/projects/${projectId}/files`); }
  deleteFile(projectId: string, fileId: string) { return this.request<void>(`/projects/${projectId}/files/${fileId}`, { method: 'DELETE' }); }

  // Dataset
  previewDataset(projectId: string) { return this.request<{ job_id: string }>(`/projects/${projectId}/dataset/preview`, { method: 'POST' }); }
  generateDataset(projectId: string) { return this.request<{ job_id: string }>(`/projects/${projectId}/dataset/generate`, { method: 'POST' }); }
  getDatasetExamples(projectId: string, params?: Record<string, string>) { /* ... */ }
  getDatasetStats(projectId: string) { return this.request<DatasetStats>(`/projects/${projectId}/dataset/stats`); }

  // Hardware
  checkHardware() { return this.request<HardwareInfo>('/hardware/check'); }

  // Training
  startTraining(projectId: string, config: TrainingConfig) { return this.request<{ job_id: string; run_id: string }>(`/projects/${projectId}/train/start`, { method: 'POST', body: JSON.stringify(config) }); }
  getTrainingStatus(projectId: string) { return this.request<TrainingStatus>(`/projects/${projectId}/train/status`); }
  cancelTraining(projectId: string) { return this.request<void>(`/projects/${projectId}/train/cancel`, { method: 'POST' }); }

  // SSE
  streamTrainingLogs(projectId: string, onEvent: (event: any) => void): EventSource {
    const es = new EventSource(`${this.base}/projects/${projectId}/train/logs/stream`);
    es.addEventListener('log', (e) => onEvent({ type: 'log', data: JSON.parse(e.data) }));
    es.addEventListener('eval', (e) => onEvent({ type: 'eval', data: JSON.parse(e.data) }));
    es.addEventListener('progress', (e) => onEvent({ type: 'progress', data: JSON.parse(e.data) }));
    es.addEventListener('complete', (e) => onEvent({ type: 'complete', data: JSON.parse(e.data) }));
    es.addEventListener('error', (e) => onEvent({ type: 'error', data: JSON.parse((e as MessageEvent).data) }));
    return es;
  }

  // Export
  exportArtifacts(projectId: string) { return this.request<{ job_id: string }>(`/projects/${projectId}/export`, { method: 'POST' }); }
  getExportFiles(projectId: string) { return this.request<{ files: any[] }>(`/projects/${projectId}/export/files`); }
  downloadExport(projectId: string) { return `${this.base}/projects/${projectId}/export/download`; }

  // Jobs
  getJob(jobId: string) { return this.request<Job>(`/jobs/${jobId}`); }
  listJobs(projectId: string) { return this.request<{ jobs: Job[] }>(`/projects/${projectId}/jobs`); }
}

export const api = new ApiClient();
```

## 9. Composants réutilisables

| Composant | Usage |
|-----------|-------|
| `JobProgress` | Barre de progression + polling job status |
| `FileUploadZone` | Drag & drop + liste fichiers |
| `DataTable` | Tableau paginé + filtres + sélection |
| `StatusBadge` | Badge coloré selon le statut |
| `LossChart` | Graphique recharts pour les courbes de loss |
| `LogStream` | Affichage temps réel des logs SSE |
| `ErrorToast` | Notification d'erreur |
| `ConfirmDialog` | Dialog de confirmation (delete, cancel) |
| `HyperparamForm` | Formulaire des hyperparamètres avec defaults |
