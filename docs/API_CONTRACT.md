# FineTuneFlow — Contrat API (FastAPI)

> Base URL: `http://localhost:8000`  
> Format: JSON  
> Pas d'authentification (app locale)

## 1. Health

### `GET /health`
Vérifie que le backend + DB + Redis sont up.

**Response 200:**
```json
{
  "status": "ok",
  "db": "connected",
  "redis": "connected",
  "version": "0.1.0"
}
```

---

## 2. Projects

### `POST /projects`
Créer un nouveau projet.

**Request Body:**
```json
{
  "name": "Mon projet Q&A",
  "description": "Fine-tuning sur mes docs internes",
  "task_type": "qa",
  "config": {
    "num_examples_target": 2000,
    "max_seq_length": 2048,
    "train_eval_split": 0.9
  }
}
```

**Response 201:**
```json
{
  "id": "uuid",
  "name": "Mon projet Q&A",
  "description": "...",
  "task_type": "qa",
  "base_model_id": null,
  "model_info": null,
  "status": "draft",
  "config": {...},
  "created_at": "2026-02-14T10:00:00Z",
  "updated_at": "2026-02-14T10:00:00Z"
}
```

### `GET /projects`
Liste tous les projets.

**Query params:**
- `status` (optional): filtrer par statut
- `limit` (default: 50)
- `offset` (default: 0)

**Response 200:**
```json
{
  "items": [
    {
      "id": "uuid",
      "name": "...",
      "task_type": "instruction",
      "status": "training",
      "base_model_id": "meta-llama/Llama-3.1-8B",
      "created_at": "...",
      "updated_at": "...",
      "stats": {
        "file_count": 3,
        "chunk_count": 45,
        "example_count": 1800,
        "last_job_status": "running"
      }
    }
  ],
  "total": 5,
  "limit": 50,
  "offset": 0
}
```

### `GET /projects/{project_id}`
Détail d'un projet.

**Response 200:** Même structure que POST response + `stats` enrichi.

### `PATCH /projects/{project_id}`
Mise à jour partielle d'un projet.

**Request Body (partial):**
```json
{
  "name": "Nouveau nom",
  "task_type": "instruction",
  "config": {"num_examples_target": 3000}
}
```

### `DELETE /projects/{project_id}`
Supprimer un projet et tous ses fichiers/données.

**Response 204:** No content.

---

## 3. Model Resolution

### `POST /projects/{project_id}/model/resolve`
Résoudre un modèle HuggingFace et estimer la VRAM.

**Request Body:**
```json
{
  "model_id": "meta-llama/Llama-3.1-8B"
}
```

**Response 200:**
```json
{
  "model_id": "meta-llama/Llama-3.1-8B",
  "model_type": "llama",
  "num_parameters": 8030000000,
  "estimated_vram_fp16_gb": 16.06,
  "estimated_vram_4bit_gb": 5.52,
  "architecture": "LlamaForCausalLM",
  "vocab_size": 128256,
  "max_position_embeddings": 131072,
  "license": "llama3.1",
  "valid": true,
  "warnings": []
}
```

**Response 404:**
```json
{
  "detail": "Model 'xxx' not found on HuggingFace Hub"
}
```

---

## 4. Files

### `POST /projects/{project_id}/files/upload`
Upload un ou plusieurs fichiers.

**Request:** `multipart/form-data`
- `files`: un ou plusieurs fichiers
- `kind` (form field): `raw_doc` | `dataset_upload`

**Contraintes (validation):**
- Taille max par fichier: 50 MB (configurable)
- Taille max totale par projet: 500 MB
- Types MIME acceptés:
  - `raw_doc`: `application/pdf`, `application/vnd.openxmlformats-officedocument.wordprocessingml.document`, `text/plain`, `text/markdown`
  - `dataset_upload`: `application/json`, `application/jsonl`, `text/csv`, `text/plain`
- Nom de fichier: sanitisé (pas de `..`, pas de `/`, alphanum + `._-` seulement)

**Response 201:**
```json
{
  "files": [
    {
      "id": "uuid",
      "filename": "document.pdf",
      "kind": "raw_doc",
      "status": "ready",
      "mime_type": "application/pdf",
      "size_bytes": 1245678,
      "created_at": "..."
    }
  ]
}
```

**Response 400:**
```json
{
  "detail": "File 'malware.exe' has unsupported MIME type 'application/x-executable'"
}
```

**Response 413:**
```json
{
  "detail": "File exceeds maximum size of 50 MB"
}
```

### `GET /projects/{project_id}/files`
Liste les fichiers d'un projet.

**Query params:**
- `kind` (optional): filtrer par type

**Response 200:**
```json
{
  "files": [
    {
      "id": "uuid",
      "filename": "doc1.pdf",
      "kind": "raw_doc",
      "status": "ready",
      "mime_type": "application/pdf",
      "size_bytes": 1234567,
      "created_at": "..."
    }
  ]
}
```

### `DELETE /projects/{project_id}/files/{file_id}`
Supprimer un fichier.

**Response 204:** No content.

---

## 5. Chunks

### `POST /projects/{project_id}/chunks/generate`
Déclencher le chunking des documents uploadés.

**Request Body (optional):**
```json
{
  "chunk_size_tokens": 512,
  "chunk_overlap_tokens": 50
}
```

**Response 202:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Chunking job enqueued"
}
```

### `GET /projects/{project_id}/chunks`
Liste les chunks.

**Query params:**
- `source_file_id` (optional)
- `limit`, `offset`

**Response 200:**
```json
{
  "items": [
    {
      "id": "uuid",
      "source_file_id": "uuid",
      "chunk_index": 0,
      "content": "Premier paragraphe du document...",
      "token_count": 487,
      "metadata": {"page_start": 1, "page_end": 2}
    }
  ],
  "total": 45
}
```

---

## 6. Dataset

### `POST /projects/{project_id}/dataset/preview`
Lancer la génération de 10 exemples preview.

**Request Body (optional):**
```json
{
  "num_examples": 10,
  "ollama_model": "llama3.1:70b"
}
```

**Response 202:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Dataset preview job enqueued"
}
```

### `POST /projects/{project_id}/dataset/generate`
Lancer la génération complète du dataset.

**Request Body (optional):**
```json
{
  "num_examples_target": 2000,
  "examples_per_chunk": 5,
  "ollama_model": "llama3.1:70b"
}
```

**Response 202:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Dataset generation job enqueued"
}
```

### `GET /projects/{project_id}/dataset/examples`
Récupérer les exemples du dataset (paginé).

**Query params:**
- `split`: `preview` | `train` | `eval` (default: all)
- `is_valid`: `true` | `false` (default: all)
- `limit` (default: 50)
- `offset` (default: 0)

**Response 200:**
```json
{
  "items": [
    {
      "id": "uuid",
      "split": "train",
      "data": {
        "instruction": "Explique le concept de backpropagation.",
        "input": "",
        "output": "La backpropagation est un algorithme..."
      },
      "is_valid": true,
      "token_count": 245,
      "created_at": "..."
    }
  ],
  "total": 1800,
  "stats": {
    "total": 2000,
    "valid": 1800,
    "invalid": 200,
    "train": 1620,
    "eval": 180,
    "avg_token_count": 312
  }
}
```

### `DELETE /projects/{project_id}/dataset/examples/{example_id}`
Supprimer un exemple individuel.

**Response 204:** No content.

### `GET /projects/{project_id}/dataset/stats`
Statistiques du dataset.

**Response 200:**
```json
{
  "total": 2000,
  "valid": 1800,
  "invalid": 200,
  "by_split": {
    "train": {"count": 1620, "avg_tokens": 310, "min_tokens": 45, "max_tokens": 1024},
    "eval": {"count": 180, "avg_tokens": 315, "min_tokens": 52, "max_tokens": 980}
  },
  "validation_errors": {
    "output_too_short": 120,
    "json_invalid": 45,
    "duplicate": 35
  }
}
```

---

## 7. Hardware

### `GET /hardware/check`
Probe hardware de la machine (GPU, CUDA, etc.).

**Response 200:**
```json
{
  "has_nvidia_smi": true,
  "gpu_name": "NVIDIA RTX 4090",
  "gpu_count": 1,
  "vram_total_gb": 24.0,
  "vram_free_gb": 22.5,
  "driver_version": "550.54.14",
  "cuda_runtime": "12.4",
  "torch_version": "2.5.0",
  "torch_cuda": "12.4",
  "cuda_available": true,
  "bnb_available": true,
  "recommended_method": "qlora",
  "recommendation_reason": "24GB VRAM detected. QLoRA recommended for 8B+ models to save memory.",
  "notes": ["All checks passed"],
  "warnings": []
}
```

**Response 200 (pas de GPU):**
```json
{
  "has_nvidia_smi": false,
  "gpu_name": null,
  "cuda_available": false,
  "bnb_available": false,
  "recommended_method": null,
  "recommendation_reason": "No CUDA GPU detected. GPU fine-tuning not possible on this machine.",
  "notes": [],
  "warnings": ["No NVIDIA GPU found. Consider using a cloud GPU instance."]
}
```

---

## 8. Training

### `POST /projects/{project_id}/train/start`
Démarrer le fine-tuning.

**Request Body:**
```json
{
  "method": "qlora",
  "hyperparams": {
    "num_epochs": 3,
    "learning_rate": 2e-4,
    "per_device_batch_size": 4,
    "gradient_accumulation_steps": 4,
    "max_seq_length": 2048,
    "lora_r": 16,
    "lora_alpha": 32,
    "lora_dropout": 0.05,
    "warmup_ratio": 0.03,
    "weight_decay": 0.01
  }
}
```

**Response 202:**
```json
{
  "job_id": "uuid",
  "run_id": "uuid",
  "status": "queued",
  "message": "Training job enqueued on gpu_queue"
}
```

**Response 409:**
```json
{
  "detail": "A training job is already running for this project"
}
```

### `GET /projects/{project_id}/train/status`
Statut du training en cours ou dernier.

**Response 200:**
```json
{
  "run_id": "uuid",
  "job_id": "uuid",
  "status": "running",
  "method": "qlora",
  "progress_pct": 45,
  "current_metrics": {
    "step": 450,
    "total_steps": 1000,
    "train_loss": 0.52,
    "eval_loss": 0.48,
    "learning_rate": 1.8e-4,
    "epoch": 1.35
  },
  "started_at": "2026-02-14T10:30:00Z",
  "elapsed_seconds": 3600
}
```

### `GET /projects/{project_id}/train/logs/stream`
Stream SSE des logs de training en temps réel.

**Response:** `text/event-stream`
```
event: log
data: {"step": 1, "train_loss": 2.45, "learning_rate": 2e-4, "epoch": 0.003}

event: log
data: {"step": 2, "train_loss": 2.31, "learning_rate": 2e-4, "epoch": 0.006}

event: progress
data: {"progress_pct": 1, "message": "Step 2/1000"}

event: eval
data: {"step": 100, "eval_loss": 1.85, "perplexity": 6.36}

event: checkpoint
data: {"step": 500, "path": "checkpoint-500"}

event: complete
data: {"status": "success", "duration_seconds": 7200, "final_train_loss": 0.32}

event: error
data: {"status": "failed", "error": "CUDA out of memory"}
```

### `POST /projects/{project_id}/train/cancel`
Annuler le training en cours.

**Response 200:**
```json
{
  "status": "cancelled",
  "message": "Training job cancelled"
}
```

---

## 9. Export

### `POST /projects/{project_id}/export`
Lancer l'export des artifacts.

**Response 202:**
```json
{
  "job_id": "uuid",
  "status": "queued",
  "message": "Export job enqueued"
}
```

### `GET /projects/{project_id}/export/files`
Liste les fichiers exportés.

**Response 200:**
```json
{
  "files": [
    {"filename": "adapter_model.safetensors", "size_bytes": 123456},
    {"filename": "adapter_config.json", "size_bytes": 456},
    {"filename": "training_config.json", "size_bytes": 789},
    {"filename": "metrics.json", "size_bytes": 234},
    {"filename": "report.md", "size_bytes": 5678},
    {"filename": "finetuneflow_export.zip", "size_bytes": 130000}
  ]
}
```

### `GET /projects/{project_id}/export/download/{filename}`
Télécharger un fichier export.

**Response 200:** Binary file download.

### `GET /projects/{project_id}/export/download`
Télécharger le zip complet.

**Response 200:** Binary file download (`finetuneflow_export.zip`).

---

## 10. Jobs

### `GET /projects/{project_id}/jobs`
Liste les jobs d'un projet.

**Query params:**
- `type` (optional): filtrer par type
- `status` (optional): filtrer par statut

**Response 200:**
```json
{
  "jobs": [
    {
      "id": "uuid",
      "type": "dataset_generate",
      "status": "success",
      "progress_pct": 100,
      "result_summary": {"total_generated": 2000, "valid": 1800},
      "created_at": "...",
      "started_at": "...",
      "finished_at": "...",
      "error_message": null
    }
  ]
}
```

### `GET /jobs/{job_id}`
Détail d'un job.

### `POST /jobs/{job_id}/cancel`
Annuler un job en cours.

**Response 200:**
```json
{
  "status": "cancelled"
}
```

---

## 11. Codes d'erreur standards

| Code | Signification |
|------|---------------|
| 400 | Bad request (validation échouée) |
| 404 | Resource not found |
| 409 | Conflict (ex: training déjà en cours) |
| 413 | Payload too large (upload) |
| 422 | Validation error (Pydantic) |
| 429 | Rate limit exceeded |
| 500 | Internal server error |
| 503 | Service unavailable (DB ou Redis down) |

**Format d'erreur standard:**
```json
{
  "detail": "Human-readable error message",
  "error_code": "TRAINING_ALREADY_RUNNING",
  "context": {}
}
```

## 12. Rate Limiting

| Endpoint pattern | Limite |
|-----------------|--------|
| `POST /*/upload` | 10 req/min |
| `POST /*/generate` | 5 req/min |
| `POST /*/train/*` | 3 req/min |
| `GET /*` | 60 req/min |
| Autres `POST` | 30 req/min |
